#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply system-design updates to PIRN-CMPT/CMDIAD-main.

This patcher is intentionally text-based so it can be applied to the current
GitHub repository without requiring a full fork from this environment. It adds:
  1. explicit K-shot/few-shot training split support;
  2. true missing-modality test loading for RGB-only or SN-only inputs;
  3. paper-aligned MNC default switch;
  4. clear ablation switches for CMPT / shared prototypes / APR / MNC / pseudo prototypes.

Usage:
    cd PIRN-CMPT/CMDIAD-main
    python tools/apply_system_design_updates.py .

The script creates *.bak_system_design backups before editing.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


MARK = "# === PIRN-CMPT system-design update ==="
END_MARK = "# === end PIRN-CMPT system-design update ==="


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def backup(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".bak_system_design")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def ensure_exists(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {relative} under {root}")
    return path


def already_patched(text: str, token: str) -> bool:
    return token in text


def insert_after(text: str, anchor: str, addition: str, *, label: str) -> str:
    if addition.strip() in text:
        return text
    idx = text.find(anchor)
    if idx < 0:
        raise RuntimeError(f"Anchor not found while inserting {label}: {anchor[:120]!r}")
    idx += len(anchor)
    return text[:idx] + addition + text[idx:]


def insert_before(text: str, anchor: str, addition: str, *, label: str) -> str:
    if addition.strip() in text:
        return text
    idx = text.find(anchor)
    if idx < 0:
        raise RuntimeError(f"Anchor not found while inserting {label}: {anchor[:120]!r}")
    return text[:idx] + addition + text[idx:]


def regex_sub_once(text: str, pattern: str, replacement: str, *, label: str, flags: int = re.DOTALL) -> str:
    new_text, n = re.subn(pattern, replacement, text, count=1, flags=flags)
    if n == 0:
        raise RuntimeError(f"Pattern not found while patching {label}: {pattern[:160]!r}")
    return new_text



def insert_after_call_block(text: str, start_needle: str, addition: str, *, label: str) -> str:
    """Insert after a possibly multi-line function call block such as parser.add_argument(...)."""
    if addition.strip() and addition.strip() in text:
        return text
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if start_needle in line:
            indent = re.match(r"\s*", line).group(0)
            j = i
            balance = line.count('(') - line.count(')')
            while balance > 0 and j + 1 < len(lines):
                j += 1
                balance += lines[j].count('(') - lines[j].count(')')
            # Normalize addition indentation to the argument line indentation.
            add = addition
            if not add.startswith('\n'):
                add = '\n' + add
            lines.insert(j + 1, add)
            return ''.join(lines)
    raise RuntimeError(f"Call block starting with {start_needle!r} not found while inserting {label}")

def patch_main(path: Path) -> None:
    backup(path)
    text = read_text(path)

    fewshot_args = """
    parser.add_argument('--few_shot_k', default=0, type=int,
                        help='K-shot normal samples per class. If > 0, the training split is deterministically sampled to K normal samples.')
    parser.add_argument('--shot_seed', default=0, type=int,
                        help='Random seed used for deterministic few-shot support-set sampling.')
    parser.add_argument('--save_fewshot_list', default=False, action='store_true',
                        help='Save the selected K-shot support-set file list for reproducibility.')
    parser.add_argument('--fewshot_list_dir', default='fewshot_splits', type=str,
                        help='Directory used to save few-shot support-set manifests.')
    parser.add_argument('--allow_true_missing_modality', default=False, action='store_true',
                        help='For test split, allow RGB-only or SN-only files instead of forcing paired RGB/XYZ samples.')
"""
    if "--few_shot_k" not in text:
        text = insert_after_call_block(
            text,
            "parser.add_argument('--max_sample'",
            fewshot_args,
            label="few-shot argparse options",
        )

    ablation_args = """
    parser.add_argument('--paper_mnc', default=False, action='store_true',
                        help='Use the thesis/paper two-stage MNC path: mnc_strong=True and at least two stages.')
    parser.add_argument('--disable_cmpt', default=False, action='store_true',
                        help='Ablation: disable CMPT training and pseudo-modality scoring.')
    parser.add_argument('--disable_shared_proto', default=False, action='store_true',
                        help='Ablation: remove shared prototypes from prototype banks.')
    parser.add_argument('--disable_apr', default=False, action='store_true',
                        help='Ablation: use raw k-means prototypes without APR refinement/update.')
    parser.add_argument('--disable_mnc', default=False, action='store_true',
                        help='Ablation: disable MNC cross-modal normality communication.')
    parser.add_argument('--disable_pseudo_proto', default=False, action='store_true',
                        help='Ablation: disable pseudo prototypes and pseudo-modality scoring.')
"""
    if "--disable_cmpt" not in text:
        text = insert_after_call_block(
            text,
            "parser.add_argument('--mnc_pseudo_loss_weight'",
            ablation_args,
            label="ablation argparse options",
        )

    post_parse = f"""
    {MARK}
    # Align the default PIRN_CMPT command with the thesis diagram: SPA -> MNC stage1 -> MNC stage2.
    # Baseline/ablation runs can still disable this path with --disable_mnc.
    if args.paper_mnc or args.method_name in ['PIRN_CMPT', 'CMPT']:
        if not args.disable_mnc:
            args.mnc_strong = True
            args.mnc_stages = max(2, args.mnc_stages)
    if args.few_shot_k and args.few_shot_k > 0:
        # The dataset itself is sampled to K items; this prevents the old max_sample loop
        # from truncating the K-shot support set accidentally.
        args.max_sample = max(args.max_sample, args.few_shot_k)
    {END_MARK}
"""
    if MARK not in text:
        text = insert_after(text, "args = parser.parse_args()", post_parse, label="post-parse system config")

    write_text(path, text)


def patch_cmdiad_runner(path: Path) -> None:
    backup(path)
    text = read_text(path)
    if "few_shot_k" not in text:
        text = text.replace(
            "self.count = args.max_sample",
            "self.count = args.few_shot_k if getattr(args, 'few_shot_k', 0) else args.max_sample",
            1,
        )
    write_text(path, text)


DATASET_HELPERS = r'''

# === PIRN-CMPT system-design update ===
def _path_stem(path):
    return Path(path).stem if path is not None else ''


def _stable_fewshot_indices(total, k, seed, class_name):
    """Deterministically sample K indices for the support set."""
    import random
    if k <= 0 or total <= k:
        return list(range(total))
    salt = sum(ord(ch) for ch in str(class_name))
    rng = random.Random(int(seed) + salt)
    indices = list(range(total))
    rng.shuffle(indices)
    return sorted(indices[:k])


def _save_fewshot_manifest(dataset, indices, args, split):
    if not getattr(args, 'save_fewshot_list', False):
        return
    out_dir = Path(getattr(args, 'fewshot_list_dir', 'fewshot_splits'))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset.cls}_{split}_K{args.few_shot_k}_seed{args.shot_seed}.csv"
    with out_path.open('w', encoding='utf-8') as f:
        f.write('rank,original_index,rgb_path,xyz_path,label\n')
        for rank, idx in enumerate(indices):
            item = dataset.img_paths[idx]
            if isinstance(item, (tuple, list)):
                rgb_path = item[0] if len(item) > 0 else ''
                xyz_path = item[1] if len(item) > 1 else ''
            else:
                rgb_path, xyz_path = item, ''
            label = dataset.labels[idx] if hasattr(dataset, 'labels') else 0
            f.write(f'{rank},{idx},{rgb_path},{xyz_path},{label}\n')
    print(f'[FewShot] Saved support set manifest to {out_path}')


def apply_fewshot_subset(dataset, args, split):
    """Apply an explicit K-shot support set to train/train_validation datasets."""
    k = int(getattr(args, 'few_shot_k', 0) or 0)
    if split not in ['train', 'train_validation'] or k <= 0:
        return dataset
    total = len(dataset.img_paths)
    if total == 0:
        return dataset
    indices = _stable_fewshot_indices(total, k, getattr(args, 'shot_seed', 0), dataset.cls)
    _save_fewshot_manifest(dataset, indices, args, split)
    dataset.img_paths = [dataset.img_paths[i] for i in indices]
    dataset.labels = [dataset.labels[i] for i in indices]
    print(f'[FewShot] Class {dataset.cls}: using {len(indices)}/{total} normal training samples '
          f'(K={k}, seed={getattr(args, "shot_seed", 0)}).')
    return dataset


class MissingModalityTestDataset(TestDataset):
    """Test dataset that supports physically missing RGB or XYZ/SN files.

    The original TestDataset zips RGB and XYZ files, so if one modality is absent
    the sample disappears. This subclass keeps the available modality as the
    primary key and fills the missing modality with zeros. The model will only
    consume the available branch when --main_modality is rgb or sn.
    """

    def __init__(self, class_name, rgb_size, xyz_size, gt_size, dataset_path, img_process_method, args):
        self.args = args
        super().__init__(class_name, rgb_size, xyz_size, gt_size, dataset_path, img_process_method)

    @staticmethod
    def _dict_by_stem(paths):
        return {Path(path).stem: path for path in paths}

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        main_modality = getattr(self.args, 'main_modality', '')
        if main_modality not in {'rgb', 'sn'}:
            return super().load_dataset()

        defect_types = os.listdir(self.img_path)
        for defect_type in defect_types:
            rgb_paths = sorted(Path(self.img_path, defect_type, 'rgb').glob('*.png'))
            tiff_paths = sorted(Path(self.img_path, defect_type, 'xyz').glob('*.tiff'))
            gt_paths = sorted(Path(self.img_path, defect_type, 'gt').glob('*.png')) if defect_type != 'good' else []
            rgb_by_stem = self._dict_by_stem(rgb_paths)
            tiff_by_stem = self._dict_by_stem(tiff_paths)
            gt_by_stem = self._dict_by_stem(gt_paths)

            primary_paths = rgb_paths if main_modality == 'rgb' else tiff_paths
            for primary in primary_paths:
                stem = Path(primary).stem
                rgb_path = rgb_by_stem.get(stem)
                tiff_path = tiff_by_stem.get(stem)
                # Keep the available modality even when the other modality file is absent.
                if main_modality == 'rgb' and rgb_path is None:
                    continue
                if main_modality == 'sn' and tiff_path is None:
                    continue
                img_tot_paths.append((rgb_path, tiff_path))
                if defect_type == 'good':
                    gt_tot_paths.append(0)
                    tot_labels.append(0)
                else:
                    gt_tot_paths.append(gt_by_stem.get(stem, 0))
                    tot_labels.append(1)

        if not img_tot_paths:
            raise FileNotFoundError(
                f'No samples found for true missing-modality test. '
                f'class={self.cls}, main_modality={main_modality}, split_root={self.img_path}'
            )
        return img_tot_paths, gt_tot_paths, tot_labels

    def __getitem__(self, idx):
        import torch
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path, tiff_path = img_path
        ref_path = str(rgb_path if rgb_path is not None else tiff_path)

        if rgb_path is not None:
            img_original = Image.open(rgb_path).convert('RGB')
            img = self.rgb_transform(img_original)
        else:
            img = torch.zeros(3, self.rgb_size, self.rgb_size, dtype=torch.float32)

        if tiff_path is not None:
            organized_pc = read_tiff_organized_pc(tiff_path)
            depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
            resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
            resized_organized_pc = resize_organized_pc(
                organized_pc, target_height=self.xyz_size, target_width=self.xyz_size
            ).clone().detach().float()
        else:
            resized_organized_pc = torch.zeros(3, self.xyz_size, self.xyz_size, dtype=torch.float32)
            resized_depth_map_3channel = torch.zeros(3, self.gt_size, self.gt_size, dtype=torch.float32)

        if gt == 0:
            gt_tensor = torch.zeros([1, self.gt_size, self.gt_size], dtype=torch.float32)
        else:
            gt_img = Image.open(gt).convert('L')
            gt_tensor = self.gt_transform(gt_img)
            gt_tensor = torch.where(gt_tensor > 0.5, 1., .0)
        return (img, resized_organized_pc, resized_depth_map_3channel), gt_tensor[:1], label, ref_path
# === end PIRN-CMPT system-design update ===
'''


def patch_dataset(path: Path) -> None:
    backup(path)
    text = read_text(path)
    if "MissingModalityTestDataset" not in text:
        text = insert_before(text, "def get_data_loader", DATASET_HELPERS, label="dataset helpers")

    if "dataset = apply_fewshot_subset(dataset, args, split)" not in text:
        text = regex_sub_once(
            text,
            r"(elif\s+split\s+in\s+\['test'\]:\s*dataset\s*=\s*TestDataset\([^\n]*?\)\s*else:\s*raise\s+ValueError)",
            "elif split in ['test']:\n"
            "        if getattr(args, 'allow_true_missing_modality', False) and getattr(args, 'main_modality', '') in {'rgb', 'sn'}:\n"
            "            dataset = MissingModalityTestDataset(class_name=class_name, rgb_size=rgb_size, xyz_size=xyz_size, gt_size=gt_size, dataset_path=args.dataset_path, img_process_method=args.img_process_method, args=args)\n"
            "        else:\n"
            "            dataset = TestDataset(class_name=class_name, rgb_size=rgb_size, xyz_size=xyz_size, gt_size=gt_size, dataset_path=args.dataset_path, img_process_method=args.img_process_method)\n"
            "    else:\n"
            "        raise ValueError",
            label="missing-modality test loader",
            flags=re.DOTALL,
        )
        text = regex_sub_once(
            text,
            r"(data_loader\s*=\s*DataLoader\(dataset=dataset,)",
            "dataset = apply_fewshot_subset(dataset, args, split)\n    \\1",
            label="few-shot dataset hook",
            flags=re.DOTALL,
        )

    write_text(path, text)


def _insert_after_line_contains(text: str, needle: str, addition_lines: list[str], *, label: str) -> str:
    """Insert lines after the first line containing needle, preserving indentation."""
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if needle in line:
            indent = re.match(r"\s*", line).group(0)
            body_indent = indent + "    "
            addition = "".join(body_indent + item + "\n" for item in addition_lines)
            lines.insert(i + 1, addition)
            return "".join(lines)
    raise RuntimeError(f"Line containing {needle!r} not found while patching {label}")


def _insert_before_line_contains(text: str, needle: str, addition_lines: list[str], *, label: str) -> str:
    """Insert lines before the first line containing needle, preserving indentation."""
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if needle in line:
            indent = re.match(r"\s*", line).group(0)
            addition = "".join(indent + item + "\n" for item in addition_lines)
            lines.insert(i, addition)
            return "".join(lines)
    raise RuntimeError(f"Line containing {needle!r} not found while patching {label}")


def _insert_after_assignment_block(text: str, assignment_start: str, addition_lines: list[str], *, label: str) -> str:
    """Insert after a single-line or simple multi-line assignment block."""
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if assignment_start in line:
            indent = re.match(r"\s*", line).group(0)
            j = i
            paren_balance = line.count('(') - line.count(')')
            while paren_balance > 0 and j + 1 < len(lines):
                j += 1
                paren_balance += lines[j].count('(') - lines[j].count(')')
            addition = "".join(indent + item + "\n" for item in addition_lines)
            lines.insert(j + 1, addition)
            return "".join(lines)
    raise RuntimeError(f"Assignment {assignment_start!r} not found while patching {label}")


def patch_cmpt_features(path: Path) -> None:
    backup(path)
    text = read_text(path)

    if "[Ablation] CMPT disabled" not in text:
        text = _insert_after_line_contains(
            text,
            "def _fit_cmpt_if_needed(self):",
            [
                "if getattr(self.args, 'disable_cmpt', False):",
                "    self.cmpt.eval()",
                "    self.cmpt_trained = True",
                "    print('[Ablation] CMPT disabled: pseudo-modality transfer/scoring will be skipped.')",
                "    return",
            ],
            label="disable CMPT hook",
        )

    if "disable_apr" not in text:
        text = _insert_before_line_contains(
            text,
            "train_tokens = sample_tokens",
            [
                "if getattr(self.args, 'disable_apr', False):",
                "    return prototypes",
            ],
            label="disable APR hook",
        )

    if "disable_shared_proto" not in text:
        text = _insert_after_assignment_block(
            text,
            "self.shared_prototypes = self._build_single_prototype_set",
            [
                "if getattr(self.args, 'disable_shared_proto', False):",
                "    self.shared_prototypes = None",
                "    print('[Ablation] Shared prototypes disabled.')",
            ],
            label="disable shared prototypes",
        )
        text = text.replace(
            "return torch.cat(prototypes, dim=0)",
            "return torch.cat([p for p in prototypes if p is not None], dim=0)",
        )

    if "disable_pseudo_proto" not in text:
        text = _insert_after_line_contains(
            text,
            "def _transfer_prototypes(self, source_prototypes, target_prototypes, out_type):",
            [
                "if getattr(self.args, 'disable_cmpt', False) or getattr(self.args, 'disable_pseudo_proto', False):",
                "    return torch.nn.functional.normalize(source_prototypes, dim=-1)",
            ],
            label="disable pseudo prototypes hook",
        )

    if "disable_mnc" not in text:
        text = _insert_before_line_contains(
            text,
            "if cross_modal is not None:",
            [
                "if getattr(self.args, 'disable_mnc', False):",
                "    cross_modal = None",
            ],
            label="disable MNC hook",
        )

    if "CMPT_OR_PSEUDO_DISABLED" not in text:
        text = _insert_after_line_contains(
            text,
            "if self.args.main_modality == 'rgb':",
            [
                "CMPT_OR_PSEUDO_DISABLED = getattr(self.args, 'disable_cmpt', False) or getattr(self.args, 'disable_pseudo_proto', False)",
                "if CMPT_OR_PSEUDO_DISABLED:",
                "    _, rgb_patch56 = self._extract_rgb_patches(sample)",
                "    s_rgb, smap_rgb = self._reconstruct_with_prototypes(rgb_patch56, 'rgb', cross_modal=None)",
                "    return s_rgb, smap_rgb, {'rgb_error': smap_rgb, 'fused': smap_rgb}",
            ],
            label="RGB pseudo/CMPT disabled scoring hook",
        )
        text = _insert_after_line_contains(
            text,
            "elif self.args.main_modality == 'sn':",
            [
                "CMPT_OR_PSEUDO_DISABLED = getattr(self.args, 'disable_cmpt', False) or getattr(self.args, 'disable_pseudo_proto', False)",
                "if CMPT_OR_PSEUDO_DISABLED:",
                "    _, sn_patch56 = self._extract_sn_patches(sample)",
                "    s_sn, smap_sn = self._reconstruct_with_prototypes(sn_patch56, 'sn', cross_modal=None)",
                "    smap_sn = self._mask_sn_error_map(smap_sn, sample)",
                "    s_sn = self._score_from_map(smap_sn)",
                "    return s_sn, smap_sn, {'sn_error': smap_sn, 'fused': smap_sn}",
            ],
            label="SN pseudo/CMPT disabled scoring hook",
        )

    write_text(path, text)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('repo_root', nargs='?', default='.', help='Path to CMDIAD-main or PIRN-CMPT root')
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    if (root / 'CMDIAD-main').exists():
        root = root / 'CMDIAD-main'

    files = {
        'main': ensure_exists(root, 'main.py'),
        'runner': ensure_exists(root, 'cmdiad_runner.py'),
        'dataset': ensure_exists(root, 'dataset.py'),
        'cmpt': ensure_exists(root, 'feature_extractors/cmpt_features.py'),
    }

    patch_main(files['main'])
    patch_cmdiad_runner(files['runner'])
    patch_dataset(files['dataset'])
    patch_cmpt_features(files['cmpt'])

    print('[OK] Applied PIRN-CMPT system-design code updates.')
    print('Backups were saved as *.bak_system_design next to edited files.')
    print('Run `python main.py --help | grep -E "few_shot|disable_|paper_mnc|missing"` to verify new switches.')


if __name__ == '__main__':
    main()
