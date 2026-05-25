import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
try:
    import timm
except ImportError as exc:
    raise ImportError(
        'CMPT uses DINOv2 through timm, but timm is not installed in this Python environment. '
        'Install the project dependencies with `pip install -r requirements.txt` or at least '
        '`pip install timm==0.9.12`.'
    ) from exc

from feature_extractors.features import Features
from models.pirn_modules import (
    adaptive_prototype_refinement,
    adaptive_prototype_memory_update,
    balanced_sinkhorn_assignment,
    gated_cross_modal_reconstruction,
    kmeans_prototypes,
    multi_stage_normality_communication,
    reconstruction_error_map,
    sample_tokens,
    structured_prototype_assignment,
)
from utils.au_pro_util import calculate_au_pro
from utils.mvtec3d_util import organized_pc_to_surface_normal_map, resize_organized_pc
from utils.utils import KNNGaussianBlur


def resolve_checkpoint_path(checkpoint_path, role):
    def _is_candidate(path):
        name = path.name.lower()
        return (
            path.is_file()
            and ('dinov2' in name or 'dino_v2' in name or 'vit_base_patch14' in name or name in {'model.pth', 'pytorch_model.bin', 'model.safetensors'})
            and path.suffix.lower() in {'.pth', '.pt', '.bin', '.safetensors'}
        )

    if checkpoint_path == '':
        search_roots = [
            Path('checkpoints'),
            Path('weights'),
            Path('pretrained'),
            Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints',
            Path.home() / '.cache' / 'huggingface' / 'hub',
            Path('/root/autodl-tmp'),
        ]
        for root in search_roots:
            if not root.exists():
                continue
            for path in root.rglob('*'):
                if _is_candidate(path):
                    print(f'[CMPT] Auto-found {role} checkpoint `{path}`.')
                    return str(path)
        return ''

    path = Path(checkpoint_path).expanduser()
    if path.is_file():
        return str(path)

    if path.is_dir():
        candidates = [
            'dinov2_vitb14_pretrain.pth',
            'vit_base_patch14_dinov2.lvd142m.pth',
            'vit_base_patch14_dinov2_lvd142m.pth',
            'model.pth',
            'pytorch_model.bin',
            'model.safetensors',
        ]
        for name in candidates:
            candidate = path / name
            if candidate.is_file():
                print(f'[CMPT] Using {role} checkpoint `{candidate}`.')
                return str(candidate)

        files = sorted([p.name for p in path.iterdir() if p.is_file()])
        raise FileNotFoundError(
            f'No supported {role} checkpoint was found in directory `{path}`. '
            f'Files in this directory: {files[:20]}. '
            f'Pass the exact checkpoint file path with --{role}_checkpoint_path.'
        )

    raise FileNotFoundError(
        f'{role} checkpoint path `{checkpoint_path}` does not exist. '
        f'Check the path with `ls -lh {path.parent}` or omit the checkpoint path for a smoke test. '
        f'For real experiments, download/copy DINOv2 ViT-B/14 weights and pass the actual file path.'
    )


def load_vit_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        for key in ('model', 'state_dict', 'teacher'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(f'Unsupported checkpoint format at `{checkpoint_path}`.')

    state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        if key.startswith('backbone.'):
            key = key[len('backbone.'):]
        if key.startswith('teacher.'):
            key = key[len('teacher.'):]
        state_dict[key] = value

    incompatible = model.load_state_dict(state_dict, strict=False)
    print(f'[CMPT] Loaded checkpoint `{checkpoint_path}` with strict=False: {incompatible}')


class FrozenViTFeatureExtractor(nn.Module):
    def __init__(self, model_name, img_size, checkpoint_path='', allow_pretrained_download=False):
        super().__init__()
        use_checkpoint = checkpoint_path != ''
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=allow_pretrained_download and not use_checkpoint,
            img_size=img_size,
        )
        if use_checkpoint:
            load_vit_checkpoint(self.model, checkpoint_path)
        if not use_checkpoint and not allow_pretrained_download:
            print(f'[CMPT] {model_name} initialized without pretrained weights. '
                  f'Pass --rgb_checkpoint_path/--sn_checkpoint_path for real experiments, '
                  f'or --allow_pretrained_download to let timm download weights.')
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def _patch_grid(self, x):
        patch_size = self.model.patch_embed.patch_size
        if isinstance(patch_size, int):
            patch_h = patch_w = patch_size
        else:
            patch_h, patch_w = patch_size
        return x.shape[-2] // patch_h, x.shape[-1] // patch_w

    def _tokens_to_feature_map(self, features, x):
        if isinstance(features, dict):
            if 'x_norm_patchtokens' in features:
                patch_tokens = features['x_norm_patchtokens']
            elif 'patch_tokens' in features:
                patch_tokens = features['patch_tokens']
            else:
                token_values = [value for value in features.values() if torch.is_tensor(value) and value.ndim == 3]
                if not token_values:
                    raise RuntimeError(f'Cannot find patch tokens in keys: {features.keys()}')
                patch_tokens = token_values[-1]
        else:
            patch_tokens = features

        grid_h, grid_w = self._patch_grid(x)
        patch_count = grid_h * grid_w
        patch_tokens = patch_tokens[:, -patch_count:, :]
        return patch_tokens.transpose(1, 2).contiguous().view(x.shape[0], -1, grid_h, grid_w)

    def forward(self, x):
        self.model.eval()
        return self._tokens_to_feature_map(self.model.forward_features(x), x)


class CrossModalFeatureTransfer(nn.Module):
    def __init__(self, dim=768, hidden_ratio=2.5, mlp_depth=2):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.rgb_norm = nn.LayerNorm(dim)
        self.sn_norm = nn.LayerNorm(dim)
        self.rgb_to_sn = self._mlp(dim, hidden, dim, mlp_depth)
        self.sn_to_rgb = self._mlp(dim, hidden, dim, mlp_depth)

    @staticmethod
    def _mlp(dim, hidden, out_dim, depth):
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(dim if i == 0 else hidden, hidden))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden, out_dim))
        return nn.Sequential(*layers)

    def forward(self, rgb_feature=None, sn_feature=None, out_type='train'):
        if out_type == 'sn':
            return self.rgb_to_sn(self.rgb_norm(rgb_feature))
        if out_type == 'rgb':
            return self.sn_to_rgb(self.sn_norm(sn_feature))
        return self.rgb_to_sn(self.rgb_norm(rgb_feature)), self.sn_to_rgb(self.sn_norm(sn_feature))


class LearnableMNCDecoder(nn.Module):
    def __init__(self, dim=768, hidden_ratio=2.0, num_heads=8, dropout=0.0):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.rgb_to_sn = self._make_branch(dim, hidden, num_heads, dropout)
        self.sn_to_rgb = self._make_branch(dim, hidden, num_heads, dropout)
        self.rgb_to_pseudo_sn = self._make_branch(dim, hidden, num_heads, dropout)
        self.sn_to_pseudo_rgb = self._make_branch(dim, hidden, num_heads, dropout)

    @staticmethod
    def _make_branch(dim, hidden, num_heads, dropout):
        return nn.ModuleDict({
            'query_norm': nn.LayerNorm(dim),
            'memory_norm': nn.LayerNorm(dim),
            'cross_attn': nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            ),
            'gate': nn.Sequential(
                nn.LayerNorm(dim * 3),
                nn.Linear(dim * 3, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
                nn.Sigmoid(),
            ),
            'decoder': nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, dim),
            ),
        })

    @staticmethod
    def _select_branch(modal, cross_modal):
        if modal == 'rgb' and cross_modal == 'sn':
            return 'rgb_to_sn'
        if modal == 'sn' and cross_modal == 'rgb':
            return 'sn_to_rgb'
        if modal == 'rgb' and cross_modal == 'pseudo_sn':
            return 'rgb_to_pseudo_sn'
        if modal == 'pseudo_sn' and cross_modal == 'rgb':
            return 'rgb_to_pseudo_sn'
        if modal == 'sn' and cross_modal == 'pseudo_rgb':
            return 'sn_to_pseudo_rgb'
        if modal == 'pseudo_rgb' and cross_modal == 'sn':
            return 'sn_to_pseudo_rgb'
        raise ValueError(f'Unsupported learnable MNC route: {modal} <- {cross_modal}')

    def forward(self, tokens, own_reconstruction, cross_memory, modal, cross_modal):
        branch = self[self._select_branch(modal, cross_modal)]
        query = branch['query_norm'](own_reconstruction)
        memory = branch['memory_norm'](cross_memory)
        context, _ = branch['cross_attn'](query, memory, memory, need_weights=False)
        gate_input = torch.cat([tokens, own_reconstruction, context], dim=-1)
        gate = branch['gate'](gate_input)
        communicated = own_reconstruction + gate * context
        decoded = branch['decoder'](communicated)
        return F.normalize(own_reconstruction + decoded, dim=-1)

    def __getitem__(self, key):
        return getattr(self, key)


class CMPTFeatures(Features):
    def __init__(self, args):
        args.skip_deep_feature_extractor = True
        super().__init__(args)

        sn_backbone_name = getattr(args, 'sn_backbone_name', args.rgb_backbone_name)
        rgb_checkpoint_path = resolve_checkpoint_path(args.rgb_checkpoint_path, 'rgb')
        sn_checkpoint_arg = getattr(args, 'sn_checkpoint_path', args.rgb_checkpoint_path)
        sn_checkpoint_path = resolve_checkpoint_path(sn_checkpoint_arg, 'sn')
        allow_download = getattr(args, 'allow_pretrained_download', False)
        self.rgb_encoder = FrozenViTFeatureExtractor(
            args.rgb_backbone_name,
            args.rgb_size,
            rgb_checkpoint_path,
            allow_pretrained_download=allow_download,
        ).to(self.device)
        self.sn_encoder = FrozenViTFeatureExtractor(
            sn_backbone_name,
            args.rgb_size,
            sn_checkpoint_path,
            allow_pretrained_download=allow_download,
        ).to(self.device)

        self.cmpt = CrossModalFeatureTransfer(dim=args.feature_dim, hidden_ratio=args.cmpt_hidden_ratio,
                                              mlp_depth=args.cmpt_mlp_depth).to(self.device)
        if args.cmpt_checkpoint_path:
            ckpt = torch.load(args.cmpt_checkpoint_path, map_location=self.device)
            state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
            incompatible = self.cmpt.load_state_dict(state_dict, strict=False)
            print('[CMPT]', incompatible)
            self.cmpt.eval()
        else:
            self.cmpt.train()
        self.cmpt_trained = bool(args.cmpt_checkpoint_path)

        self.learnable_mnc = LearnableMNCDecoder(
            dim=args.feature_dim,
            hidden_ratio=args.mnc_decoder_hidden_ratio,
            num_heads=args.mnc_num_heads,
            dropout=args.mnc_dropout,
        ).to(self.device)
        self.use_learnable_mnc = bool(args.mnc_learnable or args.mnc_checkpoint_path)
        if args.mnc_checkpoint_path:
            ckpt = torch.load(args.mnc_checkpoint_path, map_location=self.device)
            if isinstance(ckpt, dict):
                state_dict = ckpt.get('mnc', ckpt.get('model', ckpt))
            else:
                state_dict = ckpt
            incompatible = self.learnable_mnc.load_state_dict(state_dict, strict=False)
            print('[MNC]', incompatible)
            self.learnable_mnc.eval()
        else:
            self.learnable_mnc.train()
        self.mnc_trained = bool(args.mnc_checkpoint_path)
        if self.use_learnable_mnc and args.main_modality in {'rgb', 'sn'} and not args.mnc_train_pseudo and not args.mnc_checkpoint_path:
            print('[MNC] Missing-modality learnable MNC is enabled without --mnc_train_pseudo. '
                  'Pseudo routes will be weakly trained; add --mnc_train_pseudo for missing-modality experiments.')

        self.rgb_specific_prototypes = None
        self.sn_specific_prototypes = None
        self.shared_prototypes = None
        self.pseudo_sn_prototypes = None
        self.pseudo_rgb_prototypes = None
        self.prototype_ready = False

        self.patch_rgb56_train = []
        self.patch_sn56_train = []
        self.sn_foreground_masks56_train = []
        self.cmpt_feature_pool = torch.nn.AdaptiveAvgPool2d(
            (args.cmpt_feature_grid, args.cmpt_feature_grid)
        )
        self.blur = KNNGaussianBlur(4)
        self.saved_heatmaps = 0

    @staticmethod
    def _normalize_map(score_map):
        score_map = np.asarray(score_map, dtype=np.float32)
        min_value = float(score_map.min())
        max_value = float(score_map.max())
        if max_value - min_value < 1e-12:
            return np.zeros_like(score_map, dtype=np.float32)
        return (score_map - min_value) / (max_value - min_value)

    @staticmethod
    def _jet_colormap(score_map):
        x = np.clip(score_map, 0.0, 1.0)
        r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
        return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)

    def _save_prediction_heatmap(self, s_map, mask, label, rgb_path):
        max_num = getattr(self.args, 'save_heatmap_num', 0)
        if not getattr(self.args, 'save_heatmaps', False) or self.saved_heatmaps >= max_num:
            return

        rgb_path = rgb_path[0] if isinstance(rgb_path, (list, tuple, np.ndarray)) else rgb_path
        rgb_path = str(rgb_path)
        image = Image.open(rgb_path).convert('RGB').resize((self.gt_size, self.gt_size), Image.BICUBIC)
        image_np = np.asarray(image, dtype=np.uint8)

        score_map = s_map.detach().cpu().squeeze().numpy()
        score_map = self._normalize_map(score_map)
        heatmap_np = self._jet_colormap(score_map)
        overlay_np = np.clip(image_np.astype(np.float32) * 0.55 + heatmap_np.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)

        gt_np = mask.detach().cpu().squeeze().numpy()
        gt_np = (gt_np > 0.5).astype(np.uint8) * 255

        defect_type = Path(rgb_path).parents[1].name if len(Path(rgb_path).parents) > 1 else 'unknown'
        label_name = 'anomaly' if int(label) == 1 else 'good'
        out_dir = Path(
            getattr(self.args, 'heatmap_dir', 'visualization'),
            self.args.experiment_note or 'default',
            self.class_name,
            f'{self.args.main_modality or "full"}_{label_name}',
            defect_type,
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(rgb_path).stem
        Image.fromarray(image_np).save(out_dir / f'{self.saved_heatmaps:03d}_{stem}_rgb.png')
        Image.fromarray(gt_np).save(out_dir / f'{self.saved_heatmaps:03d}_{stem}_gt.png')
        Image.fromarray(heatmap_np).save(out_dir / f'{self.saved_heatmaps:03d}_{stem}_heatmap.png')
        Image.fromarray(overlay_np).save(out_dir / f'{self.saved_heatmaps:03d}_{stem}_overlay.png')
        self.saved_heatmaps += 1

    def _segmentation_tensor_path(self, rgb_path, component='fused'):
        mode = self.args.main_modality if self.args.main_modality else 'full'
        method_dir = f'{self.args.method_name.lower()}_{mode}'
        rgb_path = Path(str(rgb_path))
        component = component.replace('/', '_')

        try:
            dataset_root = Path(self.args.dataset_path).expanduser().resolve()
            rel_path = rgb_path.expanduser().resolve().relative_to(dataset_root)
            return dataset_root.parent / 'segmentation' / method_dir / component / rel_path.with_suffix('.pt')
        except (OSError, RuntimeError, ValueError):
            pass

        parts = list(rgb_path.parts)
        for dataset_name in ('mvtec_3d', 'data'):
            if dataset_name in parts:
                idx = parts.index(dataset_name)
                return Path(*parts[:idx], 'segmentation', method_dir, component, *parts[idx + 1:]).with_suffix('.pt')

        return Path('segmentation', method_dir, component, rgb_path.name).with_suffix('.pt')

    def _save_segmentation_result(self, s_map, rgb_path, component='fused'):
        if not getattr(self.args, 'save_seg_results', False):
            return
        rgb_path = rgb_path[0] if isinstance(rgb_path, (list, tuple, np.ndarray)) else rgb_path
        seg_save_path = self._segmentation_tensor_path(rgb_path, component=component)
        seg_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(s_map.detach().cpu(), seg_save_path)

    def _save_component_maps(self, component_maps, rgb_path):
        if not getattr(self.args, 'save_component_maps', False):
            return
        for component, score_map in component_maps.items():
            self._save_segmentation_result(score_map, rgb_path, component=component)

    def _format_cmpt_path(self, path):
        if not path:
            return ''
        modality = self.args.main_modality if self.args.main_modality else 'full'
        return path.format(
            class_name=self.class_name,
            cls=self.class_name,
            modality=modality,
            mode=modality,
        )

    def _sn_tensor_from_sample(self, sample):
        organized_pc = sample[1].squeeze().permute(1, 2, 0).cpu().numpy()
        foreground = np.all(organized_pc != 0, axis=2)
        sn_map = organized_pc_to_surface_normal_map(organized_pc)
        sn_map[~foreground] = np.asarray(self.args.sn_background, dtype=np.float32)
        sn_tensor = resize_organized_pc(
            sn_map,
            target_height=self.rgb_size,
            target_width=self.rgb_size,
            mode=self.args.sn_resize_mode,
        ).float()
        if self.args.sn_smooth_kernel > 1:
            kernel = self.args.sn_smooth_kernel
            sn_tensor = torch.nn.functional.avg_pool2d(
                sn_tensor.unsqueeze(0),
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
            ).squeeze(0)
        sn_tensor = sn_tensor.clamp(0.0, 1.0)
        mean = torch.tensor(self.args.sn_mean, dtype=sn_tensor.dtype).view(3, 1, 1)
        std = torch.tensor(self.args.sn_std, dtype=sn_tensor.dtype).view(3, 1, 1)
        return ((sn_tensor - mean) / std).unsqueeze(0)

    def _sn_foreground_mask_from_sample(self, sample):
        organized_pc = sample[1].squeeze().permute(1, 2, 0)
        foreground = torch.all(organized_pc != 0, dim=2).float().unsqueeze(0).unsqueeze(0)
        grid = self.args.cmpt_feature_grid
        mask = torch.nn.functional.interpolate(foreground, size=(grid, grid), mode='nearest')
        return mask.squeeze().reshape(-1) > self.args.sn_foreground_threshold

    def _sn_foreground_map_from_sample(self, sample):
        organized_pc = sample[1].squeeze().permute(1, 2, 0)
        foreground = torch.all(organized_pc != 0, dim=2).float().unsqueeze(0).unsqueeze(0)
        mask = torch.nn.functional.interpolate(foreground, size=(self.gt_size, self.gt_size), mode='nearest')
        return mask.cpu()

    def _mask_sn_error_map(self, s_map, sample):
        if not self.args.sn_mask_error_map:
            return s_map
        return s_map * self._sn_foreground_map_from_sample(sample)

    def _patches_from_map(self, feature_map, resize_for_cmpt=False):
        if resize_for_cmpt:
            feature_map = self.cmpt_feature_pool(feature_map)
        patch = feature_map.squeeze(0).reshape(feature_map.shape[1], -1).T
        return patch.cpu()

    def _extract_rgb_patches(self, sample):
        rgb = sample[0].to(self.device)
        with torch.no_grad():
            rgb_map = self.rgb_encoder(rgb)
        rgb_patch = self._patches_from_map(rgb_map)
        rgb_patch56 = self._patches_from_map(rgb_map, resize_for_cmpt=True)
        return rgb_patch, rgb_patch56

    def _extract_sn_patches(self, sample):
        sn = self._sn_tensor_from_sample(sample).to(self.device)
        with torch.no_grad():
            sn_map = self.sn_encoder(sn)
        sn_patch = self._patches_from_map(sn_map)
        sn_patch56 = self._patches_from_map(sn_map, resize_for_cmpt=True)
        return sn_patch, sn_patch56

    def _extract_patches(self, sample):
        rgb_patch, rgb_patch56 = self._extract_rgb_patches(sample)
        sn_patch, sn_patch56 = self._extract_sn_patches(sample)
        return rgb_patch, sn_patch, rgb_patch56, sn_patch56

    def _pseudo_sn_from_rgb(self, rgb_patch56):
        with torch.no_grad():
            return self.cmpt(rgb_feature=rgb_patch56.unsqueeze(0).to(self.device), out_type='sn').squeeze(0).cpu()

    def _pseudo_rgb_from_sn(self, sn_patch56):
        with torch.no_grad():
            return self.cmpt(sn_feature=sn_patch56.unsqueeze(0).to(self.device), out_type='rgb').squeeze(0).cpu()

    def _fit_cmpt_if_needed(self):
        if self.cmpt_trained:
            return
        if getattr(self.args, 'disable_cmpt', False):
            self.cmpt.eval()
            self.cmpt_trained = True
            print('[Ablation] CMPT disabled: pseudo-modality transfer/scoring is skipped.')
            return

        rgb, sn = self._paired_train_tokens(self.args.cmpt_max_train_tokens)
        rgb = rgb.to(self.device)
        sn = sn.to(self.device)
        optimizer = torch.optim.AdamW(self.cmpt.parameters(), lr=self.args.cmpt_lr, weight_decay=self.args.cmpt_weight_decay)
        loss_fn = nn.SmoothL1Loss()

        self.cmpt.train()
        for _ in range(self.args.cmpt_epochs):
            perm = torch.randperm(rgb.shape[0], device=self.device)
            for start in range(0, rgb.shape[0], self.args.cmpt_batch_size):
                idx = perm[start:start + self.args.cmpt_batch_size]
                rgb_batch = rgb[idx].unsqueeze(0)
                sn_batch = sn[idx].unsqueeze(0)
                pseudo_sn, pseudo_rgb = self.cmpt(rgb_batch, sn_batch, out_type='train')
                loss = loss_fn(pseudo_sn, sn_batch) + loss_fn(pseudo_rgb, rgb_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        cmpt_save_path = self._format_cmpt_path(self.args.cmpt_save_path)
        if cmpt_save_path:
            save_path = Path(cmpt_save_path)
            if save_path.parent != Path('.'):
                save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model': self.cmpt.state_dict()}, save_path)
        self.cmpt.eval()
        self.cmpt_trained = True

    def _save_learnable_mnc(self):
        mnc_save_path = self._format_cmpt_path(self.args.mnc_save_path)
        if not mnc_save_path:
            return
        save_path = Path(mnc_save_path)
        if save_path.parent != Path('.'):
            save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'mnc': self.learnable_mnc.state_dict()}, save_path)

    def _spa_reconstruction_tensor(self, tokens, modal):
        prototypes = self._prototype_bank(modal)
        reconstruction, _ = structured_prototype_assignment(
            tokens,
            prototypes,
            temperature=self.args.spa_temperature,
            sinkhorn_iters=self.args.spa_sinkhorn_iters,
        )
        return reconstruction

    def _cross_memory_tensor(self, cross_modal):
        cross_bank = self._prototype_bank(cross_modal)
        if self.shared_prototypes is not None and cross_modal in {'rgb', 'sn', 'pseudo_rgb', 'pseudo_sn'}:
            return torch.cat([self._specific_prototype_bank(cross_modal), self.shared_prototypes], dim=0)
        return cross_bank

    def _mnc_training_route(self, tokens, modal, cross_modal):
        spa_reconstruction = self._spa_reconstruction_tensor(tokens, modal)
        cross_memory = self._cross_memory_tensor(cross_modal).unsqueeze(0)
        return self.learnable_mnc(
            tokens.unsqueeze(0),
            spa_reconstruction.unsqueeze(0),
            cross_memory,
            modal,
            cross_modal,
        ).squeeze(0)

    def _fit_learnable_mnc_if_needed(self, rgb_tokens, sn_tokens):
        if not self.use_learnable_mnc or self.mnc_trained:
            return
        if not self.prototype_ready:
            raise RuntimeError('Learnable MNC requires prototypes before training.')

        train_count = min(rgb_tokens.shape[0], sn_tokens.shape[0])
        max_tokens = self.args.mnc_max_train_tokens
        if max_tokens and train_count > max_tokens:
            train_count = max_tokens
            idx = torch.linspace(0, rgb_tokens.shape[0] - 1, steps=train_count, device=rgb_tokens.device).long()
            rgb_tokens = rgb_tokens[idx]
            sn_tokens = sn_tokens[idx]
        else:
            rgb_tokens = rgb_tokens[:train_count]
            sn_tokens = sn_tokens[:train_count]

        optimizer = torch.optim.AdamW(
            self.learnable_mnc.parameters(),
            lr=self.args.mnc_lr,
            weight_decay=self.args.mnc_weight_decay,
        )
        loss_fn = nn.SmoothL1Loss()
        self.learnable_mnc.train()
        for _ in range(self.args.mnc_epochs):
            perm = torch.randperm(train_count, device=self.device)
            for start in range(0, train_count, self.args.mnc_batch_size):
                idx = perm[start:start + self.args.mnc_batch_size]
                rgb_batch = rgb_tokens[idx]
                sn_batch = sn_tokens[idx]

                rgb_rec = self._mnc_training_route(rgb_batch, 'rgb', 'sn')
                sn_rec = self._mnc_training_route(sn_batch, 'sn', 'rgb')
                loss = loss_fn(rgb_rec, F.normalize(rgb_batch, dim=-1)) + loss_fn(sn_rec, F.normalize(sn_batch, dim=-1))

                if self.args.mnc_train_pseudo:
                    with torch.no_grad():
                        pseudo_sn = self.cmpt(rgb_feature=rgb_batch.unsqueeze(0), out_type='sn').squeeze(0)
                        pseudo_rgb = self.cmpt(sn_feature=sn_batch.unsqueeze(0), out_type='rgb').squeeze(0)
                    pseudo_sn_rec = self._mnc_training_route(pseudo_sn, 'pseudo_sn', 'rgb')
                    pseudo_rgb_rec = self._mnc_training_route(pseudo_rgb, 'pseudo_rgb', 'sn')
                    rgb_with_pseudo_sn_rec = self._mnc_training_route(rgb_batch, 'rgb', 'pseudo_sn')
                    sn_with_pseudo_rgb_rec = self._mnc_training_route(sn_batch, 'sn', 'pseudo_rgb')
                    loss = loss + self.args.mnc_pseudo_loss_weight * (
                        loss_fn(pseudo_sn_rec, F.normalize(sn_batch, dim=-1))
                        + loss_fn(pseudo_rgb_rec, F.normalize(rgb_batch, dim=-1))
                        + loss_fn(rgb_with_pseudo_sn_rec, F.normalize(rgb_batch, dim=-1))
                        + loss_fn(sn_with_pseudo_rgb_rec, F.normalize(sn_batch, dim=-1))
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.learnable_mnc.eval()
        self.mnc_trained = True
        self._save_learnable_mnc()

    @staticmethod
    def _feature_hw(patch):
        side = int(round(patch.shape[0] ** 0.5))
        if side * side != patch.shape[0]:
            raise ValueError(f'Patch token count {patch.shape[0]} cannot be reshaped to a square feature map.')
        return side, side

    @staticmethod
    def _score_from_map(s_map):
        return s_map.flatten().max()

    def _normal_token_memory(self, modal):
        if modal in {'rgb', 'pseudo_rgb'}:
            banks = self.patch_rgb56_train
        elif modal in {'sn', 'pseudo_sn'}:
            banks = self.patch_sn56_train
        else:
            raise ValueError(f'Unsupported token memory: {modal}')
        if not banks:
            raise RuntimeError(f'No normal token memory is available for `{modal}`.')
        memory = torch.cat(banks, 0).to(self.device)
        max_tokens = getattr(self.args, 'no_proto_memory_tokens', 4096)
        if max_tokens and memory.shape[0] > max_tokens:
            idx = torch.linspace(0, memory.shape[0] - 1, steps=max_tokens, device=memory.device).long()
            memory = memory[idx]
        return torch.nn.functional.normalize(memory, dim=-1)

    def _paired_train_tokens(self, max_tokens):
        if not self.patch_rgb56_train or not self.patch_sn56_train:
            raise RuntimeError(
                f'No training tokens were collected for class `{self.class_name}`. '
                f'Check --dataset_path and preprocessing output. Expected training files under '
                f'`<dataset_path>/{self.class_name}/train/good/rgb/*.png` and '
                f'`<dataset_path>/{self.class_name}/train/good/xyz/*.tiff`.'
            )
        rgb = torch.cat(self.patch_rgb56_train, 0)
        sn = torch.cat(self.patch_sn56_train, 0)
        if self.args.sn_foreground_only and self.sn_foreground_masks56_train:
            mask = torch.cat(self.sn_foreground_masks56_train, 0)
            if mask.any():
                rgb = rgb[mask]
                sn = sn[mask]
        if max_tokens and rgb.shape[0] > max_tokens:
            idx = torch.linspace(0, rgb.shape[0] - 1, steps=max_tokens).long()
            rgb = rgb[idx]
            sn = sn[idx]
        return rgb, sn

    def add_sample_to_mem_bank(self, sample, class_name=None):
        self.class_name = class_name
        _, _, rgb_patch56, sn_patch56 = self._extract_patches(sample)
        sn_mask56 = self._sn_foreground_mask_from_sample(sample)

        self.patch_rgb56_train.append(rgb_patch56)
        self.patch_sn56_train.append(sn_patch56)
        self.sn_foreground_masks56_train.append(sn_mask56)

    def _build_single_prototype_set(self, tokens, num_prototypes):
        prototypes = kmeans_prototypes(
            tokens,
            num_prototypes=num_prototypes,
            iters=self.args.prototype_kmeans_iters,
            max_tokens=self.args.prototype_max_tokens,
        )
        if getattr(self.args, 'disable_apr', False):
            return prototypes
        train_tokens = sample_tokens(tokens, max_tokens=self.args.prototype_max_tokens)
        if self.args.apr_memory_update_iters > 0:
            return adaptive_prototype_memory_update(
                train_tokens,
                prototypes,
                temperature=self.args.spa_temperature,
                sinkhorn_iters=self.args.spa_sinkhorn_iters,
                update_rate=self.args.apr_update_rate,
                confidence_threshold=self.args.apr_confidence_threshold,
                update_iters=self.args.apr_memory_update_iters,
            )
        _, assignment = structured_prototype_assignment(
            train_tokens,
            prototypes,
            temperature=self.args.spa_temperature,
            sinkhorn_iters=self.args.spa_sinkhorn_iters,
        )
        return adaptive_prototype_refinement(
            train_tokens,
            prototypes,
            assignment,
            update_rate=self.args.apr_update_rate,
            confidence_threshold=self.args.apr_confidence_threshold,
        )

    def _transfer_prototypes(self, source_prototypes, target_prototypes, out_type):
        if getattr(self.args, 'disable_cmpt', False) or getattr(self.args, 'disable_pseudo_proto', False):
            return torch.nn.functional.normalize(target_prototypes.clone(), dim=-1)
        assignment = balanced_sinkhorn_assignment(
            source_prototypes,
            target_prototypes,
            temperature=self.args.cmpt_ot_temperature,
            iters=self.args.cmpt_ot_iters,
        )
        ot_context = assignment @ target_prototypes
        with torch.no_grad():
            if out_type == 'sn':
                transferred = self.cmpt(rgb_feature=source_prototypes.unsqueeze(0), out_type='sn').squeeze(0)
            elif out_type == 'rgb':
                transferred = self.cmpt(sn_feature=source_prototypes.unsqueeze(0), out_type='rgb').squeeze(0)
            else:
                raise ValueError(f'Unsupported prototype transfer target: {out_type}')
        transferred = torch.nn.functional.normalize(transferred, dim=-1)
        ot_context = torch.nn.functional.normalize(ot_context, dim=-1)
        return torch.nn.functional.normalize(
            self.args.cmpt_proto_mlp_weight * transferred + (1.0 - self.args.cmpt_proto_mlp_weight) * ot_context,
            dim=-1,
        )

    def run_coreset(self):
        self._fit_cmpt_if_needed()
        rgb_tokens, sn_tokens = self._paired_train_tokens(self.args.prototype_max_tokens)
        rgb_tokens = rgb_tokens.to(self.device)
        sn_tokens = sn_tokens.to(self.device)
        shared_tokens = torch.nn.functional.normalize((rgb_tokens + sn_tokens) * 0.5, dim=-1)

        self.rgb_specific_prototypes = self._build_single_prototype_set(
            rgb_tokens, self.args.num_rgb_prototypes
        )
        self.sn_specific_prototypes = self._build_single_prototype_set(
            sn_tokens, self.args.num_sn_prototypes
        )
        self.shared_prototypes = self._build_single_prototype_set(
            shared_tokens, self.args.num_shared_prototypes
        )
        if getattr(self.args, 'disable_shared_proto', False):
            self.shared_prototypes = None
            print('[Ablation] Shared prototypes disabled.')
        self.pseudo_sn_prototypes = self._transfer_prototypes(
            self.rgb_specific_prototypes, self.sn_specific_prototypes, out_type='sn'
        )
        self.pseudo_rgb_prototypes = self._transfer_prototypes(
            self.sn_specific_prototypes, self.rgb_specific_prototypes, out_type='rgb'
        )
        self.prototype_ready = True
        self._fit_learnable_mnc_if_needed(rgb_tokens, sn_tokens)

    def _prototype_bank(self, modal):
        if not self.prototype_ready:
            raise RuntimeError('CMPT prototypes are not ready. Call run_coreset() before prediction.')
        if getattr(self.args, 'only_shared_proto', False):
            if self.shared_prototypes is None:
                raise RuntimeError('only_shared_proto requires shared prototypes.')
            return self.shared_prototypes
        if modal == 'rgb':
            prototypes = [self.rgb_specific_prototypes, self.shared_prototypes]
        elif modal == 'sn':
            prototypes = [self.sn_specific_prototypes, self.shared_prototypes]
        elif modal == 'pseudo_sn':
            prototypes = [self.pseudo_sn_prototypes, self.shared_prototypes]
        elif modal == 'pseudo_rgb':
            prototypes = [self.pseudo_rgb_prototypes, self.shared_prototypes]
        else:
            raise ValueError(f'Unsupported prototype bank: {modal}')
        prototypes = [prototype for prototype in prototypes if prototype is not None]
        if not prototypes:
            raise RuntimeError(f'No prototypes are available for bank `{modal}`.')
        return torch.cat(prototypes, dim=0)

    def _specific_prototype_bank(self, modal):
        if modal == 'rgb':
            return self.rgb_specific_prototypes
        if modal == 'sn':
            return self.sn_specific_prototypes
        if modal == 'pseudo_sn':
            return self.pseudo_sn_prototypes
        if modal == 'pseudo_rgb':
            return self.pseudo_rgb_prototypes
        raise ValueError(f'Unsupported prototype bank: {modal}')

    def _reconstruct_with_prototypes(self, patch, modal, cross_modal=None):
        tokens = patch.to(self.device)
        if getattr(self.args, 'disable_prototypes', False):
            memory = self._normal_token_memory(modal)
            similarity = torch.nn.functional.normalize(tokens, dim=-1) @ memory.t()
            nearest = similarity.argmax(dim=-1)
            reconstruction = memory[nearest]
            s_map = reconstruction_error_map(
                tokens,
                reconstruction,
                out_size=self.gt_size,
                feature_hw=self._feature_hw(patch),
            ).cpu()
            s_map = self.blur(s_map)
            return self._score_from_map(s_map), s_map

        prototypes = self._prototype_bank(modal)
        reconstruction, _ = structured_prototype_assignment(
            tokens,
            prototypes,
            temperature=self.args.spa_temperature,
            sinkhorn_iters=self.args.spa_sinkhorn_iters,
        )

        if self.args.apr_inference_refine:
            _, assignment = structured_prototype_assignment(
                tokens,
                prototypes,
                temperature=self.args.spa_temperature,
                sinkhorn_iters=self.args.spa_sinkhorn_iters,
            )
            refined_prototypes = adaptive_prototype_refinement(
                tokens,
                prototypes,
                assignment,
                update_rate=self.args.apr_update_rate,
                confidence_threshold=self.args.apr_confidence_threshold,
            )
            reconstruction, _ = structured_prototype_assignment(
                tokens,
                refined_prototypes,
                temperature=self.args.spa_temperature,
                sinkhorn_iters=self.args.spa_sinkhorn_iters,
            )

        if getattr(self.args, 'disable_mnc', False):
            cross_modal = None

        if cross_modal is not None:
            if self.use_learnable_mnc:
                if not self.mnc_trained:
                    raise RuntimeError('Learnable MNC is enabled but not trained or loaded.')
                cross_memory = self._cross_memory_tensor(cross_modal).unsqueeze(0)
                with torch.no_grad():
                    reconstruction = self.learnable_mnc(
                        tokens.unsqueeze(0),
                        reconstruction.unsqueeze(0),
                        cross_memory,
                        modal,
                        cross_modal,
                    ).squeeze(0)
            elif self.args.mnc_strong:
                _, reconstruction = multi_stage_normality_communication(
                    tokens,
                    reconstruction,
                    own_prototypes=self._prototype_bank(modal),
                    cross_prototypes=self._specific_prototype_bank(cross_modal),
                    shared_prototypes=self.shared_prototypes,
                    stage1_weight=self.args.mnc_stage1_weight,
                    stage2_weight=self.args.mnc_stage2_weight,
                    temperature=self.args.mnc_temperature,
                )
                for _ in range(max(0, self.args.mnc_stages - 2)):
                    _, reconstruction = multi_stage_normality_communication(
                        tokens,
                        reconstruction,
                        own_prototypes=self._prototype_bank(modal),
                        cross_prototypes=self._specific_prototype_bank(cross_modal),
                        shared_prototypes=self.shared_prototypes,
                        stage1_weight=self.args.mnc_stage1_weight,
                        stage2_weight=self.args.mnc_stage2_weight,
                        temperature=self.args.mnc_temperature,
                    )
            else:
                cross_prototypes = self._prototype_bank(cross_modal)
                for _ in range(self.args.mnc_stages):
                    reconstruction = gated_cross_modal_reconstruction(
                        tokens,
                        reconstruction,
                        cross_prototypes,
                        weight=self.args.mnc_cross_weight,
                    )

        s_map = reconstruction_error_map(
            tokens,
            reconstruction,
            out_size=self.gt_size,
            feature_hw=self._feature_hw(patch),
        ).cpu()
        s_map = self.blur(s_map)
        return self._score_from_map(s_map), s_map

    def _scores_from_sample(self, sample):
        if self.args.main_modality == 'rgb':
            _, rgb_patch56 = self._extract_rgb_patches(sample)
            if getattr(self.args, 'disable_cmpt', False) or getattr(self.args, 'disable_pseudo_proto', False):
                s_rgb, smap_rgb = self._reconstruct_with_prototypes(rgb_patch56, 'rgb', cross_modal=None)
                return s_rgb, smap_rgb, {'rgb_error': smap_rgb, 'fused': smap_rgb}
            pseudo_sn_patch = self._pseudo_sn_from_rgb(rgb_patch56)
            s_rgb, smap_rgb = self._reconstruct_with_prototypes(rgb_patch56, 'rgb', cross_modal='pseudo_sn')
            s_sn, smap_sn = self._reconstruct_with_prototypes(pseudo_sn_patch, 'pseudo_sn', cross_modal='rgb')
            s = self.args.rgb_s_lambda * s_rgb + self.args.cmpt_s_lambda * s_sn
            s_map = self.args.rgb_smap_lambda * smap_rgb + self.args.cmpt_smap_lambda * smap_sn
            component_maps = {
                'rgb_error': smap_rgb,
                'pseudo_sn_error': smap_sn,
                'fused': s_map,
            }
            return s, s_map, component_maps
        elif self.args.main_modality == 'sn':
            _, sn_patch56 = self._extract_sn_patches(sample)
            if getattr(self.args, 'disable_cmpt', False) or getattr(self.args, 'disable_pseudo_proto', False):
                s_sn, smap_sn = self._reconstruct_with_prototypes(sn_patch56, 'sn', cross_modal=None)
                smap_sn = self._mask_sn_error_map(smap_sn, sample)
                s_sn = self._score_from_map(smap_sn)
                return s_sn, smap_sn, {'sn_error': smap_sn, 'fused': smap_sn}
            pseudo_rgb_patch = self._pseudo_rgb_from_sn(sn_patch56)
            s_sn, smap_sn = self._reconstruct_with_prototypes(sn_patch56, 'sn', cross_modal='pseudo_rgb')
            s_rgb, smap_rgb = self._reconstruct_with_prototypes(pseudo_rgb_patch, 'pseudo_rgb', cross_modal='sn')
            smap_sn = self._mask_sn_error_map(smap_sn, sample)
            smap_rgb = self._mask_sn_error_map(smap_rgb, sample)
            s_sn = self._score_from_map(smap_sn)
            s_rgb = self._score_from_map(smap_rgb)
            s = self.args.sn_s_lambda * s_sn + self.args.cmpt_s_lambda * s_rgb
            s_map = self.args.sn_smap_lambda * smap_sn + self.args.cmpt_smap_lambda * smap_rgb
            component_maps = {
                'sn_error': smap_sn,
                'pseudo_rgb_error': smap_rgb,
                'fused': s_map,
            }
            return s, s_map, component_maps
        else:
            _, _, rgb_patch56, sn_patch56 = self._extract_patches(sample)
            s_rgb, smap_rgb = self._reconstruct_with_prototypes(rgb_patch56, 'rgb', cross_modal='sn')
            s_sn, smap_sn = self._reconstruct_with_prototypes(sn_patch56, 'sn', cross_modal='rgb')
            smap_sn = self._mask_sn_error_map(smap_sn, sample)
            s_sn = self._score_from_map(smap_sn)
            s = (
                self.args.rgb_s_lambda * s_rgb
                + self.args.sn_s_lambda * s_sn
            )
            s_map = (
                self.args.rgb_smap_lambda * smap_rgb
                + self.args.sn_smap_lambda * smap_sn
            )
            component_maps = {
                'rgb_error': smap_rgb,
                'sn_error': smap_sn,
                'fused': s_map,
            }
        return s, s_map, component_maps

    def add_sample_to_late_fusion_mem_bank(self, sample):
        return

    def predict(self, sample, mask, label, rgb_path):
        s, s_map, component_maps = self._scores_from_sample(sample)
        self._save_segmentation_result(s_map, rgb_path)
        self._save_component_maps(component_maps, rgb_path)
        self._save_prediction_heatmap(s_map, mask, label, rgb_path)

        self.image_preds.append(np.asarray([float(s.detach().cpu())]))
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().detach().cpu().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())
        self.img_name.append(rgb_path)

    def run_late_fusion(self):
        return

    def calculate_metrics(self):
        self.image_preds = np.stack(self.image_preds)
        self.image_labels = np.stack(self.image_labels)
        self.pixel_preds = np.array(self.pixel_preds)
        self.img_name = np.stack(self.img_name)

        if self.args.save_raw_results:
            txt_to_save = np.concatenate((self.image_preds, self.image_labels, self.img_name), axis=1)
            np.savetxt(f'./visualization/{self.args.experiment_note}/{self.class_name}_raw_results.csv',
                       txt_to_save, delimiter=',', fmt="%s")

        from sklearn.metrics import roc_auc_score
        self.image_rocauc = roc_auc_score(self.image_labels, self.image_preds)
        self.pixel_rocauc = roc_auc_score(self.pixel_labels, self.pixel_preds)
        self.au_pro, _ = calculate_au_pro(self.gts, self.predictions)
        self.au_pro_001, _ = calculate_au_pro(self.gts, self.predictions, 0.01)
