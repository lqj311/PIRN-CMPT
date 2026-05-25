import os
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils.mvtec3d_util import *


def eyecandies_classes():
    return [
        'CandyCane',
        'ChocolateCookie',
        'ChocolatePraline',
        'Confetto',
        'GummyBear',
        'HazelnutTruffle',
        'LicoriceSandwich',
        'Lollipop',
        'Marshmallow',
        'PeppermintCandy',
    ]


def mvtec3d_classes():
    return [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]


RGB_SIZE = 224


def _format_missing_dataset_message(class_name, split_root, rgb_dir, xyz_dir):
    return (
        f'No paired RGB/XYZ samples found for class `{class_name}` at `{split_root}`. '
        f'Expected RGB files in `{rgb_dir}` and XYZ TIFF files in `{xyz_dir}`. '
        f'Check --dataset_path, unzip location, and whether `python utils/preprocessing.py '
        f'--dataset_path <path>` has been run.'
    )


def _path_stem(path):
    return Path(path).stem


def _stable_fewshot_indices(total, k, seed, class_name):
    if k <= 0 or k >= total:
        return list(range(total))
    offset = sum(ord(c) for c in class_name)
    rng = np.random.default_rng(seed + offset)
    return sorted(rng.choice(total, size=k, replace=False).tolist())


def _save_fewshot_manifest(dataset, indices, args, split):
    if not getattr(args, 'save_fewshot_list', False):
        return
    out_dir = Path(args.fewshot_list_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{dataset.cls}_{split}_k{args.few_shot_k}_seed{args.shot_seed}.txt'
    with open(out_path, 'w', encoding='utf-8') as fp:
        for idx in indices:
            item = dataset.img_paths[idx]
            if isinstance(item, tuple):
                fp.write('\t'.join(str(p) for p in item) + '\n')
            else:
                fp.write(str(item) + '\n')


def apply_fewshot_subset(dataset, args, split):
    k = getattr(args, 'few_shot_k', 0)
    if split not in ['train', 'train_validation'] or k <= 0:
        return dataset
    total = len(dataset.img_paths)
    indices = _stable_fewshot_indices(total, k, getattr(args, 'shot_seed', 0), dataset.cls)
    _save_fewshot_manifest(dataset, indices, args, split)
    dataset.img_paths = [dataset.img_paths[i] for i in indices]
    dataset.labels = [dataset.labels[i] for i in indices]
    print(f'[FewShot] {dataset.cls} {split}: selected {len(indices)}/{total} normal training samples '
          f'(K={k}, seed={getattr(args, "shot_seed", 0)}).')
    return dataset


class BaseAnomalyDetectionDataset(Dataset):
    def __init__(self, split, class_name, rgb_size, xyz_size, gt_size, dataset_path, img_process_method):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.cls = class_name
        self.rgb_size = rgb_size
        self.xyz_size = xyz_size
        self.gt_size = gt_size
        # self.img_path = os.path.join(dataset_path, self.cls, split)
        if split == 'train_validation':
            self.img_path = str(Path(dataset_path, self.cls, 'train'))
            self.img_path2 = str(Path(dataset_path, self.cls, 'validation'))
        else:
            self.img_path = str(Path(dataset_path, self.cls, split))
        # maybe change this to GPU computation or opencv version later
        self.img_process_method = img_process_method
        if self.img_process_method == 'cpu_v1':
            self.rgb_transform = transforms.Compose(
                [transforms.Resize((self.rgb_size, self.rgb_size), interpolation=transforms.InterpolationMode.BICUBIC),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])
        elif self.img_process_method == 'cpu_v2':
            self.rgb_transform = v2.Compose(
                [v2.Resize((self.rgb_size, self.rgb_size), interpolation=v2.InterpolationMode.BICUBIC),
                 v2.ToTensor(),
                 v2.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])


class TrainDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, rgb_size, xyz_size, gt_size, dataset_path, img_process_method):
        super().__init__(split="train", class_name=class_name, rgb_size=rgb_size, xyz_size=xyz_size, gt_size=gt_size,
                         dataset_path=dataset_path, img_process_method=img_process_method)
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        # rgb_paths = glob.glob(os.path.join(self.img_path, 'good', 'rgb') + "/*.png")
        # tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff")
        rgb_dir = Path(self.img_path, 'good', 'rgb')
        xyz_dir = Path(self.img_path, 'good', 'xyz')
        rgb_paths = list(rgb_dir.glob("*.png"))
        tiff_paths = list(xyz_dir.glob("*.tiff"))
        rgb_paths.sort()
        tiff_paths.sort()
        sample_paths = list(zip(rgb_paths, tiff_paths))
        if not sample_paths:
            raise FileNotFoundError(_format_missing_dataset_message(self.cls, self.img_path, rgb_dir, xyz_dir))
        img_tot_paths.extend(sample_paths)
        # fill 0 as label for normal samples
        tot_labels.extend([0] * len(sample_paths))
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        # maybe change this to GPU computation or opencv version later
        if self.img_process_method == 'cpu_v1' or self.img_process_method == 'cpu_v2':
            img = Image.open(rgb_path).convert('RGB')
            img = self.rgb_transform(img)

        organized_pc = read_tiff_organized_pc(tiff_path)

        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.xyz_size, target_width=self.xyz_size)
        resized_organized_pc = resized_organized_pc.clone().detach().float()

        return (img, resized_organized_pc, resized_depth_map_3channel), label


class TrainValidationDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, rgb_size, xyz_size, gt_size, dataset_path, img_process_method):
        super().__init__(split="train_validation", class_name=class_name, rgb_size=rgb_size, xyz_size=xyz_size,
                         gt_size=gt_size, dataset_path=dataset_path, img_process_method=img_process_method)
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        # rgb_paths = glob.glob(os.path.join(self.img_path, 'good', 'rgb') + "/*.png")
        # tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff")
        rgb_dir = Path(self.img_path, 'good', 'rgb')
        xyz_dir = Path(self.img_path, 'good', 'xyz')
        rgb_val_dir = Path(self.img_path2, 'good', 'rgb')
        xyz_val_dir = Path(self.img_path2, 'good', 'xyz')
        rgb_paths = list(rgb_dir.glob("*.png"))
        rgb_paths2 = list(rgb_val_dir.glob("*.png"))
        rgb_paths = rgb_paths + rgb_paths2
        tiff_paths = list(xyz_dir.glob("*.tiff"))
        tiff_paths2 = list(xyz_val_dir.glob("*.tiff"))
        tiff_paths = tiff_paths + tiff_paths2
        rgb_paths.sort()
        tiff_paths.sort()
        sample_paths = list(zip(rgb_paths, tiff_paths))
        if not sample_paths:
            raise FileNotFoundError(
                _format_missing_dataset_message(self.cls, self.img_path, rgb_dir, xyz_dir)
                + f' Also checked validation dirs `{rgb_val_dir}` and `{xyz_val_dir}`.'
            )
        img_tot_paths.extend(sample_paths)
        # fill 0 as label for normal samples
        tot_labels.extend([0] * len(sample_paths))
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        # maybe change this to GPU computation or opencv version later
        if self.img_process_method == 'cpu_v1' or self.img_process_method == 'cpu_v2':
            img = Image.open(rgb_path).convert('RGB')
            img = self.rgb_transform(img)

        organized_pc = read_tiff_organized_pc(tiff_path)

        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.xyz_size, target_width=self.xyz_size)
        resized_organized_pc = resized_organized_pc.clone().detach().float()

        return (img, resized_organized_pc, resized_depth_map_3channel), label


class TestDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, rgb_size, xyz_size, gt_size, dataset_path, img_process_method):
        super().__init__(split="test", class_name=class_name, rgb_size=rgb_size, xyz_size=xyz_size,
                         gt_size=gt_size, dataset_path=dataset_path, img_process_method=img_process_method)
        # maybe change this to GPU computation or opencv version later and why it's different between test and train
        if self.img_process_method == 'cpu_v1':
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.gt_size, self.gt_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()])
        elif self.img_process_method == 'cpu_v2':
            self.gt_transform = v2.Compose(
                [v2.Resize((self.gt_size, self.gt_size), interpolation=v2.InterpolationMode.NEAREST),
                 v2.ToTensor()])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                # rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                # tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                rgb_paths = list(Path(self.img_path, defect_type, 'rgb').glob("*.png"))
                tiff_paths = list(Path(self.img_path, defect_type, 'xyz').glob("*.tiff"))
                rgb_paths.sort()
                tiff_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            else:
                # rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                # tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                # gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                rgb_paths = list(Path(self.img_path, defect_type, 'rgb').glob("*.png"))
                tiff_paths = list(Path(self.img_path, defect_type, 'xyz').glob("*.tiff"))
                gt_paths = list(Path(self.img_path, defect_type, 'gt').glob("*.png"))
                rgb_paths.sort()
                tiff_paths.sort()
                gt_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))

                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(sample_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path = str(img_path[0])
        tiff_path = str(img_path[1])
        # maybe change this to GPU computation or opencv version later
        if self.img_process_method == 'cpu_v1' or self.img_process_method == 'cpu_v2':
            img_original = Image.open(rgb_path).convert('RGB')
            img = self.rgb_transform(img_original)

        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.xyz_size, target_width=self.xyz_size)
        resized_organized_pc = resized_organized_pc.clone().detach().float()

        if gt == 0:
            gt = torch.zeros(
                [1, resized_depth_map_3channel.size()[-2], resized_depth_map_3channel.size()[-2]])
        else:
            if self.img_process_method == 'cpu_v1' or self.img_process_method == 'cpu_v2':
                gt = Image.open(gt).convert('L')
                gt = self.gt_transform(gt)
                gt = torch.where(gt > 0.5, 1., .0)

        # only need 1 dimension gt instead of 3
        return (img, resized_organized_pc, resized_depth_map_3channel), gt[:1], label, rgb_path


class MissingModalityTestDataset(TestDataset):
    def __init__(self, class_name, rgb_size, xyz_size, gt_size, dataset_path, img_process_method, main_modality):
        self.main_modality = main_modality
        super().__init__(
            class_name=class_name,
            rgb_size=rgb_size,
            xyz_size=xyz_size,
            gt_size=gt_size,
            dataset_path=dataset_path,
            img_process_method=img_process_method,
        )

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.img_path)
        main_modality = self.main_modality

        for defect_type in defect_types:
            defect_root = Path(self.img_path, defect_type)
            rgb_paths = sorted(defect_root.joinpath('rgb').glob('*.png'))
            tiff_paths = sorted(defect_root.joinpath('xyz').glob('*.tiff'))
            rgb_by_stem = {_path_stem(path): path for path in rgb_paths}
            tiff_by_stem = {_path_stem(path): path for path in tiff_paths}
            if main_modality == 'rgb':
                stems = sorted(rgb_by_stem)
            elif main_modality == 'sn':
                stems = sorted(tiff_by_stem)
            else:
                stems = sorted(set(rgb_by_stem) & set(tiff_by_stem))

            if defect_type == 'good':
                gt_paths = [0] * len(stems)
                labels = [0] * len(stems)
            else:
                gt_by_stem = {_path_stem(path): path for path in sorted(defect_root.joinpath('gt').glob('*.png'))}
                gt_paths = [gt_by_stem.get(stem, 0) for stem in stems]
                labels = [1] * len(stems)

            for stem, gt_path, label in zip(stems, gt_paths, labels):
                rgb_path = rgb_by_stem.get(stem)
                tiff_path = tiff_by_stem.get(stem)
                if main_modality == 'rgb' and rgb_path is None:
                    continue
                if main_modality == 'sn' and tiff_path is None:
                    continue
                if main_modality == '' and (rgb_path is None or tiff_path is None):
                    continue
                img_tot_paths.append((rgb_path, tiff_path))
                gt_tot_paths.append(gt_path)
                tot_labels.append(label)

        if not img_tot_paths:
            raise FileNotFoundError(
                f'No test samples found for class `{self.cls}` at `{self.img_path}` with '
                f'main_modality `{main_modality}`.'
            )
        return img_tot_paths, gt_tot_paths, tot_labels

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]

        if rgb_path is None:
            img = torch.zeros(3, self.rgb_size, self.rgb_size)
        else:
            img = self.rgb_transform(Image.open(rgb_path).convert('RGB'))

        if tiff_path is None:
            resized_organized_pc = torch.zeros(3, self.xyz_size, self.xyz_size).float()
            resized_depth_map_3channel = torch.zeros(3, self.xyz_size, self.xyz_size).float()
        else:
            organized_pc = read_tiff_organized_pc(tiff_path)
            depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
            resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
            resized_organized_pc = resize_organized_pc(
                organized_pc,
                target_height=self.xyz_size,
                target_width=self.xyz_size,
            )
            resized_organized_pc = resized_organized_pc.clone().detach().float()

        if gt == 0:
            gt = torch.zeros([1, self.gt_size, self.gt_size])
        else:
            gt = self.gt_transform(Image.open(gt).convert('L'))
            gt = torch.where(gt > 0.5, 1., .0)

        ref_path = str(rgb_path if rgb_path is not None else tiff_path)
        return (img, resized_organized_pc, resized_depth_map_3channel), gt[:1], label, ref_path


class PreTrainTensorDataset(Dataset):
    #  patch = torch.cat([xyz_patch, rgb_patch_resize], dim=1)  # 3136 768+1152
    def __init__(self, root_path):
        super().__init__()
        self.root_path = root_path
        self.tensor_paths = os.listdir(self.root_path)

    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[idx]

        # tensor = torch.load(os.path.join(self.root_path, tensor_path))
        tensor = torch.load(Path(self.root_path, tensor_path), map_location='cuda')

        label = 0

        return tensor, label


class FeatureToInputPreTrainTensorDataset(Dataset):
    def __init__(self, root_path, data_type):
        super().__init__()
        self.root_path = root_path
        self.data_type = data_type
        if data_type == 'rgb_fxyz':
            self.rgb_root_path = Path(root_path, 'rgb')
            self.fxyz_root_path = Path(root_path, 'fxyz')

            self.rgb_paths = list(self.rgb_root_path.glob('*.pt'))
            self.rgb_paths.sort()
            self.fxyz_paths = list(self.fxyz_root_path.glob('*hfxyz.pt'))
            self.fxyz_paths.sort()

            assert len(self.rgb_paths) == len(self.fxyz_paths)
            self.len = len(self.rgb_paths)

        elif data_type == 'xyz_frgb':
            self.frgb_root_path = Path(root_path, 'frgb')
            self.xyz_root_path = Path(root_path, 'xyz')

            self.frgb_paths = list(self.frgb_root_path.glob('*.pt'))
            self.frgb_paths.sort()
            self.xyz_paths = list(self.xyz_root_path.glob('*.pt'))
            self.xyz_paths.sort()
            assert len(self.frgb_paths) == len(self.xyz_paths)
            self.len = len(self.frgb_paths)


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.data_type == 'rgb_fxyz':
            rgb_path = self.rgb_paths[idx]
            fxyz_path = self.fxyz_paths[idx]
            # tensor = torch.load(os.path.join(self.root_path, tensor_path))
            rgb = torch.load(rgb_path, map_location='cuda')
            fxyz = torch.load(fxyz_path, map_location='cuda')
            return fxyz, rgb
        elif self.data_type == 'xyz_frgb':
            # frgb 3136 768 xyz 3 224 224
            frgb_path = self.frgb_paths[idx]
            xyz_path = self.xyz_paths[idx]
            # tensor = torch.load(os.path.join(self.root_path, tensor_path))
            frgb = torch.load(frgb_path, map_location='cuda')
            xyz = torch.load(xyz_path, map_location='cuda')
            return frgb, xyz


class InputToFeaturePreTrainTensorDataset(Dataset):
    def __init__(self, root_path, data_type):
        super().__init__()
        self.data_type = data_type
        self.root_path = root_path
        if data_type == 'rgb_fxyz':
            self.rgb_root_path = Path(root_path, 'rgb')
            self.fxyz_root_path = Path(root_path, 'fxyz')
            self.rgb_paths = list(self.rgb_root_path.glob('*.pt'))
            self.rgb_paths.sort()
            self.fxyz_paths = list(self.fxyz_root_path.glob('*hfxyz.pt'))
            self.fxyz_paths.sort()
            assert len(self.rgb_paths) == len(self.fxyz_paths)
            self.len = len(self.rgb_paths)
        elif data_type == 'xyz_frgb':
            self.frgb_root_path = Path(root_path, 'frgb')
            self.xyz_root_path = Path(root_path, 'xyz')
            self.frgb_paths = list(self.frgb_root_path.glob('*.pt'))
            self.frgb_paths.sort()
            self.xyz_paths = list(self.xyz_root_path.glob('*.pt'))
            self.xyz_paths.sort()
            assert len(self.frgb_paths) == len(self.xyz_paths)
            self.len = len(self.frgb_paths)
        else:
            raise NotImplementedError

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.data_type == 'rgb_fxyz':
            # frgb 3136 768 xyz 3 224 224 fxyz 3136 768 rgb 3 224 224
            rgb_path = self.rgb_paths[idx]
            fxyz_path = self.fxyz_paths[idx]
            # tensor = torch.load(os.path.join(self.root_path, tensor_path))
            rgb = torch.load(rgb_path)
            fxyz = torch.load(fxyz_path)
            return rgb, fxyz
        elif self.data_type == 'xyz_frgb':
            xyz_path = self.xyz_paths[idx]
            frgb_path = self.frgb_paths[idx]
            frgb = torch.load(frgb_path)
            xyz = torch.load(xyz_path)
            return xyz, frgb


def get_data_loader(split, class_name, rgb_size, xyz_size, gt_size, args):
    if split in ['train']:
        dataset = TrainDataset(class_name=class_name, rgb_size=rgb_size, xyz_size=xyz_size, gt_size=gt_size,
                               dataset_path=args.dataset_path, img_process_method=args.img_process_method)
    elif split in ['train_validation']:
        dataset = TrainValidationDataset(class_name=class_name, rgb_size=rgb_size, xyz_size=xyz_size, gt_size=gt_size,
                               dataset_path=args.dataset_path, img_process_method=args.img_process_method)
    elif split in ['test']:
        if (
            getattr(args, 'allow_true_missing_modality', False)
            and getattr(args, 'main_modality', '') in ['rgb', 'sn']
        ):
            dataset = MissingModalityTestDataset(
                class_name=class_name,
                rgb_size=rgb_size,
                xyz_size=xyz_size,
                gt_size=gt_size,
                dataset_path=args.dataset_path,
                img_process_method=args.img_process_method,
                main_modality=args.main_modality,
            )
        else:
            dataset = TestDataset(class_name=class_name, rgb_size=rgb_size, xyz_size=xyz_size, gt_size=gt_size,
                                  dataset_path=args.dataset_path, img_process_method=args.img_process_method)
    else:
        raise ValueError

    dataset = apply_fewshot_subset(dataset, args, split)

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=6, drop_last=False,
                             prefetch_factor=6, pin_memory=True)
    # train (img, resized_organized_pc, resized_depth_map_3channel), label
    # img = B RGB3 224 224 resized_organized_pc= B XYZ3 224 224
    return data_loader
