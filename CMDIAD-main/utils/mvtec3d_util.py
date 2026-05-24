import tifffile as tiff
import torch
import numpy as np


def organized_pc_to_unorganized_pc(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])


def read_tiff_organized_pc(path):
    tiff_img = tiff.imread(path)
    return tiff_img


def resize_organized_pc(organized_pc, target_height=224, target_width=224, tensor_out=True, mode='nearest'):
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0).contiguous()
    # need to be improved by change to another method instead of nn functional
    if mode in {'linear', 'bilinear', 'bicubic', 'trilinear'}:
        torch_resized_organized_pc = torch.nn.functional.interpolate(
            torch_organized_pc,
            size=(target_height, target_width),
            mode=mode,
            align_corners=False,
        )
    else:
        torch_resized_organized_pc = torch.nn.functional.interpolate(
            torch_organized_pc,
            size=(target_height, target_width),
            mode=mode,
        )
    if tensor_out:
        return torch_resized_organized_pc.squeeze(dim=0).contiguous()
    else:
        return torch_resized_organized_pc.squeeze().permute(1, 2, 0).contiguous().numpy()


def organized_pc_to_depth_map(organized_pc):
    return organized_pc[:, :, 2]


def organized_pc_to_surface_normal_map(organized_pc, eps=1e-8):
    pc = np.asarray(organized_pc, dtype=np.float32)
    valid = np.all(pc != 0, axis=2)

    dx = np.zeros_like(pc, dtype=np.float32)
    dy = np.zeros_like(pc, dtype=np.float32)
    dx[:, 1:-1, :] = pc[:, 2:, :] - pc[:, :-2, :]
    dx[:, 0, :] = pc[:, 1, :] - pc[:, 0, :]
    dx[:, -1, :] = pc[:, -1, :] - pc[:, -2, :]
    dy[1:-1, :, :] = pc[2:, :, :] - pc[:-2, :, :]
    dy[0, :, :] = pc[1, :, :] - pc[0, :, :]
    dy[-1, :, :] = pc[-1, :, :] - pc[-2, :, :]

    normals = np.cross(dx, dy)
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normal_valid = valid & (norm[..., 0] > eps)
    normals = normals / np.maximum(norm, eps)

    flip = normals[..., 2] < 0
    normals[flip] *= -1
    normals[~normal_valid] = 0

    sn_map = (normals + 1.0) * 0.5
    sn_map[~normal_valid] = 0
    return sn_map.astype(np.float32)
