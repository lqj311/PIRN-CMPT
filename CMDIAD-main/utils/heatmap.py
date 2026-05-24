import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def load_score_map(path):
    data = torch.load(path, map_location='cpu')
    if isinstance(data, dict):
        for key in ('s_map', 'score_map', 'heatmap', 'prediction'):
            if key in data:
                data = data[key]
                break
    if torch.is_tensor(data):
        data = data.detach().cpu().squeeze().numpy()
    else:
        data = np.asarray(data).squeeze()
    if data.ndim != 2:
        raise ValueError(f'Expected a 2D score map in `{path}`, got shape {data.shape}.')
    return data


def render_heatmap(input_path, output_path, cmap='YlGnBu_r', dpi=300, vmin=None, vmax=None, cbar=False):
    score_map = load_score_map(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)
    sns.heatmap(
        data=score_map,
        ax=ax,
        cmap=cmap,
        cbar=cbar,
        xticklabels=[],
        yticklabels=[],
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_axis_off()
    plt.savefig(output_path, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close(fig)


def iter_inputs(args):
    if args.input:
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else input_path.with_suffix('.png')
        yield input_path, output_path

    if args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) if args.output_dir else input_dir
        for input_path in sorted(input_dir.rglob(args.glob)):
            rel_path = input_path.relative_to(input_dir).with_suffix('.png')
            yield input_path, output_dir / rel_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render CMDIAD-style heatmaps from saved segmentation score tensors.'
    )
    parser.add_argument('--input', default='', type=str, help='Single .pt score map to render.')
    parser.add_argument('--output', default='', type=str, help='Output PNG for --input.')
    parser.add_argument('--input_dir', default='', type=str, help='Directory containing saved .pt score maps.')
    parser.add_argument('--output_dir', default='', type=str, help='Directory for rendered PNG heatmaps.')
    parser.add_argument('--glob', default='*.pt', type=str, help='File pattern under --input_dir.')
    parser.add_argument('--cmap', default='YlGnBu_r', type=str, help='Seaborn/matplotlib colormap.')
    parser.add_argument('--dpi', default=300, type=int)
    parser.add_argument('--vmin', default=None, type=float)
    parser.add_argument('--vmax', default=None, type=float)
    parser.add_argument('--cbar', default=False, action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    count = 0
    for input_path, output_path in iter_inputs(args):
        render_heatmap(input_path, output_path, args.cmap, args.dpi, args.vmin, args.vmax, args.cbar)
        count += 1
    if count == 0:
        raise SystemExit('No input heatmap tensors were found. Pass --input or --input_dir.')
    print(f'Rendered {count} heatmap(s).')


if __name__ == '__main__':
    main()
