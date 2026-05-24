# PIRN-Style Experiment Change Log

Date: 2026-05-24

## Why This Change Was Added

OpenReview reviews and author responses for PIRN indicate that the original method relies on a compact normality bottleneck:

- prototype count `K=10`;
- patch token count around `N=256`;
- BPA as balanced OT;
- APR as conservative GRU-style gated prototype update;
- MNC as prototype-level graph/cross attention followed by gated cross-modal normality injection.

The previous PIRN_CMPT runs used `K=128` and a `56 x 56` token grid (`N=3136`). This can weaken the prototype bottleneck and let anomalies be reconstructed too well, which is consistent with the observed image-level AUROC drop when learnable MNC is enabled.

## Code Change

A reversible argument was added:

```bash
--cmpt_feature_grid
```

Default:

```bash
--cmpt_feature_grid 56
```

This preserves the previous behavior exactly.

PIRN-style diagnostic setting:

```bash
--cmpt_feature_grid 16
--num_rgb_prototypes 10
--num_sn_prototypes 10
--num_shared_prototypes 10
```

This uses a compact 16x16 token grid, matching the `N=256` setting described in PIRN's author response.

## Files Changed

- `main.py`
  - added `--cmpt_feature_grid`, default `56`.
- `feature_extractors/cmpt_features.py`
  - added `self.cmpt_feature_pool = AdaptiveAvgPool2d((cmpt_feature_grid, cmpt_feature_grid))`;
  - replaced fixed `resize56` use in CMPT/PIRN_CMPT token extraction with the configurable pool;
  - replaced fixed `56 x 56` SN foreground mask with `cmpt_feature_grid x cmpt_feature_grid`.

## Rollback

No code rollback is required for normal experiments.

To recover the previous behavior, omit `--cmpt_feature_grid` or explicitly set:

```bash
--cmpt_feature_grid 56
```

Also keep the previous prototype setting:

```bash
--num_rgb_prototypes 128
--num_sn_prototypes 128
--num_shared_prototypes 128
```

## First Experiment To Run

Full modality, no learnable MNC, PIRN-style compact bottleneck:

```bash
python main.py \
--method_name PIRN_CMPT \
--main_modality "" \
--dataset_type mvtec3d \
--dataset_path data \
--classes rope \
--rgb_backbone_name vit_base_patch14_dinov2.lvd142m \
--sn_backbone_name vit_base_patch14_dinov2.lvd142m \
--rgb_checkpoint_path /root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth \
--sn_checkpoint_path /root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth \
--rgb_size 518 \
--max_sample 10 \
--cmpt_feature_grid 16 \
--cmpt_epochs 80 \
--cmpt_lr 1e-4 \
--num_rgb_prototypes 10 \
--num_sn_prototypes 10 \
--num_shared_prototypes 10 \
--rgb_s_lambda 1.0 \
--rgb_smap_lambda 1.0 \
--sn_s_lambda 1.0 \
--sn_smap_lambda 1.0 \
--cmpt_save_path checkpoints/cmpt_rope_10shot_full_grid16_k10.pth \
--experiment_note pirn_cmpt_rope_10shot_full_grid16_k10 \
--save_seg_results \
--save_component_maps
```

Then run the same setting with learnable MNC:

```bash
--mnc_learnable \
--mnc_epochs 40 \
--mnc_lr 1e-4 \
--mnc_batch_size 2048 \
--mnc_max_train_tokens 50000 \
--mnc_save_path checkpoints/mnc_rope_10shot_full_grid16_k10.pth
```

Do not combine this first pass with SN preprocessing, foreground masking, or altered fusion weights.

## Result And Decision

The first `grid16 + K10` full-modality rope 10-shot run produced:

| Setting | Image AUROC | Pixel AUROC | AU-PRO | AU-PRO-0.01 |
|---|---:|---:|---:|---:|
| full, grid16, K10, no learnable MNC | 0.829 | 0.916 | 0.724 | 0.251 |

This is much worse than the previous stable baseline:

| Setting | Image AUROC | Pixel AUROC | AU-PRO | AU-PRO-0.01 |
|---|---:|---:|---:|---:|
| full, grid56, K128, no learnable MNC | 0.976 | 0.985 | 0.945 | 0.403 |

Decision:

- Do not use `grid16 + K10` as the main setting.
- The result indicates under-capacity in the current implementation.
- PIRN's original `K=10` depends on its full learnable layer-wise decoder, GRU APR, and GAT-based MNC. In the current simplified prototype reconstruction implementation, reducing both token grid and prototype count at once removes too much capacity.
- Roll back experiments by omitting `--cmpt_feature_grid` and using `K=128`.

Recommended recovery command settings:

```bash
# omit --cmpt_feature_grid, or explicitly use:
--cmpt_feature_grid 56 \
--num_rgb_prototypes 128 \
--num_sn_prototypes 128 \
--num_shared_prototypes 128
```

If further diagnosis is needed, isolate one factor at a time:

```bash
# test prototype count only
--cmpt_feature_grid 56 --num_rgb_prototypes 10 --num_sn_prototypes 10 --num_shared_prototypes 10

# test token grid only
--cmpt_feature_grid 16 --num_rgb_prototypes 128 --num_sn_prototypes 128 --num_shared_prototypes 128
```
