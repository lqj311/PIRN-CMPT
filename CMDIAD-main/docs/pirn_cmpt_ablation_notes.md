# PIRN-CMPT Experiment And Ablation Notes

Date: 2026-05-24

## Design Position

This project should be treated as a PIRN-based method with the following changes:

- BPA is replaced by SPA.
- APR and MNC should stay close to PIRN's original intent.
- CMPT is added only for missing-modality completion.
- When RGB and SN are both available, the full pipeline should use the PIRN-style real RGB/SN reconstruction flow and should not use pseudo-modal CMPT branches for scoring.
- When one modality is missing, CMPT completes pseudo features and pseudo prototypes, then the missing branch can still produce a reconstruction error map.

## Current Code Switches

Baseline behavior is intentionally kept stable:

- `--mnc_strong` is off by default.
- `--apr_memory_update_iters` defaults to `0`.
- `--sn_resize_mode nearest`, `--sn_smooth_kernel 1`, `--sn_foreground_only False`, `--sn_mask_error_map False` restore the previous SN behavior.
- Full mode `--main_modality ""` uses real RGB and real SN only.
- Missing-modality modes:
  - `--main_modality rgb`: real RGB plus CMPT pseudo SN.
  - `--main_modality sn`: real SN plus CMPT pseudo RGB.

## Recorded Results

All listed runs use DINOv2 ViT-B/14 checkpoints and MVTec 3D-AD.

| Class | Setting | Main Modality | Shots | Strong MNC | APR Memory | Image AUROC | Pixel AUROC | AU-PRO | AU-PRO-0.01 | Note |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| rope | baseline full | RGB+SN | 10 | no | 0 | 0.976 | 0.985 | 0.945 | 0.403 | current best full baseline |
| rope | strong PIRN approx | RGB+SN | 10 | yes | 3 | 0.979 | 0.981 | 0.936 | 0.399 | image improves slightly, localization drops |
| rope | SN-fixed only | SN | 10 | no | 0 | 0.962 | 0.848 | 0.798 | 0.331 | SN preprocessing/mask combination hurts pixel ranking |
| rope | full with SN-fixed | RGB+SN | 10 | no | 0 | 0.975 | 0.882 | 0.856 | 0.366 | masking/preprocess hurts pixel AUROC |
| tire | baseline full | RGB+SN | 10 | no | 0 | 0.702 | 0.807 | 0.530 | 0.120 | SN harms full fusion |
| tire | RGB-only | RGB | 10 | no | 0 | 0.710 | 0.861 | 0.643 | 0.200 | best current tire localization |
| tire | SN-only | SN | 10 | no | 0 | 0.571 | 0.569 | 0.157 | 0.022 | SN branch nearly fails |
| tire | 50-shot full | RGB+SN | 50 | no | 0 | 0.743 | 0.858 | 0.616 | 0.150 | more normal coverage helps image score, not enough for localization |
| tire | strong PIRN approx | RGB+SN | 10 | yes | 3 | 0.711 | 0.790 | 0.515 | 0.119 | image improves slightly, localization drops |
| rope | learnable MNC full | RGB+SN | 10 | learnable | 0 | 0.944 | 0.991 | 0.940 | 0.411 | improves Pixel AUROC and low-FPR PRO, hurts image ranking |
| rope | learnable MNC full e20 | RGB+SN | 10 | learnable | 0 | 0.939 | 0.991 | 0.942 | 0.411 | fewer MNC epochs does not recover image AUROC |
| rope | missing SN, no learnable MNC | RGB -> pseudo SN | 10 | no | 0 | 0.971 | 0.954 | 0.900 | 0.359 | CMPT alone keeps strong image-level detection |
| rope | missing SN, learnable MNC | RGB -> pseudo SN | 10 | learnable | 0 | 0.913 | 0.984 | 0.941 | 0.387 | learnable MNC greatly improves localization but lowers image AUROC |
| rope | missing RGB, no learnable MNC | SN -> pseudo RGB | 10 | no | 0 | 0.964 | 0.936 | 0.790 | 0.298 | CMPT alone is weaker when starting from SN |
| rope | missing RGB, learnable MNC | SN -> pseudo RGB | 10 | learnable | 0 | 0.957 | 0.979 | 0.858 | 0.351 | learnable MNC improves localization with minor image drop |

## Main Observations

1. The strongest current full result is still the baseline full mode on rope: `0.976 / 0.985 / 0.945`.
2. The strong PIRN approximation is not a default improvement. It slightly improves image-level AUROC but decreases pixel-level localization.
3. Tire exposes the weakness of the current SN branch. RGB-only is better than full and much better than SN-only.
4. SN preprocessing and foreground masking should not be enabled by default. The tested combination improves neither Pixel AUROC nor full-mode stability.
5. CMPT should be evaluated under missing-modality settings only. It should not affect full-mode baseline scoring.
6. Learnable MNC behaves like a localization enhancer. It improves Pixel AUROC and AU-PRO-0.01 but can reduce image-level AUROC because the decoder changes and smooths the residual distribution.

## How To Run Ablations

Only change one factor at a time. Do not mix APR, strong MNC, SN preprocessing, prototype number, and fusion weights in the same diagnostic run.

### A. Baseline

Use default APR/MNC/SN behavior:

```bash
--mnc_stages 2 \
--rgb_s_lambda 1.0 \
--rgb_smap_lambda 1.0 \
--sn_s_lambda 1.0 \
--sn_smap_lambda 1.0
```

Do not pass:

```bash
--mnc_strong
--apr_memory_update_iters 3
--sn_mask_error_map True
```

### B. APR-Only

Purpose: test prototype memory update without changing MNC.

```bash
--apr_memory_update_iters 3
```

Do not pass `--mnc_strong`.

### C. MNC-Only

Purpose: test explicit two-stage MNC without APR memory update.

```bash
--mnc_strong \
--apr_memory_update_iters 0 \
--mnc_temperature 0.05 \
--mnc_stage1_weight 0.5 \
--mnc_stage2_weight 0.5
```

### D. APR + Strong MNC

Purpose: test the closest current approximation to PIRN's APR + MNC.

```bash
--mnc_strong \
--apr_memory_update_iters 3 \
--mnc_temperature 0.05 \
--mnc_stage1_weight 0.5 \
--mnc_stage2_weight 0.5
```

This is not the default because current results show localization degradation.

### D2. Learnable MNC + Decoder

Purpose: test a closer PIRN-style MNC implementation than the previous parameter-free approximation.

This variant adds a trainable gated cross-attention module and a feature reconstruction decoder. It is enabled only with:

```bash
--mnc_learnable \
--mnc_epochs 20 \
--mnc_lr 1e-4 \
--mnc_batch_size 2048 \
--mnc_max_train_tokens 50000 \
--mnc_num_heads 8 \
--mnc_decoder_hidden_ratio 2.0 \
--mnc_save_path checkpoints/mnc_{class_name}_{modality}_learnable.pth
```

For missing-modality ablation, also train pseudo routes:

```bash
--mnc_train_pseudo \
--mnc_pseudo_loss_weight 0.5
```

Do not mix this with `--mnc_strong` in the same diagnostic run. `--mnc_learnable` takes precedence over the old strong approximation during reconstruction.

### E. Prototype Number

Few-shot prototype count should be tested separately:

```bash
--num_rgb_prototypes 10  --num_sn_prototypes 10  --num_shared_prototypes 10
--num_rgb_prototypes 16  --num_sn_prototypes 16  --num_shared_prototypes 16
--num_rgb_prototypes 32  --num_sn_prototypes 32  --num_shared_prototypes 32
--num_rgb_prototypes 64  --num_sn_prototypes 64  --num_shared_prototypes 64
--num_rgb_prototypes 128 --num_sn_prototypes 128 --num_shared_prototypes 128
```

Run these with baseline MNC first. Do not combine prototype-number sweeps with strong MNC in the first pass.

### F. Modality Contribution

For each hard class, run:

```bash
# RGB-only
--rgb_s_lambda 1.0 --rgb_smap_lambda 1.0 --sn_s_lambda 0.0 --sn_smap_lambda 0.0

# SN-only
--rgb_s_lambda 0.0 --rgb_smap_lambda 0.0 --sn_s_lambda 1.0 --sn_smap_lambda 1.0

# Full
--rgb_s_lambda 1.0 --rgb_smap_lambda 1.0 --sn_s_lambda 1.0 --sn_smap_lambda 1.0

# RGB-dominant
--rgb_s_lambda 1.0 --rgb_smap_lambda 1.0 --sn_s_lambda 0.1 --sn_smap_lambda 0.1
```

This is mandatory for categories like tire, where SN-only can be unreliable.

### G. Missing-Modality CMPT

Use CMPT only for missing-modality experiments:

```bash
# missing SN
--main_modality rgb

# missing RGB
--main_modality sn
```

Full mode should remain:

```bash
--main_modality ""
```

## Why Strong PIRN Approximation Can Drop

The result drop is plausible even though the intended PIRN mechanism should improve performance.

1. PIRN's MNC is a learned gated cross-attention module. The OpenReview abstract describes MNC as exchanging high-level normal cues via gated cross-attention. Our current strong MNC is a parameter-free approximation over frozen DINOv2 tokens and prototypes, so it has no trained decoder weights to learn when cross-modal cues should be suppressed.

2. PIRN's APR is a gated prototype update mechanism. Our memory update is still a lightweight functional update, not a learned GRU-style prototype memory. In few-shot settings, overly aggressive updates can make prototypes fit limited normal samples too tightly and reduce pixel-level generalization.

3. DINOv2 is trained for general visual features on curated natural image data. Its paper describes large-scale visual pretraining and all-purpose image features. Surface normal maps are not natural RGB images, so SN tokens may not be as reliable as RGB tokens, especially on texture-dense categories like tire.

4. MVTec 3D-AD contains geometric defects and uses only anomaly-free samples for training. This makes few-shot normal coverage critical. For tire, dense repeated grooves and curved geometry make SN normal patterns diverse even for normal samples, so 10-shot normal prototypes can under-cover normal variation.

5. CFM and other multimodal methods do not simply add raw modality error maps. CFM maps features from one modality to another and detects inconsistencies between observed and mapped features. This supports the point that cross-modal interaction must be learned/calibrated, not only hand-mixed.

6. The observed metrics fit this explanation: strong MNC slightly increases image AUROC but reduces Pixel AUROC and AU-PRO. This means global anomaly separation can improve while localization becomes less sharp or more spatially spread.

## Working Conclusion

Use the baseline full mode as the current official result unless APR-only or MNC-only ablations prove otherwise.

For the current rope 10-shot ablation:

- Full-modality official result should remain the baseline full mode: `0.976 / 0.985 / 0.945 / 0.403`.
- Learnable MNC should be reported as a localization-oriented ablation: full-mode Pixel AUROC rises to `0.991` and AU-PRO-0.01 rises to `0.411`, while Image AUROC drops to `0.944` or `0.939`.
- Missing-SN CMPT is valid even without learnable MNC: `0.971 / 0.954 / 0.900 / 0.359`.
- Missing-SN with learnable MNC is better for localization: `0.913 / 0.984 / 0.941 / 0.387`.
- Missing-RGB with learnable MNC improves localization over no learnable MNC: Pixel AUROC `0.936 -> 0.979`, AU-PRO `0.790 -> 0.858`, AU-PRO-0.01 `0.298 -> 0.351`.
- The next technical fix should target image-level scoring, not the heatmap pipeline. Current `max` scoring is too sensitive to decoder-induced residual smoothing; top-k average or calibrated branch-wise scoring should be tested next.

For the paper narrative:

- SPA + CMPT design is valid.
- CMPT is for missing modality, not full modality.
- APR and MNC need to be reported carefully:
  - baseline uses stable prototype reconstruction and standard gated cross-modal reconstruction;
  - strong APR/MNC approximation is under ablation;
  - learnable MNC uses gated cross-attention plus decoder and should be reported as the closer PIRN-style implementation;
  - full PIRN-level gains require learnable gated cross-attention/decoder training, not just parameter-free reconstruction.

## References

- PIRN OpenReview page: https://openreview.net/forum?id=7L7kmHHfgf
- DINOv2 paper: https://arxiv.org/abs/2304.07193
- MVTec 3D-AD dataset paper: https://arxiv.org/abs/2112.09045
- Crossmodal Feature Mapping, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Costanzino_Multimodal_Industrial_Anomaly_Detection_by_Crossmodal_Feature_Mapping_CVPR_2024_paper.html
