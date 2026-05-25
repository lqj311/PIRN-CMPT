# Final three ablation plan

Date: 2026-05-25

The final thesis experiments follow PIRN's ablation style: use compact module tables and controlled scenario comparisons. We only borrow the experiment logic, not the exact hyperparameters.

The final ablation section contains three groups:

1. Module ablation.
2. Few-shot experiment.
3. Missing-modality experiment.

## 1. Module Ablation

Purpose: verify that APR, CMPT, SPA, and MNC each contribute to the complete model.

Table format:

| APR | CMPT | SPA | MNC | AUROC-I | AUROC-P | AUPRO |
|:--:|:--:|:--:|:--:|--:|--:|--:|
| x | x | x | x | replace | replace | replace |
| x | check | check | check | replace | replace | replace |
| check | x | check | check | replace | replace | replace |
| check | check | x | check | replace | replace | replace |
| check | check | check | x | replace | replace | replace |
| check | check | check | check | replace | replace | replace |

Command mapping:

```bash
# Base
--disable_apr --disable_cmpt --disable_spa --disable_mnc --paper_mnc

# w/o APR
--disable_apr --paper_mnc

# w/o CMPT
--disable_cmpt --paper_mnc

# w/o SPA
--disable_spa --paper_mnc

# w/o MNC
--disable_mnc --paper_mnc

# Full
--paper_mnc
```

Implementation notes:

- `--disable_spa` means nearest-prototype reconstruction instead of structured OT assignment.
- `--disable_mnc` disables MNC stage2 only. CMPT-NC is still active if CMPT is enabled.
- CMPT includes both missing-modality completion and cross-modal normality communication.

## 2. Few-Shot Experiment

Purpose: verify that the complete model is stable when normal training samples are limited.

Use Full Model only:

| K-shot | AUROC-I | AUROC-P | AUPRO |
|--:|--:|--:|--:|
| 1 | replace | replace | replace |
| 2 | replace | replace | replace |
| 4 | replace | replace | replace |
| 8 | replace | replace | replace |
| 16 | replace | replace | replace |

Default thesis main result uses K=16. K=16 is reused as the Full row in module ablation.

## 3. Missing-Modality Experiment

Purpose: verify that CMPT improves testing when RGB or SN is missing.

Table format:

| Test modality | CMPT | AUROC-I | AUROC-P | AUPRO |
|---|:--:|--:|--:|--:|
| RGB-only | x | replace | replace | replace |
| RGB-only | check | replace | replace | replace |
| SN-only | x | replace | replace | replace |
| SN-only | check | replace | replace | replace |

Command mapping:

```bash
# RGB-only Full
--main_modality rgb --allow_true_missing_modality --paper_mnc

# RGB-only w/o CMPT
--main_modality rgb --allow_true_missing_modality --disable_cmpt --paper_mnc

# SN-only Full
--main_modality sn --allow_true_missing_modality --paper_mnc

# SN-only w/o CMPT
--main_modality sn --allow_true_missing_modality --disable_cmpt --paper_mnc
```

## Final Script

```bash
cd /root/autodl-tmp/PIRN-CMPT/CMDIAD-main
bash scripts/run_thesis_final_15configs.sh
```

Debug run:

```bash
CLASSES=rope,tire,bagel SEEDS="0" bash scripts/run_thesis_final_15configs.sh
python scripts/show_thesis_results.py --filter final_
```
