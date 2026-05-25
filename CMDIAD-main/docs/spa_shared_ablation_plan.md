# SPA shared-prototype ablation plan

Date: 2026-05-25

## Rationale

The thesis no longer treats shared prototypes as an isolated on/off component. Shared prototypes are part of the structured normal prototype memory, and their effect depends on how patch tokens are assigned to prototypes.

Therefore, the third core ablation is changed from:

```text
w/o prototype / specific only / shared only / specific + shared
```

to:

```text
Nearest prototype / Top-k prototype / Softmax assignment / Structured OT assignment
```

All SPA ablations keep the full prototype design, including RGB-specific, SN-specific, and shared prototypes. The controlled variable is only the assignment mechanism.

## Thesis interpretation

This experiment answers:

> Does structured prototype assignment make shared normal prototypes useful under few-shot cross-modal anomaly detection?

Expected analysis:

1. Nearest assignment is too hard and can overfit limited K-shot prototypes.
2. Top-k and softmax improve smoothness but do not enforce balanced prototype usage.
3. Structured OT encourages balanced and structure-aware assignment, so shared prototypes contribute as a stable normal-structure constraint rather than competing noisily with specific prototypes.

## Script notes

`scripts/run_thesis_final_15configs.sh` now uses:

- `final_k16_seed*_spa_nearest`
- `final_k16_seed*_spa_topk`
- `final_k16_seed*_spa_softmax`
- `final_k16_seed*_full_model` as structured OT full model

The old prototype on/off notes are no longer part of the final thesis plan.

## Paper table style

Use a module-style ablation table, following the PIRN paper format:

| SPA-NN | SPA-TopK | SPA-Soft | SPA-OT | AUROC-I | AUROC-P | AUPRO |
|:--:|:--:|:--:|:--:|--:|--:|--:|
| x | x | x | x | replace | replace | replace |
| check | x | x | x | replace | replace | replace |
| x | check | x | x | replace | replace | replace |
| x | x | check | x | replace | replace | replace |
| x | x | x | check | replace | replace | replace |

Recommended row mapping:

1. `w/o SPA`: use `--disable_prototypes` only if a pure non-prototype baseline is required.
2. `SPA-NN`: `--spa_assignment nearest`.
3. `SPA-TopK`: `--spa_assignment topk --spa_topk 5`.
4. `SPA-Soft`: `--spa_assignment softmax`.
5. `SPA-OT`: full model, default `--spa_assignment structured_ot`.

In the final 15-config script, rows 2-5 are used. Row 1 can be added only if the thesis needs a non-prototype lower-bound baseline.

Suggested Chinese caption:

```text
表5-x SPA结构化原型分配模块消融实验
```

Suggested analysis:

```text
最近邻分配仅选择单一原型，容易受到少样本原型偏差影响；Top-k和Softmax分配能够利用多个原型进行平滑重建，但缺少全局结构约束。结构化OT分配在保持多原型表达能力的同时约束原型使用分布，使共享原型能够作为稳定的跨模态正常结构参与重建，因此在像素级定位和区域级定位指标上表现更稳定。
```
