# CMPT training upgrade

Date: 2026-05-25

## Goal

Make CMPT contribute more clearly under the thesis setting:

- few-shot normal training;
- RGB/SN cross-modal transfer;
- missing-modality inference.

The previous CMPT objective only minimized direct feature regression:

```text
RGB -> pseudo SN ~= SN
SN  -> pseudo RGB ~= RGB
```

This is weak in few-shot training because the MLP can overfit paired tokens without preserving normal-token geometry.

## New objective

CMPT now trains with four complementary losses:

1. Direct feature regression.
2. Cosine feature alignment.
3. Cycle consistency:

```text
RGB -> pseudo SN -> cycle RGB ~= RGB
SN  -> pseudo RGB -> cycle SN  ~= SN
```

4. Relational consistency over sampled normal tokens, which preserves the pairwise similarity structure of normal local patterns.

Default weights:

```bash
--cmpt_cycle_loss_weight 0.25
--cmpt_cosine_loss_weight 0.1
--cmpt_relation_loss_weight 0.05
--cmpt_relation_tokens 512
```

## Reliability gate change

CMPT reliability is now estimated as a 0-1 cycle-consistency score on normal training tokens. Missing-modality scoring still multiplies it by `--cmpt_aux_weight`, while full-modality CMPT consistency uses the raw reliability score with `--cmpt_full_consistency_weight`.

This prevents the full-modality CMPT consistency term from being accidentally down-weighted twice.

## Ablation interpretation

In the module table, `w/o CMPT` removes:

- missing-modality feature/prototype completion;
- CMPT normality communication;
- CMPT full-modality consistency scoring.

This keeps the comparison aligned with the thesis claim that CMPT is the cross-modal module, not just an auxiliary pseudo-branch.
