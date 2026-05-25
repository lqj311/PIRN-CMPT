# CMPT-NC change log

Date: 2026-05-25

## Goal

Move CMPT from a missing-modality auxiliary branch into the main PIRN reconstruction path.

The new paper path is:

```text
SPA -> CMPT-NC -> MNC stage2 -> error map
```

instead of:

```text
SPA -> MNC stage1 -> MNC stage2 -> error map
```

## What changed

1. Added `--cmpt_replace_mnc1`.

   This uses CMPT as the first-stage cross-modal normality communication module.

2. Added `--classic_pirn_mnc1`.

   This keeps the old PIRN-style MNC stage1 for rollback or comparison.

3. Added CMPT-NC gates:

   - `--cmpt_nc_weight`, default `0.1`
   - `--cmpt_nc_confidence_threshold`, default `0.6`
   - `--cmpt_nc_safe_margin`, default `0.0`

4. Updated CMPT-NC to a conservative residual communication block.

   The candidate reconstruction produced through CMPT must satisfy four gates before it is injected into the original SPA reconstruction:

   - cross-modal prototype assignment confidence;
   - consistency with the original patch token;
   - consistency with the original SPA reconstruction;
   - non-worse normal-prototype compatibility.

   This follows the PIRN-style "gated normality communication" idea: cross-modal information can refine normal reconstruction, but it is not allowed to overwrite a stronger single-branch reconstruction.

5. APR now has a safety check.

   After prototype refinement, the refined prototypes are evaluated on normal training tokens. If refinement increases normal reconstruction error, the module automatically falls back to the original k-means prototypes.

6. Added full-modality CMPT consistency scoring.

   When RGB and SN are both available, CMPT now also computes RGB-to-SN and SN-to-RGB consistency maps. The map is gated by CMPT reliability estimated on normal training tokens and added with a small default maximum weight:

   - `--cmpt_full_consistency_weight`, default `0.15`

   This gives CMPT a measurable cross-modal consistency role in the full-modality setting, while `--disable_cmpt` removes both missing-modality completion and this consistency constraint.

7. `--paper_mnc` now enables `--cmpt_replace_mnc1` by default unless `--classic_pirn_mnc1` is passed.

## Interpretation

With the new path, CMPT includes two functions:

1. Missing modality completion.
2. Cross-modal normality communication, replacing MNC stage1.

Therefore, in ablation experiments, `--disable_cmpt` removes both pseudo-modality completion and the first-stage cross-modal normality communication. MNC stage2 is still kept, so the baseline remains a valid PIRN-style self-normalization path.

## Rollback

To recover the previous two-stage PIRN MNC behavior, add:

```bash
--classic_pirn_mnc1
```

To remove cross-modal normality communication entirely, use:

```bash
--disable_cmpt
```

or:

```bash
--disable_mnc
```
