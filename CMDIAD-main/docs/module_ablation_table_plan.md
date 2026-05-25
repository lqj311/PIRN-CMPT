# Module ablation table plan

Date: 2026-05-25

## Table style

Use the same incremental module table style as PIRN:

| APR | CMPT | SPA | MNC | AUROC-I | AUROC-P | AUPRO |
|:--:|:--:|:--:|:--:|--:|--:|--:|
| x | x | x | x | replace | replace | replace |
| x | check | check | check | replace | replace | replace |
| check | x | check | check | replace | replace | replace |
| check | check | x | check | replace | replace | replace |
| check | check | check | x | replace | replace | replace |
| check | check | check | check | replace | replace | replace |

## Row definitions

1. Base:

```bash
--disable_apr --disable_cmpt --disable_spa --disable_mnc --paper_mnc
```

2. w/o APR:

```bash
--disable_apr --paper_mnc
```

3. w/o CMPT:

```bash
--disable_cmpt --paper_mnc
```

4. w/o SPA:

```bash
--disable_spa --paper_mnc
```

5. w/o MNC:

```bash
--disable_mnc --paper_mnc
```

6. Full:

```bash
--paper_mnc
```

## Important implementation note

`--disable_mnc` disables only the MNC stage2 self-normality correction. It does not disable CMPT-NC. This keeps the `w/o MNC` row from accidentally removing CMPT.

`--disable_spa` replaces structured OT assignment with nearest-prototype reconstruction. Therefore, `SPA x` is a nearest-prototype baseline, while `SPA check` is the structured prototype assignment.

## Test script

```bash
cd /root/autodl-tmp/PIRN-CMPT/CMDIAD-main
CLASSES=rope SEEDS="0" K=16 bash scripts/run_module_ablation_table.sh
```
