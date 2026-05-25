#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   cd PIRN-CMPT/CMDIAD-main
#   bash /path/to/pirn_cmpt_code_update/scripts/run_kshot_system_experiments.sh

DATASET_PATH=${DATASET_PATH:-datasets/mvtec_3d}
RGB_CKPT=${RGB_CKPT:-checkpoints/dinov2_vitb14_pretrain.pth}
SN_CKPT=${SN_CKPT:-checkpoints/dinov2_vitb14_pretrain.pth}
CLASSES=${CLASSES:-""}     # e.g. "bagel,rope"; empty means all classes
SEEDS=${SEEDS:-"0 1 2"}
K_LIST=${K_LIST:-"1 2 4 8 16"}

COMMON=(
  --method_name PIRN_CMPT
  --dataset_path "$DATASET_PATH"
  --rgb_checkpoint_path "$RGB_CKPT"
  --sn_checkpoint_path "$SN_CKPT"
  --paper_mnc
  --save_results True
  --save_fewshot_list
)

if [[ -n "$CLASSES" ]]; then
  COMMON+=(--classes "$CLASSES")
fi

for K in $K_LIST; do
  for SEED in $SEEDS; do
    for MODE in rgb sn ""; do
      if [[ "$MODE" == "rgb" ]]; then
        NOTE="K${K}_seed${SEED}_rgb_missing_sn_full"
        python main.py "${COMMON[@]}" \
          --main_modality rgb \
          --allow_true_missing_modality \
          --few_shot_k "$K" \
          --shot_seed "$SEED" \
          --experiment_note "$NOTE"
      elif [[ "$MODE" == "sn" ]]; then
        NOTE="K${K}_seed${SEED}_sn_missing_rgb_full"
        python main.py "${COMMON[@]}" \
          --main_modality sn \
          --allow_true_missing_modality \
          --few_shot_k "$K" \
          --shot_seed "$SEED" \
          --experiment_note "$NOTE"
      else
        NOTE="K${K}_seed${SEED}_full_modality_full"
        python main.py "${COMMON[@]}" \
          --main_modality '' \
          --few_shot_k "$K" \
          --shot_seed "$SEED" \
          --experiment_note "$NOTE"
      fi
    done
  done
done
