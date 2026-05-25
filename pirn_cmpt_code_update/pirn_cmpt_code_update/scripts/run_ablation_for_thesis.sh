#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   cd PIRN-CMPT/CMDIAD-main
#   K=4 SEED=0 MODE=rgb bash /path/to/pirn_cmpt_code_update/scripts/run_ablation_for_thesis.sh

DATASET_PATH=${DATASET_PATH:-datasets/mvtec_3d}
RGB_CKPT=${RGB_CKPT:-checkpoints/dinov2_vitb14_pretrain.pth}
SN_CKPT=${SN_CKPT:-checkpoints/dinov2_vitb14_pretrain.pth}
CLASSES=${CLASSES:-""}
K=${K:-4}
SEED=${SEED:-0}
MODE=${MODE:-rgb}   # rgb, sn, or full

COMMON=(
  --method_name PIRN_CMPT
  --dataset_path "$DATASET_PATH"
  --rgb_checkpoint_path "$RGB_CKPT"
  --sn_checkpoint_path "$SN_CKPT"
  --few_shot_k "$K"
  --shot_seed "$SEED"
  --save_fewshot_list
  --paper_mnc
  --save_results True
)

if [[ -n "$CLASSES" ]]; then
  COMMON+=(--classes "$CLASSES")
fi

if [[ "$MODE" == "full" ]]; then
  MODALITY_ARGS=(--main_modality '')
else
  MODALITY_ARGS=(--main_modality "$MODE" --allow_true_missing_modality)
fi

run_one() {
  local name=$1
  shift
  python main.py "${COMMON[@]}" "${MODALITY_ARGS[@]}" \
    --experiment_note "K${K}_seed${SEED}_${MODE}_${name}" "$@"
}

run_one full
run_one wo_cmpt --disable_cmpt
run_one wo_shared --disable_shared_proto
run_one wo_apr --disable_apr
run_one wo_mnc --disable_mnc
run_one wo_pseudo --disable_pseudo_proto
