#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   cd /root/autodl-tmp/PIRN-CMPT/CMDIAD-main
#   CLASSES=rope K=4 SEED=0 MODE=rgb bash scripts/run_ablation_for_thesis.sh
#
# MODE:
#   full : RGB+SN both available
#   rgb  : RGB available, SN missing, CMPT supplies pseudo SN unless ablated
#   sn   : SN available, RGB missing, CMPT supplies pseudo RGB unless ablated

DATASET_PATH=${DATASET_PATH:-data}
DINO_CKPT=${DINO_CKPT:-/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth}
RGB_CKPT=${RGB_CKPT:-$DINO_CKPT}
SN_CKPT=${SN_CKPT:-$DINO_CKPT}
CLASSES=${CLASSES:-""}
K=${K:-10}
SEED=${SEED:-0}
MODE=${MODE:-rgb}
CMPT_EPOCHS=${CMPT_EPOCHS:-80}
PROTOTYPES=${PROTOTYPES:-128}
RGB_SIZE=${RGB_SIZE:-518}

COMMON=(
  --method_name PIRN_CMPT
  --dataset_type mvtec3d
  --dataset_path "$DATASET_PATH"
  --rgb_backbone_name vit_base_patch14_dinov2.lvd142m
  --sn_backbone_name vit_base_patch14_dinov2.lvd142m
  --rgb_checkpoint_path "$RGB_CKPT"
  --sn_checkpoint_path "$SN_CKPT"
  --rgb_size "$RGB_SIZE"
  --few_shot_k "$K"
  --shot_seed "$SEED"
  --cmpt_epochs "$CMPT_EPOCHS"
  --cmpt_lr 1e-4
  --num_rgb_prototypes "$PROTOTYPES"
  --num_sn_prototypes "$PROTOTYPES"
  --num_shared_prototypes "$PROTOTYPES"
  --rgb_s_lambda 1.0
  --rgb_smap_lambda 1.0
  --sn_s_lambda 1.0
  --sn_smap_lambda 1.0
  --cmpt_s_lambda 1.0
  --cmpt_smap_lambda 1.0
  --save_fewshot_list
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
    --cmpt_save_path "checkpoints/cmpt_ablation_${name}_k${K}_seed${SEED}_${MODE}_{class_name}.pth" \
    --experiment_note "ablation_${name}_k${K}_seed${SEED}_${MODE}" "$@"
}

run_one full --paper_mnc
run_one no_cmpt --disable_cmpt --paper_mnc
run_one no_shared --disable_shared_proto --paper_mnc
run_one only_shared --only_shared_proto --paper_mnc
run_one no_apr --disable_apr --paper_mnc
run_one no_mnc --disable_mnc
run_one no_pseudo --disable_pseudo_proto --paper_mnc
