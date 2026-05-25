#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   cd /root/autodl-tmp/PIRN-CMPT/CMDIAD-main
#   CLASSES=rope K_LIST="1 2 4 8 16" SEEDS="0 1 2" bash scripts/run_kshot_system_experiments.sh

DATASET_PATH=${DATASET_PATH:-data}
DINO_CKPT=${DINO_CKPT:-/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth}
RGB_CKPT=${RGB_CKPT:-$DINO_CKPT}
SN_CKPT=${SN_CKPT:-$DINO_CKPT}
CLASSES=${CLASSES:-""}
SEEDS=${SEEDS:-"0 1 2"}
K_LIST=${K_LIST:-"1 2 4 8 16"}
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
  --cmpt_epochs "$CMPT_EPOCHS"
  --cmpt_lr 1e-4
  --cmpt_cycle_loss_weight 0.25
  --cmpt_cosine_loss_weight 0.1
  --cmpt_relation_loss_weight 0.05
  --cmpt_aux_weight 0.35
  --cmpt_aux_confidence_threshold 0.50
  --cmpt_aux_mode both
  --cmpt_full_consistency_weight 0.25
  --cmpt_full_calibration_std -1.0
  --cmpt_full_map_gain 1.0
  --cmpt_nc_weight 0.20
  --cmpt_nc_confidence_threshold 0.55
  --cmpt_nc_safe_margin 0.02
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

for K in $K_LIST; do
  for SEED in $SEEDS; do
    python main.py "${COMMON[@]}" \
      --main_modality '' \
      --few_shot_k "$K" \
      --shot_seed "$SEED" \
      --cmpt_save_path "checkpoints/cmpt_k${K}_seed${SEED}_full_{class_name}.pth" \
      --experiment_note "k${K}_seed${SEED}_full"

    python main.py "${COMMON[@]}" \
      --main_modality rgb \
      --allow_true_missing_modality \
      --few_shot_k "$K" \
      --shot_seed "$SEED" \
      --cmpt_save_path "checkpoints/cmpt_k${K}_seed${SEED}_missing_sn_{class_name}.pth" \
      --experiment_note "k${K}_seed${SEED}_rgb_missing_sn"

    python main.py "${COMMON[@]}" \
      --main_modality sn \
      --allow_true_missing_modality \
      --few_shot_k "$K" \
      --shot_seed "$SEED" \
      --cmpt_save_path "checkpoints/cmpt_k${K}_seed${SEED}_missing_rgb_{class_name}.pth" \
      --experiment_note "k${K}_seed${SEED}_sn_missing_rgb"
  done
done
