#!/usr/bin/env bash
set -euo pipefail

# Module ablation table in PIRN-paper style:
#   Base + leave-one-out APR/CMPT/SPA/MNC + Full

DATASET_PATH=${DATASET_PATH:-data}
DINO_CKPT=${DINO_CKPT:-/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth}
RGB_CKPT=${RGB_CKPT:-$DINO_CKPT}
SN_CKPT=${SN_CKPT:-$DINO_CKPT}
CLASSES=${CLASSES:-rope}
SEEDS=${SEEDS:-0}
K=${K:-16}
CMPT_EPOCHS=${CMPT_EPOCHS:-80}
PROTOTYPES=${PROTOTYPES:-128}
RGB_SIZE=${RGB_SIZE:-518}
RUN_SET=${RUN_SET:-full}

COMMON=(
  --method_name PIRN_CMPT
  --main_modality ''
  --dataset_type mvtec3d
  --dataset_path "$DATASET_PATH"
  --classes "$CLASSES"
  --rgb_backbone_name vit_base_patch14_dinov2.lvd142m
  --sn_backbone_name vit_base_patch14_dinov2.lvd142m
  --rgb_checkpoint_path "$RGB_CKPT"
  --sn_checkpoint_path "$SN_CKPT"
  --rgb_size "$RGB_SIZE"
  --few_shot_k "$K"
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
)

run_exp() {
  local note=$1
  shift
  python main.py "${COMMON[@]}" \
    --cmpt_save_path "checkpoints/${note}_{class_name}.pth" \
    --experiment_note "$note" "$@"
}

for SEED in $SEEDS; do
  if [[ "$RUN_SET" == "minimal" ]]; then
    run_exp "module_k${K}_seed${SEED}_base" \
      --shot_seed "$SEED" \
      --disable_apr \
      --disable_cmpt \
      --disable_spa \
      --disable_mnc \
      --paper_mnc

    run_exp "module_k${K}_seed${SEED}_wo_cmpt" \
      --shot_seed "$SEED" \
      --disable_cmpt \
      --paper_mnc

    run_exp "module_k${K}_seed${SEED}_full" \
      --shot_seed "$SEED" \
      --paper_mnc
    continue
  fi

  run_exp "module_k${K}_seed${SEED}_base" \
    --shot_seed "$SEED" \
    --disable_apr \
    --disable_cmpt \
    --disable_spa \
    --disable_mnc \
    --paper_mnc

  run_exp "module_k${K}_seed${SEED}_wo_apr" \
    --shot_seed "$SEED" \
    --disable_apr \
    --paper_mnc

  run_exp "module_k${K}_seed${SEED}_wo_cmpt" \
    --shot_seed "$SEED" \
    --disable_cmpt \
    --paper_mnc

  run_exp "module_k${K}_seed${SEED}_wo_spa" \
    --shot_seed "$SEED" \
    --disable_spa \
    --paper_mnc

  run_exp "module_k${K}_seed${SEED}_wo_mnc" \
    --shot_seed "$SEED" \
    --disable_mnc \
    --paper_mnc

  run_exp "module_k${K}_seed${SEED}_full" \
    --shot_seed "$SEED" \
    --paper_mnc
done
