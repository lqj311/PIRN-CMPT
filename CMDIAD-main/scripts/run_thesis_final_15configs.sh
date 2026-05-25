#!/usr/bin/env bash
set -euo pipefail

# Final thesis experiment plan:
#   Exp-0 Main: Full Model, K=16, full modality
#   Exp-1 K-shot: K=1,2,4,8,16, full modality
#   Exp-2 Cross-modal missing: RGB-only/SN-only/full modality with and without CMPT
#   Exp-3 Shared prototype: no prototype/specific only/shared only/specific+shared
#
# Unique configs:
#   K-shot full-model gives K=16 main result, so total unique configs = 15.
#
# Usage:
#   cd /root/autodl-tmp/PIRN-CMPT/CMDIAD-main
#   bash scripts/run_thesis_final_15configs.sh
#
# Debug:
#   CLASSES=rope SEEDS="0" bash scripts/run_thesis_final_15configs.sh

DATASET_PATH=${DATASET_PATH:-data}
DINO_CKPT=${DINO_CKPT:-/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth}
RGB_CKPT=${RGB_CKPT:-$DINO_CKPT}
SN_CKPT=${SN_CKPT:-$DINO_CKPT}
CLASSES=${CLASSES:-"bagel,cable_gland,carrot,cookie,dowel,foam,peach,potato,rope,tire"}
SEEDS=${SEEDS:-"0 1 2"}
CMPT_EPOCHS=${CMPT_EPOCHS:-80}
PROTOTYPES=${PROTOTYPES:-128}
RGB_SIZE=${RGB_SIZE:-518}

COMMON=(
  --method_name PIRN_CMPT
  --dataset_type mvtec3d
  --dataset_path "$DATASET_PATH"
  --classes "$CLASSES"
  --rgb_backbone_name vit_base_patch14_dinov2.lvd142m
  --sn_backbone_name vit_base_patch14_dinov2.lvd142m
  --rgb_checkpoint_path "$RGB_CKPT"
  --sn_checkpoint_path "$SN_CKPT"
  --rgb_size "$RGB_SIZE"
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

run_exp() {
  local note=$1
  shift
  python main.py "${COMMON[@]}" \
    --cmpt_save_path "checkpoints/${note}_{class_name}.pth" \
    --experiment_note "$note" "$@"
}

for SEED in $SEEDS; do
  # Exp-1 K-shot. K=16 is also Exp-0 main result.
  for K in 1 2 4 8 16; do
    run_exp "final_k${K}_seed${SEED}_full_model" \
      --main_modality '' \
      --few_shot_k "$K" \
      --shot_seed "$SEED" \
      --paper_mnc
  done

  # Exp-2 cross-modal missing at K=16.
  run_exp "final_k16_seed${SEED}_rgb_missing_sn_full" \
    --main_modality rgb \
    --allow_true_missing_modality \
    --few_shot_k 16 \
    --shot_seed "$SEED" \
    --paper_mnc

  run_exp "final_k16_seed${SEED}_rgb_missing_sn_no_cmpt" \
    --main_modality rgb \
    --allow_true_missing_modality \
    --few_shot_k 16 \
    --shot_seed "$SEED" \
    --disable_cmpt

  run_exp "final_k16_seed${SEED}_sn_missing_rgb_full" \
    --main_modality sn \
    --allow_true_missing_modality \
    --few_shot_k 16 \
    --shot_seed "$SEED" \
    --paper_mnc

  run_exp "final_k16_seed${SEED}_sn_missing_rgb_no_cmpt" \
    --main_modality sn \
    --allow_true_missing_modality \
    --few_shot_k 16 \
    --shot_seed "$SEED" \
    --disable_cmpt

  run_exp "final_k16_seed${SEED}_full_modality_no_cmpt" \
    --main_modality '' \
    --few_shot_k 16 \
    --shot_seed "$SEED" \
    --disable_cmpt

  # Full modality full model is reused from final_k16_seed*_full_model.

  # Exp-3 shared prototype at K=16.
  run_exp "final_k16_seed${SEED}_proto_none" \
    --main_modality '' \
    --few_shot_k 16 \
    --shot_seed "$SEED" \
    --disable_prototypes

  run_exp "final_k16_seed${SEED}_proto_specific_only" \
    --main_modality '' \
    --few_shot_k 16 \
    --shot_seed "$SEED" \
    --disable_shared_proto \
    --paper_mnc

  run_exp "final_k16_seed${SEED}_proto_shared_only" \
    --main_modality '' \
    --few_shot_k 16 \
    --shot_seed "$SEED" \
    --only_shared_proto \
    --paper_mnc

  # Specific + shared is reused from final_k16_seed*_full_model.
done
