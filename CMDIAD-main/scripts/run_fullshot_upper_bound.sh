#!/usr/bin/env bash
set -euo pipefail

# Full-shot upper-bound reference for the thesis K-shot table.
#
# Usage:
#   cd /root/autodl-tmp/PIRN-CMPT/CMDIAD-main
#   bash scripts/run_fullshot_upper_bound.sh

DATASET_PATH=${DATASET_PATH:-data}
DINO_CKPT=${DINO_CKPT:-/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth}
RGB_CKPT=${RGB_CKPT:-$DINO_CKPT}
SN_CKPT=${SN_CKPT:-$DINO_CKPT}
CLASSES=${CLASSES:-"bagel,cable_gland,carrot,cookie,dowel,foam,peach,potato,rope,tire"}
SEEDS=${SEEDS:-"0"}
CMPT_EPOCHS=${CMPT_EPOCHS:-80}
PROTOTYPES=${PROTOTYPES:-128}
RGB_SIZE=${RGB_SIZE:-518}

for SEED in $SEEDS; do
  python main.py \
    --method_name PIRN_CMPT \
    --main_modality "" \
    --dataset_type mvtec3d \
    --dataset_path "$DATASET_PATH" \
    --classes "$CLASSES" \
    --rgb_backbone_name vit_base_patch14_dinov2.lvd142m \
    --sn_backbone_name vit_base_patch14_dinov2.lvd142m \
    --rgb_checkpoint_path "$RGB_CKPT" \
    --sn_checkpoint_path "$SN_CKPT" \
    --rgb_size "$RGB_SIZE" \
    --max_sample 500 \
    --cmpt_epochs "$CMPT_EPOCHS" \
    --cmpt_lr 1e-4 \
    --cmpt_cycle_loss_weight 0.25 \
    --cmpt_cosine_loss_weight 0.1 \
    --cmpt_relation_loss_weight 0.05 \
    --cmpt_aux_weight 0.35 \
    --cmpt_aux_confidence_threshold 0.50 \
    --cmpt_aux_mode both \
    --cmpt_full_consistency_weight 0.25 \
    --cmpt_full_calibration_std -1.0 \
    --cmpt_full_map_gain 1.0 \
    --cmpt_nc_weight 0.20 \
    --cmpt_nc_confidence_threshold 0.55 \
    --cmpt_nc_safe_margin 0.02 \
    --num_rgb_prototypes "$PROTOTYPES" \
    --num_sn_prototypes "$PROTOTYPES" \
    --num_shared_prototypes "$PROTOTYPES" \
    --rgb_s_lambda 1.0 \
    --rgb_smap_lambda 1.0 \
    --sn_s_lambda 1.0 \
    --sn_smap_lambda 1.0 \
    --cmpt_s_lambda 1.0 \
    --cmpt_smap_lambda 1.0 \
    --paper_mnc \
    --cmpt_save_path "checkpoints/final_fullshot_seed${SEED}_full_model_{class_name}.pth" \
    --experiment_note "final_fullshot_seed${SEED}_full_model"
done
