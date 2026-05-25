#!/usr/bin/env bash
set -euo pipefail

# Thesis-oriented ablations over all MVTec 3D-AD classes.
# The groups are organized by the three paper claims instead of arbitrary toggles:
#   - few-shot normal prototype learning
#   - cross-modal completion under missing modality
#   - shared prototype memory
#
# Usage:
#   cd /root/autodl-tmp/PIRN-CMPT/CMDIAD-main
#   bash scripts/run_all_classes_core_ablation.sh
#
# Short debug:
#   CLASSES=rope K=4 SEEDS="0" bash scripts/run_all_classes_core_ablation.sh

DATASET_PATH=${DATASET_PATH:-data}
DINO_CKPT=${DINO_CKPT:-/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth}
RGB_CKPT=${RGB_CKPT:-$DINO_CKPT}
SN_CKPT=${SN_CKPT:-$DINO_CKPT}
CLASSES=${CLASSES:-"bagel,cable_gland,carrot,cookie,dowel,foam,peach,potato,rope,tire"}
K=${K:-4}
SEEDS=${SEEDS:-"0 1 2"}
CMPT_EPOCHS=${CMPT_EPOCHS:-80}
PROTOTYPES=${PROTOTYPES:-128}
RGB_SIZE=${RGB_SIZE:-518}
DONE_DIR=${DONE_DIR:-results/done_markers/core_ablation}
SKIP_DONE=${SKIP_DONE:-1}

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

IFS=',' read -r -a CLASS_ARRAY <<< "$CLASSES"
mkdir -p "$DONE_DIR"

run_exp() {
  local seed=$1
  local group=$2
  local note=$3
  shift 3
  for CLASS_NAME in "${CLASS_ARRAY[@]}"; do
    CLASS_NAME="${CLASS_NAME//[[:space:]]/}"
    if [[ -z "$CLASS_NAME" ]]; then
      continue
    fi
    local marker="${DONE_DIR}/ablation_${group}_k${K}_seed${seed}_${note}_${CLASS_NAME}.done"
    if [[ "$SKIP_DONE" == "1" && -f "$marker" ]]; then
      echo "[Skip] ablation_${group} seed=${seed} ${note} ${CLASS_NAME} already completed: ${marker}"
      continue
    fi
    python main.py "${COMMON[@]}" \
      --classes "$CLASS_NAME" \
      --shot_seed "$seed" \
      --cmpt_save_path "checkpoints/ablation_${group}_k${K}_seed${seed}_{class_name}.pth" \
      --experiment_note "ablation_${group}_k${K}_seed${seed}_${note}_${CLASS_NAME}" "$@"
    touch "$marker"
  done
}

for SEED in $SEEDS; do
  # Main incremental ablation: answer whether the complete design is better than removing each core part.
  run_exp "$SEED" "full" "full_modality" \
    --main_modality '' \
    --paper_mnc

  run_exp "$SEED" "no_mnc" "full_modality" \
    --main_modality '' \
    --disable_mnc

  run_exp "$SEED" "no_cmpt" "full_modality" \
    --main_modality '' \
    --disable_cmpt \
    --paper_mnc

  run_exp "$SEED" "no_shared" "full_modality" \
    --main_modality '' \
    --disable_shared_proto \
    --paper_mnc

  run_exp "$SEED" "only_shared" "full_modality" \
    --main_modality '' \
    --only_shared_proto \
    --paper_mnc

  run_exp "$SEED" "no_apr" "full_modality" \
    --main_modality '' \
    --disable_apr \
    --paper_mnc

  # Cross-modal missing-modality claim.
  run_exp "$SEED" "rgb_missing_sn_full" "rgb_missing_sn" \
    --main_modality rgb \
    --allow_true_missing_modality \
    --paper_mnc

  run_exp "$SEED" "rgb_missing_sn_no_cmpt" "rgb_missing_sn" \
    --main_modality rgb \
    --allow_true_missing_modality \
    --disable_cmpt \
    --paper_mnc

  run_exp "$SEED" "rgb_missing_sn_no_shared" "rgb_missing_sn" \
    --main_modality rgb \
    --allow_true_missing_modality \
    --disable_shared_proto \
    --paper_mnc

  run_exp "$SEED" "rgb_missing_sn_no_pseudo" "rgb_missing_sn" \
    --main_modality rgb \
    --allow_true_missing_modality \
    --disable_pseudo_proto \
    --paper_mnc

  run_exp "$SEED" "sn_missing_rgb_full" "sn_missing_rgb" \
    --main_modality sn \
    --allow_true_missing_modality \
    --paper_mnc

  run_exp "$SEED" "sn_missing_rgb_no_cmpt" "sn_missing_rgb" \
    --main_modality sn \
    --allow_true_missing_modality \
    --disable_cmpt \
    --paper_mnc

  run_exp "$SEED" "sn_missing_rgb_no_shared" "sn_missing_rgb" \
    --main_modality sn \
    --allow_true_missing_modality \
    --disable_shared_proto \
    --paper_mnc

  run_exp "$SEED" "sn_missing_rgb_no_pseudo" "sn_missing_rgb" \
    --main_modality sn \
    --allow_true_missing_modality \
    --disable_pseudo_proto \
    --paper_mnc
done
