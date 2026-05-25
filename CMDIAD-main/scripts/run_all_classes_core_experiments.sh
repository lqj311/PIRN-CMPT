#!/usr/bin/env bash
set -euo pipefail

# Full 10-class experiments for the thesis core claims:
#   1) few-shot learning
#   2) cross-modal missing-modality completion
#   3) shared prototypes
#
# Usage:
#   cd /root/autodl-tmp/PIRN-CMPT/CMDIAD-main
#   bash scripts/run_all_classes_core_experiments.sh
#
# Typical shorter debug run:
#   CLASSES=rope K_LIST="4" SEEDS="0" bash scripts/run_all_classes_core_experiments.sh

DATASET_PATH=${DATASET_PATH:-data}
DINO_CKPT=${DINO_CKPT:-/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth}
RGB_CKPT=${RGB_CKPT:-$DINO_CKPT}
SN_CKPT=${SN_CKPT:-$DINO_CKPT}
CLASSES=${CLASSES:-"bagel,cable_gland,carrot,cookie,dowel,foam,peach,potato,rope,tire"}
K_LIST=${K_LIST:-"1 2 4 8 16"}
SEEDS=${SEEDS:-"0 1 2"}
CMPT_EPOCHS=${CMPT_EPOCHS:-80}
PROTOTYPES=${PROTOTYPES:-128}
RGB_SIZE=${RGB_SIZE:-518}
RUN_FULL_SHOT=${RUN_FULL_SHOT:-1}
DONE_DIR=${DONE_DIR:-results/done_markers/core_experiments}
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

IFS=',' read -r -a CLASS_ARRAY <<< "$CLASSES"
mkdir -p "$DONE_DIR"

run_exp() {
  local note=$1
  shift
  for CLASS_NAME in "${CLASS_ARRAY[@]}"; do
    CLASS_NAME="${CLASS_NAME//[[:space:]]/}"
    if [[ -z "$CLASS_NAME" ]]; then
      continue
    fi
    local marker="${DONE_DIR}/${note}_${CLASS_NAME}.done"
    if [[ "$SKIP_DONE" == "1" && -f "$marker" ]]; then
      echo "[Skip] ${note} ${CLASS_NAME} already completed: ${marker}"
      continue
    fi
    python main.py "${COMMON[@]}" \
      --classes "$CLASS_NAME" \
      --experiment_note "${note}_${CLASS_NAME}" "$@"
    touch "$marker"
  done
}

for K in $K_LIST; do
  for SEED in $SEEDS; do
    run_exp "core_k${K}_seed${SEED}_full_modality" \
      --main_modality '' \
      --few_shot_k "$K" \
      --shot_seed "$SEED" \
      --paper_mnc \
      --cmpt_save_path "checkpoints/core_k${K}_seed${SEED}_full_{class_name}.pth"

    run_exp "core_k${K}_seed${SEED}_rgb_missing_sn" \
      --main_modality rgb \
      --allow_true_missing_modality \
      --few_shot_k "$K" \
      --shot_seed "$SEED" \
      --paper_mnc \
      --cmpt_save_path "checkpoints/core_k${K}_seed${SEED}_rgb_missing_sn_{class_name}.pth"

    run_exp "core_k${K}_seed${SEED}_sn_missing_rgb" \
      --main_modality sn \
      --allow_true_missing_modality \
      --few_shot_k "$K" \
      --shot_seed "$SEED" \
      --paper_mnc \
      --cmpt_save_path "checkpoints/core_k${K}_seed${SEED}_sn_missing_rgb_{class_name}.pth"
  done
done

if [[ "$RUN_FULL_SHOT" == "1" ]]; then
  run_exp "core_fullshot_full_modality" \
    --main_modality '' \
    --max_sample 500 \
    --paper_mnc \
    --cmpt_save_path "checkpoints/core_fullshot_full_{class_name}.pth"
fi
