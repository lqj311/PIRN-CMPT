# PIRN-CMPT 论文系统设计代码更新包

这份更新包把代码从“能跑 PIRN-CMPT”进一步改成论文里更好交代的系统版本，重点对齐三个核心设计：

1. **少样本 K-shot**：每类只抽取 K 张正常样本作为 support set，并可保存抽样清单。
2. **跨模态缺失输入**：测试时支持真实 RGB-only 或 SN-only 文件，不再必须通过 RGB/XYZ 文件 zip 配对才能得到样本。
3. **共享原型与消融实验**：加入 shared prototype、CMPT、APR、MNC、pseudo prototype 的显式消融开关。

## 1. 使用方法

把本更新包复制到你的仓库目录后执行：

```bash
cd PIRN-CMPT/CMDIAD-main
python /path/to/pirn_cmpt_code_update/tools/apply_system_design_updates.py .
```

脚本会直接修改以下文件，并在同目录生成备份：

- `main.py.bak_system_design`
- `cmdiad_runner.py.bak_system_design`
- `dataset.py.bak_system_design`
- `feature_extractors/cmpt_features.py.bak_system_design`

应用后检查参数：

```bash
python main.py --help | grep -E "few_shot|shot_seed|disable_|paper_mnc|missing"
```

## 2. 新增参数

### 少样本实验

```bash
--few_shot_k 4
--shot_seed 0
--save_fewshot_list
--fewshot_list_dir fewshot_splits
```

含义：每个类别只使用 K 张正常训练样本，抽样由 `shot_seed` 控制，并保存 CSV 清单，方便论文中说明“4-shot 的训练样本到底是哪几张”。

### 真实缺失模态测试

```bash
--allow_true_missing_modality
--main_modality rgb
```

当 `main_modality=rgb` 时，测试集可以只有 RGB 图像；当 `main_modality=sn` 时，测试集可以只有 XYZ/SN 来源文件。原始代码会把 RGB 和 XYZ 按排序 `zip` 成样本，一旦某个模态文件不存在就会丢样本；这个更新解决了这个问题。

### 论文图中的两阶段 MNC

```bash
--paper_mnc
```

该参数会启用 `mnc_strong=True`，并保证 `mnc_stages>=2`，对应论文图里的 `MNC stage1 -> MNC stage2`。

为了让默认系统设计与论文图一致，应用更新后 `PIRN_CMPT` 和 `CMPT` 默认也会走 paper MNC 路径；若要做消融，可用 `--disable_mnc`。

### 消融实验开关

```bash
--disable_cmpt
--disable_shared_proto
--disable_apr
--disable_mnc
--disable_pseudo_proto
```

建议论文表格设置：

| 实验名 | 参数 |
|---|---|
| Full | 无消融参数 |
| w/o CMPT | `--disable_cmpt` |
| w/o Shared | `--disable_shared_proto` |
| w/o APR | `--disable_apr` |
| w/o MNC | `--disable_mnc` |
| w/o Pseudo | `--disable_pseudo_proto` |

## 3. 推荐运行命令

RGB 可用、SN 缺失，4-shot：

```bash
python main.py \
  --method_name PIRN_CMPT \
  --main_modality rgb \
  --few_shot_k 4 \
  --shot_seed 0 \
  --save_fewshot_list \
  --paper_mnc \
  --allow_true_missing_modality \
  --rgb_checkpoint_path checkpoints/dinov2_vitb14_pretrain.pth \
  --sn_checkpoint_path checkpoints/dinov2_vitb14_pretrain.pth \
  --experiment_note K4_rgb_missing_sn_full
```

SN 可用、RGB 缺失，4-shot：

```bash
python main.py \
  --method_name PIRN_CMPT \
  --main_modality sn \
  --few_shot_k 4 \
  --shot_seed 0 \
  --save_fewshot_list \
  --paper_mnc \
  --allow_true_missing_modality \
  --rgb_checkpoint_path checkpoints/dinov2_vitb14_pretrain.pth \
  --sn_checkpoint_path checkpoints/dinov2_vitb14_pretrain.pth \
  --experiment_note K4_sn_missing_rgb_full
```

双模态完整，4-shot：

```bash
python main.py \
  --method_name PIRN_CMPT \
  --main_modality '' \
  --few_shot_k 4 \
  --shot_seed 0 \
  --save_fewshot_list \
  --paper_mnc \
  --rgb_checkpoint_path checkpoints/dinov2_vitb14_pretrain.pth \
  --sn_checkpoint_path checkpoints/dinov2_vitb14_pretrain.pth \
  --experiment_note K4_full_modality_full
```

## 4. 论文中可以这样描述代码实现

> 为保证少样本实验可复现，本文在训练数据加载阶段引入 K-shot 支持集采样机制。对于每个工业类别，仅从正常训练样本中按固定随机种子抽取 K 个样本参与 CMPT 训练、原型构建和正常模式记忆维护，并保存抽样文件列表。为验证真实缺失模态场景，测试数据加载器不再强制要求 RGB 与 XYZ 文件成对出现，而是以当前可用模态为主键构造样本，并对缺失分支使用占位张量，使模型推理路径与实际传感器缺失情形一致。

## 5. 注意事项

- 训练阶段仍然需要 RGB/SN 成对样本，因为 CMPT 要学习跨模态映射。
- 测试阶段才支持真实单模态输入。
- 如果 `main_modality=sn` 且测试时没有 RGB 图像，建议不要开启 `--save_heatmaps`，因为叠加热力图需要 RGB 原图作为背景。仍然可以保存数值结果和 segmentation tensor。
- 本更新包不包含数据集和 DINOv2 权重。
