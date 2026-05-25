# PIRN-CMPT 更新包合并记录

更新时间：2026-05-25

## 来源

本次根据 `E:/PIRN-CMPT/pirn_cmpt_code_update.zip` 中的说明与脚本，手动合并系统级更新。没有直接运行 `tools/apply_system_design_updates.py`，原因是自动脚本会对当前已经调通的 DINOv2、CMPT、learnable MNC 和默认 baseline 逻辑做文本替换，存在覆盖风险。

## 已合并内容

1. K-shot / few-shot 支持

- 新增参数：`--few_shot_k`、`--shot_seed`、`--save_fewshot_list`、`--fewshot_list_dir`。
- 训练集在 `dataset.py` 中按类别名和 seed 做确定性抽样。
- 选中的样本会保存为 `results/fewshot_lists/<class>_<split>_k<K>_seed<seed>.txt`。

2. 真实缺模态测试支持

- 新增参数：`--allow_true_missing_modality`。
- 当 `--main_modality rgb --allow_true_missing_modality` 时，测试 loader 允许只存在 RGB 文件。
- 当 `--main_modality sn --allow_true_missing_modality` 时，测试 loader 允许只存在 XYZ/SN 文件。
- 缺失模态会在 dataset 层用零张量占位，模型侧仍只使用可用模态和 CMPT 伪模态分支。

3. Paper MNC 与消融开关

- 新增参数：`--paper_mnc`。显式传入时启用当前代码中的两阶段 PIRN-style MNC 近似：`mnc_strong=True`，`mnc_stages>=2`。
- 新增消融参数：
  - `--disable_cmpt`
  - `--disable_shared_proto`
  - `--disable_apr`
  - `--disable_mnc`
  - `--disable_pseudo_proto`

4. 实验脚本

- 新增 `scripts/run_kshot_system_experiments.sh`：K-shot、多 seed、完整/缺模态系统实验。
- 新增 `scripts/run_ablation_for_thesis.sh`：论文消融实验。

## 与更新包不同的保留点

更新包的自动脚本会在 `PIRN_CMPT` / `CMPT` 方法下自动打开 paper MNC。这里没有这样做。

原因：前期实验显示，当前稳定 baseline 在 rope 10-shot 上达到：

- Image ROCAUC: 0.976
- Pixel ROCAUC: 0.985
- AU-PRO: 0.945
- AU-PRO-0.01: 0.403

而强 MNC / paper MNC 近似不总是提高指标。因此默认仍保持原稳定逻辑；只有显式传入 `--paper_mnc` 时才启用，便于作为论文结构对应实验或消融项。

## 回退范围

本次涉及文件：

- `main.py`
- `cmdiad_runner.py`
- `dataset.py`
- `feature_extractors/cmpt_features.py`
- `scripts/run_kshot_system_experiments.sh`
- `scripts/run_ablation_for_thesis.sh`
- `docs/system_design_update_merge.md`

若要回退本次更新，应只回退以上文件中对应本记录的修改，避免影响此前已经验证的 DINOv2、CMPT 和 heatmap 输出逻辑。
