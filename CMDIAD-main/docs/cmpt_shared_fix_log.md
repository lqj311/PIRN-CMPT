# CMPT 与共享原型修正记录

时间：2026-05-25

## 问题

最终实验结果显示：

- `w/o CMPT` 优于 `Full Model`。
- `w/o Prototype` 优于结构化原型版本。
- `Shared only` 明显偏低。

这说明当前实现中 CMPT 和 shared prototypes 虽然参与了流程，但没有形成稳定正增益。

## 原因判断

1. CMPT 伪模态被等权当作真实分支评分。

缺模态时，原流程把 pseudo SN / pseudo RGB 的 error map 与真实可用模态 error map 直接加权融合。若伪特征存在分布偏移，伪分支会把噪声直接写进最终 heatmap。

2. shared prototypes 与 specific prototypes 简单拼接。

原流程把 shared prototypes 和 modality-specific prototypes 直接拼成一个 prototype bank。这样 shared prototypes 会和模态特定原型竞争分配权重；若 shared 表达不够稳定，会干扰 specific reconstruction。

## 修改

1. shared prototypes 改为置信度门控辅助重构。

默认 prototype bank 只包含 modality-specific prototypes。shared prototypes 不再直接拼接，而是单独重构 shared context，并通过置信度门控混入：

```text
z = (1 - gate) * z_specific + gate * z_shared
```

新增参数：

- `--shared_proto_gate`，默认 `0.25`
- `--shared_proto_confidence_threshold`，默认 `0.35`

2. CMPT 伪模态改为可靠性门控辅助评分。

缺模态时，真实可用模态仍是主分支；pseudo modality 只作为辅助分支，其权重由真实特征与伪特征的一致性决定：

```text
score = score_real + gate * score_pseudo
map = map_real + gate * map_pseudo
```

新增参数：

- `--cmpt_aux_weight`，默认 `0.25`
- `--cmpt_aux_confidence_threshold`，默认 `0.55`

## 预期影响

- 当 CMPT 生成质量较好时，伪模态分支提供补充异常线索。
- 当 CMPT 生成质量较差时，门控会降低伪分支权重，避免拖累真实模态。
- shared prototypes 从竞争式主原型变为辅助式结构约束，更符合“共享正常结构”的角色。

## 回退方式

若需要近似回到旧行为：

- 将 `--shared_proto_gate` 调大，并降低 `--shared_proto_confidence_threshold`。
- 将 `--cmpt_aux_weight` 调为 `1.0`，并降低 `--cmpt_aux_confidence_threshold`。

若需要完全对比：

- `--disable_shared_proto`
- `--disable_cmpt`
