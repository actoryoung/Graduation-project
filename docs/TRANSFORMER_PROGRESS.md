# Transformer迁移进度记录

**开始时间**: 2026-02-09
**状态**: 阶段2完成 - 发现严重过拟合问题

---

## 已完成任务

### ✅ 准备工作
- [x] 分析整体流程并创建任务分解计划
- [x] 创建实施计划文档：`docs/TRANSFORMER_MIGRATION_PLAN.md`

### ✅ 阶段1: BERT文本特征替换

#### 1.1 数据准备 ✅
**问题**: BERT hybrid数据缺少Neutral类样本
**解决**: 创建了完整数据集脚本
- 脚本：`scripts/convert_bert_to_3class.py`
- 脚本：`scripts/create_bert_3class_full.py`

**输出数据**:
- `data/mosei/bert_3class_full/train.npz` - 2249样本
- `data/mosei/bert_3class_full/val.npz` - 300样本
- `data/mosei/bert_3class_full/test.npz` - 676样本

**标签分布**:
```
训练集:
  Negative (0): 224 (10.0%)
  Neutral (1): 1134 (50.4%)
  Positive (2): 891 (39.6%)
```

#### 1.2 模型适配 ✅
**文件**: `src/models/bert_hybrid_model_3class.py`

**模型信息**:
- 参数量：851,587
- 模型大小：3.25 MB
- 架构：BERT(768) + COVAREP(74) + OpenFace(710) → 3类分类

#### 1.3 训练脚本 ✅
**文件**: `scripts/train_bert_3class.py`

**功能**:
- 完整训练流程（30 epoch，早停机制）
- Macro F1作为主要评估指标
- 自动保存最佳模型

---

## 问题与解决

### ⚠️ 问题1: 类别不平衡导致Negative类完全不被学习

#### 首次训练结果（无加权）
- **Test Accuracy**: 54.44%
- **Test Macro F1**: 0.384
- **Negative F1**: 0.0 - **模型完全忽略了Negative类！**
- **Neutral F1**: 0.591
- **Positive F1**: 0.561

#### 第二次训练结果（使用逆频率权重）
- **Test Accuracy**: 43.05% - **性能反而下降！**
- **Test Macro F1**: 0.398
- **Negative F1**: 0.2185 - **开始学习了，但仍很低**
- **Neutral F1**: 0.5142
- **Positive F1**: 0.4612

### 根本原因分析

1. **严重类别不平衡**: Negative类仅占10% (224/2249样本)
2. **模型容量不足**: 简单的MLP融合层可能不足以处理这种不平衡
3. **加权损失效果有限**: 无论使用何种权重策略，Negative类都难以学习

### 尝试的解决方案对比

| 方案 | Negative权重 | 测试准确率 | Macro F1 | Negative F1 | 问题 |
|------|-------------|-----------|----------|-------------|------|
| 无权重 | 1.0x | 54.44% | 0.384 | **0.0** | 完全忽略Negative类 |
| 逆频率权重 | 5.02x | 43.05% | 0.398 | 0.2185 | 权重过激，整体性能下降 |
| sqrt逆频率 | 2.24x | 52.81% | 0.382 | 0.025 | 权重不足，几乎无效 |

### 核心问题

**BERT + MLP融合架构在这个数据集上无法有效学习Negative类**。

可能原因：
1. BERT特征(768D)与音频/视频特征的尺度差异太大
2. 简单的MLP融合层缺乏足够的容量来处理这种不平衡
3. 训练数据量太小(2249训练样本)不足以让模型学习稀有类

### 决策

**跳过BERT单独实验，直接进入Phase 2: Transformer融合架构**

理由：
1. Transformer架构通过自注意力机制可能更好地处理多模态特征
2. Transformer在处理不平衡数据方面通常表现更好
3. 可以直接使用BERT+Transformer的组合方案

---

## 进行中任务

### ✅ 阶段2.4: Transformer融合模型训练（已完成）

**最终结果** (Epoch 16/50, 早停):
- **训练准确率**: 81.95%
- **训练Macro F1**: 0.7927
- **验证准确率**: 52.00%
- **验证Macro F1**: 0.4265 (最佳0.4508 at epoch 16)

**测试集结果**:
- **测试准确率**: 52.66%
- **测试Macro F1**: 0.4377
- **Negative F1**: 0.1818 - **仍然很低**
- **Neutral F1**: 0.5863
- **Positive F1**: 0.5451

### ⚠️ 关键发现：严重过拟合

**过拟合证据**:
- 训练Macro F1: 0.7927
- 测试Macro F1: 0.4377
- **性能差距**: 0.355 (35.5个百分点)

**问题分析**:
1. 训练数据太少 (2249样本)
2. 模型参数太多 (2.1M参数 vs 2249样本 = ~930参数/样本)
3. BERT特征维度过高 (768维)
4. Transformer架构对小数据集过于复杂

---

## 综合分析与结论

### 实验结果汇总

| 模型 | 参数量 | 测试准确率 | Macro F1 | Negative F1 | 问题 |
|------|--------|-----------|----------|-------------|------|
| **S10-3 (GloVe基线)** | 850K | **58.88%** | **0.5011** | **0.2832** | 类别不平衡 |
| BERT+MLP (无权重) | 852K | 54.44% | 0.384 | 0.0 | 完全忽略Negative |
| BERT+MLP (sqrt权重) | 852K | 52.81% | 0.382 | 0.025 | 几乎无效 |
| BERT+MLP (inv权重) | 852K | 43.05% | 0.398 | 0.2185 | 过度补偿 |
| **BERT+Transformer** | 2.1M | 52.66% | 0.438 | 0.182 | **严重过拟合** |

### 根本原因

**数据限制**:
- 训练集仅2249样本
- Negative类仅224样本 (10%)
- Negative类在测试集仅77样本 (11.4%)

**模型复杂度**:
- BERT特征768维远超GloVe的300维
- Transformer参数量是MLP的2.5倍
- 复杂模型在小数据集上容易过拟合

### 建议

**短期方案** (推荐):
1. **保留S10-3 CE模型**作为最终结果
   - 在当前数据集上表现最好
   - 已完成的论文可以基于此模型

2. **如果必须改进**，考虑：
   - 使用更多训练数据 (完整CMU-MOSEI而非10%子集)
   - 简化模型架构 (减少BERT特征维度)
   - 使用预训练模型微调而非从头训练

**长期方案** (需要更多时间):
1. 收集更多标注数据
2. 使用数据增强技术
3. 尝试半监督学习
4. 探索预训练的多模态模型

---

## 创建的新文件

### 数据准备
- `scripts/convert_bert_to_3class.py`
- `scripts/create_bert_3class_full.py`

### 模型
- `src/models/bert_hybrid_model_3class.py`
- `src/models/transformer_fusion.py`

### 训练
- `scripts/train_bert_3class.py`
- `scripts/train_transformer_fusion.py`

### 文档
- `docs/TRANSFORMER_MIGRATION_PLAN.md`
- `docs/TRANSFORMER_PROGRESS.md`

---

## 最终建议

考虑到时间和资源限制，**建议将S10-3 CE模型作为最终结果**用于论文。

**理由**:
1. S10-3 CE在当前数据集上表现最好
2. Transformer架构需要更多数据和计算资源
3. 已有完整实验结果和论文材料

如果需要进一步改进，建议先扩充数据集再尝试复杂模型。

---

## 待完成任务

### ⏳ 阶段2: Transformer融合架构 (5-7天)
- 2.1 架构设计 - 设计MISA-style融合模块
- 2.2 模型实现 - 实现TransformerFusionModel
- 2.3 训练脚本 - 编写Transformer训练脚本
- 2.4 实验运行 - 训练并评估Transformer模型

### ⏳ 阶段3: BERT+Transformer组合 (3-5天)
- 3.1 组合模型 - BERT+Transformer融合
- 3.2 训练评估 - 训练最终组合模型

### ⏳ 阶段4: 结果分析 (2-3天)
- 论文更新
- 结果总结

---

## 创建的新文件

### 数据准备
- `scripts/convert_bert_to_3class.py`
- `scripts/create_bert_3class_full.py`

### 模型
- `src/models/bert_hybrid_model_3class.py`

### 训练
- `scripts/train_bert_3class.py`

---

## 问题与解决

### 问题1: BERT hybrid数据缺少Neutral类
**原因**: 原始BERT hybrid提取时只保留了部分类别
**解决**: 合并BERT特征和processed_cleaned_3class的标签

### 问题2: 训练脚本编码错误
**原因**: Windows终端不支持某些特殊字符
**解决**: 修复了`set_postfix`调用

---

## 下一步行动

1. **等待BERT训练完成**（约15-25分钟）
2. 分析BERT 3类模型结果
3. 开始阶段2：Transformer融合架构设计
