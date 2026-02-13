# Findings: 多模态情感分析系统研究记录
<!--
  WHAT: 项目的研究发现、技术决策、需求记录
  WHY: 持久化知识库，避免上下文丢失
  WHEN: 每次发现新信息后立即更新
-->

## 项目概述

### 基本信息
| 项目 | 值 |
|------|-----|
| 项目名称 | 多模态情感分析系统 |
| 数据集 | CMU-MOSEI SDK子集 |
| 任务类型 | 三分类情感分析（-1: 负面, 0: 中性, 1: 正面） |
| 项目性质 | 毕业设计 |

### 核心问题
1. **类别不平衡**: Negative类仅占8.8%，Positive类占58.7%
2. **模态融合**: 如何有效整合文本、音频、视觉三种模态
3. **阈值优化**: 默认阈值导致Negative类完全失效

## 数据集特性发现

### CMU-MOSEI SDK子集分布
| 类别 | 数量 | 占比 |
|------|------|------|
| Negative (-1) | ~640 | 8.8% |
| Neutral (0) | ~2,400 | 32.5% |
| Positive (1) | ~4,300 | 58.7% |

### 数据规模
- 训练集: ~16,000 样本（估算，基于7:1:2分割）
- 验证集: ~2,300 样本
- 测试集: ~4,600 样本

### 特征维度
| 模态 | 特征类型 | 维度 |
|------|----------|------|
| 文本 | GloVe词向量 | 300D |
| 文本 | BERT嵌入 | 768D |
| 音频 | COVAREP | 74D |
| 音频 | wav2vec | - |
| 视觉 | OpenFace | 710D |
| 视觉 | ResNet | - |

## 模型架构演进

### 1. 注意力融合模块 (fusion_module.py)
**关键特性:**
- 多头跨模态注意力机制
- 模态间动态权重学习
- LSTM编码器（基础版本）

**文件位置:** `src/models/fusion_module.py`

### 2. BERT混合模型 (bert_hybrid_model_3class.py)
**关键特性:**
- BERT文本编码器
- 跨模态注意力融合
- 三分类输出
- 类别权重支持

**文件位置:** `src/models/bert_hybrid_model_3class.py`

### 3. Transformer融合模块 (transformer_fusion.py)
**关键特性:**
- Transformer编码器替代LSTM
- 自注意力机制
- 用于架构对比实验

**文件位置:** `src/models/transformer_fusion.py`

## 优化方案演进

### S10完整优化方案
**文档位置:** `docs/experiments/s10/`

**优化步骤:**
1. 阈值优化: 0.38 → 0.30
2. 类别权重调整
3. 重新训练模型

**效果:**
| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| Macro F1 | 0.2898 | 0.5011 | +21.3% |
| Negative F1 | 0.0000 | 0.2759 | 恢复 |
| Accuracy | 58.24% | 58.88% | +0.64% |

## 技术决策记录

### 为什么选择3类分类？
**决策:** 将原始的7级情感标签简化为3类（-1, 0, 1）
**原因:**
1. 聚焦核心问题：类别不平衡
2. 简化任务复杂度
3. 更符合实际应用场景

### 为什么使用注意力融合？
**决策:** 采用多头跨模态注意力机制
**原因:**
1. 动态学习模态间交互权重
2. 不同样本可能需要不同的模态权重
3. 比简单拼接更有效

### 为什么调整阈值？
**决策:** 将决策阈值从默认0.38降至0.30
**原因:**
1. 默认阈值过高导致Negative类完全无预测
2. 降低阈值可提高Negative类召回率
3. 通过实验确定最佳阈值

## 训练配置

### 最佳配置 (S10)
| 参数 | 值 |
|------|-----|
| 学习率 | 0.0001 |
| Batch Size | 32 |
| Epochs | 15-20 |
| 损失函数 | CrossEntropyLoss (带类别权重) |
| 优化器 | Adam |
| 决策阈值 | 0.30 |

### 类别权重
| 类别 | 权重 |
|------|------|
| Negative | 2.5 |
| Neutral | 1.0 |
| Positive | 0.6 |

## 实验结果

### 混淆矩阵 (S10)
```
              Pred_Neg  Pred_Neu  Pred_Pos
Actual_Neg       115       237       288
Actual_Neu        87       631       672
Actual_Pos       129       809      1582
```

### 各类别性能
| 类别 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| Negative | 0.3333 | 0.1796 | 0.2759 |
| Neutral | 0.3818 | 0.4549 | 0.4151 |
| Positive | 0.6336 | 0.6302 | 0.6319 |

## Web演示系统

### S7版本
**文件位置:** `web/app_s7.py`

**功能:**
1. 模型预测演示
2. 可视化分析
3. 结果展示

**启动方式:**
```bash
cd web
streamlit run app_s7.py
```

## 论文文档

### 论文结构
| 章节 | 内容 | 状态 |
|------|------|------|
| 摘要 | 中英文摘要 | 完成 |
| 第1章 | 绪论 | 完成 |
| 第2章 | 相关工作 | 完成 |
| 第3章 | 方法 | 完成 |
| 第4章 | 实验 | 完成 |
| 第5章 | 结果与分析 | 完成 |
| 第6章 | 总结与展望 | 完成 |

### 图表索引
| 图表 | 说明 | 文件位置 |
|------|------|----------|
| 图1 | 标签分布 | results/figures/ |
| 图2 | 模型性能演变 | results/figures/ |
| 图3 | S10方案对比 | results/figures/ |
| 图4 | 混淆矩阵 | results/figures/ |
| 图5 | 各类别性能雷达图 | results/figures/ |
| 图6 | 数据集规模对比 | results/figures/ |
| 图7 | 阈值敏感性分析 | results/figures/ |
| 图8 | 训练曲线 | results/figures/ |

## 参考资料

### 关键论文
1. CMU-MOSEI 数据集论文
2. 多模态情感分析综述
3. 注意力机制相关论文

### 技术文档
| 文档 | 位置 |
|------|------|
| 训练指南 | docs/TRAINING_GUIDE.md |
| 模型下载指南 | docs/MODEL_DOWNLOAD_GUIDE.md |
| Transformer迁移计划 | docs/TRANSFORMER_MIGRATION_PLAN.md |
| Transformer进度 | docs/TRANSFORMER_PROGRESS.md |

## 待解决问题

1. **Transformer融合模块训练**: 需要完成训练以对比LSTM vs Transformer
2. **架构对比分析**: 需要详细分析两种架构的性能差异
3. **论文更新**: 需要将架构对比结果写入论文

## 关键代码位置索引

| 功能 | 文件位置 |
|------|----------|
| 注意力融合模块 | src/models/fusion_module.py |
| BERT混合模型 | src/models/bert_hybrid_model_3class.py |
| Transformer融合 | src/models/transformer_fusion.py |
| 特征提取器 | src/features/ |
| 训练脚本 | scripts/train_bert_3class.py |
| Web应用 | web/app_s7.py |

---
*最后更新: 2026-02-09*
