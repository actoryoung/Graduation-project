# 多模态情感分析系统

基于CMU-MOSEI数据集的多模态情感分析毕业设计项目，采用注意力融合机制处理类别不平衡问题。

## 项目概述

本项目实现了一个完整的多模态情感分析系统，整合文本、音频、视觉三种模态进行情感分类。

### 核心特性

- **跨模态注意力融合**: 动态学习模态间交互权重
- **类别不平衡处理**: 系统性优化方案，Macro F1提升21.3%
- **Web演示系统**: 基于Streamlit的交互式界面
- **完整论文**: 34,000字，6章节，含中英文摘要

### 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 准确率 | 58.88% | 测试集 |
| Macro F1 | 0.5011 | +21.3% 提升 |
| Negative F1 | 0.2759 | 从失效恢复 |

## 项目结构

```
biyesheji/
├── src/                    # 核心源代码
│   ├── models/            # 模型定义（注意力融合等）
│   ├── features/          # 特征提取（GloVe、COVAREP、OpenFace）
│   └── inference/         # 推理模块
├── web/                    # Streamlit Web应用
├── scripts/                # 训练和数据处理脚本
├── docs/                   # 文档
│   ├── thesis/            # 论文相关（6章节+摘要）
│   └── experiments/       # 实验报告
├── results/                # 实验结果和图表
│   └── figures/           # 8张学术图表
└── requirements.txt        # 依赖清单
```

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

### 运行Web演示

```bash
cd web
streamlit run app_s7.py
```

### 训练模型

```bash
python scripts/train_model.py
```

## 技术栈

- **深度学习**: PyTorch 2.5.0
- **特征提取**: GloVe (300D), COVAREP (74D), OpenFace (710D)
- **Web框架**: Streamlit 1.28.0
- **数据处理**: NumPy, Pandas, NLTK

## 主要创新点

1. **SDK子集数据特性发现**: 首次系统分析CMU-MOSEI SDK子集特性
2. **注意力融合机制**: 多头跨模态注意力实现动态模态交互
3. **系统性类别不平衡解决方案**: 阈值优化+损失函数对比+重新训练
4. **工程化实现**: 完整的Web演示系统

## 论文文档

- [论文定稿](docs/thesis/论文定稿.md) - 完整论文框架
- [各章节](docs/thesis/) - 第1-6章 + 摘要
- [参考文献](docs/thesis/参考文献列表.md) - 35篇文献
- [实验报告](docs/experiments/) - S10优化全过程

## 图表索引

| 图表 | 说明 |
|------|------|
| 图1 | 标签分布 |
| 图2 | 模型性能演变 |
| 图3 | S10方案对比 |
| 图4 | 混淆矩阵 |
| 图5 | 各类别性能雷达图 |
| 图6 | 数据集规模对比 |
| 图7 | 阈值敏感性分析 |
| 图8 | 训练曲线 |

## 许可证

本项目仅供学术研究使用。

## 联系方式

GitHub: https://github.com/actoryoung/Graduation-project
