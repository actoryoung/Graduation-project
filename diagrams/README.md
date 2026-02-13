# 多模态情感分析系统架构流程图

本项目包含三种格式的系统架构流程图，分别适用于不同场景。

## 📁 文件列表

| 文件 | 格式 | 用途 | 工具 |
|------|------|------|------|
| `tikz_multimodal_architecture.tex` | LaTeX TikZ | 论文发表、学术文档 | TeX Live / MiKTeX |
| `drawio_multimodal_architecture.drawio` | Draw.io XML | 可视化编辑、演示 | diagrams.net |
| `mermaid_multimodal_architecture.mmd` | Mermaid | 文档嵌入、快速预览 | 支持 Mermaid 的编辑器 |

## 🎨 使用方式

### 1. TikZ 流程图（论文级质量）

**编译方式**：
```bash
pdflatex tikz_multimodal_architecture.tex
```

**特点**：
- 矢量输出，无限缩放
- 与LaTeX文档样式完美融合
- 适合发表在顶级会议/期刊
- 支持中文标签（需安装 CJK 字体）

**依赖**：
- TeX Live 或 MiKTeX
- 宏包：`tikz`, `xeCJK`, `amssymb`

### 2. Draw.io 流程图（可视化编辑）

**打开方式**：
1. 访问 https://app.diagrams.net/
2. 选择 "File" → "Open From" → "Device"
3. 选择 `drawio_multimodal_architecture.drawio` 文件

**特点**：
- 所见即所得的图形编辑器
- 支持导出为 PNG、SVG、PDF 等格式
- 可以自由调整布局和样式
- 支持团队协作

**导出选项**：
- PNG（用于 PPT）
- SVG（用于 Web）
- PDF（用于文档）

### 3. Mermaid 流程图（轻量快速）

**渲染方式**：
- 在线编辑器：https://mermaid.live/
- VS Code 插件：Markdown Preview Mermaid Support
- GitHub/GitLab 原生支持

**特点**：
- 纯文本格式，版本控制友好
- 语法简洁，易于修改
- 可直接嵌入 Markdown 文档

---

## 🏗️ 系统架构说明

### 输入层
- **文本**: BERT 特征提取器，768 维
- **音频**: COVAREP 声学特征，74 维
- **视频**: OpenFace 视觉特征，710 维

### 特征编码层
- **文本编码器**: Linear(768 → 256) + LayerNorm + ReLU + Dropout
- **音频编码器**: Linear(74 → 128) + LayerNorm + ReLU + Dropout
- **视频编码器**: Linear(710 → 256) + LayerNorm + ReLU + Dropout
- **音频投影层**: Linear(128 → 256) - 统一维度

### 跨模态融合层
- **位置编码**: 为 3 个模态添加位置信息
- **Transformer 融合**:
  - 多头自注意力机制 (Multi-Head Self-Attention)
  - 4 个注意力头，2 层 Transformer
  - 跨模态交互学习
- **融合聚合**: 平均池化 + MLP 融合网络

### 输出层
- **分类器**: Linear(128 → 7)
- **7 类情感**: 强烈负面、负面、弱负面、中性、弱正面、正面、强烈正面
- **置信度**: Softmax 概率分布

---

## 🎨 颜色编码

| 模态 | 颜色 | HEX 代码 |
|------|------|----------|
| 文本 | 蓝色 | `#E3F2FD` / `#1976D2` |
| 音频 | 绿色 | `#E8F5E9` / `#43A047` |
| 视频 | 红色 | `#FFEBEE` / `#E53935` |
| 融合 | 紫色 | `#F3E5F5` / `#7B1FA2` |
| 操作 | 灰蓝 | `#E1F5FE` / `#0277BD` |
| 分类 | 橙色 | `#FFF3E0` / `#FF9800` |

---

## 📊 推荐使用场景

| 场景 | 推荐格式 |
|------|----------|
| 发表论文 | TikZ (PDF) |
| PPT 演示 | Draw.io (PNG) |
| 技术文档 | Mermaid / Draw.io (SVG) |
| 快速分享 | Mermaid (在线预览) |
| 版本控制 | Mermaid (.mmd) / TikZ (.tex) |

---

## 🔧 自定义修改

### 修改颜色
- **TikZ**: 修改颜色定义部分 (`\definecolor`)
- **Draw.io**: 选中图形 → 右侧样式面板
- **Mermaid**: 修改 `classDef` 中的颜色值

### 添加新模块
- **TikZ**: 使用 `\node` 命令定义新节点
- **Draw.io**: 从左侧工具栏拖拽图形
- **Mermaid**: 添加新的节点定义和连接

### 调整布局
- **TikZ**: 修改 `node distance` 和 `positioning` 参数
- **Draw.io**: 直接拖拽图形调整位置
- **Mermaid**: 调整节点排列方向（`TD`/`LR`）和间距

---

## 📝 许可

本架构图仅供学术研究使用。
