# Transformer架构迁移实施计划

**目标**: 实现BERT文本特征替换 + Transformer融合架构，提升模型性能至65-75%准确率

**当前基线**: S10-3 CE模型，58.88%准确率，Macro F1=0.5011

---

## 整体架构对比

### 当前架构 (S10-3)
```
文本 → GloVe(300D) ──┐
                    ├──→ MLP融合 → 分类器 → 3类情感
音频 → COVAREP(74D) ─┤
                    │
视频 → OpenFace(710D)┘
```

### 目标架构1 (BERT替换)
```
文本 → BERT(768D) ───┐
                    ├──→ MLP融合 → 分类器 → 3类情感
音频 → COVAREP(74D) ─┤
                    │
视频 → OpenFace(710D)┘
```

### 目标架构2 (Transformer融合)
```
文本 → GloVe(300D) ──┐
                    ├──→ Transformer融合 → 分类器 → 3类情感
音频 → COVAREP(74D) ─┤
                    │
视频 → OpenFace(710D)┘
```

### 目标架构3 (完整方案)
```
文本 → BERT(768D) ───┐
                    ├──→ Transformer融合 → 分类器 → 3类情感
音频 → COVAREP(74D) ─┤
                    │
视频 → OpenFace(710D)┘
```

---

## 阶段1: BERT文本特征替换 (预计3-5天)

### 1.1 数据准备 (1天)

**任务**: 提取BERT特征并转换为3类标签

**输入**: SDK子集原始数据
**输出**: `data/bert_3class_{train,val,test}.npz`

**步骤**:
1. 使用现有的BERT提取脚本 (`scripts/extract_bert_hybrid.py`)
2. 将7类标签转换为3类:
   - strong_negative, negative, weak_negative → Negative (0)
   - neutral → Neutral (1)
   - weak_positive, positive, strong_positive → Positive (2)
3. 保存为NPZ格式

**验证**: 检查标签分布是否符合预期

**依赖**: 现有的BERT特征提取代码

**风险**: 低 - 已有现成代码

---

### 1.2 模型适配 (1天)

**任务**: 修改BERTHybridModel为3类分类

**文件**: `src/models/bert_hybrid_model_3class.py`

**修改点**:
```python
class BERTHybridModel3Class(nn.Module):
    def __init__(
        self,
        text_dim=768,
        audio_dim=74,
        video_dim=710,
        fusion_dim=512,
        num_classes=3,  # 改为3类
        dropout_rate=0.3
    ):
        # ... 其余代码保持不变
```

**验证**: 模型可以正常初始化和前向传播

**依赖**: 现有的BERTHybridModel

**风险**: 低 - 简单修改

---

### 1.3 训练脚本 (1天)

**任务**: 编写BERT-3类训练脚本

**文件**: `scripts/train_bert_3class.py`

**关键配置**:
```python
# 数据路径
DATA_PATH = 'data/bert_3class_train.npz'

# 模型配置
model = BERTHybridModel3Class(
    text_dim=768,
    audio_dim=74,
    video_dim=710,
    fusion_dim=512,
    num_classes=3,
    dropout_rate=0.3
)

# 训练配置
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)

# 训练参数
num_epochs = 30
batch_size = 16
```

**评估指标**:
- Accuracy
- Macro F1
- 各类别F1 (Negative, Neutral, Positive)

**验证**: 脚本可以正常启动训练

**依赖**: 1.1和1.2完成

**风险**: 低 - 参考现有训练脚本

---

### 1.4 实验运行 (1-2天)

**任务**: 训练并评估BERT-3类模型

**步骤**:
1. 运行训练脚本
2. 监控训练过程（loss, Macro F1）
3. 在验证集上选择最佳模型
4. 在测试集上评估最终性能

**预期结果**:
- Accuracy: 63-66%
- Macro F1: 0.55-0.60
- Negative F1: 0.35-0.45

**对比基线**:
| 指标 | S10-3 (GloVe) | BERT-3类 (预期) | 提升 |
|------|--------------|----------------|------|
| Accuracy | 58.88% | 63-66% | +4-7% |
| Macro F1 | 0.50 | 0.55-0.60 | +0.05-0.10 |
| Negative F1 | 0.28 | 0.35-0.45 | +0.07-0.17 |

**依赖**: 1.3完成

**风险**: 中 - 需要GPU资源和训练时间

---

## 阶段2: Transformer融合架构 (预计5-7天)

### 2.1 架构设计 (1-2天)

**任务**: 设计MISA-style融合模块

**参考论文**:
- MISA: Modality-Invariant and -Specific Representations
- Mulan: Multi-task Learning for Multimodal Sentiment Analysis

**设计要点**:

1. **模态编码器**: 将各模态编码到统一空间
   ```python
   self.text_encoder = nn.Linear(300, 256)
   self.audio_encoder = nn.Linear(74, 256)
   self.video_encoder = nn.Linear(710, 256)
   ```

2. **模态不变表示**: 跨模态注意力学习共享特征
   ```python
   self.cross_attention = nn.MultiheadAttention(256, num_heads=4)
   ```

3. **模态特定表示**: 保留各模态独特特征
   ```python
   self.modality_specific = nn.ModuleList([
       nn.Linear(256, 256) for _ in range(3)
   ])
   ```

4. **融合层**: 组合不变和特定表示
   ```python
   self.fusion = nn.Linear(256 * 2, 256)
   ```

**验证**: 架构设计合理，可训练参数适中

**依赖**: 无

**风险**: 中 - 需要理解Transformer原理

---

### 2.2 模型实现 (2-3天)

**任务**: 实现TransformerFusionModel

**文件**: `src/models/transformer_fusion.py`

**核心代码**:
```python
class TransformerFusionModel(nn.Module):
    """Transformer融合的多模态情感分析模型"""

    def __init__(
        self,
        text_dim=300,
        audio_dim=74,
        video_dim=710,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        num_classes=3,
        dropout_rate=0.3
    ):
        super().__init__()

        # 模态编码器
        self.text_encoder = nn.Linear(text_dim, hidden_dim)
        self.audio_encoder = nn.Linear(audio_dim, hidden_dim)
        self.video_encoder = nn.Linear(video_dim, hidden_dim)

        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim)

        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text, audio, video):
        # 编码各模态
        t_feat = self.text_encoder(text)
        a_feat = self.audio_encoder(audio)
        v_feat = self.video_encoder(video)

        # 堆叠为序列 [batch, 3, hidden]
        modalities = torch.stack([t_feat, a_feat, v_feat], dim=1)

        # 添加位置编码
        modalities = self.pos_encoding(modalities)

        # Transformer处理
        fused = self.transformer(modalities)

        # 聚合 (平均池化)
        fused = fused.mean(dim=1)

        # 融合和分类
        output = self.fusion(fused)
        logits = self.classifier(output)

        return logits
```

**验证**: 模型可以正常训练，梯度传播正常

**依赖**: 2.1完成

**风险**: 中 - Transformer实现较复杂

---

### 2.3 训练脚本 (1天)

**任务**: 编写Transformer训练脚本

**文件**: `scripts/train_transformer_fusion.py`

**关键配置**:
```python
# 模型配置
model = TransformerFusionModel(
    text_dim=300,
    audio_dim=74,
    video_dim=710,
    hidden_dim=256,
    num_heads=4,
    num_layers=2,
    num_classes=3,
    dropout_rate=0.3
)

# 优化器配置
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

# 学习率调度
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6
)
```

**验证**: 脚本可以正常启动训练

**依赖**: 2.2完成

**风险**: 低 - 参考现有训练脚本

---

### 2.4 实验运行 (1-2天)

**任务**: 训练并评估Transformer模型

**预期结果**:
- Accuracy: 64-70%
- Macro F1: 0.58-0.65
- Negative F1: 0.40-0.50

**对比基线**:
| 指标 | S10-3 (MLP) | Transformer (预期) | 提升 |
|------|------------|-------------------|------|
| Accuracy | 58.88% | 64-70% | +5-11% |
| Macro F1 | 0.50 | 0.58-0.65 | +0.08-0.15 |
| Negative F1 | 0.28 | 0.40-0.50 | +0.12-0.22 |

**依赖**: 2.3完成

**风险**: 高 - Transformer训练需要更多资源和时间

---

## 阶段3: BERT+Transformer组合 (预计3-5天)

### 3.1 组合模型 (1-2天)

**任务**: 实现BERT+Transformer融合模型

**文件**: `src/models/bert_transformer_model.py`

**架构**:
```python
class BERTTransformerModel(nn.Module):
    """BERT文本特征 + Transformer融合"""

    def __init__(
        self,
        text_dim=768,    # BERT特征
        audio_dim=74,
        video_dim=710,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        num_classes=3,
        dropout_rate=0.3
    ):
        # ... 类似TransformerFusionModel
        # 但text_dim=768 (BERT特征)
```

**验证**: 模型可以正常训练

**依赖**: 阶段1和阶段2完成

**风险**: 中 - 需要整合两个阶段的代码

---

### 3.2 训练评估 (2-3天)

**任务**: 训练最终组合模型

**预期结果**:
- Accuracy: 68-75%
- Macro F1: 0.62-0.70
- Negative F1: 0.45-0.55

**最终对比**:
| 模型 | Accuracy | Macro F1 | Negative F1 |
|------|----------|----------|-------------|
| S10-3 (基线) | 58.88% | 0.50 | 0.28 |
| BERT-3类 | 63-66% | 0.55-0.60 | 0.35-0.45 |
| Transformer融合 | 64-70% | 0.58-0.65 | 0.40-0.50 |
| **BERT+Transformer** | **68-75%** | **0.62-0.70** | **0.45-0.55** |

**依赖**: 3.1完成

**风险**: 高 - 最终实验，时间和资源消耗大

---

## 阶段4: 结果分析与论文更新 (预计2-3天)

### 4.1 结果整理

**任务**: 整理所有实验结果

**输出**:
- 性能对比表
- 混淆矩阵
- 训练曲线
- 各类别F1对比

### 4.2 论文更新

**更新章节**:
- 第四章: 添加Transformer实现细节
- 第五章: 添加BERT和Transformer实验结果
- 第六章: 更新创新点和结论

---

## 总体时间估算

| 阶段 | 任务 | 预计时间 | 累计时间 |
|------|------|---------|---------|
| 1 | BERT文本特征替换 | 3-5天 | 3-5天 |
| 2 | Transformer融合架构 | 5-7天 | 8-12天 |
| 3 | BERT+Transformer组合 | 3-5天 | 11-17天 |
| 4 | 结果分析与论文更新 | 2-3天 | 13-20天 |

**总计**: 约2-3周（取决于每天投入时间和资源可用性）

---

## 资源需求

### 计算资源
- GPU: 至少8GB显存（推荐16GB）
- 内存: 至少16GB
- 存储: 至少20GB（模型和数据）

### 软件依赖
```bash
# BERT相关
pip install transformers

# Transformer训练
pip install torch>=2.0
```

---

## 风险与缓解

### 高风险项
1. **Transformer训练不稳定**
   - 缓解: 使用梯度裁剪、学习率warmup
   - 备选: 减少Transformer层数

2. **GPU内存不足**
   - 缓解: 减小batch size，使用梯度累积
   - 备选: 使用更小的模型

3. **过拟合（小数据集）**
   - 缓解: 增强dropout，使用数据增强
   - 备选: 使用预训练模型并微调

### 中风险项
1. **BERT特征提取时间**
   - 缓解: 使用已提取的特征（如有）
   - 备选: 使用更小的BERT模型（distilbert）

2. **训练时间过长**
   - 缓解: 使用早停，减少epoch数
   - 备选: 使用更小的数据集验证

---

## 成功标准

### 最低目标
- [ ] BERT-3类模型训练成功
- [ ] Transformer融合模型训练成功
- [ ] 准确率提升至65%以上

### 理想目标
- [ ] BERT+Transformer组合模型训练成功
- [ ] 准确率提升至70%以上
- [ ] Macro F1提升至0.65以上
- [ ] Negative F1提升至0.50以上

---

## 下一步行动

**立即开始**: 阶段1.1 - 数据准备

需要我帮您开始实施吗？
