# 多模态情感分析模型训练指南

本文档提供多模态情感分析系统在CMU-MOSEI数据集上的完整训练指南。

## 目录

1. [环境准备](#环境准备)
2. [数据集下载](#数据集下载)
3. [训练配置](#训练配置)
4. [训练模型](#训练模型)
5. [监控训练](#监控训练)
6. [故障排查](#故障排查)

---

## 环境准备

### 1. 安装依赖

确保已安装所有必要的依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：
- `torch >= 2.0.0` - 深度学习框架
- `transformers >= 4.30.0` - 预训练模型（BERT、wav2vec）
- `opencv-python` - 视频处理
- `librosa` - 音频处理
- `pandas`, `numpy` - 数据处理
- `tensorboard` - 训练监控
- `tqdm` - 进度条
- `scikit-learn` - 评估指标

### 2. 可选依赖

如果使用CMU官方的mmsdk库下载数据集：

```bash
pip install mmsdk
```

如果使用Kaggle下载数据集：

```bash
pip install kaggle
```

---

## 数据集下载

### 方式1: 使用mmsdk库（推荐）

```bash
python scripts/download_mosei.py --mmsdk
```

### 方式2: 使用Kaggle API

```bash
# 配置Kaggle API凭证
# 1. 访问 https://www.kaggle.com/settings
# 2. 下载 kaggle.json
# 3. 放到 ~/.kaggle/ 目录

python scripts/download_mosei.py --kaggle
```

### 方式3: 创建模拟数据（测试用）

```bash
# 创建100个样本的模拟数据集
python scripts/download_mosei.py --mock --num-samples 100
```

### 方式4: 手动下载

```bash
python scripts/download_mosei.py --manual
```

按照说明手动下载数据集文件后，放到 `data/mosei/` 目录。

---

## 训练配置

### 默认配置

使用默认配置训练：

```bash
python scripts/train_model.py
```

### 自定义配置

可以通过命令行参数覆盖默认配置：

```bash
# 自定义批次大小和学习率
python scripts/train_model.py --batch-size 32 --lr 1e-3

# 自定义训练轮数
python scripts/train_model.py --epochs 100

# 指定数据目录
python scripts/train_model.py --data-dir /path/to/mosei

# 指定设备
python scripts/train_model.py --device cuda
```

### 快速测试模式

使用少量epoch进行快速测试：

```bash
python scripts/train_model.py --fast
```

### 从检查点恢复

从已保存的检查点继续训练：

```bash
python scripts/train_model.py --resume checkpoints/checkpoint_epoch_10.pth
```

---

## 训练模型

### 完整训练流程

1. **准备数据集**

```bash
# 下载CMU-MOSEI数据集
python scripts/download_mosei.py --mock
```

2. **开始训练**

```bash
# 使用默认配置
python scripts/train_model.py

# 或使用快速模式测试
python scripts/train_model.py --fast
```

3. **监控训练**

训练会自动：
- 保存检查点到 `checkpoints/` 目录
- 保存日志到 `logs/tensorboard/` 目录
- 在控制台显示训练进度

### 训练输出

训练过程中会输出：

```
Epoch 1/50
----------------------------------------------------------------------
Train: 100%|██████████| 500/500 [02:15<00:00,  3.68it/s, loss=2.1234]
Val:   100%|██████████| 100/100 [00:20<00:00,  4.92it/s, loss=1.8765]

训练 - Loss: 2.1234, Acc: 0.2500, F1: 0.2200
验证 - Loss: 1.8765, Acc: 0.3200, F1: 0.2900

检查点已保存: checkpoints/checkpoint_epoch_1.pth
```

---

## 监控训练

### 使用TensorBoard

训练期间，使用TensorBoard监控训练进度：

```bash
# 启动TensorBoard
tensorboard --logdir logs/tensorboard

# 在浏览器打开
# http://localhost:6006
```

TensorBoard提供：
- 损失曲线（训练/验证）
- 准确率曲线
- F1分数曲线
- 学习率变化
- 梯度分布（如果启用）

### 关键指标

监控以下指标：

| 指标 | 说明 | 良好范围 |
|------|------|----------|
| 训练损失 | 模型在训练集上的损失 | 逐渐下降 |
| 验证损失 | 模型在验证集上的损失 | 逐渐下降且不高于训练损失 |
| 训练准确率 | 训练集分类准确率 | 逐渐上升 |
| 验证准确率 | 验证集分类准确率 | 逐渐上升 |
| F1分数 | 综合评估指标 | > 0.7 为良好 |

### 早停机制

默认启用早停：
- 监控验证损失
- 耐心值：5个epoch
- 最小改善阈值：1e-4

---

## 训练结果

### 模型文件

训练完成后，模型保存在：

```
checkpoints/
├── best_model.pth              # 最佳模型（验证损失最低）
├── checkpoint_epoch_5.pth      # 第5轮检查点
├── checkpoint_epoch_10.pth     # 第10轮检查点
└── ...
```

### 日志文件

```
logs/
├── tensorboard/                # TensorBoard日志
│   └── 20240129_153045/
└── training_results_20240129_153045.json  # 训练结果摘要
```

### 测试结果

训练结束后，会在测试集上进行最终评估：

```
测试集结果:
  - Loss: 0.8234
  - Accuracy: 0.7250
  - Precision: 0.7180
  - Recall: 0.7120
  - F1: 0.7150
```

---

## 使用训练好的模型

### 加载模型进行推理

```python
from src.models.fusion_module import MultimodalFusionModule
import torch

# 创建模型
model = MultimodalFusionModule()

# 加载检查点
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 使用模型预测
from src.inference.predictor import Predictor

predictor = Predictor(model=model)
result = predictor.predict(text="I love this movie!")
print(result['emotion'])  # 'positive'
print(result['confidence'])  # 0.85
```

---

## 故障排查

### 问题1: 数据加载失败

**错误信息:**
```
FileNotFoundError: 无法找到train数据集的索引文件或数据
```

**解决方案:**
```bash
# 先下载或创建数据集
python scripts/download_mosei.py --mock
```

### 问题2: CUDA内存不足

**错误信息:**
```
RuntimeError: CUDA out of memory
```

**解决方案:**
```bash
# 减小批次大小
python scripts/train_model.py --batch-size 8

# 或使用CPU训练
python scripts/train_model.py --device cpu
```

### 问题3: 训练不收敛

**可能原因和解决方案:**

1. **学习率过大**
   ```bash
   python scripts/train_model.py --lr 1e-5
   ```

2. **数据未归一化**
   - 检查预处理代码

3. **类别不平衡**
   - 使用加权损失函数

### 问题4: 过拟合

**症状:** 训练准确率高，验证准确率低

**解决方案:**
1. 增加Dropout比率
2. 增加数据增强
3. 减少模型复杂度

### 问题5: 训练速度慢

**优化建议:**
1. 增加数据加载进程数
2. 使用混合精度训练
3. 减小日志记录频率

```bash
# 增加数据加载进程
# 在train_config.py中设置:
num_workers = 8
pin_memory = True
```

---

## 训练配置参考

### 快速测试（调试用）

```python
batch_size = 8
num_epochs = 5
early_stopping_patience = 2
learning_rate = 1e-4
```

### 标准训练

```python
batch_size = 16
num_epochs = 50
early_stopping_patience = 5
learning_rate = 1e-4
weight_decay = 1e-5
```

### 大规模训练

```python
batch_size = 32
num_epochs = 100
early_stopping_patience = 10
learning_rate = 1e-4
weight_decay = 1e-4
```

---

## 下一步

训练完成后，可以：

1. **在测试集上评估详细性能**
   ```bash
   python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
   ```

2. **可视化训练曲线**
   ```bash
   tensorboard --logdir logs/tensorboard
   ```

3. **导出模型用于部署**
   ```python
   # 导出为TorchScript
   model = MultimodalFusionModule()
   checkpoint = torch.load('checkpoints/best_model.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()

   example_input = (
       torch.randn(1, 768),  # text
       torch.randn(1, 768),  # audio
       torch.randn(1, 25)    # video
   )
   traced_model = torch.jit.trace(model, example_input)
   traced_model.save('model_deploy.pt')
   ```

---

## 附录

### A. 训练配置文件

完整配置请参考 `scripts/train_config.py`。

### B. 数据集格式

CMU-MOSEI数据集的JSON索引格式：

```json
[
  {
    "video_id": "some_video_id",
    "text_features": [0.1, 0.2, ...],  // 768维
    "audio_features": [0.1, 0.2, ...],  // 768维
    "video_features": [0.1, 0.2, ...],  // 25维
    "label": 1,  // -3到+3，映射到0-6
    "split": "train"
  }
]
```

### C. 情感标签映射

```python
LABEL_MAP = {
    -3: 0,  # strong_negative (强烈负面)
    -2: 1,  # negative (负面)
    -1: 2,  # weak_negative (弱负面)
     0: 3,  # neutral (中性)
     1: 4,  # weak_positive (弱正面)
     2: 5,  # positive (正面)
     3: 6   # strong_positive (强烈正面)
}
```

---

**文档更新日期:** 2026-01-29
