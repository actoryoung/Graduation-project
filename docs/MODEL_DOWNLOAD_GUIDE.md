# 预训练模型下载指南

## 快速开始（使用镜像）

### 对于中国大陆用户

使用HF Mirror镜像站点可以大幅加速下载：

```bash
# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"

# Windows CMD
set HF_ENDPOINT=https://hf-mirror.com

# Linux/Mac
export HF_ENDPOINT=https://hf-mirror.com
```

### 测试下载

设置镜像后，运行测试脚本自动下载所需模型：

```bash
# 测试BERT（约420MB）
python src/features/bert_extractor.py

# 测试wav2vec（约360MB）
python src/features/wav2vec_extractor.py

# 测试ResNet（约98MB）
python src/features/resnet_extractor.py

# 测试融合模块（无需下载）
python src/models/fusion_module_v2.py
```

---

## 模型详情

### 1. BERT (bert-base-uncased)

- **大小**: ~420MB
- **用途**: 文本特征提取（768维）
- **首次运行会自动下载以下文件**:
  - `config.json` (~600B)
  - `pytorch_model.bin` (~420MB)
  - `tokenizer_config.json` (~300B)
  - `vocab.txt` (~230KB)
  - `tokenizer.json` (~500KB)

**缓存位置**: `~/.cache/huggingface/hub/models--bert-base-uncased/`

### 2. wav2vec 2.0 (facebook/wav2vec2-base)

- **大小**: ~360MB
- **用途**: 音频特征提取（768维）
- **首次运行会自动下载以下文件**:
  - `config.json` (~2KB)
  - `pytorch_model.bin` (~360MB)
  - `preprocessor_config.json` (~200B)
  - `vocab.json` (~400B)

**缓存位置**: `~/.cache/huggingface/hub/models--facebook--wav2vec2-base/`

### 3. ResNet-50

- **大小**: ~98MB
- **用途**: 视频特征提取（2048维）
- **首次运行会自动下载**:
  - `resnet50-0676ba61.pth` (~98MB)

**缓存位置**: `~/.cache/torch/hub/checkpoints/`

---

## 验证下载

下载完成后，验证模型是否正确加载：

```bash
# 融合模块测试（验证所有编码器输出维度正确）
python src/models/fusion_module_v2.py
```

预期输出：
```
======================================================================
多模态融合模块 v2 测试（预训练编码器版本）
======================================================================
...
======================================================================
所有测试通过!
======================================================================
```

---

## 故障排除

### 问题1: SSL连接错误

```
SSL: CERTIFICATE_VERIFY_FAILED
```

**解决方法**: 禁用SSL验证（不推荐用于生产环境）

```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

### 问题2: 连接超时

```
ConnectTimeoutError: Connection to huggingface.co timed out
```

**解决方法**:
1. 使用HF镜像（推荐）
2. 使用VPN
3. 手动下载模型文件

### 问题3: 磁盘空间不足

```
OSError: [Errno 28] No space left on device
```

**解决方法**: 清理缓存或更改缓存位置

```bash
# 更改Hugging Face缓存位置
export HF_HOME="/path/to/new/cache"

# 更改torch缓存位置
export TORCH_HOME="/path/to/new/cache"
```

---

## 手动下载

如果自动下载失败，可以手动下载模型文件：

### BERT手动下载

1. 访问: https://huggingface.co/bert-base-uncased/tree/main
2. 下载以下文件到 `models/bert-base-uncased/`:
   - `config.json`
   - `pytorch_model.bin`
   - `tokenizer_config.json`
   - `vocab.txt`

3. 使用时指定本地路径:
```python
extractor = BERTTextExtractor(model_name='models/bert-base-uncased')
```

### wav2vec手动下载

1. 访问: https://huggingface.co/facebook/wav2vec2-base/tree/main
2. 下载以下文件到 `models/wav2vec2-base/`:
   - `config.json`
   - `pytorch_model.bin`
   - `preprocessor_config.json`

3. 使用时指定本地路径:
```python
extractor = Wav2VecAudioExtractor(model_name='models/wav2vec2-base')
```

### ResNet手动下载

1. 访问: https://download.pytorch.org/models/resnet50-0676ba61.pth
2. 下载到 `~/.cache/torch/hub/checkpoints/`

---

## 网络要求

- **最低带宽**: 10 Mbps
- **预计下载时间**:
  - BERT: ~5-10分钟 (10 Mbps)
  - wav2vec: ~5-8分钟 (10 Mbps)
  - ResNet: ~1-2分钟 (10 Mbps)
  - **总计**: ~15-20分钟

---

## 下载后训练

所有模型下载完成后，开始训练：

```bash
python scripts/train_pretrained.py \
    --data-dir data/mosei/processed_cleaned_correct \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --device cuda
```

---

**提示**: 模型只需下载一次，后续运行会自动使用缓存。
