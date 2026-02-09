# S7-v1集成Web应用使用指南

**版本**: S7-v1集成版本
**创建日期**: 2026-02-06
**测试准确率**: 59.47%
**Negative准确率**: 23.38%

---

## 快速启动

### Windows用户

1. 双击运行: `web/run_s7.bat`

2. 或者在命令行中运行:
```bash
cd web
streamlit run app_s7.py
```

3. 浏览器会自动打开: `http://localhost:8501`

---

## 系统特性

### ✅ 已实现功能

| 功能 | 状态 | 说明 |
|------|------|------|
| 文本输入 | ✅ 完全支持 | 使用GloVe词向量 (300维) |
| S7-v1集成 | ✅ 已实现 | 双模型加权投票 |
| 情感分类 | ✅ 3类情感 | Negative/Neutral/Positive |
| 置信度显示 | ✅ 显示 | 预测置信度百分比 |
| 概率分布图 | ✅ 可视化 | 柱状图展示各类别概率 |
| 模型说明 | ✅ 交互式 | 可展开查看详细说明 |

### ⚠️ 部分支持功能

| 功能 | 状态 | 说明 |
|------|------|------|
| 音频输入 | ⚠️ 需配置 | 需要COVAREP特征提取工具 |
| 视频输入 | ⚠️ 需配置 | 需要OpenFace特征提取工具 |

---

## 模型配置

### S7-v1集成模型

```python
模型1: S3+S4 (注意力融合 + 类别权重)
  文件: baseline_attention_3class_weighted_best_model.pth
  权重: 1.5
  测试准确率: 56.80%
  Negative准确率: 25.97%

模型2: S3 (注意力融合)
  文件: baseline_attention_3class_ce_best_model.pth
  权重: 1.0
  测试准确率: 59.17%
  Negative准确率: 0.00%

集成方式: 加权投票
最终准确率: 59.47%
最终Negative准确率: 23.38%
```

### 性能指标

| 指标 | 值 | 说明 |
|------|-----|------|
| 测试准确率 | 59.47% | 所有模型中最高 |
| Negative准确率 | 23.38% (18/77) | 可接受水平 |
| Neutral准确率 | 65.10% (222/341) | 表现良好 |
| Positive准确率 | 62.79% (162/258) | 表现良好 |
| 宏平均F1 | 0.5070 | 类别平衡指标 |

---

## 使用步骤

### 1. 启动应用

```bash
# 方法1: 双击bat文件
web/run_s7.bat

# 方法2: 命令行启动
cd web
streamlit run app_s7.py
```

### 2. 界面操作

1. 在左侧查看系统信息和模型配置
2. 在"输入区域"输入要分析的文本
3. 点击"开始分析"按钮
4. 查看"分析结果"区域的预测结果和可视化

### 3. 结果说明

**预测情感**: 系统预测的情感类别
- Negative (负面)
- Neutral (中性)
- Positive (正面)

**置信度**: 预测的置信程度
- 接近100%表示模型很确定
- 接近33%表示模型不太确定（三分类随机基线）

**概率分布**: 各类别的预测概率
- 总和为100%
- 可以看到模型对其他类别的倾向

---

## 与原版本对比

| 特性 | 原版本 (app.py) | S7版本 (app_s7.py) |
|------|-----------------|-------------------|
| 模型 | BERT混合 (50.44%) | S7-v1集成 (59.47%) |
| Negative准确率 | ~20% | 23.38% |
| 模型类型 | 单模型 | 双模型集成 |
| 论文适配度 | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 故障排除

### 问题1: 模型加载失败

**错误信息**: "模型文件不存在"

**解决方案**:
1. 确认模型文件存在:
   - `checkpoints/baseline_attention_3class_weighted_best_model.pth`
   - `checkpoints/baseline_attention_3class_ce_best_model.pth`

2. 如果文件不存在，运行训练脚本:
```bash
python scripts/train_baseline.py --data-dir data/mosei/processed_cleaned_3class --mode attention --weight-strategy sqrt
python scripts/train_baseline.py --data-dir data/mosei/processed_cleaned_3class --mode attention --use-ce
```

### 问题2: CUDA内存不足

**错误信息**: "CUDA out of memory"

**解决方案**:
1. 在代码中切换到CPU模式
2. 或者在启动脚本中添加:
```bash
streamlit run app_s7.py --device.cpu
```

### 问题3: 端口被占用

**错误信息**: "Port 8501 is already in use"

**解决方案**:
```bash
# 方法1: 使用其他端口
streamlit run app_s7.py --server.port 8502

# 方法2: 关闭占用端口的进程
netstat -ano | findstr :8501
taskkill /PID <进程ID> /F
```

---

## 论文截图准备

### 需要截图的界面

1. **主界面** - 显示整体布局
2. **系统信息** - 侧边栏的模型配置
3. **输入区域** - 文本输入框
4. **分析结果** - 预测情感 + 置信度 + 概率分布图
5. **模型说明** - 展开后的技术说明

### 截图建议

- 使用系统自带截图工具 (Win + Shift + S)
- 保存为PNG格式
- 在论文中标注关键信息

---

## 技术架构

```
Streamlit Web界面
        │
        ▼
S7集成预测器 (ensemble_predictor.py)
        │
        ├─→ 模型1: S3+S4 (权重1.5)
        │       - 注意力融合
        │       - 类别权重
        │
        └─→ 模型2: S3 (权重1.0)
                - 注意力融合

        │
        ▼
    预测结果
        │
        ├─→ 情感类别
        ├─→ 置信度
        └─→ 概率分布
```

---

## 更新日志

**v1.0** (2026-02-06)
- 创建S7-v1集成版本的Web应用
- 使用最优模型 (59.47%准确率)
- 添加概率分布可视化
- 添加模型说明交互组件

---

**联系人**: 如有问题，请查看项目主文档
