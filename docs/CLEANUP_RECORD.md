# 项目文件整理记录

**整理日期**: 2026-02-06
**整理原因**: 执行项目原则3 - 临时文件清理

---

## 整理内容

### 1. Log文件整理 (8个)

**主目录 → logs/archive/**

| 文件 | 说明 |
|------|------|
| cleaning_run.log | 数据清洗运行日志 |
| cleaning_final.log | 数据清洗最终日志 |
| quick_clean_final.log | 快速清洗日志 |
| training_correct_output.log | 正确标签训练输出 |
| baseline_training.log | 基线训练日志 |
| bert_training.log | BERT训练日志 |
| bert_training_correct.log | BERT正确训练日志 |
| raw_baseline_training.log | 原始基线训练日志 |

### 2. JSON文件整理 (9个)

**主目录 → results/**

| 文件 | 说明 | 保留原因 |
|------|------|---------|
| s4_variants_analysis.json | S4变体分析结果 | 实验数据 |
| ensemble_results.json | S7集成结果 | 实验数据 |

**logs/ → logs/archive/results/**

| 文件 | 说明 | 归档原因 |
|------|------|---------|
| training_results_20260130_092721.json | 旧训练结果 | 过时 |
| training_results_20260204_114721.json | 旧训练结果 | 过时 |
| training_results_20260204_123631.json | 旧训练结果 | 过时 |
| training_results_20260204_131827.json | 旧训练结果 | 过时 |
| training_results_20260204_134643.json | 旧训练结果 | 过时 |
| training_results_20260204_150837.json | 旧训练结果 | 过时 |
| training_results_20260204_152418.json | 旧训练结果 | 过时 |

### 3. 新建目录

| 目录 | 用途 |
|------|------|
| `results/` | 存放实验结果JSON文件 |
| `logs/archive/results/` | 存档旧的训练结果 |

---

## 整理效果

| 项目 | 整理前 | 整理后 |
|------|--------|--------|
| 主目录log文件 | 6个 | 0个 |
| 主目录json文件 | 2个 | 0个 |
| logs/根目录文件 | 9个 | 0个 |

---

## 目录结构

```
项目根目录/
├── results/                        # 实验结果
│   ├── s4_variants_analysis.json
│   └── ensemble_results.json
│
├── logs/                           # 日志目录
│   └── archive/                    # 归档日志
│       ├── results/                # 旧训练结果
│       │   └── training_results_*.json (7个)
│       ├── *.log                   # 各种训练日志 (8个)
│       └── (已存在的归档文件)
```

---

## 项目原则已添加

原则已添加到 `EXPERIMENT_JOURNEY.md`:

1. **主文档最小化** - 减少token消耗
2. **文件归档管理** - 保持主目录整洁
3. **临时文件清理** - 及时清理临时文件

---

**整理完成**: 2026-02-06
**项目状态**: 主目录整洁，原则已建立
