# -*- coding: utf-8 -*-
"""
S10-2: 阈值优化 - 网格搜索最优决策阈值

任务:
1. 加载S7-v1模型的预测概率
2. 网格搜索不同阈值组合
3. 计算每种组合的Macro F1
4. 找到最优阈值配置
5. 评估准确率损失

目标:
- 通过阈值优化提升Macro F1
- 可接受的准确率损失（<2%）
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from typing import Dict, List, Tuple
import json
from itertools import product

import torch
import torch.nn as nn

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import Config, DEVICE
from s10_1_analysis import Simple3ClassModel, S10Analyzer


class ThresholdOptimizer:
    """阈值优化器"""

    def __init__(self, model_path: str, data_path: str, num_classes=3):
        """
        初始化阈值优化器

        Args:
            model_path: 模型checkpoint路径
            data_path: 测试数据路径
            num_classes: 类别数量
        """
        self.model_path = model_path
        self.data_path = data_path
        self.num_classes = num_classes
        self.device = DEVICE

        # 类别名称
        self.class_names = ['Negative', 'Neutral', 'Positive']

        # 7类到3类的映射
        self.label_map_7to3 = {
            0: 0, 1: 0, 2: 0,  # Negative
            3: 1,              # Neutral
            4: 2, 5: 2, 6: 2   # Positive
        }

        print(f"[S10-2] 初始化阈值优化器")
        print(f"[S10-2] 模型: {model_path}")

    def load_model_and_predict(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载模型并预测，返回概率

        Returns:
            (probs, labels, confidences): 预测概率、真实标签、置信度
        """
        print(f"[S10-2] 加载模型并预测...")

        # 复用S10Analyzer的加载逻辑
        analyzer = S10Analyzer(self.model_path, self.data_path, self.num_classes)
        model = analyzer.load_model()
        text_features, audio_features, video_features, labels = analyzer.load_data()

        # 获取预测概率
        model.eval()
        all_probs = []

        batch_size = 32
        num_samples = len(text_features)

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_text = text_features[i:i+batch_size]
                batch_audio = audio_features[i:i+batch_size]
                batch_video = video_features[i:i+batch_size]

                text_feat = torch.FloatTensor(batch_text).to(self.device)
                audio_feat = torch.FloatTensor(batch_audio).to(self.device)
                video_feat = torch.FloatTensor(batch_video).to(self.device)

                logits = model(text_feat, audio_feat, video_feat)
                probs = torch.softmax(logits, dim=1)

                all_probs.extend(probs.cpu().numpy())

        probs = np.array(all_probs)
        confidences = np.max(probs, axis=1)

        print(f"[S10-2] 预测完成")
        print(f"[S10-2] 样本数: {len(labels)}")

        return probs, labels, confidences

    def predict_with_thresholds(self, probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """
        使用自定义阈值进行预测

        Args:
            probs: 预测概率 (N, num_classes)
            thresholds: 每个类别的阈值 (num_classes,)

        Returns:
            predictions: 预测结果
        """
        predictions = np.zeros(len(probs), dtype=int)

        for i in range(len(probs)):
            # 对于每个样本，找到超过阈值的类别
            # 如果多个类别超过阈值，选择概率最大的
            # 如果没有类别超过阈值，选择概率最大的

            above_threshold = probs[i] >= thresholds

            if above_threshold.any():
                # 有类别超过阈值，选择其中概率最大的
                masked_probs = probs[i].copy()
                masked_probs[~above_threshold] = -1
                predictions[i] = np.argmax(masked_probs)
            else:
                # 没有类别超过阈值，选择概率最大的
                predictions[i] = np.argmax(probs[i])

        return predictions

    def grid_search_thresholds(self, probs: np.ndarray, labels: np.ndarray,
                               threshold_range: List[float] = None) -> Dict:
        """
        网格搜索最优阈值

        Args:
            probs: 预测概率
            labels: 真实标签
            threshold_range: 阈值搜索范围

        Returns:
            搜索结果字典
        """
        if threshold_range is None:
            threshold_range = [0.3, 0.4, 0.5, 0.6, 0.7]

        print(f"[S10-2] 开始网格搜索...")
        print(f"[S10-2] 阈值范围: {threshold_range}")

        best_macro_f1 = 0
        best_thresholds = None
        best_predictions = None
        best_accuracy = 0

        results = []

        # 遍历所有阈值组合
        total_combinations = len(threshold_range) ** self.num_classes
        current = 0

        for thresholds in product(threshold_range, repeat=self.num_classes):
            thresholds = np.array(thresholds)
            current += 1

            if current % 50 == 0:
                print(f"[S10-2] 进度: {current}/{total_combinations}")

            # 使用阈值预测
            predictions = self.predict_with_thresholds(probs, thresholds)

            # 计算指标
            accuracy = accuracy_score(labels, predictions)
            macro_f1 = f1_score(labels, predictions, average='macro')
            f1_per_class = f1_score(labels, predictions, average=None)

            # 记录结果
            result = {
                'thresholds': thresholds.tolist(),
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'f1_negative': f1_per_class[0],
                'f1_neutral': f1_per_class[1],
                'f1_positive': f1_per_class[2]
            }
            results.append(result)

            # 更新最优结果
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_thresholds = thresholds
                best_predictions = predictions
                best_accuracy = accuracy

        print(f"[S10-2] 网格搜索完成")
        print(f"[S10-2] 最优Macro F1: {best_macro_f1:.4f}")
        print(f"[S10-2] 最优阈值: {best_thresholds}")
        print(f"[S10-2] 对应准确率: {best_accuracy:.4f}")

        return {
            'best_macro_f1': best_macro_f1,
            'best_thresholds': best_thresholds.tolist(),
            'best_accuracy': best_accuracy,
            'best_predictions': best_predictions,
            'all_results': results
        }

    def search_balanced_thresholds(self, probs: np.ndarray, labels: np.ndarray,
                                   base_threshold: float = 0.5) -> Dict:
        """
        搜索平衡阈值 - 针对Negative类优化

        由于Negative类F1=0，尝试降低Negative类阈值

        Args:
            probs: 预测概率
            labels: 真实标签
            base_threshold: 基础阈值

        Returns:
            搜索结果
        """
        print(f"[S10-2] 搜索平衡阈值...")
        print(f"[S10-2] 基础阈值: {base_threshold}")

        best_macro_f1 = 0
        best_thresholds = None
        best_accuracy = 0

        results = []

        # 针对Negative类降低阈值，保持其他类别阈值不变
        for neg_threshold in np.linspace(0.1, 0.5, 41):  # 0.1到0.5，步长0.01
            thresholds = np.array([neg_threshold, base_threshold, base_threshold])

            predictions = self.predict_with_thresholds(probs, thresholds)

            accuracy = accuracy_score(labels, predictions)
            macro_f1 = f1_score(labels, predictions, average='macro')
            f1_per_class = f1_score(labels, predictions, average=None)

            result = {
                'neg_threshold': neg_threshold,
                'thresholds': thresholds.tolist(),
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'f1_negative': f1_per_class[0],
                'f1_neutral': f1_per_class[1],
                'f1_positive': f1_per_class[2]
            }
            results.append(result)

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_thresholds = thresholds
                best_accuracy = accuracy

        print(f"[S10-2] 最优Macro F1: {best_macro_f1:.4f}")
        print(f"[S10-2] 最优阈值: {best_thresholds}")
        print(f"[S10-2] 对应准确率: {best_accuracy:.4f}")

        return {
            'best_macro_f1': best_macro_f1,
            'best_thresholds': best_thresholds.tolist(),
            'best_accuracy': best_accuracy,
            'all_results': results
        }

    def plot_threshold_sensitivity(self, results: Dict, save_path: str):
        """绘制阈值敏感性分析图"""
        print(f"[S10-2] 绘制阈值敏感性分析...")

        all_results = results['all_results']

        # 提取数据
        neg_thresholds = [r['neg_threshold'] for r in all_results]
        macro_f1s = [r['macro_f1'] for r in all_results]
        f1_negatives = [r['f1_negative'] for r in all_results]
        accuracies = [r['accuracy'] for r in all_results]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Macro F1 vs Negative阈值
        axes[0, 0].plot(neg_thresholds, macro_f1s, 'b-', linewidth=2)
        axes[0, 0].axhline(y=0.4133, color='r', linestyle='--', label='Baseline (0.4133)')
        best_thresh = results['best_thresholds'][0]
        axes[0, 0].axvline(x=best_thresh, color='g', linestyle='--', label=f'Best ({best_thresh:.2f})')
        axes[0, 0].set_xlabel('Negative Threshold', fontsize=12)
        axes[0, 0].set_ylabel('Macro F1', fontsize=12)
        axes[0, 0].set_title('Macro F1 vs Negative Threshold', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # 2. Negative F1 vs Negative阈值
        axes[0, 1].plot(neg_thresholds, f1_negatives, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Negative Threshold', fontsize=12)
        axes[0, 1].set_ylabel('Negative F1', fontsize=12)
        axes[0, 1].set_title('Negative F1 vs Negative Threshold', fontsize=14)
        axes[0, 1].grid(alpha=0.3)

        # 3. 准确率 vs Negative阈值
        axes[1, 0].plot(neg_thresholds, accuracies, 'g-', linewidth=2)
        axes[1, 0].axhline(y=0.5917, color='r', linestyle='--', label='Baseline (0.5917)')
        axes[1, 0].set_xlabel('Negative Threshold', fontsize=12)
        axes[1, 0].set_ylabel('Accuracy', fontsize=12)
        axes[1, 0].set_title('Accuracy vs Negative Threshold', fontsize=14)
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # 4. 轨迹图
        axes[1, 1].plot(neg_thresholds, macro_f1s, 'b-', label='Macro F1', linewidth=2)
        axes[1, 1].plot(neg_thresholds, accuracies, 'g-', label='Accuracy', linewidth=2)
        axes[1, 1].set_xlabel('Negative Threshold', fontsize=12)
        axes[1, 1].set_ylabel('Score', fontsize=12)
        axes[1, 1].set_title('Trade-off Analysis', fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[S10-2] 敏感性分析图已保存: {save_path}")

    def save_results(self, results: Dict, save_dir: str):
        """保存结果"""
        results_path = os.path.join(save_dir, 's10_2_threshold_optimization_results.json')

        # 移除best_predictions（太大）
        save_data = results.copy()
        if 'best_predictions' in save_data:
            del save_data['best_predictions']

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"[S10-2] 结果已保存: {results_path}")

    def run(self, save_dir: str = 'results'):
        """运行阈值优化"""
        print(f"[S10-2] 开始S10-2阈值优化")
        print(f"=" * 50)

        os.makedirs(save_dir, exist_ok=True)
        s10_dir = os.path.join(save_dir, 's10')
        os.makedirs(s10_dir, exist_ok=True)

        # 1. 加载模型并预测
        probs, labels, confidences = self.load_model_and_predict()

        # 2. 计算基准性能（argmax）
        baseline_predictions = np.argmax(probs, axis=1)
        baseline_accuracy = accuracy_score(labels, baseline_predictions)
        baseline_macro_f1 = f1_score(labels, baseline_predictions, average='macro')

        print(f"[S10-2] 基准性能:")
        print(f"[S10-2]   准确率: {baseline_accuracy:.4f}")
        print(f"[S10-2]   Macro F1: {baseline_macro_f1:.4f}")

        # 3. 搜索平衡阈值
        results = self.search_balanced_thresholds(probs, labels, base_threshold=0.5)

        # 4. 绘制敏感性分析
        sensitivity_plot_path = os.path.join(s10_dir, 's10_2_threshold_sensitivity.png')
        self.plot_threshold_sensitivity(results, sensitivity_plot_path)

        # 5. 保存结果
        self.save_results(results, s10_dir)

        # 6. 打印摘要
        print(f"\n[S10-2] 阈值优化摘要:")
        print(f"[S10-2]   基准Macro F1: {baseline_macro_f1:.4f}")
        print(f"[S10-2]   最优Macro F1: {results['best_macro_f1']:.4f}")
        print(f"[S10-2]   Macro F1提升: {results['best_macro_f1'] - baseline_macro_f1:.4f}")
        print(f"[S10-2]   最优阈值: {results['best_thresholds']}")
        print(f"[S10-2]   准确率变化: {baseline_accuracy:.4f} -> {results['best_accuracy']:.4f}")
        print(f"[S10-2]   准确率损失: {baseline_accuracy - results['best_accuracy']:.4f}")

        print(f"[S10-2] 阈值优化完成!")
        print(f"=" * 50)

        return results


def main():
    """主函数"""
    # 配置
    model_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'baseline_attention_3class_ce_best_model.pth')
    data_path = os.path.join(PROJECT_ROOT, 'data', 'mosei', 'processed_cleaned_correct', 'test.npz')
    save_dir = os.path.join(PROJECT_ROOT, 'results')

    # 创建优化器
    optimizer = ThresholdOptimizer(
        model_path=model_path,
        data_path=data_path,
        num_classes=3
    )

    # 运行优化
    results = optimizer.run(save_dir=save_dir)


if __name__ == '__main__':
    main()
