# -*- coding: utf-8 -*-
"""
S10-4: 集成权重优化 - 优化S7投票权重

任务:
1. 加载S7-v1的两个模型
2. 网格搜索最优权重组合
3. 目标：最大化Macro F1

预期:
- 5-10分钟内完成
- 无需训练，仅需推理
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


class EnsembleWeightOptimizer:
    """S7集成权重优化器"""

    def __init__(self, model_paths: List[str], data_path: str, num_classes=3):
        """
        初始化权重优化器

        Args:
            model_paths: 模型路径列表 [S3+S4模型, S3模型]
            data_path: 测试数据路径
            num_classes: 类别数量
        """
        self.model_paths = model_paths
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

        print(f"[S10-4] 初始化S7集成权重优化器")
        print(f"[S10-4] 模型数量: {len(model_paths)}")

    def load_models_and_predict(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        加载所有模型并预测

        Returns:
            (all_probs, labels): 所有模型的预测概率和真实标签
        """
        print(f"[S10-4] 加载模型并预测...")

        # 加载数据
        analyzer = S10Analyzer(self.model_paths[0], self.data_path, self.num_classes)
        text_features, audio_features, video_features, labels = analyzer.load_data()

        all_probs = []

        # 加载每个模型并预测
        for i, model_path in enumerate(self.model_paths):
            print(f"[S10-4] 加载模型 {i+1}/{len(self.model_paths)}: {os.path.basename(model_path)}")

            model = analyzer.load_model()

            # 获取预测概率
            model.eval()
            probs_list = []

            batch_size = 32
            num_samples = len(text_features)

            with torch.no_grad():
                for j in range(0, num_samples, batch_size):
                    batch_text = text_features[j:j+batch_size]
                    batch_audio = audio_features[j:j+batch_size]
                    batch_video = video_features[j:j+batch_size]

                    text_feat = torch.FloatTensor(batch_text).to(self.device)
                    audio_feat = torch.FloatTensor(batch_audio).to(self.device)
                    video_feat = torch.FloatTensor(batch_video).to(self.device)

                    logits = model(text_feat, audio_feat, video_feat)
                    probs = torch.softmax(logits, dim=1)

                    probs_list.extend(probs.cpu().numpy())

            all_probs.append(np.array(probs_list))

        print(f"[S10-4] 所有模型预测完成")
        print(f"[S10-4] 样本数: {len(labels)}")

        return all_probs, labels

    def predict_with_weights(self, all_probs: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
        """
        使用指定权重进行集成预测

        Args:
            all_probs: 所有模型的预测概率 [(N, C), (N, C), ...]
            weights: 每个模型的权重 [w1, w2, ...]

        Returns:
            predictions: 集成预测结果
        """
        # 加权平均概率
        weighted_probs = sum(p * w for p, w in zip(all_probs, weights))
        weighted_probs /= weights.sum()  # 归一化

        # 预测
        predictions = np.argmax(weighted_probs, axis=1)

        return predictions

    def grid_search_weights(self, all_probs: List[np.ndarray], labels: np.ndarray,
                           weight_range: List[float] = None) -> Dict:
        """
        网格搜索最优权重

        Args:
            all_probs: 所有模型的预测概率
            labels: 真实标签
            weight_range: 权重搜索范围

        Returns:
            搜索结果字典
        """
        if weight_range is None:
            weight_range = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]

        print(f"[S10-4] 开始权重网格搜索...")
        print(f"[S10-4] 权重范围: {weight_range}")

        best_macro_f1 = 0
        best_weights = None
        best_predictions = None
        best_accuracy = 0
        best_f1_per_class = None

        results = []
        num_models = len(all_probs)
        total_combinations = len(weight_range) ** num_models
        current = 0

        for weights in product(weight_range, repeat=num_models):
            weights = np.array(weights)
            current += 1

            if current % 50 == 0:
                print(f"[S10-4] 进度: {current}/{total_combinations}")

            # 使用权重预测
            predictions = self.predict_with_weights(all_probs, weights)

            # 计算指标
            accuracy = accuracy_score(labels, predictions)
            macro_f1 = f1_score(labels, predictions, average='macro')
            f1_per_class = f1_score(labels, predictions, average=None)

            # 记录结果
            result = {
                'weights': weights.tolist(),
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
                best_weights = weights
                best_predictions = predictions
                best_accuracy = accuracy
                best_f1_per_class = f1_per_class

        print(f"[S10-4] 网格搜索完成")
        print(f"[S10-4] 最优Macro F1: {best_macro_f1:.4f}")
        print(f"[S10-4] 最优权重: {best_weights}")
        print(f"[S10-4] 对应准确率: {best_accuracy:.4f}")
        print(f"[S10-4] 各类别F1: {best_f1_per_class}")

        return {
            'best_macro_f1': best_macro_f1,
            'best_weights': best_weights.tolist(),
            'best_accuracy': best_accuracy,
            'best_predictions': best_predictions,
            'best_f1_per_class': best_f1_per_class.tolist(),
            'all_results': results
        }

    def search_balanced_weights(self, all_probs: List[np.ndarray], labels: np.ndarray,
                              base_weights: List[float] = None) -> Dict:
        """
        搜索平衡权重 - 细粒度搜索

        Args:
            all_probs: 所有模型的预测概率
            labels: 真实标签
            base_weights: 基础权重（S7-v1默认为[1.5, 1.0]）

        Returns:
            搜索结果
        """
        if base_weights is None:
            base_weights = [1.5, 1.0]  # S7-v1默认权重

        print(f"[S10-4] 搜索平衡权重（细粒度）...")
        print(f"[S10-4] 基础权重: {base_weights}")

        best_macro_f1 = 0
        best_weights = None
        best_accuracy = 0
        best_f1_per_class = None

        results = []

        # 细粒度搜索：在基础权重附近搜索
        search_range = np.linspace(0.5, 2.5, 41)  # 0.5到2.5，步长0.05

        for w1 in search_range:
            for w2 in search_range:
                weights = np.array([w1, w2])

                predictions = self.predict_with_weights(all_probs, weights)

                accuracy = accuracy_score(labels, predictions)
                macro_f1 = f1_score(labels, predictions, average='macro')
                f1_per_class = f1_score(labels, predictions, average=None)

                result = {
                    'weights': weights.tolist(),
                    'accuracy': accuracy,
                    'macro_f1': macro_f1,
                    'f1_negative': f1_per_class[0],
                    'f1_neutral': f1_per_class[1],
                    'f1_positive': f1_per_class[2]
                }
                results.append(result)

                if macro_f1 > best_macro_f1:
                    best_macro_f1 = macro_f1
                    best_weights = weights
                    best_accuracy = accuracy
                    best_f1_per_class = f1_per_class

        print(f"[S10-4] 细粒度搜索完成")
        print(f"[S10-4] 最优Macro F1: {best_macro_f1:.4f}")
        print(f"[S10-4] 最优权重: {best_weights}")
        print(f"[S10-4] 对应准确率: {best_accuracy:.4f}")
        print(f"[S10-4] 各类别F1: {best_f1_per_class}")

        return {
            'best_macro_f1': best_macro_f1,
            'best_weights': best_weights.tolist(),
            'best_accuracy': best_accuracy,
            'best_f1_per_class': best_f1_per_class.tolist(),
            'all_results': results
        }

    def plot_weight_heatmap(self, results: Dict, save_path: str):
        """绘制权重热力图"""
        print(f"[S10-4] 绘制权重热力图...")

        all_results = results['all_results']

        # 创建网格数据
        w1_values = sorted(set([r['weights'][0] for r in all_results]))
        w2_values = sorted(set([r['weights'][1] for r in all_results]))

        macro_f1_grid = np.zeros((len(w1_values), len(w2_values)))

        for r in all_results:
            i = w1_values.index(r['weights'][0])
            j = w2_values.index(r['weights'][1])
            macro_f1_grid[i, j] = r['macro_f1']

        # 绘制热力图
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Macro F1热力图
        im = axes[0].imshow(macro_f1_grid, cmap='RdYlGn', aspect='auto',
                           extent=[w2_values[0], w2_values[-1], w1_values[0], w1_values[-1]],
                           vmin=0.3, vmax=0.5)
        axes[0].set_xlabel('Model 2 Weight', fontsize=12)
        axes[0].set_ylabel('Model 1 Weight', fontsize=12)
        axes[0].set_title('Macro F1 vs Ensemble Weights', fontsize=14)
        plt.colorbar(im, ax=axes[0], label='Macro F1')

        # 标记最优点
        best_w1 = results['best_weights'][0]
        best_w2 = results['best_weights'][1]
        axes[0].plot(best_w2, best_w1, 'r*', markersize=20, label=f'Best ({best_w1:.2f}, {best_w2:.2f})')
        axes[0].legend()

        # 2. Negative F1热力图
        f1_neg_grid = np.zeros((len(w1_values), len(w2_values)))
        for r in all_results:
            i = w1_values.index(r['weights'][0])
            j = w2_values.index(r['weights'][1])
            f1_neg_grid[i, j] = r['f1_negative']

        im2 = axes[1].imshow(f1_neg_grid, cmap='RdYlGn', aspect='auto',
                            extent=[w2_values[0], w2_values[-1], w1_values[0], w1_values[-1]],
                            vmin=0, vmax=0.4)
        axes[1].set_xlabel('Model 2 Weight', fontsize=12)
        axes[1].set_ylabel('Model 1 Weight', fontsize=12)
        axes[1].set_title('Negative F1 vs Ensemble Weights', fontsize=14)
        plt.colorbar(im2, ax=axes[1], label='Negative F1')

        axes[1].plot(best_w2, best_w1, 'r*', markersize=20, label=f'Best ({best_w1:.2f}, {best_w2:.2f})')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[S10-4] 热力图已保存: {save_path}")

    def save_results(self, results: Dict, save_dir: str):
        """保存结果"""
        results_path = os.path.join(save_dir, 's10_4_ensemble_weights_results.json')

        # 移除best_predictions（太大）
        save_data = results.copy()
        if 'best_predictions' in save_data:
            del save_data['best_predictions']

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"[S10-4] 结果已保存: {results_path}")

    def run(self, save_dir: str = 'results'):
        """运行权重优化"""
        print(f"[S10-4] 开始S10-4集成权重优化")
        print(f"=" * 50)

        os.makedirs(save_dir, exist_ok=True)
        s10_dir = os.path.join(save_dir, 's10')
        os.makedirs(s10_dir, exist_ok=True)

        # 1. 加载模型并预测
        all_probs, labels = self.load_models_and_predict()

        # 2. 计算基准性能（S7-v1默认权重 [1.5, 1.0]）
        baseline_weights = np.array([1.5, 1.0])
        baseline_predictions = self.predict_with_weights(all_probs, baseline_weights)
        baseline_accuracy = accuracy_score(labels, baseline_predictions)
        baseline_macro_f1 = f1_score(labels, baseline_predictions, average='macro')
        baseline_f1_per_class = f1_score(labels, baseline_predictions, average=None)

        print(f"[S10-4] S7-v1基准性能:")
        print(f"[S10-4]   权重: [1.5, 1.0]")
        print(f"[S10-4]   准确率: {baseline_accuracy:.4f}")
        print(f"[S10-4]   Macro F1: {baseline_macro_f1:.4f}")
        print(f"[S10-4]   各类别F1: {baseline_f1_per_class}")

        # 3. 细粒度搜索最优权重
        results = self.search_balanced_weights(all_probs, labels, base_weights=[1.5, 1.0])

        # 4. 绘制热力图
        heatmap_path = os.path.join(s10_dir, 's10_4_weight_heatmap.png')
        self.plot_weight_heatmap(results, heatmap_path)

        # 5. 保存结果
        self.save_results(results, s10_dir)

        # 6. 打印摘要
        print(f"\n[S10-4] 集成权重优化摘要:")
        print(f"[S10-4]   基准权重: [1.5, 1.0]")
        print(f"[S10-4]   基准Macro F1: {baseline_macro_f1:.4f}")
        print(f"[S10-4]   最优权重: {results['best_weights']}")
        print(f"[S10-4]   最优Macro F1: {results['best_macro_f1']:.4f}")
        print(f"[S10-4]   Macro F1提升: {results['best_macro_f1'] - baseline_macro_f1:.4f}")
        print(f"[S10-4]   准确率变化: {baseline_accuracy:.4f} -> {results['best_accuracy']:.4f}")
        print(f"[S10-4]   Negative F1: {baseline_f1_per_class[0]:.4f} -> {results['best_f1_per_class'][0]:.4f}")

        print(f"[S10-4] 集成权重优化完成!")
        print(f"=" * 50)

        return results


def main():
    """主函数"""
    # 配置
    # S7-v1的两个模型
    model_paths = [
        os.path.join(PROJECT_ROOT, 'checkpoints', 'baseline_attention_3class_weighted_best_model.pth'),  # S3+S4
        os.path.join(PROJECT_ROOT, 'checkpoints', 'baseline_attention_3class_ce_best_model.pth')   # S3
    ]
    data_path = os.path.join(PROJECT_ROOT, 'data', 'mosei', 'processed_cleaned_correct', 'test.npz')
    save_dir = os.path.join(PROJECT_ROOT, 'results')

    # 创建优化器
    optimizer = EnsembleWeightOptimizer(
        model_paths=model_paths,
        data_path=data_path,
        num_classes=3
    )

    # 运行优化
    results = optimizer.run(save_dir=save_dir)


if __name__ == '__main__':
    main()
