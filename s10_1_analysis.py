# -*- coding: utf-8 -*-
"""
S10-1: 数据分析 - 分析模型预测分布

任务:
1. 加载S7-v1模型（使用单个模型作为代表）
2. 在测试集上预测，获取各类别置信度
3. 绘制置信度分布直方图
4. 生成混淆矩阵
5. 计算当前Macro F1

输出:
- 预测置信度分布图
- 混淆矩阵
- 当前Macro F1基准
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from typing import Dict, List, Tuple
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import Config, DEVICE
from src.models.fusion_module import MultimodalFusionModule


class Simple3ClassModel(nn.Module):
    """
    简单的3类多模态融合模型（带注意力机制）
    用于加载已有的3类模型checkpoint
    """

    def __init__(self, text_dim=300, audio_dim=74, video_dim=710, num_classes=3):
        super(Simple3ClassModel, self).__init__()

        # 编码器维度
        text_encoded_dim = 128
        audio_encoded_dim = 64
        video_encoded_dim = 128

        # 编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, text_encoded_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, audio_encoded_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, video_encoded_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 投影层（用于注意力）
        self.text_proj = nn.Linear(text_encoded_dim, 64)
        self.audio_proj = nn.Linear(audio_encoded_dim, 64)
        self.video_proj = nn.Linear(video_encoded_dim, 64)

        # 交叉注意力
        self.cross_attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        # 融合层 (匹配checkpoint结构)
        fused_dim = 64 * 3  # 192
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 分类器
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, text_feat, audio_feat, video_feat):
        # 编码各模态
        text_encoded = self.text_encoder(text_feat)
        audio_encoded = self.audio_encoder(audio_feat)
        video_encoded = self.video_encoder(video_feat)

        # 投影到注意力维度
        text_proj = self.text_proj(text_encoded)  # (B, 64)
        audio_proj = self.audio_proj(audio_encoded)  # (B, 64)
        video_proj = self.video_proj(video_encoded)  # (B, 64)

        # 堆叠为序列 (B, 3, 64)
        seq = torch.stack([text_proj, audio_proj, video_proj], dim=1)

        # 交叉注意力
        attn_out, _ = self.cross_attention(seq, seq, seq)  # (B, 3, 64)

        # 展平
        fused = attn_out.flatten(1)  # (B, 192)

        # 融合
        fused = self.fusion(fused)

        # 分类
        logits = self.classifier(fused)

        return logits


class S10Analyzer:
    """S10数据分析器"""

    def __init__(self, model_path: str, data_path: str, num_classes=3):
        """
        初始化分析器

        Args:
            model_path: 模型checkpoint路径
            data_path: 测试数据路径 (.npz文件)
            num_classes: 类别数量
        """
        self.model_path = model_path
        self.data_path = data_path
        self.num_classes = num_classes
        self.device = DEVICE

        # 类别名称
        self.class_names = ['Negative', 'Neutral', 'Positive']

        # 7类到3类的映射
        # 7类: [0,1,2,3,4,5,6] -> [-3,-2,-1,0,1,2,3]
        # 3类: Negative [0,1,2]->0, Neutral [3]->1, Positive [4,5,6]->2
        self.label_map_7to3 = {
            0: 0, 1: 0, 2: 0,  # Negative
            3: 1,              # Neutral
            4: 2, 5: 2, 6: 2   # Positive
        }

        print(f"[S10-1] 初始化分析器")
        print(f"[S10-1] 模型: {model_path}")
        print(f"[S10-1] 数据: {data_path}")
        print(f"[S10-1] 设备: {self.device}")

    def load_model(self) -> nn.Module:
        """加载模型"""
        print(f"[S10-1] 加载模型...")

        # 创建模型
        model = Simple3ClassModel(num_classes=self.num_classes)

        # 加载checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        model.to(self.device)

        print(f"[S10-1] 模型加载完成")

        return model

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        加载测试数据

        Returns:
            (text_features, audio_features, video_features, labels): 特征和标签
        """
        print(f"[S10-1] 加载数据...")

        data = np.load(self.data_path)

        # 加载各模态特征
        text_features = data['text_features']  # (N, 300)
        audio_features = data['audio_features']  # (N, 74)
        video_features = data['video_features']  # (N, 710)
        labels_7class = data['labels']  # (N,) - 7类标签

        # 转换7类标签到3类
        labels = np.array([self.label_map_7to3[int(l)] for l in labels_7class])

        print(f"[S10-1] 数据加载完成")
        print(f"[S10-1] 样本数: {len(labels)}")
        print(f"[S10-1] 文本特征: {text_features.shape}")
        print(f"[S10-1] 音频特征: {audio_features.shape}")
        print(f"[S10-1] 视频特征: {video_features.shape}")
        print(f"[S10-1] 原始标签分布(7类): {np.bincount(labels_7class)}")
        print(f"[S10-1] 转换后标签分布(3类): {np.bincount(labels)}")

        return text_features, audio_features, video_features, labels

    def predict_with_confidence(self, model: nn.Module, text_features: np.ndarray,
                                audio_features: np.ndarray, video_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测并获取置信度

        Args:
            model: 模型
            text_features: 文本特征 (N, 300)
            audio_features: 音频特征 (N, 74)
            video_features: 视频特征 (N, 710)

        Returns:
            (predictions, confidences): 预测结果和置信度
        """
        print(f"[S10-1] 进行预测...")

        model.eval()
        all_predictions = []
        all_confidences = []
        all_probs = []

        batch_size = 32
        num_samples = len(text_features)

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_text = text_features[i:i+batch_size]
                batch_audio = audio_features[i:i+batch_size]
                batch_video = video_features[i:i+batch_size]

                # 转换为tensor
                text_feat = torch.FloatTensor(batch_text).to(self.device)
                audio_feat = torch.FloatTensor(batch_audio).to(self.device)
                video_feat = torch.FloatTensor(batch_video).to(self.device)

                # 前向传播
                logits = model(text_feat, audio_feat, video_feat)

                # 获取概率
                probs = torch.softmax(logits, dim=1)

                # 获取预测和置信度
                predictions = torch.argmax(probs, dim=1)
                confidences = torch.max(probs, dim=1)[0]

                all_predictions.extend(predictions.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        predictions = np.array(all_predictions)
        confidences = np.array(all_confidences)
        probs = np.array(all_probs)

        print(f"[S10-1] 预测完成")
        print(f"[S10-1] 预测数: {len(predictions)}")

        return predictions, confidences, probs

    def compute_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict:
        """
        计算评估指标

        Args:
            predictions: 预测结果
            labels: 真实标签

        Returns:
            指标字典
        """
        print(f"[S10-1] 计算指标...")

        # 计算准确率
        accuracy = (predictions == labels).mean()

        # 计算F1分数
        macro_f1 = f1_score(labels, predictions, average='macro')
        weighted_f1 = f1_score(labels, predictions, average='weighted')

        # 各类别F1
        f1_per_class = f1_score(labels, predictions, average=None)

        metrics = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'f1_per_class': f1_per_class.tolist()
        }

        print(f"[S10-1] 准确率: {accuracy:.4f}")
        print(f"[S10-1] Macro F1: {macro_f1:.4f}")
        print(f"[S10-1] Weighted F1: {weighted_f1:.4f}")
        for i, class_name in enumerate(self.class_names):
            print(f"[S10-1] {class_name} F1: {f1_per_class[i]:.4f}")

        return metrics

    def plot_confidence_distribution(self, confidences: np.ndarray, labels: np.ndarray, save_path: str):
        """
        绘制置信度分布图

        Args:
            confidences: 置信度数组
            labels: 真实标签
            save_path: 保存路径
        """
        print(f"[S10-1] 绘制置信度分布...")

        plt.figure(figsize=(12, 5))

        # 子图1: 整体置信度分布
        plt.subplot(1, 2, 1)
        plt.hist(confidences, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(confidences.mean(), color='red', linestyle='--', linewidth=2, label=f'平均值: {confidences.mean():.3f}')
        plt.xlabel('置信度', fontsize=12)
        plt.ylabel('样本数', fontsize=12)
        plt.title('预测置信度分布', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)

        # 子图2: 各类别置信度分布
        plt.subplot(1, 2, 2)
        for i, class_name in enumerate(self.class_names):
            mask = (labels == i)
            if mask.sum() > 0:
                plt.hist(confidences[mask], bins=30, alpha=0.5, label=class_name)

        plt.xlabel('置信度', fontsize=12)
        plt.ylabel('样本数', fontsize=12)
        plt.title('各类别置信度分布', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[S10-1] 置信度分布图已保存: {save_path}")

    def plot_confusion_matrix(self, predictions: np.ndarray, labels: np.ndarray, save_path: str):
        """
        绘制混淆矩阵

        Args:
            predictions: 预测结果
            labels: 真实标签
            save_path: 保存路径
        """
        print(f"[S10-1] 绘制混淆矩阵...")

        cm = confusion_matrix(labels, predictions)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar_kws={'label': '样本数'})
        plt.xlabel('预测类别', fontsize=12)
        plt.ylabel('真实类别', fontsize=12)
        plt.title('混淆矩阵', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[S10-1] 混淆矩阵已保存: {save_path}")

    def save_results(self, metrics: Dict, save_dir: str):
        """保存结果"""
        results_path = os.path.join(save_dir, 's10_1_baseline_results.json')

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"[S10-1] 结果已保存: {results_path}")

    def run(self, save_dir: str = 'results'):
        """运行完整分析流程"""
        print(f"[S10-1] 开始S10-1数据分析")
        print(f"=" * 50)

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 1. 加载模型
        model = self.load_model()

        # 2. 加载数据
        text_features, audio_features, video_features, labels = self.load_data()

        # 3. 预测
        predictions, confidences, probs = self.predict_with_confidence(
            model, text_features, audio_features, video_features
        )

        # 4. 计算指标
        metrics = self.compute_metrics(predictions, labels)

        # 5. 绘制置信度分布
        confidence_plot_path = os.path.join(save_dir, 's10_1_confidence_distribution.png')
        self.plot_confidence_distribution(confidences, labels, confidence_plot_path)

        # 6. 绘制混淆矩阵
        cm_plot_path = os.path.join(save_dir, 's10_1_confusion_matrix.png')
        self.plot_confusion_matrix(predictions, labels, cm_plot_path)

        # 7. 保存结果
        self.save_results(metrics, save_dir)

        print(f"[S10-1] 分析完成!")
        print(f"=" * 50)

        return metrics


def main():
    """主函数"""
    # 配置
    model_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'baseline_attention_3class_ce_best_model.pth')
    data_path = os.path.join(PROJECT_ROOT, 'data', 'mosei', 'processed_cleaned_correct', 'test.npz')
    save_dir = os.path.join(PROJECT_ROOT, 'results', 's10')

    # 创建分析器
    analyzer = S10Analyzer(
        model_path=model_path,
        data_path=data_path,
        num_classes=3
    )

    # 运行分析
    metrics = analyzer.run(save_dir=save_dir)

    # 打印摘要
    print(f"\n[S10-1] 基准指标摘要:")
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")


if __name__ == '__main__':
    main()
