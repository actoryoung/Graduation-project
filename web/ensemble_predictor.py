# -*- coding: utf-8 -*-
"""
S7-v1集成预测器 - 用于Web应用

支持S7-v1双模型加权集成:
- 模型1: baseline_attention_3class_weighted_best_model.pth (S3+S4, 权重1.5)
- 模型2: baseline_attention_3class_ce_best_model.pth (S3, 权重1.0)

测试准确率: 59.47%
Negative准确率: 23.38%
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.attention_fusion_model import AttentionFusionModel


class S7EnsemblePredictor:
    """
    S7-v1集成预测器

    使用两个模型进行加权投票:
    - S3+S4模型 (权重1.5)
    - S3模型 (权重1.0)
    """

    def __init__(self,
                 model_paths: List[str],
                 weights: List[float],
                 num_classes: int = 3,
                 device: str = 'cuda'):
        """
        初始化集成预测器

        Args:
            model_paths: 模型路径列表
            weights: 权重列表
            num_classes: 类别数量
            device: 设备 ('cuda' or 'cpu')
        """
        self.num_classes = num_classes
        self.device = device if torch.cuda.is_available() else 'cpu'

        # 验证输入
        if len(model_paths) != len(weights):
            raise ValueError(f"模型数量({len(model_paths)})与权重数量({len(weights)})不匹配")

        # 加载模型
        self.models = []
        self.weights = weights

        for model_path in model_paths:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            model = self._load_model(model_path)
            model.eval()
            model.to(self.device)
            self.models.append(model)

        print(f"[S7Ensemble] 已加载 {len(self.models)} 个模型")
        print(f"[S7Ensemble] 权重: {self.weights}")
        print(f"[S7Ensemble] 设备: {self.device}")

    def _load_model(self, model_path: str) -> AttentionFusionModel:
        """加载单个模型"""
        # 根据模型维度初始化
        checkpoint = torch.load(model_path, map_location=self.device)

        # 获取模型配置
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # 尝试从state_dict推断维度
            text_dim = state_dict['text_encoder.0.weight'].shape[1]
            audio_dim = state_dict['audio_encoder.0.weight'].shape[1]
            video_dim = state_dict['video_encoder.0.weight'].shape[1]
        else:
            # 默认配置
            text_dim = 300
            audio_dim = 74
            video_dim = 710

        # 创建模型
        model = AttentionFusionModel(
            text_dim=text_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
            num_classes=self.num_classes
        )

        # 加载权重
        model.load_state_dict(state_dict)
        return model

    def predict(self,
                text_features: Optional[np.ndarray] = None,
                audio_features: Optional[np.ndarray] = None,
                video_features: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        集成预测

        Args:
            text_features: 文本特征 (1, 300) or (batch, 300)
            audio_features: 音频特征 (1, 74) or (batch, 74)
            video_features: 视频特征 (1, 710) or (batch, 710)

        Returns:
            预测结果字典:
            {
                'emotion': 情感类别,
                'emotion_zh': 中文情感,
                'probabilities': 各类别概率,
                'confidence': 置信度
            }
        """
        # 验证输入
        if text_features is None and audio_features is None and video_features is None:
            raise ValueError("至少需要提供一种模态的特征")

        # 转换为tensor并移动到设备
        text_tensor = self._prepare_input(text_features, 300)
        audio_tensor = self._prepare_input(audio_features, 74)
        video_tensor = self._prepare_input(video_features, 710)

        # 对每个模型预测
        probs_list = []
        with torch.no_grad():
            for model, weight in zip(self.models, self.weights):
                # 模型推理
                outputs = model(text_tensor, audio_tensor, video_tensor)
                # Softmax
                probs = torch.softmax(outputs, dim=1)
                # 加权
                probs_list.append(probs * weight)

        # 加权平均
        ensemble_probs = torch.stack(probs_list).sum(dim=0)

        # 获取预测结果
        prob_array = ensemble_probs.cpu().numpy()[0]  # (num_classes,)
        pred_class = int(prob_array.argmax())
        confidence = float(prob_array.max())

        # 类别映射
        emotions = ['Negative', 'Neutral', 'Positive']
        emotions_zh = ['负面', '中性', '正面']

        return {
            'emotion': emotions[pred_class],
            'emotion_zh': emotions_zh[pred_class],
            'probabilities': {
                'Negative': float(prob_array[0]),
                'Neutral': float(prob_array[1]),
                'Positive': float(prob_array[2])
            },
            'confidence': confidence
        }

    def _prepare_input(self, features: Optional[np.ndarray], expected_dim: int) -> Optional[torch.Tensor]:
        """准备输入tensor"""
        if features is None:
            return None

        # 转换为tensor
        tensor = torch.FloatTensor(features)

        # 检查维度
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)  # (1, dim)

        # 移动到设备
        return tensor.to(self.device)


def create_s7v1_predictor(device: str = 'cuda') -> S7EnsemblePredictor:
    """
    创建S7-v1集成预测器

    Returns:
        S7EnsemblePredictor实例
    """
    model_paths = [
        os.path.join(PROJECT_ROOT, 'checkpoints', 'baseline_attention_3class_weighted_best_model.pth'),
        os.path.join(PROJECT_ROOT, 'checkpoints', 'baseline_attention_3class_ce_best_model.pth')
    ]

    weights = [1.5, 1.0]

    return S7EnsemblePredictor(
        model_paths=model_paths,
        weights=weights,
        num_classes=3,
        device=device
 )


# ==============================================================================
# 测试代码
# ==============================================================================

if __name__ == '__main__':
    # 测试集成预测器
    print("=" * 60)
    print("S7-v1集成预测器测试")
    print("=" * 60)

    try:
        # 创建预测器
        predictor = create_s7v1_predictor()

        # 模拟输入
        text_features = np.random.randn(1, 300)
        audio_features = np.random.randn(1, 74)
        video_features = np.random.randn(1, 710)

        # 预测
        result = predictor.predict(
            text_features=text_features,
            audio_features=audio_features,
            video_features=video_features
        )

        print("\n预测结果:")
        print(f"  情感类别: {result['emotion']}")
        print(f"  中文情感: {result['emotion_zh']}")
        print(f"  置信度: {result['confidence']:.4f}")
        print(f"  概率分布:")
        for emotion, prob in result['probabilities'].items():
            print(f"    {emotion}: {prob:.4f}")

        print("\n✅ 集成预测器测试成功!")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
