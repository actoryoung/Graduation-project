# -*- coding: utf-8 -*-
"""
预测器模块 - 统一推理接口

本模块实现了多模态情感分析的统一推理接口，协调各特征提取器和融合模型
进行预测。采用依赖注入模式，支持灵活的模态组合和优雅的异常处理。

主要功能:
    - 依赖注入模式：通过构造函数注入所有依赖
    - 统一预测接口：支持可选的文本/音频/视频输入
    - 缺失模态处理：自动处理缺失的模态
    - 异常处理：特征提取失败时优雅降级
    - 批量预测：支持批量处理多个输入

数据源支持:
    - cmu_mosei: 使用GloVe(300) + COVAREP(74) + OpenFace(25)预提取特征
    - default: 使用BERT(768) + wav2vec(768) + OpenFace(25)实时特征

示例:
    >>> from src.inference.predictor import Predictor
    >>> predictor = Predictor()
    >>> result = predictor.predict(text="I'm very happy today!")
    >>> print(result['emotion'])  # 'positive'
"""

import sys
import os
from typing import Optional, List, Dict, Union
import warnings

import numpy as np
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config, EMOTIONS


class Predictor:
    """
    多模态情感分析预测器

    统一的推理接口，协调各特征提取器和融合模型进行预测。
    采用依赖注入模式，便于测试和替换组件。

    Attributes:
        text_extractor: 文本特征提取器（BERT或GloVe）
        audio_extractor: 音频特征提取器（wav2vec或COVAREP）
        video_extractor: 视频特征提取器（OpenFace）
        model: 多模态融合模型
        device: 运行设备
        data_source: 数据源配置

    Examples:
        >>> predictor = Predictor()
        >>> result = predictor.predict(text="Hello world")
        >>> print(result['emotion'])
        'neutral'
    """

    def __init__(
        self,
        text_extractor = None,
        audio_extractor = None,
        video_extractor = None,
        model = None,
        model_path: str = None
    ):
        """
        初始化预测器（依赖注入模式）

        通过构造函数注入所有依赖组件。所有参数都是可选的，
        但至少需要提供一个特征提取器和一个模型才能正常工作。

        Args:
            text_extractor: 文本特征提取器，默认根据Config.data_source自动创建
            audio_extractor: 音频特征提取器，默认根据Config.data_source自动创建
            video_extractor: 视频特征提取器，默认为None时自动创建
            model: 多模态融合模型，默认为None时自动创建
            model_path: 预训练模型路径，如果提供则加载模型权重

        Examples:
            >>> # 使用默认组件（根据Config.data_source自动选择）
            >>> predictor = Predictor()
            >>>
            >>> # 使用训练好的模型
            >>> predictor = Predictor(model_path='checkpoints/best_model.pth')
        """
        # 检测是否是临时模型（best_model_temp.pth）
        self.is_temp_model = (
            model_path and
            'temp' in model_path.lower() and
            os.path.exists(model_path)
        )

        # 如果是临时模型，使用临时模型架构
        if self.is_temp_model:
            from src.models.temp_model import ImprovedModel
            self.model = model or ImprovedModel()
        elif model_path and 'bert_hybrid' in model_path.lower():
            # BERT混合模型使用特定架构
            from src.models.bert_hybrid_model import BERTHybridModel
            self.model = model or BERTHybridModel()
        else:
            # 动态导入（避免循环导入）
            from src.models.fusion_module import MultimodalFusionModule
            self.model = model or MultimodalFusionModule()

        # 获取设备（在加载模型前）
        self.device = self.model.device

        # 加载预训练模型（如果提供）
        if model_path is not None:
            self._load_model(model_path)

        # 保存数据源配置
        self.data_source = Config.data_source

        # 根据数据源初始化文本特征提取器
        if text_extractor is None:
            # 强制使用BERT特征提取器（用于BERT混合模型）
            from src.features.text_features import BERTFeatureExtractor
            self.text_extractor = BERTFeatureExtractor()
        else:
            self.text_extractor = text_extractor

        # 临时模型跳过音频/视频提取器初始化（避免下载大型模型）
        if self.is_temp_model:
            self.audio_extractor = None
            self.video_extractor = None
        else:
            # 初始化可选的组件（允许失败）
            try:
                from src.features.audio_features import Wav2VecFeatureExtractor
                self.audio_extractor = audio_extractor or Wav2VecFeatureExtractor()
            except Exception as e:
                warnings.warn(f"音频特征提取器初始化失败: {e}，音频功能将不可用")
                self.audio_extractor = None

            try:
                from src.features.video_features import OpenFaceFeatureExtractor
                self.video_extractor = video_extractor or OpenFaceFeatureExtractor()
            except Exception as e:
                warnings.warn(f"视频特征提取器初始化失败: {e}，视频功能将不可用")
                self.video_extractor = None

        # 确保模型在评估模式
        self.model.eval()

    def _load_model(self, model_path: str):
        """加载预训练模型

        Args:
            model_path: 模型checkpoint路径
        """
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[OK] 已加载训练模型: {model_path}")
            if 'epoch' in checkpoint:
                print(f"  - 训练轮数: {checkpoint['epoch']}")
            if 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                print(f"  - 准确率: {metrics.get('val_acc', 'N/A')}")
        else:
            # 兼容纯state_dict格式
            self.model.load_state_dict(checkpoint)
            print(f"[OK] 已加载模型权重: {model_path}")

    def predict(
        self,
        text: Optional[str] = None,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None
    ) -> Dict:
        """
        统一的预测接口

        支持可选的文本/音频/视频输入，自动处理缺失模态。
        至少需要提供一个模态输入，否则抛出ValueError。

        Args:
            text: 文本输入
            audio_path: 音频文件路径
            video_path: 视频文件路径

        Returns:
            包含以下键的字典:
                - emotion (str): 预测的情感类别
                - confidence (float): 预测置信度 (0-1)
                - probabilities (dict): 各情感类别的概率
                - available_modalities (list): 成功提取的模态列表

        Raises:
            ValueError: 所有输入都为空时

        Examples:
            >>> predictor = Predictor()
            >>>
            >>> # 仅文本
            >>> result = predictor.predict(text="I love this!")
            >>>
            >>> # 多模态
            >>> result = predictor.predict(
            ...     text="Great movie!",
            ...     audio_path="audio.wav",
            ...     video_path="video.mp4"
            ... )
            >>>
            >>> print(result['emotion'])
            'positive'
            >>> print(f"{result['confidence']:.2%}")
            '85.00%'
        """
        # 验证至少有一个输入
        if text is None and audio_path is None and video_path is None:
            raise ValueError(
                "至少需要提供一个输入（text/audio_path/video_path）"
            )

        # 初始化特征和可用模态列表
        text_feat = None
        audio_feat = None
        video_feat = None
        available_modalities = []

        # 提取文本特征
        if text is not None:
            try:
                text_feat = self._extract_text_feature(text)
                available_modalities.append('text')
            except Exception as e:
                warnings.warn(f"文本特征提取失败: {e}，将使用零向量")

        # 提取音频特征
        if audio_path is not None:
            if self.audio_extractor is not None:
                try:
                    audio_feat = self._extract_audio_feature(audio_path)
                    available_modalities.append('audio')
                except Exception as e:
                    warnings.warn(f"音频特征提取失败: {e}，将使用零向量")
            else:
                warnings.warn("音频特征提取器未初始化，跳过音频输入")

        # 提取视频特征
        if video_path is not None:
            if self.video_extractor is not None:
                try:
                    video_feat = self._extract_video_feature(video_path)
                    available_modalities.append('video')
                except Exception as e:
                    warnings.warn(f"视频特征提取失败: {e}，将使用零向量")
            else:
                warnings.warn("视频特征提取器未初始化，跳过视频输入")

        # 检查是否至少成功提取了一个模态
        if not available_modalities:
            raise RuntimeError(
                "所有模态特征提取均失败，请检查输入是否有效"
            )

        # 转换为torch张量并添加batch维度
        text_feat_tensor = self._to_tensor(text_feat) if text_feat is not None else None
        audio_feat_tensor = self._to_tensor(audio_feat) if audio_feat is not None else None
        video_feat_tensor = self._to_tensor(video_feat) if video_feat is not None else None

        # 使用模型预测
        result = self.model.predict(
            text_feat=text_feat_tensor,
            audio_feat=audio_feat_tensor,
            video_feat=video_feat_tensor
        )

        # 添加可用模态信息
        result['available_modalities'] = available_modalities

        return result

    def batch_predict(self, inputs: List[Dict]) -> List[Dict]:
        """
        批量预测

        对多个输入进行批量预测。每个输入是一个字典，包含text/audio_path/video_path键。

        Args:
            inputs: 输入列表，每个元素是包含以下可选键的字典:
                - text (str): 文本输入
                - audio_path (str): 音频文件路径
                - video_path (str): 视频文件路径

        Returns:
            预测结果列表，每个元素是predict()返回的字典

        Raises:
            ValueError: inputs为空或格式不正确时

        Examples:
            >>> predictor = Predictor()
            >>> inputs = [
            ...     {'text': 'I love this!'},
            ...     {'text': 'This is terrible', 'audio_path': 'audio.wav'},
            ...     {'text': 'Amazing!', 'audio_path': 'a.wav', 'video_path': 'v.mp4'}
            ... ]
            >>> results = predictor.batch_predict(inputs)
            >>> print(len(results))
            3
            >>> print(results[0]['emotion'])
            'positive'
        """
        if not inputs:
            raise ValueError("inputs列表不能为空")

        results = []
        for i, input_data in enumerate(inputs):
            try:
                # 验证输入格式
                if not isinstance(input_data, dict):
                    warnings.warn(
                        f"inputs[{i}] 不是字典类型，跳过"
                    )
                    continue

                # 提取参数
                text = input_data.get('text')
                audio_path = input_data.get('audio_path')
                video_path = input_data.get('video_path')

                # 预测
                result = self.predict(text, audio_path, video_path)
                results.append(result)

            except Exception as e:
                warnings.warn(f"inputs[{i}] 预测失败: {e}")
                # 添加错误结果
                results.append({
                    'emotion': 'unknown',
                    'confidence': 0.0,
                    'probabilities': {},
                    'available_modalities': [],
                    'error': str(e)
                })

        return results

    def _create_glove_extractor(self):
        """
        创建GloVe文本特征提取器

        优先使用真实的GloVe词向量文件，如果不可用则使用简化版本。

        Returns:
            文本特征提取器对象
        """
        import warnings

        try:
            # 尝试使用真实的GloVe特征提取器
            from src.features.glove_extractor import GloVeFeatureExtractor
            extractor = GloVeFeatureExtractor()

            # 检查是否成功加载了词向量
            if extractor.glove.is_loaded():
                print("[OK] 使用真实GloVe词向量进行特征提取")
                return extractor
            else:
                print("[警告] GloVe词向量文件不存在，使用简化版特征提取器")
                raise ImportError("GloVe词向量未加载")

        except (ImportError, Exception) as e:
            warnings.warn(f"无法加载真实GloVe特征提取器: {e}")
            warnings.warn("将使用简化版特征提取器（预测结果不可信）")

            # 降级到简化版
            class SimpleGloVeExtractor:
                """简化的GloVe文本特征提取器"""
                def __init__(self, dim=300):
                    self.dim = dim

                def extract_from_raw(self, text: str) -> np.ndarray:
                    """从原始文本提取GloVe特征（简化版）"""
                    # 返回小的随机向量作为占位符
                    np.random.seed(hash(text) % 1000000)
                    return np.random.randn(self.dim).astype(np.float32) * 0.01

            return SimpleGloVeExtractor(dim=Config.text_dim)

    def _extract_text_feature(self, text: str) -> np.ndarray:
        """
        提取文本特征

        Args:
            text: 文本字符串

        Returns:
            文本特征向量 (300 for GloVe, 768 for BERT)
        """
        return self.text_extractor.extract_from_raw(text)

    def _extract_audio_feature(self, audio_path: str) -> np.ndarray:
        """
        提取音频特征

        Args:
            audio_path: 音频文件路径

        Returns:
            音频特征向量 (74 for COVAREP, 768 for wav2vec)
        """
        return self.audio_extractor.extract_from_raw(audio_path)

    def _extract_video_feature(self, video_path: str) -> np.ndarray:
        """
        提取视频特征

        Args:
            video_path: 视频文件路径

        Returns:
            视频特征向量 (25 for OpenFace)
        """
        return self.video_extractor.extract_from_raw(video_path)

    def _to_tensor(self, features: np.ndarray) -> torch.Tensor:
        """
        将numpy数组转换为torch tensor并添加batch维度

        Args:
            features: numpy特征数组

        Returns:
            torch tensor，形状为(1, feature_dim)
        """
        # 转换为torch tensor
        tensor = torch.from_numpy(features).float()

        # 添加batch维度 (feature_dim,) -> (1, feature_dim)
        tensor = tensor.unsqueeze(0)

        # 移到正确设备
        tensor = tensor.to(self.device)

        return tensor

    def set_model_eval_mode(self):
        """将模型设置为评估模式"""
        self.model.eval()

    def get_model(self):
        """获取模型实例（用于加载checkpoint等操作）"""
        return self.model

    def get_device(self) -> str:
        """获取当前设备"""
        return self.device


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == '__main__':
    # 设置Windows控制台编码
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print("=" * 70)
    print("预测器模块测试")
    print("=" * 70)

    # 测试1: 初始化预测器
    print("\n[测试1] 初始化预测器")
    print("-" * 70)
    try:
        predictor = Predictor()
        print(f"[OK] 预测器初始化成功")
        print(f"  - 设备: {predictor.device}")
        print(f"  - 文本提取器: {type(predictor.text_extractor).__name__}")
        print(f"  - 音频提取器: {type(predictor.audio_extractor).__name__ if predictor.audio_extractor else 'None (不可用)'}")
        print(f"  - 视频提取器: {type(predictor.video_extractor).__name__ if predictor.video_extractor else 'None (不可用)'}")
        print(f"  - 融合模型: {type(predictor.model).__name__}")
    except Exception as e:
        print(f"[FAIL] 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # 测试2: 仅文本输入
    print("\n[测试2] 仅文本输入")
    print("-" * 70)
    test_texts = [
        "I'm very happy and excited today!",
        "This is the worst day of my life.",
        "I feel neutral about this situation.",
        "Absolutely amazing! Love it so much!"
    ]

    for i, text in enumerate(test_texts, 1):
        try:
            print(f"\n测试用例 {i}: \"{text[:50]}...\"")
            result = predictor.predict(text=text)

            print(f"  - 预测情感: {result['emotion']}")
            print(f"  - 置信度: {result['confidence']:.4f}")
            print(f"  - 可用模态: {result['available_modalities']}")

            # 显示概率分布
            print(f"  - 概率分布:")
            for emotion, prob in sorted(
                result['probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]:  # 只显示前3个
                print(f"      {emotion}: {prob:.4f}")

        except Exception as e:
            print(f"  [FAIL] 预测失败: {e}")

    # 测试3: 多模态输入（使用模拟文件）
    print("\n[测试3] 多模态输入（模拟）")
    print("-" * 70)
    print("注意: 由于没有真实的音频/视频文件，此测试主要验证异常处理")

    # 测试不存在的文件（应该优雅降级）
    try:
        result = predictor.predict(
            text="This is a test with audio and video",
            audio_path="nonexistent_audio.wav",
            video_path="nonexistent_video.mp4"
        )
        print(f"[OK] 预测成功（音频/视频降级）")
        print(f"  - 预测情感: {result['emotion']}")
        print(f"  - 置信度: {result['confidence']:.4f}")
        print(f"  - 可用模态: {result['available_modalities']}")
    except Exception as e:
        print(f"✗ 预测失败: {e}")

    # 测试4: 异常情况 - 所有输入为空
    print("\n[测试4] 异常情况 - 所有输入为空")
    print("-" * 70)
    try:
        result = predictor.predict()
        print(f"[FAIL] 应该抛出ValueError但没有")
    except ValueError as e:
        print(f"[OK] 正确捕获ValueError: {e}")
    except Exception as e:
        print(f"[FAIL] 捕获到意外的异常: {e}")

    # 测试5: 批量预测
    print("\n[测试5] 批量预测")
    print("-" * 70)
    batch_inputs = [
        {'text': 'I love this product!'},
        {'text': 'This is terrible quality'},
        {'text': 'It is okay, nothing special'},
        {'text': 'Amazing experience! Highly recommend!'},
        {'text': 'Not worth the money'}
    ]

    try:
        results = predictor.batch_predict(batch_inputs)
        print(f"[OK] 批量预测成功")
        print(f"  - 输入数量: {len(batch_inputs)}")
        print(f"  - 输出数量: {len(results)}")

        # 显示前3个结果
        for i, result in enumerate(results[:3], 1):
            print(f"\n  结果 {i}:")
            print(f"    - 情感: {result['emotion']}")
            print(f"    - 置信度: {result['confidence']:.4f}")

    except Exception as e:
        print(f"[FAIL] 批量预测失败: {e}")

    # 测试6: 批量预测（包含无效输入）
    print("\n[测试6] 批量预测（包含无效输入）")
    print("-" * 70)
    mixed_inputs = [
        {'text': 'This is valid'},
        {'text': 'Also valid'},
        "invalid input type",  # 无效类型
        None,  # 无效类型
        {'text': 'Another valid one'}
    ]

    try:
        results = predictor.batch_predict(mixed_inputs)
        print(f"[OK] 批量预测成功（包含无效输入）")
        print(f"  - 输入数量: {len(mixed_inputs)}")
        print(f"  - 输出数量: {len(results)}")
        print(f"  - 成功预测: {sum(1 for r in results if 'error' not in r)}")
        print(f"  - 失败预测: {sum(1 for r in results if 'error' in r)}")

    except Exception as e:
        print(f"[FAIL] 批量预测失败: {e}")

    # 测试7: 模型方法测试
    print("\n[测试7] 模型方法测试")
    print("-" * 70)
    try:
        model = predictor.get_model()
        device = predictor.get_device()

        print(f"[OK] 获取模型和设备成功")
        print(f"  - 模型类型: {type(model).__name__}")
        print(f"  - 设备: {device}")

        # 测试eval模式
        predictor.set_model_eval_mode()
        print(f"  [OK] 模型已设置为评估模式")

    except Exception as e:
        print(f"[FAIL] 模型方法测试失败: {e}")

    # 测试总结
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
    print("\n使用方法:")
    print("  from src.inference.predictor import Predictor")
    print("  predictor = Predictor()")
    print("  result = predictor.predict(text='Your text here')")
    print("  print(result['emotion'])  # 预测的情感")
    print("  print(result['confidence'])  # 置信度")
    print("\n批量预测:")
    print("  inputs = [")
    print("      {'text': 'First text'},")
    print("      {'text': 'Second text', 'audio_path': 'audio.wav'}")
    print("  ]")
    print("  results = predictor.batch_predict(inputs)")
