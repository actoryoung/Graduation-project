# -*- coding: utf-8 -*-
"""
BERT文本特征提取器模块

本模块实现了基于BERT预训练模型的文本特征提取器，用于将文本字符串
转换为768维的固定长度特征向量。使用[CLS] token的隐藏状态作为句子级
表示，适用于下游的情感分析等任务。

主要功能:
    - 使用BERT-base-uncased预训练模型
    - 自动分词和预处理
    - 输出768维特征向量
    - 支持CPU/CUDA设备切换
    - 异常处理（空文本、超长文本）

示例:
    >>> from src.features.text_features import BERTFeatureExtractor
    >>> extractor = BERTFeatureExtractor()
    >>> text = "This is a sample sentence for feature extraction."
    >>> features = extractor.extract_from_raw(text)
    >>> print(features.shape)
    (768,)
"""

from typing import Any, Dict, Union
import warnings

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from config import Config, TEXT_MODEL, MAX_TEXT_LENGTH, TEXT_DIM
from src.features.base import FeatureExtractor


class BERTFeatureExtractor(FeatureExtractor):
    """
    BERT文本特征提取器

    使用BERT-base-uncased预训练模型提取文本特征。提取器将输入文本
    通过BERT编码器，并使用[CLS] token的最终隐藏状态作为句子表示。

    类属性:
        model_name (str): BERT模型名称
        max_length (int): 最大序列长度
        feature_dim (int): 输出特征维度（768）

    示例:
        >>> extractor = BERTFeatureExtractor()
        >>> features = extractor.extract_from_raw("Hello world")
        >>> print(features.shape)
        (768,)
    """

    def __init__(self, device: str = None, model_name: str = None):
        """
        初始化BERT特征提取器

        加载预训练的BERT模型和分词器，并将模型移动到指定设备。

        Args:
            device: 计算设备 ('cuda' 或 'cpu')，默认为None时使用Config.DEVICE
            model_name: BERT模型名称，默认为None时使用Config.TEXT_MODEL

        Raises:
            ValueError: 当device不是'cuda'或'cpu'时
            RuntimeError: 当模型加载失败时
        """
        super().__init__(device)

        # 设置模型名称
        self.model_name = model_name or TEXT_MODEL
        self.max_length = MAX_TEXT_LENGTH
        self.feature_dim = TEXT_DIM

        # 加载分词器和模型（使用镜像站加速）
        try:
            import os
            # 设置Hugging Face镜像站（中国用户）
            os.environ['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                self.model = BertModel.from_pretrained(self.model_name)

            # 将模型移动到指定设备并设置为评估模式
            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            raise RuntimeError(f"加载BERT模型失败: {e}")

    def preprocess(self, raw_input: Union[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        预处理文本输入

        将文本字符串转换为BERT模型所需的输入格式（input_ids,
        attention_mask, token_type_ids）。如果输入已经是字典格式，
        则直接返回（确保所有张量在正确的设备上）。

        Args:
            raw_input: 原始输入，可以是：
                - str: 文本字符串
                - dict: 已分词的输入字典（包含input_ids等）

        Returns:
            包含以下键的字典:
                - input_ids (torch.Tensor): token IDs，形状为(1, seq_length)
                - attention_mask (torch.Tensor): 注意力掩码，形状为(1, seq_length)
                - token_type_ids (torch.Tensor): token类型IDs，形状为(1, seq_length)

        Raises:
            ValueError: 当输入为空字符串或格式不正确时
            TypeError: 当输入类型不匹配时

        示例:
            >>> extractor = BERTFeatureExtractor()
            >>> processed = extractor.preprocess("Hello world")
            >>> print(processed['input_ids'].shape)
            torch.Size([1, 5])
        """
        # 如果输入已经是字典格式（已分词），确保张量在正确设备上
        if isinstance(raw_input, dict):
            result = {}
            for key, value in raw_input.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.to(self.device)
                else:
                    result[key] = value
            return result

        # 验证输入类型
        if not isinstance(raw_input, str):
            raise TypeError(
                f"输入必须是字符串或字典，得到的是: {type(raw_input)}"
            )

        # 去除首尾空格
        text = raw_input.strip()

        # 验证非空
        if not text:
            raise ValueError("输入文本不能为空")

        # 验证长度限制
        if len(text) > self.max_length * 10:  # 粗略估计（一个token约4个字符）
            raise ValueError(
                f"输入文本过长（超过{self.max_length * 10}个字符），"
                f"请缩短文本或增加MAX_TEXT_LENGTH配置"
            )

        # 分词并编码
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            encoded = self.tokenizer(
                text,
                add_special_tokens=True,  # 添加[CLS]和[SEP]
                max_length=self.max_length,
                padding='max_length',      # 填充到max_length
                truncation=True,           # 截断超长序列
                return_tensors='pt',       # 返回PyTorch张量
                return_attention_mask=True,
                return_token_type_ids=True
            )

        # 将输入移动到指定设备
        return {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device),
            'token_type_ids': encoded['token_type_ids'].to(self.device)
        }

    def extract(self, data: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        从预处理的数据中提取BERT特征

        使用BERT模型处理预处理的输入，提取[CLS] token的隐藏状态
        作为句子级表示。所有计算在torch.no_grad()上下文中执行，
        不参与梯度计算。

        Args:
            data: 预处理后的输入字典，包含：
                - input_ids (torch.Tensor): token IDs
                - attention_mask (torch.Tensor): 注意力掩码
                - token_type_ids (torch.Tensor): token类型IDs

        Returns:
            np.ndarray: 768维特征向量，dtype为float32

        Raises:
            ValueError: 当输入数据缺少必需键时
            RuntimeError: 当模型推理失败时

        示例:
            >>> extractor = BERTFeatureExtractor()
            >>> processed = extractor.preprocess("Hello world")
            >>> features = extractor.extract(processed)
            >>> print(features.shape)
            (768,)
        """
        # 验证输入数据
        required_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"输入数据缺少必需的键: '{key}'")

        # 确保所有张量在正确设备上
        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)
        token_type_ids = data['token_type_ids'].to(self.device)

        # BERT推理（不计算梯度）
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

            # 提取[CLS] token的隐藏状态（第一个token）
            # outputs.last_hidden_state: (batch_size, seq_length, hidden_size)
            cls_hidden_state = outputs.last_hidden_state[0, 0, :]  # (hidden_size,)

            # 转换为numpy数组
            features = cls_hidden_state.cpu().numpy().astype(np.float32)

        except Exception as e:
            raise RuntimeError(f"BERT特征提取失败: {e}")

        return features

    def extract_batch(self, texts: list[str]) -> np.ndarray:
        """
        批量提取文本特征

        对多个文本进行批量特征提取，提高处理效率。

        Args:
            texts: 文本字符串列表

        Returns:
            np.ndarray: 特征矩阵，形状为(len(texts), 768)

        Raises:
            ValueError: 当texts为空或包含非字符串元素时

        示例:
            >>> extractor = BERTFeatureExtractor()
            >>> texts = ["Hello", "World", "Test"]
            >>> features = extractor.extract_batch(texts)
            >>> print(features.shape)
            (3, 768)
        """
        if not texts:
            raise ValueError("输入文本列表不能为空")

        if not all(isinstance(t, str) for t in texts):
            raise ValueError("所有输入必须是字符串")

        # 批量分词
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                encoded = self.tokenizer(
                    texts,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    return_attention_mask=True,
                    return_token_type_ids=True
                )

            # 移动到指定设备
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            token_type_ids = encoded['token_type_ids'].to(self.device)

            # 批量推理
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

            # 提取所有[CLS] token
            cls_hidden_states = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
            features = cls_hidden_states.cpu().numpy().astype(np.float32)

        except Exception as e:
            raise RuntimeError(f"批量特征提取失败: {e}")

        return features


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("BERT文本特征提取器测试")
    print("=" * 60)

    # 测试1: 初始化
    print("\n[测试1] 初始化特征提取器")
    print("-" * 40)
    try:
        extractor = BERTFeatureExtractor()
        print(f"✓ 初始化成功")
        print(f"  - 设备: {extractor.device}")
        print(f"  - 模型: {extractor.model_name}")
        print(f"  - 特征维度: {extractor.feature_dim}")
        print(f"  - 最大长度: {extractor.max_length}")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        exit(1)

    # 测试2: 单文本特征提取
    print("\n[测试2] 单文本特征提取")
    print("-" * 40)
    test_texts = [
        "Hello, world!",
        "This is a sample sentence for feature extraction.",
        "多模态情感分析系统使用BERT提取文本特征。",
        "Strong negative emotion expressed in anger.",
        ""
    ]

    for i, text in enumerate(test_texts, 1):
        if text == "":
            print(f"\n测试用例 {i}: 空文本（预期异常）")
            try:
                features = extractor.extract_from_raw(text)
                print(f"✗ 应该抛出异常但没有")
            except ValueError as e:
                print(f"✓ 正确捕获异常: {e}")
            continue

        print(f"\n测试用例 {i}: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")

        try:
            # 使用便捷方法
            features = extractor.extract_from_raw(text)

            # 验证维度
            print(f"  - 特征形状: {features.shape}")
            assert features.shape == (768,), f"特征维度应为(768,)，得到{features.shape}"
            print(f"  ✓ 维度正确: (768,)")

            # 验证数据类型
            assert features.dtype == np.float32, f"数据类型应为float32，得到{features.dtype}"
            print(f"  ✓ 数据类型正确: float32")

            # 验证无NaN/Inf
            assert not np.isnan(features).any(), "特征包含NaN值"
            assert not np.isinf(features).any(), "特征包含Inf值"
            print(f"  ✓ 无NaN/Inf值")

            # 输出统计信息
            print(f"  - 均值: {features.mean():.6f}")
            print(f"  - 标准差: {features.std():.6f}")
            print(f"  - 最小值: {features.min():.6f}")
            print(f"  - 最大值: {features.max():.6f}")

        except Exception as e:
            print(f"  ✗ 提取失败: {e}")

    # 测试3: 分步提取（preprocess + extract）
    print("\n[测试3] 分步提取（preprocess + extract）")
    print("-" * 40)
    test_text = "Testing preprocess and extract separately."

    try:
        processed = extractor.preprocess(test_text)
        print(f"✓ 预处理成功")
        print(f"  - input_ids形状: {processed['input_ids'].shape}")
        print(f"  - attention_mask形状: {processed['attention_mask'].shape}")
        print(f"  - token_type_ids形状: {processed['token_type_ids'].shape}")

        features = extractor.extract(processed)
        print(f"✓ 特征提取成功")
        print(f"  - 特征形状: {features.shape}")
        assert features.shape == (768,)
        print(f"  ✓ 维度正确")

    except Exception as e:
        print(f"✗ 分步提取失败: {e}")

    # 测试4: 批量提取
    print("\n[测试4] 批量特征提取")
    print("-" * 40)
    batch_texts = [
        "First sentence in the batch.",
        "Second sentence with different content.",
        "Third sentence for batch processing test."
    ]

    try:
        batch_features = extractor.extract_batch(batch_texts)
        print(f"✓ 批量提取成功")
        print(f"  - 批次大小: {len(batch_texts)}")
        print(f"  - 特征形状: {batch_features.shape}")
        assert batch_features.shape == (3, 768), f"批量特征形状应为(3, 768)，得到{batch_features.shape}"
        print(f"  ✓ 维度正确: (3, 768)")
        print(f"  - 数据类型: {batch_features.dtype}")
        assert not np.isnan(batch_features).any()
        assert not np.isinf(batch_features).any()
        print(f"  ✓ 无NaN/Inf值")

    except Exception as e:
        print(f"✗ 批量提取失败: {e}")

    # 测试5: 异常处理
    print("\n[测试5] 异常处理")
    print("-" * 40)

    # 非字符串输入
    print("测试非字符串输入:")
    try:
        extractor.preprocess(12345)
        print("  ✗ 应该抛出TypeError但没有")
    except TypeError as e:
        print(f"  ✓ 正确捕获TypeError: {e}")

    # 缺少键的输入
    print("\n测试缺少必需键的输入:")
    try:
        incomplete_data = {'input_ids': torch.randint(0, 1000, (1, 10))}
        extractor.extract(incomplete_data)
        print("  ✗ 应该抛出ValueError但没有")
    except ValueError as e:
        print(f"  ✓ 正确捕获ValueError: {e}")

    # 测试6: 重现性测试
    print("\n[测试6] 重现性测试")
    print("-" * 40)
    test_text = "Testing reproducibility of feature extraction."

    try:
        features1 = extractor.extract_from_raw(test_text)
        features2 = extractor.extract_from_raw(test_text)

        # 验证结果完全相同
        assert np.allclose(features1, features2, rtol=1e-6), "两次提取结果应相同"
        print(f"✓ 重复提取结果一致")
        print(f"  - 最大差异: {np.abs(features1 - features2).max():.10f}")

    except Exception as e:
        print(f"✗ 重现性测试失败: {e}")

    # 测试总结
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print("\n使用方法:")
    print("  from src.features.text_features import BERTFeatureExtractor")
    print("  extractor = BERTFeatureExtractor()")
    print("  features = extractor.extract_from_raw('Your text here')")
    print(f"  # features.shape = (768,)")
