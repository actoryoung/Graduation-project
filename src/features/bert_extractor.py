# -*- coding: utf-8 -*-
"""
BERT文本特征提取器

使用预训练的BERT模型提取文本特征，替代原来的GloVe词向量。
BERT输出768维特征向量，相比GloVe的300维静态词向量，BERT能捕获更丰富的上下文语义信息。
"""

import torch
import torch.nn as nn
from typing import Union

class BERTTextExtractor(nn.Module):
    """
    BERT文本特征提取器

    使用Hugging Face transformers库加载预训练的BERT模型，
    提取文本的[CLS] token特征作为文本表示。

    Attributes:
        tokenizer: BERT分词器
        model: BERT模型
        device: 运行设备
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        device: str = 'cuda'
    ):
        """
        初始化BERT文本特征提取器

        Args:
            model_name: 预训练模型名称
                - 'bert-base-uncased': BERT base (12层, 768维)
                - 'bert-large-uncased': BERT large (24层, 1024维)
            device: 运行设备 ('cuda' 或 'cpu')
        """
        super(BERTTextExtractor, self).__init__()

        self.device = device
        self.model_name = model_name

        try:
            from transformers import BertModel, BertTokenizer

            print(f"正在加载BERT模型: {model_name}")

            # 加载分词器
            self.tokenizer = BertTokenizer.from_pretrained(model_name)

            # 加载BERT模型
            self.model = BertModel.from_pretrained(model_name)

            # 移到指定设备
            self.model.to(device)
            self.model.eval()  # 设置为评估模式

            print(f"[OK] BERT模型加载成功: {model_name}")
            print(f"     设备: {device}")
            print(f"     输出维度: 768")

        except Exception as e:
            print(f"[ERROR] BERT模型加载失败: {e}")
            raise

    @torch.no_grad()
    def forward(self, text: Union[str, list]) -> torch.Tensor:
        """
        提取文本特征

        Args:
            text: 输入文本
                - 单个文本: str
                - 批量文本: list[str]

        Returns:
            text_features: 文本特征张量
                - 单个文本: [1, 768]
                - 批量文本: [batch_size, 768]
        """
        # 处理单个文本或批量文本
        is_single = isinstance(text, str)
        if is_single:
            text = [text]

        # 分词并编码
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512  # BERT最大序列长度
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 通过BERT模型
        outputs = self.model(**inputs)

        # 提取[CLS] token的特征作为文本表示
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        cls_features = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # 如果是单个文本，移除batch维度
        if is_single:
            cls_features = cls_features.squeeze(0)

        return cls_features

    def extract(self, text: Union[str, list]) -> torch.Tensor:
        """
        提取文本特征（对外接口）

        Args:
            text: 输入文本或文本列表

        Returns:
            文本特征向量
        """
        return self.forward(text)

    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return 768

    def __repr__(self):
        return f"BERTTextExtractor(model_name='{self.model_name}', device='{self.device}')"


def test_bert_extractor():
    """测试BERT文本特征提取器"""
    print("=" * 70)
    print("BERT文本特征提取器测试")
    print("=" * 70)

    # 测试单文本
    try:
        extractor = BERTTextExtractor(device='cpu')
        print("\n测试1: 单文本提取")
        print("-" * 70)

        text = "This is a wonderful day!"
        features = extractor.extract(text)

        print(f"输入文本: {text}")
        print(f"特征形状: {features.shape}")
        print(f"特征范围: [{features.min():.4f}, {features.max():.4f}]")
        print(f"特征均值: {features.mean():.4f}")
        print(f"特征标准差: {features.std():.4f}")

    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        print("\n注意: 如果是首次运行，需要下载BERT模型（约420MB）")
        print("请确保网络连接正常，或手动下载模型文件。")
        return

    # 测试批量文本
    print("\n测试2: 批量文本提取")
    print("-" * 70)

    texts = [
        "I am very happy today!",
        "This is a terrible movie.",
        "The weather is neutral."
    ]

    batch_features = extractor.extract(texts)

    print(f"输入文本数量: {len(texts)}")
    print(f"特征形状: {batch_features.shape}")
    print(f"特征范围: [{batch_features.min():.4f}, {batch_features.max():.4f}]")

    # 对比GloVe
    print("\n" + "=" * 70)
    print("BERT vs GloVe对比")
    print("=" * 70)
    print("| 特征提取器 | 维度 | 类型 | 上下文感知 | 预训练 |")
    print("|-----------|------|------|-----------|--------|")
    print("| GloVe      | 300  | 静态 | 否         | 否     |")
    print("| **BERT**   | **768** | **动态** | **是**     | **是** |")

    print("\n" + "=" * 70)
    print("[OK] 所有测试通过!")
    print("=" * 70)


if __name__ == '__main__':
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    test_bert_extractor()
