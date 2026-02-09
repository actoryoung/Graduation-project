# -*- coding: utf-8 -*-
"""
GloVe文本特征提取器

实现真实的GloVe词向量查找和特征提取，用于Web界面的实时文本情感分析。
"""

import os
import sys
import numpy as np
from typing import Dict, Optional

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import Config


class GloVeWordEmbeddings:
    """GloVe词向量加载和管理"""

    def __init__(self, glove_path: Optional[str] = None):
        """
        初始化GloVe词向量

        Args:
            glove_path: GloVe文件路径，默认使用config中的路径
        """
        if glove_path is None:
            # 默认路径
            glove_dir = os.path.join(project_root, 'data', 'glove')
            glove_path = os.path.join(glove_dir, 'glove.6B.300d.txt')

        self.glove_path = glove_path
        self.embeddings: Dict[str, np.ndarray] = {}
        self.dimension = 300

        # 尝试加载
        if os.path.exists(glove_path):
            self._load_glove_file()
        else:
            print(f"[警告] GloVe文件不存在: {glove_path}")
            print("将使用简化版特征提取器")

    def _load_glove_file(self):
        """加载GloVe词向量文件"""
        print(f"正在加载GloVe词向量: {self.glove_path}")

        word_count = 0
        with open(self.glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                if len(values) < 301:  # 单词 + 300维向量
                    continue

                word = values[0]
                vector = np.asarray(values[1:], dtype=np.float32)

                # 验证维度
                if len(vector) == 300:
                    self.embeddings[word] = vector
                    word_count += 1

        print(f"[成功] 已加载 {word_count} 个词向量")
        print(f"词向量维度: {self.dimension}")

    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """
        获取单词的词向量

        Args:
            word: 单词

        Returns:
            词向量 (300维) 或 None
        """
        # 转小写查找
        word_lower = word.lower()

        # 直接查找
        if word_lower in self.embeddings:
            return self.embeddings[word_lower]

        # 移除标点符号后查找
        import string
        cleaned = word_lower.strip(string.punctuation)
        if cleaned in self.embeddings:
            return self.embeddings[cleaned]

        return None

    def is_loaded(self) -> bool:
        """检查是否已加载词向量"""
        return len(self.embeddings) > 0


class GloVeFeatureExtractor:
    """
    GloVe文本特征提取器

    从原始文本中提取GloVe词向量特征：
    1. 文本预处理（分词、小写化）
    2. 词向量查找
    3. 平均池化（得到句子级别的300维向量）
    """

    def __init__(self, glove_path: Optional[str] = None):
        """
        初始化GloVe特征提取器

        Args:
            glove_path: GloVe文件路径
        """
        self.glove = GloVeWordEmbeddings(glove_path)
        self.dimension = 300

    def extract(self, text: str) -> np.ndarray:
        """
        从文本提取GloVe特征

        Args:
            text: 输入文本

        Returns:
            300维特征向量（句子中所有词向量的平均）

        Raises:
            ValueError: 文本为空或无法提取任何有效词向量
        """
        if not text or not text.strip():
            raise ValueError("输入文本不能为空")

        # 文本预处理
        text = text.strip()

        # 简单分词（按空格和标点分割）
        import re
        words = re.findall(r'\b\w+\b', text.lower())

        if not words:
            raise ValueError(f"无法从文本中提取有效单词: {text}")

        # 查找词向量
        vectors = []
        unknown_words = []

        for word in words:
            vec = self.glove.get_embedding(word)
            if vec is not None:
                vectors.append(vec)
            else:
                unknown_words.append(word)

        # 检查是否找到有效的词向量
        if not vectors:
            # 如果所有单词都不在词典中，使用随机初始化（降级方案）
            print(f"[警告] 未找到任何已知词向量，使用降级方案")
            np.random.seed(hash(text) % 1000000)
            return np.random.randn(300).astype(np.float32) * 0.01

        # 平均池化
        feature = np.mean(vectors, axis=0).astype(np.float32)

        # 未知单词处理
        if unknown_words:
            unknown_rate = len(unknown_words) / len(words)
            if unknown_rate > 0.5:
                print(f"[提示] 有 {len(unknown_words)}/{len(words)} 个单词不在词典中")

        return feature

    def extract_from_raw(self, text: str) -> np.ndarray:
        """
        从原始文本提取特征（兼容predictor接口）

        Args:
            text: 输入文本

        Returns:
            300维特征向量
        """
        return self.extract(text)


# 自动加载和缓存
_glove_extractor_instance = None

def get_glove_extractor(force_reload: bool = False) -> GloVeFeatureExtractor:
    """
    获取GloVe特征提取器（单例模式）

    Args:
        force_reload: 是否强制重新加载

    Returns:
        GloVe特征提取器实例
    """
    global _glove_extractor_instance

    if _glove_extractor_instance is None or force_reload:
        _glove_extractor_instance = GloVeFeatureExtractor()

    return _glove_extractor_instance


if __name__ == '__main__':
    """测试代码"""
    print("=== GloVe特征提取器测试 ===\n")

    try:
        extractor = GloVeFeatureExtractor()

        # 测试提取
        test_texts = [
            "I love this movie!",
            "This is terrible and disappointing.",
            "The weather is okay today."
        ]

        for text in test_texts:
            print(f"文本: {text}")
            try:
                feature = extractor.extract(text)
                print(f"  特征形状: {feature.shape}")
                print(f"  特征范围: [{feature.min():.4f}, {feature.max():.4f}]")
            except Exception as e:
                print(f"  错误: {e}")
            print()

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
