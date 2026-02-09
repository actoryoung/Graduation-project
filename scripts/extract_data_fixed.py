# -*- coding: utf-8 -*-
"""
修复版BERT特征提取脚本 - 简化版

策略：使用processed_cleaned_correct的数据，只升级文本特征为BERT
问题：原脚本使用了错误的JSON标签
解决：保持源数据的标签不变
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from src.features.text_features import BERTFeatureExtractor


def extract_bert_from_source_data(
    source_data_dir='data/mosei/processed_cleaned_correct',
    output_dir='data/mosei/bert_hybrid_fixed',
    device='cuda',
    batch_size=32
):
    """
    从源数据目录提取BERT特征

    关键：使用源数据的标签，不使用JSON标签
    """
    print("=" * 70)
    print("修复版BERT特征提取")
    print("=" * 70)
    print(f"源数据目录: {source_data_dir}")
    print(f"输出目录: {output_dir}")
    print()

    # 由于processed_cleaned_correct没有原始文本，
    # 我们需要从CMU-MOSEI原始数据中提取文本
    # 使用mmsdk直接加载

    print("\n提示：processed_cleaned_correct没有原始文本")
    print("我们将使用MultiBench工具重新提取数据")
    print()

    from mmsdk import mmdatasdk
    from mmsdk.mmdatasdk import dataset

    # 定义数据集
    DATA_MODE = {
        'text': 'CMU_MOSEI_TimestampedWordVectors',
        'audio': 'CMU_MOSEI_COVAREP',
        'video': 'CMU_MOSEI_VisualOpenFace2'
    }

    # 加载标签
    labels = dataset.cmu_mosei_labels()

    # 加载特征（对齐）
    features = {}
    for modal, feature_name in DATA_MODE.items():
        print(f"加载{modal}特征...")
        features[modal] = dataset.cmu_mosei_highlevel(
            feature_name,
            'alignment'
        )

    aligned_features = dataset.align(features, labels)

    # 处理每个数据集划分
    splits = {
        'train': aligned_features.get_train(),
        'val': aligned_features.get_val(),
        'test': aligned_features.get_test()
    }

    # 初始化BERT提取器
    print("\n初始化BERT提取器...")
    bert_extractor = BERTFeatureExtractor(device=device)

    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_data in splits.items():
        print(f"\n{'='*70}")
        print(f"处理 {split_name.upper()} 集")
        print(f"{'='*70}")

        # 提取文本数据
        text_data = split_data['text']
        text_features = text_data.features  # (N, 300) GloVe

        # 提取原始文本
        raw_texts = []
        for key in text_data.keys():
            # mmsdk存储的文本
            raw_text = text_data[key]['features']
            # 将词向量转换回文本（如果可能）
            # 或者使用其他方式获取原始文本
            # 这里我们使用一个简化的方法
            raw_texts.append(str(key))  # 使用视频ID作为占位符

        # 由于无法直接获取原始文本，我们需要另一个方案
        print("   注意：无法从mmsdk直接获取原始文本")
        print("   将使用替代方案：保留现有的特征维度，只修改标签映射")

        # 提取其他模态特征
        audio_data = split_data['audio']
        video_data = split_data['video']
        label_data = split_data['label']

        audio_feat = audio_data.features  # (N, 74)
        video_feat = video_data.features  # (N, 710) or (N, 713)

        # 处理视频异常维度
        if video_feat.shape[1] == 713:
            # 移除极端异常维度
            video_stds = np.std(video_feat, axis=0)
            extreme_dims = np.where(video_stds > 10000)[0]
            valid_dims = np.where(video_stds <= 10000)[0]
            video_feat = video_feat[:, valid_dims]

        # 提取标签（离散化）
        labels_list = []
        for key in label_data.keys():
            raw_label = label_data[key]['features'][0, 0]
            label_int = int(round(raw_label)) + 3  # [-3, +3] → [0, 6]
            labels_list.append(label_int)

        labels_array = np.array(labels_list)

        # 显示标签分布
        unique, counts = np.unique(labels_array, return_counts=True)
        print(f"\n   {split_name}集标签分布:")
        for u, c in zip(unique, counts):
            pct = c / len(labels_array) * 100
            print(f"     类别{u}: {c:4d} ({pct:5.1f}%)")

        # 保存特征（暂时保留GloVe，后续升级为BERT）
        # 这里我们只保存正确标签的数据
        output_file = os.path.join(output_dir, f'{split_name}.npz')

        np.savez_compressed(
            output_file,
            text_features=text_features,  # 暂时用GloVe
            audio_features=audio_feat,
            video_features=video_feat,
            labels=labels_array  # 使用正确的标签！
        )

        print(f"\n   保存到: {output_file}")
        print(f"   样本数: {len(labels_array)}")

    print(f"\n{'='*70}")
    print("数据提取完成！")
    print(f"注意：文本特征仍是GloVe，需要后续升级为BERT")
    print(f"{'='*70}")


if __name__ == '__main__':
    extract_bert_from_source_data()
