# -*- coding: utf-8 -*-
"""
将BERT混合特征数据转换为3类分类

输入: data/mosei/bert_hybrid/*.npz (7类标签: [0,2,4,6])
输出: data/mosei/bert_3class/*.npz (3类标签: [0,1,2])

标签映射:
- 0 (strong_negative) → 0 (negative)
- 2 (negative) → 0 (negative)
- 4 (positive) → 2 (positive)
- 6 (strong_positive) → 2 (positive)

注意: 当前BERT hybrid数据缺少neutral样本，这会导致类别不平衡更严重
"""

import numpy as np
import os
from pathlib import Path


def convert_bert_labels_to_3class(labels_7class):
    """
    将7类标签转换为3类

    CMU-MOSEI 7类: [0, 1, 2, 3, 4, 5, 6]
    映射到3类: [0(neg), 1(neu), 2(pos)]

    Args:
        labels_7class: 7类标签数组

    Returns:
        3类标签数组
    """
    labels_3class = []

    for label in labels_7class:
        if label <= 2:  # [0,1,2] → negative
            labels_3class.append(0)
        elif label == 3:  # 3 → neutral
            labels_3class.append(1)
        else:  # [4,5,6] → positive
            labels_3class.append(2)

    return np.array(labels_3class, dtype=np.int64)


def main():
    # 路径配置
    input_dir = 'data/mosei/bert_hybrid'
    output_dir = 'data/mosei/bert_3class'

    print("=" * 70)
    print("BERT混合特征数据转换为3类分类")
    print("=" * 70)
    print(f"\n输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理每个数据集分割
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*70}")
        print(f"处理 {split.upper()} 数据集")
        print(f"{'='*70}")

        input_file = os.path.join(input_dir, f'{split}.npz')
        output_file = os.path.join(output_dir, f'{split}.npz')

        # 检查输入文件
        if not os.path.exists(input_file):
            print(f"[错误] 文件不存在: {input_file}")
            continue

        # 加载数据
        print(f"\n1. 加载数据: {input_file}")
        data = np.load(input_file, allow_pickle=True)

        text_features = data['text_features']  # [N, 768] BERT
        audio_features = data['audio_features']  # [N, 74] COVAREP
        video_features = data['video_features']  # [N, 710] OpenFace
        labels_7class = data['labels']  # [N] 7类标签

        print(f"   样本数: {len(labels_7class)}")
        print(f"   BERT特征: {text_features.shape}")
        print(f"   COVAREP特征: {audio_features.shape}")
        print(f"   OpenFace特征: {video_features.shape}")

        # 显示原始标签分布
        print(f"\n2. 原始标签分布 (7类):")
        unique, counts = np.unique(labels_7class, return_counts=True)
        for cls, count in zip(unique, counts):
            percentage = count / len(labels_7class) * 100
            print(f"   类别{cls}: {count:4d} ({percentage:5.1f}%)")

        # 转换为3类
        print(f"\n3. 转换为3类标签...")
        labels_3class = convert_bert_labels_to_3class(labels_7class)

        # 显示3类标签分布
        print(f"\n4. 新标签分布 (3类):")
        unique_3, counts_3 = np.unique(labels_3class, return_counts=True)
        total_3 = len(labels_3class)

        class_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

        for cls, count in zip(unique_3, counts_3):
            percentage = count / total_3 * 100
            print(f"   {class_names[cls]} ({cls}): {count:4d} ({percentage:5.1f}%)")

        # 检查类别不平衡问题
        if 1 not in unique_3:  # 没有Neutral类
            print(f"\n   [警告] 数据集中没有Neutral (1) 类样本!")
            print(f"   [警告] 这会导致严重的类别不平衡问题")

        if 0 not in unique_3:  # 没有Negative类
            print(f"\n   [警告] 数据集中没有Negative (0) 类样本!")

        # 保存数据
        print(f"\n5. 保存3类数据: {output_file}")
        np.savez_compressed(
            output_file,
            text_features=text_features.astype(np.float32),
            audio_features=audio_features.astype(np.float32),
            video_features=video_features.astype(np.float32),
            labels=labels_3class
        )

        # 显示保存的信息
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"   文件大小: {file_size:.2f} MB")
        print(f"   保存成功!")

    print(f"\n{'='*70}")
    print("转换完成!")
    print(f"{'='*70}")
    print(f"\n输出目录: {output_dir}")
    print(f"\n数据集内容:")
    for split in ['train', 'val', 'test']:
        npz_file = os.path.join(output_dir, f'{split}.npz')
        if os.path.exists(npz_file):
            data = np.load(npz_file)
            labels = data['labels']
            unique, counts = np.unique(labels, return_counts=True)
            print(f"  {split}: {len(labels)} 样本, 标签: {dict(zip(unique, counts))}")

    print(f"\n下一步:")
    print(f"  1. 修改BERTHybridModel为3类分类 (num_classes=3)")
    print(f"  2. 运行训练: python scripts/train_bert_3class.py")


if __name__ == '__main__':
    main()
