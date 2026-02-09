# -*- coding: utf-8 -*-
"""
修复训练数据标签的脚本

问题：bert_hybrid/train.npz使用了错误的JSON标签
解决：使用processed_cleaned_correct的正确标签重新保存数据
"""

import numpy as np
import os


def fix_labels():
    """修复训练数据的标签"""

    print("=" * 70)
    print("修复训练数据标签")
    print("=" * 70)

    # 源数据（正确标签）
    source_dir = 'data/mosei/processed_cleaned_correct'

    # 错误数据
    wrong_dir = 'data/mosei/bert_hybrid'

    # 输出目录
    output_dir = 'data/mosei/bert_hybrid_fixed'

    os.makedirs(output_dir, exist_ok=True)

    splits = ['train', 'val', 'test']

    for split in splits:
        print(f"\n{'='*50}")
        print(f"处理 {split.upper()} 集")
        print(f"{'='*50}")

        # 加载源数据（正确标签）
        source_file = os.path.join(source_dir, f'{split}.npz')
        source_data = np.load(source_file, allow_pickle=True)

        source_labels = source_data['labels']

        print(f"源数据样本数: {len(source_labels)}")

        # 显示正确的标签分布
        unique, counts = np.unique(source_labels, return_counts=True)
        print(f"\n正确的标签分布:")
        for u, c in zip(unique, counts):
            pct = c / len(source_labels) * 100
            print(f"  类别{u}: {c:4d} ({pct:5.1f}%)")

        # 加载错误的bert_hybrid数据
        wrong_file = os.path.join(wrong_dir, f'{split}.npz')

        if not os.path.exists(wrong_file):
            print(f"\n跳过: {wrong_file} 不存在")
            continue

        wrong_data = np.load(wrong_file, allow_pickle=True)

        # 使用源数据的正确标签
        # 保留bert_hybrid的BERT特征，但替换标签

        output_file = os.path.join(output_dir, f'{split}.npz')

        # 截断到源数据的长度
        min_len = min(len(source_labels), wrong_data['text_features'].shape[0])

        np.savez_compressed(
            output_file,
            text_features=wrong_data['text_features'][:min_len],  # BERT特征
            audio_features=wrong_data['audio_features'][:min_len],  # COVAREP特征
            video_features=wrong_data['video_features'][:min_len],  # OpenFace特征
            labels=source_labels[:min_len]  # 正确的标签！
        )

        print(f"\n保存到: {output_file}")
        print(f"样本数: {min_len}")
        print(f"使用正确的标签")

    print(f"\n{'='*70}")
    print("标签修复完成！")
    print(f"输出目录: {output_dir}")
    print(f"{'='*70}")

    # 验证输出的数据
    print("\n验证修复后的数据:")
    for split in splits:
        output_file = os.path.join(output_dir, f'{split}.npz')
        if os.path.exists(output_file):
            data = np.load(output_file, allow_pickle=True)
            labels = data['labels']

            print(f"\n{split}集:")
            print(f"  样本数: {len(labels)}")

            unique, counts = np.unique(labels, return_counts=True)
            for u, c in zip(unique, counts):
                pct = c / len(labels) * 100
                print(f"    类别{u}: {c:4d} ({pct:5.1f}%)")


if __name__ == '__main__':
    fix_labels()
