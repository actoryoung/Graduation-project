# -*- coding: utf-8 -*-
"""
创建BERT 3类数据集

方案: 使用BERT hybrid的BERT特征 + processed_cleaned_3class的标签

数据来源:
- BERT特征: data/mosei/bert_hybrid/*.npz (768维)
- 音频/视频特征: data/mosei/processed_cleaned_3class/*.npz (保证样本对应)
- 3类标签: data/mosei/processed_cleaned_3class/*.npz

输出: data/mosei/bert_3class_full/*.npz
"""

import numpy as np
import os
from pathlib import Path


def main():
    # 路径配置
    bert_dir = 'data/mosei/bert_hybrid'
    glove_3class_dir = 'data/mosei/processed_cleaned_3class'
    output_dir = 'data/mosei/bert_3class_full'

    print("=" * 70)
    print("创建BERT 3类完整数据集")
    print("=" * 70)
    print(f"\nBERT特征来源: {bert_dir}")
    print(f"3类标签来源: {glove_3class_dir}")
    print(f"输出目录: {output_dir}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理每个数据集分割
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*70}")
        print(f"处理 {split.upper()} 数据集")
        print(f"{'='*70}")

        # 加载BERT数据
        print(f"\n1. 加载BERT特征...")
        bert_file = os.path.join(bert_dir, f'{split}.npz')
        if not os.path.exists(bert_file):
            print(f"[错误] BERT文件不存在: {bert_file}")
            continue

        bert_data = np.load(bert_file)
        text_features_bert = bert_data['text_features'].astype(np.float32)  # [N, 768]

        print(f"   BERT文本特征: {text_features_bert.shape}")

        # 加载GloVe 3类数据（获取音频/视频特征和标签）
        print(f"\n2. 加载3类数据标签和音频/视频特征...")
        glove_file = os.path.join(glove_3class_dir, f'{split}.npz')
        if not os.path.exists(glove_file):
            print(f"[错误] 3类数据文件不存在: {glove_file}")
            continue

        glove_data = np.load(glove_file)
        audio_features = glove_data['audio_features'].astype(np.float32)  # [N, 74]
        video_features = glove_data['video_features'].astype(np.float32)  # [N, 710]
        labels_3class = glove_data['labels'].astype(np.int64)  # [N]

        print(f"   音频特征: {audio_features.shape}")
        print(f"   视频特征: {video_features.shape}")
        print(f"   3类标签: {labels_3class.shape}")

        # 验证样本数匹配
        print(f"\n3. 验证样本数匹配...")
        n_bert = text_features_bert.shape[0]
        n_glove = audio_features.shape[0]

        if n_bert != n_glove:
            print(f"[错误] 样本数不匹配! BERT={n_bert}, GloVe={n_glove}")
            continue

        print(f"   [OK] 样本数匹配: {n_bert}")

        # 显示3类标签分布
        print(f"\n4. 3类标签分布:")
        unique, counts = np.unique(labels_3class, return_counts=True)
        total = len(labels_3class)
        class_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

        for cls, count in zip(unique, counts):
            percentage = count / total * 100
            print(f"   {class_names[cls]} ({cls}): {count:4d} ({percentage:5.1f}%)")

        # 保存合并后的数据
        print(f"\n5. 保存BERT 3类完整数据...")
        output_file = os.path.join(output_dir, f'{split}.npz')

        np.savez_compressed(
            output_file,
            text_features=text_features_bert,    # BERT特征 [N, 768]
            audio_features=audio_features,        # COVAREP特征 [N, 74]
            video_features=video_features,        # OpenFace特征 [N, 710]
            labels=labels_3class                 # 3类标签 [N]
        )

        # 显示保存信息
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"   文件大小: {file_size:.2f} MB")
        print(f"   保存成功: {output_file}")

    print(f"\n{'='*70}")
    print("BERT 3类完整数据集创建完成!")
    print(f"{'='*70}")
    print(f"\n输出目录: {output_dir}")
    print(f"\n数据集总结:")
    for split in ['train', 'val', 'test']:
        npz_file = os.path.join(output_dir, f'{split}.npz')
        if os.path.exists(npz_file):
            data = np.load(npz_file)
            labels = data['labels']
            unique, counts = np.unique(labels, return_counts=True)
            total = len(labels)
            print(f"  {split}: {total} 样本")
            for cls, count in zip(unique, counts):
                pct = count / total * 100
                print(f"    类别{cls}: {count} ({pct:.1f}%)")

    print(f"\n特征维度:")
    print(f"  文本 (BERT): 768")
    print(f"  音频 (COVAREP): 74")
    print(f"  视频 (OpenFace): 710")
    print(f"  总维度: 1652")

    print(f"\n下一步:")
    print(f"  1. 创建BERTHybridModel3Class (num_classes=3)")
    print(f"  2. 运行训练: python scripts/train_bert_3class.py")


if __name__ == '__main__':
    main()
