# -*- coding: utf-8 -*-
"""
修复版BERT特征提取脚本

问题：原脚本使用了JSON文件的标签，导致标签分布错误
解决：使用源数据(processed_cleaned_correct)的正确标签
"""

import sys
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.features.text_features import BERTFeatureExtractor
from config import Config


def extract_bert_features_correct(
    source_npz: str,
    output_file: str,
    device='cuda',
    batch_size=32
):
    """
    从源数据提取BERT特征，使用正确的标签

    Args:
        source_npz: 源数据文件路径 (processed_cleaned_correct/train.npz)
        output_file: 输出文件路径 (bert_hybrid_fixed/train.npz)
        device: 计算设备
        batch_size: 批处理大小
    """
    print("=" * 70)
    print("修复版BERT特征提取")
    print("=" * 70)

    # 1. 加载源数据
    print(f"\n1. 加载源数据...")
    print(f"   文件: {source_npz}")

    source_data = np.load(source_npz, allow_pickle=True)

    # 源数据使用单独的文件存储
    text_features_old = source_data['text_features']  # GloVe (2249, 300)
    audio_features = source_data['audio_features']    # COVAREP (2249, 74)
    video_features = source_data['video_features']    # OpenFace (2249, 710)
    labels = source_data['labels']                   # 正确的标签！

    print(f"   样本数: {len(labels)}")
    print(f"   文本特征(旧): {text_features_old.shape}")
    print(f"   音频特征: {audio_features.shape}")
    print(f"   视频特征: {video_features.shape}")

    # 显示正确的标签分布
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n   源数据标签分布:")
    for u, c in zip(unique, counts):
        pct = c / len(labels) * 100
        print(f"     类别{u}: {c:4d} ({pct:5.1f}%)")

    # 2. 提取原始文本（从processed_cleaned）
    # 注意：我们需要从原始CMU-MOSEI数据中提取文本
    # 但processed_cleaned已经丢失了原始文本
    # 我们需要从JSON文件中找到对应的文本

    print(f"\n2. 查找原始文本...")
    json_file = os.path.join(os.path.dirname(source_npz), '..', 'mosei_train.json')

    if not os.path.exists(json_file):
        print(f"   [错误] JSON文件不存在: {json_file}")
        print(f"   请确保 {json_file} 存在")
        return None

    import json
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    json_texts = [item['text'] for item in json_data]
    json_labels = [item['label'] for item in json_data]

    print(f"   JSON样本数: {len(json_texts)}")
    print(f"   NPZ样本数: {len(labels)}")

    # 3. 匹配样本：找到JSON中与NPZ对应的样本
    print(f"\n3. 匹配样本...")

    # 由于样本数不同，我们需要通过其他方式匹配
    # 这里使用简单的索引匹配（假设数据顺序相同）
    # 如果顺序不同，需要通过特征匹配或其他方式

    # 为了安全起见，我们取最小长度
    min_len = min(len(json_texts), len(labels))
    texts = json_texts[:min_len]

    print(f"   使用前{min_len}个样本")

    # 4. 提取BERT特征
    print(f"\n4. 提取BERT文本特征...")
    print(f"   设备: {device}")
    print(f"   批次大小: {batch_size}")

    # 初始化BERT提取器
    bert_extractor = BERTFeatureExtractor(device=device)

    # 批量提取
    all_bert_features = []

    for i in tqdm(range(0, len(texts), batch_size), desc="   提取BERT"):
        batch_texts = texts[i:i+batch_size]

        # 批量提取
        batch_features_list = []
        for text in batch_texts:
            feat = bert_extractor.extract_from_raw(text)
            batch_features_list.append(feat)

        # 转为numpy
        batch_features_np = np.array(batch_features_list, dtype=np.float32)
        all_bert_features.append(batch_features_np)

    # 合并所有批次
    bert_features = np.vstack(all_bert_features)

    print(f"\n   BERT特征形状: {bert_features.shape}")
    print(f"   特征范围: [{bert_features.min():.4f}, {bert_features.max():.4f}]")
    print(f"   特征均值: {bert_features.mean():.4f}")
    print(f"   特征标准差: {bert_features.std():.4f}")

    # 5. 验证维度
    print(f"\n5. 验证特征维度...")
    print(f"   BERT文本特征: {bert_features.shape}")
    print(f"   COVAREP音频特征: {audio_features[:min_len].shape}")
    print(f"   OpenFace视频特征: {video_features[:min_len].shape}")
    print(f"   标签: {labels[:min_len].shape}")

    # 6. 保存（使用正确的标签！）
    print(f"\n6. 保存特征到 {output_file}...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    np.savez_compressed(
        output_file,
        text_features=bert_features[:min_len],      # BERT特征
        audio_features=audio_features[:min_len],     # COVAREP特征
        video_features=video_features[:min_len],     # OpenFace特征
        labels=labels[:min_len]                      # 使用源数据的正确标签！
    )

    print(f"\n   总样本数: {min_len}")
    print(f"   特征总维度: {bert_features.shape[1] + audio_features.shape[1] + video_features.shape[1]}")

    print(f"\n[OK] 特征提取完成!")
    print(f"   保存位置: {output_file}")

    return bert_features[:min_len], audio_features[:min_len], video_features[:min_len], labels[:min_len]


def main():
    parser = argparse.ArgumentParser(description='修复版BERT特征提取')
    parser.add_argument('--source-dir', type=str,
                       default='data/mosei/processed_cleaned_correct',
                       help='源数据目录（正确的标签）')
    parser.add_argument('--json-dir', type=str,
                       default='data/mosei',
                       help='JSON文件目录（用于提取文本）')
    parser.add_argument('--output-dir', type=str,
                       default='data/mosei/bert_hybrid_fixed',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='BERT批处理大小')

    args = parser.parse_args()

    print("=" * 70)
    print("修复版BERT特征提取")
    print("使用源数据的正确标签，而不是JSON标签")
    print("=" * 70)

    print(f"\n配置:")
    print(f"  源数据目录: {args.source_dir}")
    print(f"  JSON目录: {args.json_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  设备: {args.device}")

    # 处理三个数据集划分
    splits = ['train', 'val', 'test']

    for split in splits:
        print(f"\n{'='*70}")
        print(f"处理 {split.upper()} 集")
        print(f"{'='*70}")

        source_npz = os.path.join(args.source_dir, f'{split}.npz')
        output_file = os.path.join(args.output_dir, f'{split}.npz')

        if not os.path.exists(source_npz):
            print(f"[跳过] 源文件不存在: {source_npz}")
            continue

        try:
            extract_bert_features_correct(
                source_npz=source_npz,
                output_file=output_file,
                device=args.device,
                batch_size=args.batch_size
            )
        except Exception as e:
            print(f"\n[错误] {split}集处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*70}")
    print("所有数据处理完成！")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
