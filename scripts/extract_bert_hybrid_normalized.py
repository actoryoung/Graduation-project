# -*- coding: utf-8 -*-
"""
从CMU-MOSEI JSON文件提取BERT文本特征（归一化版本）

与extract_bert_hybrid.py的区别：
- 对BERT特征进行Z-score归一化（基于训练集统计）
- 确保与GloVe特征在相同尺度上对比
"""
import sys
import os
import argparse
import numpy as np
import json
from tqdm import tqdm
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.features.bert_extractor import BERTTextExtractor
from config import DEVICE


def extract_bert_features_normalized(
    json_file: str,
    existing_npz: str,
    output_file: str,
    device='cuda',
    batch_size=32
):
    """
    从JSON文件提取BERT特征，进行Z-score归一化，并与现有音频/视频特征合并
    """
    print("=" * 70)
    print(f"处理文件: {os.path.basename(json_file)} (归一化版本)")
    print("=" * 70)

    # 1. 加载JSON数据
    print(f"\n1. 加载JSON文件...")
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    texts = [item['text'] for item in json_data]
    labels = [item['label'] for item in json_data]
    print(f"   加载了 {len(texts)} 个文本样本")

    # 2. 加载现有特征
    print(f"\n2. 加载现有特征...")
    existing_data = np.load(existing_npz, allow_pickle=True)

    audio_features = existing_data['audio_features'].astype(np.float32)
    video_features = existing_data['video_features'].astype(np.float32)
    old_labels = existing_data['labels'].astype(np.int64)

    print(f"   音频特征: {audio_features.shape}")
    print(f"   视频特征: {video_features.shape}")

    # 验证标签一致性
    if len(labels) != len(old_labels):
        print(f"   [警告] 标签数量不匹配: JSON({len(labels)}) vs NPZ({len(old_labels)})")
        min_len = min(len(labels), len(old_labels))
        texts = texts[:min_len]
        labels = labels[:min_len]
        audio_features = audio_features[:min_len]
        video_features = video_features[:min_len]
        print(f"   截断到 {min_len} 个样本")

    # 3. 提取BERT特征
    print(f"\n3. 提取BERT文本特征...")
    print(f"   设备: {device}")
    print(f"   批次大小: {batch_size}")

    bert_extractor = BERTTextExtractor(device=device)

    all_bert_features = []
    for i in tqdm(range(0, len(texts), batch_size), desc="   提取BERT"):
        batch_texts = texts[i:i+batch_size]
        with torch.no_grad():
            batch_features = bert_extractor.extract(batch_texts)
            batch_features_np = batch_features.cpu().numpy()
            all_bert_features.append(batch_features_np)

    bert_features = np.vstack(all_bert_features).astype(np.float32)

    print(f"\n   原始BERT特征: shape={bert_features.shape}")
    print(f"   范围: [{bert_features.min():.4f}, {bert_features.max():.4f}]")
    print(f"   均值: {bert_features.mean():.4f}")
    print(f"   标准差: {bert_features.std():.4f}")

    # 4. Z-score归一化（基于训练集统计）
    print(f"\n4. Z-score归一化...")

    # 计算训练集的均值和标准差（假设这是训练集）
    bert_mean = np.mean(bert_features, axis=0)
    bert_std = np.std(bert_features, axis=0)
    bert_std = np.where(bert_std < 1e-8, 1.0, bert_std)

    # 归一化并裁剪到±3σ
    bert_features_norm = np.clip((bert_features - bert_mean) / bert_std, -3, 3)

    print(f"   归一化后:")
    print(f"   范围: [{bert_features_norm.min():.4f}, {bert_features_norm.max():.4f}]")
    print(f"   均值: {bert_features_norm.mean():.4f}")
    print(f"   标准差: {bert_features_norm.std():.4f}")

    # 5. 验证维度匹配
    print(f"\n5. 验证特征维度...")
    print(f"   BERT文本特征: {bert_features_norm.shape}")
    print(f"   COVAREP音频特征: {audio_features.shape}")
    print(f"   OpenFace视频特征: {video_features.shape}")

    assert bert_features_norm.shape[0] == audio_features.shape[0] == video_features.shape[0]

    # 6. 保存
    print(f"\n6. 保存特征到 {output_file}...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    np.savez_compressed(
        output_file,
        text_features=bert_features_norm,  # 归一化后的BERT
        audio_features=audio_features,
        video_features=video_features,
        labels=np.array(labels, dtype=np.int64),
        bert_mean=bert_mean,  # 保存归一化参数供测试集使用
        bert_std=bert_std
    )

    print(f"\n   特征总维度: {bert_features_norm.shape[1] + audio_features.shape[1] + video_features.shape[1]}")
    print(f"   = BERT({bert_features_norm.shape[1]}) + COVAREP({audio_features.shape[1]}) + OpenFace({video_features.shape[1]})")

    print(f"\n[OK] 特征提取完成!")
    print(f"   保存位置: {output_file}")

    return bert_features_norm, audio_features, video_features, np.array(labels)


def main():
    parser = argparse.ArgumentParser(description='提取BERT特征（归一化版本）')
    parser.add_argument('--data-dir', type=str, default='data/mosei',
                       help='CMU-MOSEI数据目录')
    parser.add_argument('--json-dir', type=str, default='data/mosei',
                       help='JSON文件目录')
    parser.add_argument('--existing-dir', type=str, default='data/mosei/processed_cleaned_correct',
                       help='现有特征目录')
    parser.add_argument('--output-dir', type=str, default='data/mosei/bert_hybrid_normalized',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='BERT批处理大小')

    args = parser.parse_args()

    print("=" * 70)
    print("BERT混合特征提取（归一化版本）")
    print("=" * 70)

    print(f"\n配置:")
    print(f"  JSON目录: {args.json_dir}")
    print(f"  现有特征目录: {args.existing_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  设备: {args.device}")

    # 先处理训练集，获取归一化参数
    print("\n" + "="*70)
    print("处理 TRAIN 集（计算归一化参数）")
    print("="*70)

    train_json = os.path.join(args.json_dir, 'mosei_train.json')
    train_existing = os.path.join(args.existing_dir, 'train.npz')
    train_output = os.path.join(args.output_dir, 'train.npz')

    _, _, _, _, = extract_bert_features_normalized(
        train_json, train_existing, train_output, args.device, args.batch_size
    )

    # 读取归一化参数
    train_data = np.load(train_output)
    bert_mean = train_data['bert_mean']
    bert_std = train_data['bert_std']

    # 处理验证集和测试集（使用训练集的归一化参数）
    for split in ['val', 'test']:
        print(f"\n{'='*70}")
        print(f"处理 {split.upper()} 集（使用训练集归一化参数）")
        print(f"{'='*70}")

        json_file = os.path.join(args.json_dir, f'mosei_{split}.json')
        existing_npz = os.path.join(args.existing_dir, f'{split}.npz')
        output_file = os.path.join(args.output_dir, f'{split}.npz')

        if not os.path.exists(json_file) or not os.path.exists(existing_npz):
            print(f"[跳过] 文件不存在")
            continue

        # 重新实现，使用训练集的归一化参数
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        texts = [item['text'] for item in json_data]
        labels = [item['label'] for item in json_data]

        existing_data = np.load(existing_npz, allow_pickle=True)
        audio_features = existing_data['audio_features'].astype(np.float32)
        video_features = existing_data['video_features'].astype(np.float32)

        # 截断到一致长度
        min_len = min(len(texts), len(audio_features))
        texts = texts[:min_len]
        labels = labels[:min_len]
        audio_features = audio_features[:min_len]
        video_features = video_features[:min_len]

        # 提取BERT
        bert_extractor = BERTTextExtractor(device=args.device)
        all_bert_features = []
        for i in tqdm(range(0, len(texts), args.batch_size), desc="提取BERT"):
            batch_texts = texts[i:i+args.batch_size]
            with torch.no_grad():
                batch_features = bert_extractor.extract(batch_texts)
                all_bert_features.append(batch_features.cpu().numpy())

        bert_features = np.vstack(all_bert_features).astype(np.float32)

        # 使用训练集的归一化参数
        bert_features_norm = np.clip((bert_features - bert_mean) / bert_std, -3, 3)

        # 保存
        np.savez_compressed(
            output_file,
            text_features=bert_features_norm,
            audio_features=audio_features,
            video_features=video_features,
            labels=np.array(labels, dtype=np.int64)
        )
        print(f"[OK] 已保存到: {output_file}")

    print(f"\n{'='*70}")
    print("全部完成!")
    print(f"{'='*70}")
    print(f"\n特征已保存到: {args.output_dir}")
    print(f"\n与未归一化版本的区别:")
    print(f"  - BERT特征进行Z-score归一化")
    print(f"  - 裁剪到±3σ范围")
    print(f"  - 使用训练集的均值和标准差")


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    main()
