# -*- coding: utf-8 -*-
"""
从CMU-MOSEI JSON文件提取BERT文本特征

将GloVe(300维)升级为BERT(768维)，同时保留现有的COVAREP(74维)和OpenFace(710维)特征。
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


def extract_bert_features_from_json(
    json_file: str,
    existing_npz: str,
    output_file: str,
    device='cuda',
    batch_size=32
):
    """
    从JSON文件提取BERT特征，并与现有音频/视频特征合并

    Args:
        json_file: CMU-MOSEI JSON文件路径
        existing_npz: 现有的.npz文件路径（包含COVAREP和OpenFace特征）
        output_file: 输出.npz文件路径
        device: 计算设备
        batch_size: BERT批处理大小
    """
    print("=" * 70)
    print(f"处理文件: {os.path.basename(json_file)}")
    print("=" * 70)

    # 1. 加载JSON数据（获取文本）
    print(f"\n1. 加载JSON文件...")
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    texts = [item['text'] for item in json_data]
    labels = [item['label'] for item in json_data]

    print(f"   加载了 {len(texts)} 个文本样本")

    # 2. 加载现有特征（COVAREP和OpenFace）
    print(f"\n2. 加载现有特征...")
    existing_data = np.load(existing_npz, allow_pickle=True)

    audio_features = existing_data['audio_features'].astype(np.float32)  # [N, 74]
    video_features = existing_data['video_features'].astype(np.float32)  # [N, 710]
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

    # 初始化BERT编码器
    bert_extractor = BERTTextExtractor(device=device)

    # 批量提取BERT特征
    all_bert_features = []

    for i in tqdm(range(0, len(texts), batch_size), desc="   提取BERT"):
        batch_texts = texts[i:i+batch_size]

        with torch.no_grad():
            # 批量提取
            batch_features = bert_extractor.extract(batch_texts)

            # 转为numpy
            batch_features_np = batch_features.cpu().numpy()
            all_bert_features.append(batch_features_np)

    # 合并所有批次
    bert_features = np.vstack(all_bert_features).astype(np.float32)

    print(f"\n   BERT特征形状: {bert_features.shape}")
    print(f"   特征范围: [{bert_features.min():.4f}, {bert_features.max():.4f}]")
    print(f"   特征均值: {bert_features.mean():.4f}")
    print(f"   特征标准差: {bert_features.std():.4f}")

    # 4. 验证维度匹配
    print(f"\n4. 验证特征维度...")
    print(f"   BERT文本特征: {bert_features.shape}")
    print(f"   COVAREP音频特征: {audio_features.shape}")
    print(f"   OpenFace视频特征: {video_features.shape}")
    print(f"   标签: {np.array(labels).shape}")

    assert bert_features.shape[0] == audio_features.shape[0] == video_features.shape[0], "样本数量不匹配！"

    # 5. 保存合并后的特征
    print(f"\n5. 保存特征到 {output_file}...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    np.savez_compressed(
        output_file,
        text_features=bert_features,    # BERT特征 [N, 768]
        audio_features=audio_features,   # COVAREP特征 [N, 74]
        video_features=video_features,   # OpenFace特征 [N, 710]
        labels=np.array(labels, dtype=np.int64)
    )

    print(f"\n   特征总维度: {bert_features.shape[1] + audio_features.shape[1] + video_features.shape[1]}")
    print(f"   = BERT({bert_features.shape[1]}) + COVAREP({audio_features.shape[1]}) + OpenFace({video_features.shape[1]})")

    print(f"\n[OK] 特征提取完成!")
    print(f"   保存位置: {output_file}")

    return bert_features, audio_features, video_features, np.array(labels)


def main():
    parser = argparse.ArgumentParser(description='从JSON提取BERT特征并合并现有特征')
    parser.add_argument('--data-dir', type=str, default='data/mosei',
                       help='CMU-MOSEI数据目录')
    parser.add_argument('--json-dir', type=str, default='data/mosei',
                       help='JSON文件目录')
    parser.add_argument('--existing-dir', type=str, default='data/mosei/processed_cleaned_correct',
                       help='现有特征目录')
    parser.add_argument('--output-dir', type=str, default='data/mosei/bert_hybrid',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='BERT批处理大小')

    args = parser.parse_args()

    print("=" * 70)
    print("BERT混合特征提取")
    print("将GloVe(300维)升级为BERT(768维)，保留COVAREP和OpenFace")
    print("=" * 70)

    print(f"\n配置:")
    print(f"  JSON目录: {args.json_dir}")
    print(f"  现有特征目录: {args.existing_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  设备: {args.device}")
    print(f"  批次大小: {args.batch_size}")

    # 处理三个数据集划分
    splits = ['train', 'val', 'test']

    for split in splits:
        print(f"\n{'='*70}")
        print(f"处理 {split.upper()} 集")
        print(f"{'='*70}")

        json_file = os.path.join(args.json_dir, f'mosei_{split}.json')
        existing_npz = os.path.join(args.existing_dir, f'{split}.npz')
        output_file = os.path.join(args.output_dir, f'{split}.npz')

        # 检查文件是否存在
        if not os.path.exists(json_file):
            print(f"[跳过] JSON文件不存在: {json_file}")
            continue

        if not os.path.exists(existing_npz):
            print(f"[跳过] 现有特征不存在: {existing_npz}")
            continue

        try:
            extract_bert_features_from_json(
                json_file=json_file,
                existing_npz=existing_npz,
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
    print("全部完成!")
    print(f"{'='*70}")
    print(f"\n特征已保存到: {args.output_dir}")
    print(f"\n特征组成:")
    print(f"  文本: BERT (768维)")
    print(f"  音频: COVAREP (74维)")
    print(f"  视频: OpenFace (710维)")
    print(f"  总计: 1552维")
    print(f"\n下一步:")
    print(f"  python scripts/train_bert_hybrid.py --data-dir {args.output_dir}")


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    main()
