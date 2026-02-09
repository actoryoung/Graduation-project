# -*- coding: utf-8 -*-
"""
使用预训练编码器提取多模态特征

从原始CMU-MOSEI数据中提取预训练编码器特征：
- BERT: 768维文本特征
- wav2vec: 768维音频特征
- ResNet: 2048维视频特征

输出到新的.npz文件，用于训练预训练编码器模型。
"""

import sys
import os
import argparse
import numpy as np
from tqdm import tqdm

SDK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CMU-MultimodalSDK')
sys.path.insert(0, SDK_PATH)

import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import Config, DEVICE


# =============================================================================
# 预训练编码器导入
# =============================================================================

class PretrainedFeatureExtractor:
    """预训练特征提取器包装类"""

    def __init__(self, modality: str, device='cuda'):
        self.modality = modality
        self.device = device

        print(f"初始化{modality}编码器...")

        if modality == 'text':
            from src.features.bert_extractor import BERTTextExtractor
            self.extractor = BERTTextExtractor(device=device)
        elif modality == 'audio':
            from src.features.wav2vec_extractor import Wav2VecAudioExtractor
            self.extractor = Wav2VecAudioExtractor(device=device)
        elif modality == 'video':
            from src.features.resnet_extractor import ResNetVideoExtractor
            self.extractor = ResNetVideoExtractor(device=device, max_frames=16)
        else:
            raise ValueError(f"不支持的模态: {modality}")

    def extract(self, data):
        """提取特征"""
        return self.extractor.extract(data)


# =============================================================================
# CMU-MOSEI数据加载
# =============================================================================

def load_mosei_data(data_dir, split='train'):
    """
    加载CMU-MOSEI原始数据

    Returns:
        data: 包含text, audio, video, labels的字典
        metadata: 包含video_ids, intervals的字典
    """
    from mosei import MOSEIDataset

    # 加载数据集
    dataset = MOSEIDataset(data_dir, split=split)

    print(f"加载[{split}]数据: {len(dataset)} 个样本")

    return dataset


# =============================================================================
# 特征提取主函数
# =============================================================================

def extract_features_for_split(data_dir, split, output_dir, text_extractor,
                                audio_extractor, video_extractor, max_samples=None):
    """
    为指定数据集划分提取预训练特征

    Args:
        data_dir: CMU-MOSEI原始数据目录
        split: 数据集划分 ('train', 'val', 'test')
        output_dir: 输出目录
        text_extractor: BERT文本提取器
        audio_extractor: wav2vec音频提取器
        video_extractor: ResNet视频提取器
        max_samples: 最大样本数（用于测试）
    """
    print(f"\n{'='*70}")
    print(f"提取{split}集特征")
    print(f"{'='*70}")

    # 加载原始数据
    from mosei import MOSEIDataset
    dataset = MOSEIDataset(data_dir, split=split)

    if max_samples:
        # 限制样本数（用于快速测试）
        indices = list(range(min(max_samples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"限制样本数: {len(dataset)}")

    # 准备存储
    text_features_list = []
    audio_features_list = []
    video_features_list = []
    labels_list = []

    # 提取特征
    print(f"\n开始提取{split}集特征...")
    for idx in tqdm(range(len(dataset)), desc=f"Extracting {split}"):

        try:
            # 获取样本
            sample = dataset[idx]
            text = sample['text']  # 字符串
            audio = sample['audio']  # numpy数组 [samples]
            video = sample['video']  # numpy数组 [T, H, W, 3]
            label = sample['label']  # 标签

            # 提取文本特征
            with torch.no_grad():
                text_feat = text_extractor.extract(text)
                if text_feat.dim() == 1:
                    text_feat = text_feat.unsqueeze(0)
                text_feat_np = text_feat.cpu().numpy()

            # 提取音频特征
            try:
                # 将音频转换为tensor
                audio_tensor = torch.from_numpy(audio).float()
                with torch.no_grad():
                    audio_feat = audio_extractor.extract(audio_tensor)
                    if audio_feat.dim() == 1:
                        audio_feat = audio_feat.unsqueeze(0)
                    audio_feat_np = audio_feat.cpu().numpy()
            except Exception as e:
                print(f"\n[警告] 音频特征提取失败 (样本 {idx}): {e}")
                audio_feat_np = np.zeros((1, 768), dtype=np.float32)

            # 提取视频特征
            try:
                # 转换视频为tensor [T, 3, H, W]
                if video.ndim == 4:  # [T, H, W, 3]
                    video = video.transpose(0, 3, 1, 2)  # [T, 3, H, W]
                video_tensor = torch.from_numpy(video).float() / 255.0

                with torch.no_grad():
                    video_feat = video_extractor.extract(video_tensor)
                    if video_feat.dim() == 1:
                        video_feat = video_feat.unsqueeze(0)
                    video_feat_np = video_feat.cpu().numpy()
            except Exception as e:
                print(f"\n[警告] 视频特征提取失败 (样本 {idx}): {e}")
                video_feat_np = np.zeros((1, 2048), dtype=np.float32)

            # 存储特征
            text_features_list.append(text_feat_np)
            audio_features_list.append(audio_feat_np)
            video_features_list.append(video_feat_np)
            labels_list.append(label)

        except Exception as e:
            print(f"\n[错误] 样本 {idx} 处理失败: {e}")
            # 跳过该样本
            continue

    # 转换为numpy数组
    text_features = np.vstack(text_features_list).astype(np.float32)
    audio_features = np.vstack(audio_features_list).astype(np.float32)
    video_features = np.vstack(video_features_list).astype(np.float32)
    labels = np.array(labels_list, dtype=np.int64)

    print(f"\n特征提取完成:")
    print(f"  文本特征: {text_features.shape}")
    print(f"  音频特征: {audio_features.shape}")
    print(f"  视频特征: {video_features.shape}")
    print(f"  标签: {labels.shape}")

    # 保存到文件
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{split}_pretrained.npz')

    np.savez_compressed(
        output_file,
        text_features=text_features,
        audio_features=audio_features,
        video_features=video_features,
        labels=labels
    )

    print(f"特征已保存到: {output_file}")

    return text_features, audio_features, video_features, labels


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='提取预训练编码器特征')
    parser.add_argument('--data-dir', type=str,
                       default='data/mosei/CMU-MOSEI',
                       help='CMU-MOSEI原始数据目录')
    parser.add_argument('--output-dir', type=str,
                       default='data/mosei/pretrained_features',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda 或 cpu)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最大样本数（用于测试）')

    args = parser.parse_args()

    print("=" * 70)
    print("使用预训练编码器提取CMU-MOSEI特征")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  数据目录: {args.data_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  设备: {args.device}")
    print(f"  最大样本数: {args.max_samples or '全部'}")

    # 初始化预训练编码器
    print(f"\n初始化预训练编码器...")
    text_extractor = PretrainedFeatureExtractor('text', device=args.device)
    audio_extractor = PretrainedFeatureExtractor('audio', device=args.device)
    video_extractor = PretrainedFeatureExtractor('video', device=args.device)

    # 提取各数据集的特征
    splits = ['train', 'val', 'test']

    for split in splits:
        try:
            extract_features_for_split(
                data_dir=args.data_dir,
                split=split,
                output_dir=args.output_dir,
                text_extractor=text_extractor,
                audio_extractor=audio_extractor,
                video_extractor=video_extractor,
                max_samples=args.max_samples
            )
        except Exception as e:
            print(f"\n[错误] {split}集特征提取失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 70)
    print("特征提取完成!")
    print("=" * 70)
    print(f"\n特征文件保存在: {args.output_dir}")
    print("\n特征维度:")
    print("  文本: 768 (BERT)")
    print("  音频: 768 (wav2vec 2.0)")
    print("  视频: 2048 (ResNet-50)")
    print(f"  总计: 3584")


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    main()
