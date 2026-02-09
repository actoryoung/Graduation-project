# -*- coding: utf-8 -*-
"""
CMU-MOSEI 数据清洗和分析脚本

认真分析数据特性，进行合理的特征处理
"""
import sys
import os

SDK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CMU-MultimodalSDK')
sys.path.insert(0, SDK_PATH)

import numpy as np
from mmsdk import mmdatasdk
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from scipy import stats

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import Config, TEXT_DIM, AUDIO_DIM, VIDEO_DIM, NUM_CLASSES


def analyze_raw_features():
    """分析原始.csd数据的特性"""
    print("=" * 70)
    print("CMU-MOSEI 原始数据分析")
    print("=" * 70)

    data_dir = os.path.join(project_root, 'data', 'mosei')

    # 加载原始数据
    print("\n加载原始.csd数据...")
    highlevel_files = {
        'glove_vectors': os.path.join(data_dir, 'cmumosei_highlevel', 'CMU_MOSEI_TimestampedWordVectors.csd'),
        'COVAREP': os.path.join(data_dir, 'cmumosei_highlevel', 'CMU_MOSEI_COVAREP.csd'),
        'OpenFace_2': os.path.join(data_dir, 'cmumosei_highlevel', 'CMU_MOSEI_VisualOpenFace2.csd'),
    }
    labels_files = {
        'All Labels': os.path.join(data_dir, 'cmumosei_labels', 'CMU_MOSEI_Labels.csd')
    }

    highlevel_data = mmdatasdk.mmdataset(highlevel_files)
    labels_data = mmdatasdk.mmdataset(labels_files)

    glove_dict = highlevel_data.computational_sequences['glove_vectors'].data
    audio_dict = highlevel_data.computational_sequences['COVAREP'].data
    video_dict = highlevel_data.computational_sequences['OpenFace_2'].data
    labels_dict = labels_data.computational_sequences['All Labels'].data

    # 收集训练集特征
    from mmsdk.mmdatasdk import cmu_mosei
    train_fold = set(cmu_mosei.standard_folds.standard_train_fold)

    # 收集所有特征的原始值（不进行平均池化）
    print("\n收集训练集原始特征...")
    glove_raw = []
    audio_raw = []
    video_raw = []

    glove_keys_list = list(glove_dict.keys())
    first_key = glove_keys_list[0]

    if '[' in first_key:
        keys_to_process = [k for k in glove_dict.keys() if k.split('[')[0] in train_fold]
    else:
        keys_to_process = [k for k in glove_dict.keys() if k in train_fold]

    for key in tqdm(keys_to_process, desc="收集特征"):
        try:
            if key in glove_dict and key in audio_dict and key in video_dict:
                # 直接收集原始序列数据（不平池）
                glove_raw.append(glove_dict[key]['features'])  # (seq_len, 300)
                audio_raw.append(audio_dict[key]['features'])  # (seq_len, 74)
                video_raw.append(video_dict[key]['features'])  # (seq_len, 713)
        except:
            pass

    # 合并所有数据
    print("\n合并数据...")
    glove_all = np.vstack(glove_raw)  # (total_frames, 300)
    audio_all = np.vstack(audio_raw)  # (total_frames, 74)
    video_all = np.vstack(video_raw)  # (total_frames, 713)

    print(f"GloVe总帧数: {glove_all.shape}")
    print(f"COVAREP总帧数: {audio_all.shape}")
    print(f"OpenFace总帧数: {video_all.shape}")

    # 分析每个维度
    def analyze_dimension(features, name, save_prefix):
        print(f"\n{'='*70}")
        print(f"{name} 数据分析")
        print(f"{'='*70}")

        # 基本统计
        print(f"\n基本统计:")
        print(f"  形状: {features.shape}")
        print(f"  均值范围: [{np.mean(features, axis=0).min():.4f}, {np.mean(features, axis=0).max():.4f}]")
        print(f"  标准差范围: [{np.std(features, axis=0).min():.4f}, {np.std(features, axis=0).max():.4f}]")
        print(f"  最小值范围: [{np.min(features, axis=0).min():.4f}, {np.min(features, axis=0).max():.4f}]")
        print(f"  最大值范围: [{np.max(features, axis=0).min():.4f}, {np.max(features, axis=0).max():.4f}]")

        # 异常值统计
        has_nan = np.isnan(features).any(axis=0)
        has_inf = np.isinf(features).any(axis=0)
        has_zero_var = np.std(features, axis=0) < 1e-8

        print(f"\n维度质量检查:")
        print(f"  包含NaN的维度数: {has_nan.sum()} / {len(has_nan)}")
        print(f"  包含Inf的维度数: {has_inf.sum()} / {len(has_inf)}")
        print(f"  方差≈0的维度数: {has_zero_var.sum()} / {len(has_zero_var)}")

        # 检查极端值（超过5个标准差的值）
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)
        stds = np.where(stds < 1e-8, 1.0, stds)  # 避免除零

        # Z-score
        z_scores = (features - means) / stds
        extreme_positive = (z_scores > 5).sum(axis=0)
        extreme_negative = (z_scores < -5).sum(axis=0)

        print(f"\n极端值统计（|z|>5）:")
        print(f"  正极端值最多的维度: {extreme_positive.max()} 个")
        print(f"  负极端值最多的维度: {extreme_negative.max()} 个")
        print(f"  有>10%正极端值的维度: {(extreme_positive > features.shape[0]*0.1).sum()}")
        print(f"  有>10%负极端值的维度: {(extreme_negative > features.shape[0]*0.1).sum()}")

        # 绘制分布直方图
        save_dir = os.path.join(project_root, 'docs', 'data_analysis')
        os.makedirs(save_dir, exist_ok=True)

        # 随机选择几个维度可视化
        np.random.seed(42)
        sample_dims = np.random.choice(features.shape[1], min(9, features.shape[1]), replace=False)

        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, dim in enumerate(sample_dims):
            data = features[:, dim]
            # 过滤极端值以便可视化
            data_clipped = data[np.abs(data) < np.percentile(np.abs(data), 99)]
            axes[i].hist(data_clipped, bins=50, alpha=0.7)
            axes[i].set_title(f'维度{dim} (std={stds[dim]:.2f})')
            axes[i].set_xlabel('值')
            axes[i].set_ylabel('频数')

        plt.suptitle(f'{name} - 随机维度分布', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{save_prefix}_distribution.png'), dpi=150, bbox_inches='tight')
        print(f"\n分布图已保存: docs/data_analysis/{save_prefix}_distribution.png")

        return {
            'means': means,
            'stds': stds,
            'has_nan': has_nan,
            'has_inf': has_inf,
            'zero_var_dims': has_zero_var,
            'extreme_positive': extreme_positive,
            'extreme_negative': extreme_negative
        }

    # 分析三种模态
    glove_stats = analyze_dimension(glove_all, "GloVe (300维)", "glove")
    audio_stats = analyze_dimension(audio_all, "COVAREP (74维)", "covarep")
    video_stats = analyze_dimension(video_all, "OpenFace (713维)", "openface")

    # 保存统计信息
    stats_dir = os.path.join(project_root, 'docs', 'data_analysis')
    np.savez(os.path.join(stats_dir, 'raw_data_statistics.npz'),
             glove_means=glove_stats['means'],
             glove_stds=glove_stats['stds'],
             audio_means=audio_stats['means'],
             audio_stds=audio_stats['stds'],
             video_means=video_stats['means'],
             video_stds=video_stats['stds']
    )

    print("\n" + "=" * 70)
    print("数据分析完成！统计信息已保存")
    print("=" * 70)

    return glove_stats, audio_stats, video_stats


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    try:
        analyze_raw_features()
    except Exception as e:
        print(f"\n[ERROR] 分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
