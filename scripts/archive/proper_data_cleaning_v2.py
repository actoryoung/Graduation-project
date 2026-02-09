# -*- coding: utf-8 -*-
"""
CMU-MOSEI 正确的数据清洗脚本 v2

基于深度分析结果的针对性处理：
1. 识别并处理OpenFace的3个极端异常维度
2. 对COVAREP的-inf做精确处理
3. 使用更合理的归一化策略
"""
import sys
import os

SDK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CMU-MultimodalSDK')
sys.path.insert(0, SDK_PATH)

import numpy as np
from mmsdk import mmdatasdk
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import Config, TEXT_DIM, AUDIO_DIM, VIDEO_DIM, NUM_CLASSES


def detect_and_remove_extreme_dimensions(video_features, std_threshold=10000):
    """
    检测并移除极端异常的维度

    Args:
        video_features: (n_samples, 713) 视频特征
        std_threshold: 标准差阈值，超过此值认为极端

    Returns:
        cleaned_features: 清洗后的特征
        extreme_dims: 被移除的极端维度索引
        valid_dims: 保留的正常维度索引
    """
    stds = np.std(video_features, axis=0)

    # 找出极端维度
    extreme_dims = np.where(stds > std_threshold)[0]
    valid_dims = np.where(stds <= std_threshold)[0]

    print(f"\n检测到 {len(extreme_dims)} 个极端异常维度 (std > {std_threshold}):")
    for dim in extreme_dims:
        print(f"  维度{dim}: std={stds[dim]:.0f}, "
              f"range=[{np.min(video_features[:, dim]):.0f}, {np.max(video_features[:, dim]):.0f}]")

    print(f"\n保留 {len(valid_dims)} 个正常维度")

    return video_features[:, valid_dims], extreme_dims, valid_dims


def normalize_with_clipping(features, clip_range=3.0):
    """
    标准Z-score归一化 + 软裁剪

    Args:
        features: (n_samples, n_dims)
        clip_range: 裁剪范围（标准差倍数）

    Returns:
        normalized_features
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    normalized = (features - mean) / std

    # 软裁剪到±clip_range个标准差
    normalized = np.clip(normalized, -clip_range, clip_range)

    return normalized


def process_audio_features(audio_feat):
    """
    处理COVAREP特征的-inf值

    只有维度7有-inf，用0填充即可
    """
    # 只处理-inf，保留其他值
    audio_clean = audio_feat.copy()
    audio_clean[np.isneginf(audio_clean)] = 0.0

    return audio_clean


def clean_and_normalize_data_v2():
    """改进的数据清洗v2"""
    print("=" * 70)
    print("CMU-MOSEI 数据清洗 v2 (基于深度分析)")
    print("=" * 70)

    data_dir = os.path.join(project_root, 'data', 'mosei')
    processed_dir = os.path.join(data_dir, 'processed_cleaned_v2')
    os.makedirs(processed_dir, exist_ok=True)

    # 定义.csd文件路径
    highlevel_files = {
        'glove_vectors': os.path.join(data_dir, 'cmumosei_highlevel', 'CMU_MOSEI_TimestampedWordVectors.csd'),
        'COVAREP': os.path.join(data_dir, 'cmumosei_highlevel', 'CMU_MOSEI_COVAREP.csd'),
        'OpenFace_2': os.path.join(data_dir, 'cmumosei_highlevel', 'CMU_MOSEI_VisualOpenFace2.csd'),
    }
    labels_files = {
        'All Labels': os.path.join(data_dir, 'cmumosei_labels', 'CMU_MOSEI_Labels.csd')
    }

    # 加载数据
    print("\n加载数据文件...")
    highlevel_data = mmdatasdk.mmdataset(highlevel_files)
    labels_data = mmdatasdk.mmdataset(labels_files)

    glove_dict = highlevel_data.computational_sequences['glove_vectors'].data
    audio_dict = highlevel_data.computational_sequences['COVAREP'].data
    video_dict = highlevel_data.computational_sequences['OpenFace_2'].data
    labels_dict = labels_data.computational_sequences['All Labels'].data

    # 获取fold信息
    from mmsdk.mmdatasdk import cmu_mosei
    train_fold = set(cmu_mosei.standard_folds.standard_train_fold)
    val_fold = set(cmu_mosei.standard_folds.standard_valid_fold)
    test_fold = set(cmu_mosei.standard_folds.standard_test_fold)

    # ============ 阶段1: 收集训练集特征 ============
    print("\n" + "="*70)
    print("阶段1: 收集训练集特征")
    print("="*70)

    glove_keys_list = list(glove_dict.keys())
    first_key = glove_keys_list[0]

    if '[' in first_key:
        keys_to_process = [k for k in glove_dict.keys() if k.split('[')[0] in train_fold]
    else:
        keys_to_process = [k for k in glove_dict.keys() if k in train_fold]

    train_text_features = []
    train_audio_features = []
    train_video_features = []

    for key in tqdm(keys_to_process, desc="收集训练集"):
        try:
            if key in glove_dict and key in audio_dict and key in video_dict and key in labels_dict:
                text_feat = np.mean(glove_dict[key]['features'], axis=0).astype(np.float32)
                audio_feat = np.mean(audio_dict[key]['features'], axis=0).astype(np.float32)
                video_feat = np.mean(video_dict[key]['features'], axis=0).astype(np.float32)

                # 处理音频的-inf
                audio_feat = process_audio_features(audio_feat)

                train_text_features.append(text_feat)
                train_audio_features.append(audio_feat)
                train_video_features.append(video_feat)
        except Exception as e:
            continue

    train_text_features = np.array(train_text_features)
    train_audio_features = np.array(train_audio_features)
    train_video_features = np.array(train_video_features)

    print(f"\n收集到 {len(train_text_features)} 个训练样本")
    print(f"  文本: {train_text_features.shape}")
    print(f"  音频: {train_audio_features.shape}")
    print(f"  视频(原始): {train_video_features.shape}")

    # ============ 阶段2: 检测OpenFace极端维度 ============
    print("\n" + "="*70)
    print("阶段2: 检测并移除OpenFace极端异常维度")
    print("="*70)

    train_video_clean, extreme_dims, valid_dims = detect_and_remove_extreme_dimensions(
        train_video_features, std_threshold=10000
    )

    print(f"\n移除的维度: {extreme_dims}")
    print(f"保留的维度数: {len(valid_dims)} / 713")

    NEW_VIDEO_DIM = len(valid_dims)

    # ============ 阶段3: 计算归一化参数 ============
    print("\n" + "="*70)
    print("阶段3: 计算归一化参数")
    print("="*70)

    # GloVe: 标准Z-score，裁剪±3
    print("\nGloVe: Z-score归一化，裁剪±3σ")
    text_mean = np.mean(train_text_features, axis=0)
    text_std = np.std(train_text_features, axis=0)
    text_std = np.where(text_std < 1e-8, 1.0, text_std)

    # Audio: 标准Z-score，裁剪±3
    print("\nCOVAREP: Z-score归一化，裁剪±3σ")
    audio_mean = np.mean(train_audio_features, axis=0)
    audio_std = np.std(train_audio_features, axis=0)
    audio_std = np.where(audio_std < 1e-8, 1.0, audio_std)

    # Video (已移除极端维度): 标准Z-score，裁剪±3
    print(f"\nOpenFace (已移除{len(extreme_dims)}个极端维度): Z-score归一化，裁剪±3σ")
    video_mean = np.mean(train_video_clean, axis=0)
    video_std = np.std(train_video_clean, axis=0)
    video_std = np.where(video_std < 1e-8, 1.0, video_std)

    # ============ 阶段4: 处理各数据集 ============
    print("\n" + "="*70)
    print("阶段4: 处理并保存清洗后的数据")
    print("="*70)

    splits = [
        ('train', train_fold, os.path.join(processed_dir, 'train.npz'), keys_to_process),
        ('val', val_fold, os.path.join(processed_dir, 'val.npz'), None),
        ('test', test_fold, os.path.join(processed_dir, 'test.npz'), None)
    ]

    for split_name, fold_ids, output_path, train_keys in splits:
        print(f"\n处理 {split_name.upper()} 集...")

        # 收集该fold的keys
        if train_keys is None:
            glove_keys_list = list(glove_dict.keys())
            first_key = glove_keys_list[0]
            if '[' in first_key:
                keys_to_process = [k for k in glove_dict.keys() if k.split('[')[0] in fold_ids]
            else:
                keys_to_process = [k for k in glove_dict.keys() if k in fold_ids]

        text_features_list = []
        audio_features_list = []
        video_features_list = []
        labels_list = []
        sample_ids_list = []

        for key in tqdm(keys_to_process, desc=f"处理{split_name}"):
            try:
                if key not in glove_dict or key not in audio_dict or key not in video_dict or key not in labels_dict:
                    continue

                # 获取特征
                text_feat = np.mean(glove_dict[key]['features'], axis=0).astype(np.float32)
                audio_feat_raw = np.mean(audio_dict[key]['features'], axis=0).astype(np.float32)
                video_feat_raw = np.mean(video_dict[key]['features'], axis=0).astype(np.float32)

                # 处理音频
                audio_feat = process_audio_features(audio_feat_raw)

                # 移除视频极端维度
                video_feat = video_feat_raw[valid_dims]

                # 归一化 (统一的±3σ裁剪)
                text_feat_norm = np.clip((text_feat - text_mean) / text_std, -3, 3)
                audio_feat_norm = np.clip((audio_feat - audio_mean) / audio_std, -3, 3)
                video_feat_norm = np.clip((video_feat - video_mean) / video_std, -3, 3)

                # 检查有效性
                if np.isnan(text_feat_norm).any() or np.isinf(text_feat_norm).any():
                    continue
                if np.isnan(audio_feat_norm).any() or np.isinf(audio_feat_norm).any():
                    continue
                if np.isnan(video_feat_norm).any() or np.isinf(video_feat_norm).any():
                    continue

                # 标签 - 使用CMU-MOSEI论文的标准四舍五入离散化方法
                # 这与论文Figure 2的连续分数分布一致
                raw_label = label[0, 0]

                # 标准方法：四舍五入到最近的整数，然后偏移到[0,6]
                label_int = int(round(raw_label)) + 3

                # 边界检查（确保在[0,6]范围内）
                if label_int < 0:
                    label_int = 0
                elif label_int > 6:
                    label_int = 6

                text_features_list.append(text_feat_norm)
                audio_features_list.append(audio_feat_norm)
                video_features_list.append(video_feat_norm)
                labels_list.append(label_int)
                sample_ids_list.append(key)

            except Exception as e:
                continue

        # 保存
        if len(text_features_list) > 0:
            text_arr = np.array(text_features_list)
            audio_arr = np.array(audio_features_list)
            video_arr = np.array(video_features_list)
            labels_arr = np.array(labels_list)
            sample_ids_arr = np.array(sample_ids_list)

            print(f"\n{split_name.upper()} 统计:")
            print(f"  - 样本数: {len(labels_arr)}")
            print(f"  - 文本特征: shape={text_arr.shape}, range=[{text_arr.min():.4f}, {text_arr.max():.4f}]")
            print(f"  - 音频特征: shape={audio_arr.shape}, range=[{audio_arr.min():.4f}, {audio_arr.max():.4f}]")
            print(f"  - 视频特征: shape={video_arr.shape}, range=[{video_arr.min():.4f}, {video_arr.max():.4f}]")

            # 标签分布
            unique, counts = np.unique(labels_arr, return_counts=True)
            label_names = ['强负', '负', '弱负', '中性', '弱正', '正', '强正']
            print(f"  - 标签分布:")
            for idx, count in zip(unique, counts):
                print(f"      {label_names[idx]}: {count} ({count/len(labels_arr)*100:.1f}%)")

            np.savez_compressed(
                output_path,
                text_features=text_arr,
                audio_features=audio_arr,
                video_features=video_arr,
                labels=labels_arr,
                sample_ids=sample_ids_arr,
                new_video_dim=NEW_VIDEO_DIM,
                removed_video_dims=extreme_dims
            )
            print(f"[OK] 已保存到: {output_path}")
        else:
            print(f"[WARN] {split_name.upper()} 没有数据!")

    print("\n" + "=" * 70)
    print("[完成] 数据清洗v2完成!")
    print("=" * 70)
    print(f"\n数据保存在: {processed_dir}")
    print(f"\n关键改进:")
    print(f"  - 移除了{len(extreme_dims)}个OpenFace极端异常维度 (std > 10000)")
    print(f"  - 视频维度: {VIDEO_DIM} → {NEW_VIDEO_DIM}")
    print(f"  - 统一的归一化策略: 所有模态Z-score + 裁剪±3σ")
    print(f"  - 音频-inf精确处理: 只处理维度7")

    print(f"\n下一步: 修改config.py中的VIDEO_DIM={NEW_VIDEO_DIM}")
    print(f"      然后运行: python scripts/train_model.py --data-dir {processed_dir}")

    return NEW_VIDEO_DIM


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    try:
        new_video_dim = clean_and_normalize_data_v2()
        print(f"\n请手动修改 config.py 中的 VIDEO_DIM = {new_video_dim}")
    except Exception as e:
        print(f"\n[ERROR] 清洗失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
