# -*- coding: utf-8 -*-
"""
CMU-MOSEI 正确的数据清洗脚本

基于数据分析结果，采用合理的特征处理策略
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


def get_median_and_mad(data):
    """计算中位数和MAD（中位数绝对偏差）"""
    median = np.median(data, axis=0)
    mad = np.median(np.abs(data - median), axis=0)
    return median, mad


def robust_normalize(features, median, mad, eps=1e-8):
    """鲁棒归一化（基于中位数和MAD）"""
    # MAD转换为标准差（对于正态分布：std ≈ 1.4826 * MAD）
    std_equiv = mad * 1.4826
    std_equiv = np.where(std_equiv < eps, eps, std_equiv)

    normalized = (features - median) / std_equiv

    # 只处理真正的极端值（±5个MAD之外）
    # 使用软裁剪而非硬裁剪
    extreme_threshold = 5
    outlier_mask = np.abs(normalized) > extreme_threshold

    # 对极端值进行软裁剪（保留符号但限制幅度）
    normalized[outlier_mask] = np.sign(normalized[outlier_mask]) * extreme_threshold

    return normalized


def process_audio_features(audio_feat):
    """处理音频特征（COVAREP 74维）"""
    # 处理-inf和inf
    audio_clean = np.nan_to_num(audio_feat, nan=0.0, posinf=0.0, neginf=0.0)

    # 检查是否还有inf
    if np.isinf(audio_clean).any() or np.isnan(audio_clean).any():
        audio_clean = np.nan_to_num(audio_clean, nan=0.0, posinf=0.0, neginf=0.0)

    return audio_clean


def filter_features_by_variance(features, min_variance=1e-3):
    """过滤掉方差过小的维度（噪声维度）"""
    variances = np.var(features, axis=0)
    valid_dims = variances > min_variance

    print(f"  过滤前维度数: {features.shape[1]}")
    print(f"  过滤后维度数: {valid_dims.sum()}")
    print(f"  过滤掉{(~valid_dims).sum()}个低方差维度")

    return features[:, valid_dims], valid_dims


def clean_and_normalize_data():
    """数据清洗和归一化"""
    print("=" * 70)
    print("CMU-MOSEI 数据清洗和归一化")
    print("=" * 70)

    data_dir = os.path.join(project_root, 'data', 'mosei')
    processed_dir = os.path.join(data_dir, 'processed_cleaned')
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

    # ============ 阶段1: 收集训练集特征用于计算归一化参数 ============
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
                # 平均池化
                text_feat = np.mean(glove_dict[key]['features'], axis=0).astype(np.float32)
                audio_feat = np.mean(audio_dict[key]['features'], axis=0).astype(np.float32)
                video_feat = np.mean(video_dict[key]['features'], axis=0).astype(np.float32)

                # 处理音频异常值
                audio_feat = process_audio_features(audio_feat)

                train_text_features.append(text_feat)
                train_audio_features.append(audio_feat)
                train_video_features.append(video_feat)
        except Exception as e:
            continue

    train_text_features = np.array(train_text_features)
    train_audio_features = np.array(train_audio_features)
    train_video_features = np.array(train_video_features)

    print(f"收集到 {len(train_text_features)} 个训练样本")

    # ============ 阶段2: 分析各模态特性 ============
    print("\n" + "="*70)
    print("阶段2: 数据特性分析")
    print("="*70)

    # GloVe分析
    print("\n--- GloVe (300维) ---")
    print(f"均值范围: [{train_text_features.mean(axis=0).min():.4f}, {train_text_features.mean(axis=0).max():.4f}]")
    print(f"标准差范围: [{train_text_features.std(axis=0).min():.4f}, {train_text_features.std(axis=0).max():.4f}]")
    glove_inf = np.isinf(train_text_features).sum()
    glove_nan = np.isnan(train_text_features).sum()
    print(f"包含Inf的值: {glove_inf}")
    print(f"包含NaN的值: {glove_nan}")

    # COVAREP分析
    print("\n--- COVAREP (74维) ---")
    print(f"均值范围: [{train_audio_features.mean(axis=0).min():.4f}, {train_audio_features.mean(axis=0).max():.4f}]")
    print(f"标准差范围: [{train_audio_features.std(axis=0).min():.4f}, {train_audio_features.std(axis=0).max():.4f}]")
    audio_inf = np.isinf(train_audio_features).sum()
    audio_nan = np.isnan(train_audio_features).sum()
    print(f"包含Inf的值: {audio_inf}")
    print(f"包含NaN的值: {audio_nan}")

    # OpenFace分析
    print("\n--- OpenFace (713维) ---")
    print(f"均值范围: [{train_video_features.mean(axis=0).min():.4f}, {train_video_features.mean(axis=0).max():.4f}]")
    print(f"标准差范围: [{train_video_features.std(axis=0).min():.4f}, {train_video_features.std(axis=0).max():.4f}]")
    video_inf = np.isinf(train_video_features).sum()
    video_nan = np.isnan(train_video_features).sum()
    print(f"包含Inf的值: {video_inf}")
    print(f"包含NaN的值: {video_nan}")

    # 检查OpenFace极端值
    video_means = train_video_features.mean(axis=0)
    video_stds = train_video_features.std(axis=0)
    zero_var_video = (video_stds < 1e-8).sum()
    print(f"方差≈0的维度数: {zero_var_video} / {VIDEO_DIM}")

    extreme_video = (np.abs(train_video_features - video_means) / (video_stds + 1e-8) > 5).sum(axis=0)
    print(f"有>5%极端值的维度: {(extreme_video > len(train_video_features)*0.05).sum()}")

    # ============ 阶段3: 计算归一化参数（使用中位数和MAD）============
    print("\n" + "="*70)
    print("阶段3: 计算鲁棒归一化参数")
    print("="*70)

    # 对于GloVe：使用均值和标准差（GloVe通常分布合理）
    glove_mean = np.mean(train_text_features, axis=0)
    glove_std = np.std(train_text_features, axis=0)
    glove_std = np.where(glove_std < 1e-8, 1.0, glove_std)

    # 对于COVAREP：使用中位数和MAD（因为有-inf）
    audio_median, audio_mad = get_median_and_mad(train_audio_features)

    # 对于OpenFace：使用中位数和MAD（因为有极端值）
    video_median, video_mad = get_median_and_mad(train_video_features)

    print("\n归一化参数:")
    print(f"  GloVe - mean: [{glove_mean.min():.4f}, {glove_mean.max():.4f}]")
    print(f"  GloVe - std:  [{glove_std.min():.4f}, {glove_std.max():.4f}]")
    print(f"  COVAREP - median: [{audio_median.min():.4f}, {audio_median.max():.4f}]")
    print(f"  COVAREP - mad:    [{audio_mad.min():.4f}, {audio_mad.max():.4f}]")
    print(f"  OpenFace - median: [{video_median.min():.4f}, {video_median.max():.4f}]")
    print(f"  OpenFace - mad:    [{video_mad.min():.4f}, {video_mad.max():.4f}]")

    # ============ 阶段4: OpenFace特征过滤（过滤噪声维度）============
    print("\n" + "="*70)
    print("阶段4: OpenFace特征维度过滤")
    print("="*70)

    # 基于MAD过滤OpenFace维度（保留有变化的维度）
    video_mad_adjusted = video_mad * 1.4826  # 转换为标准差等价
    video_mad_adjusted = np.where(video_mad_adjusted < 1e-6, 1.0, video_mad_adjusted)

    # 计算变异系数（标准化方差）
    cv = video_mad_adjusted / (np.abs(video_median) + 1e-6)
    print(f"\nOpenFace维度变异系数范围: [{cv.min():.4f}, {cv.max():.4f}]")

    # 保留变异系数大于阈值的维度
    cv_threshold = 0.01  # 可调节阈值
    valid_video_dims = cv > cv_threshold

    print(f"变异系数阈值: {cv_threshold}")
    print(f"过滤前维度数: {VIDEO_DIM}")
    print(f"过滤后维度数: {valid_video_dims.sum()}")
    print(f"保留维度比例: {valid_video_dims.sum()/VIDEO_DIM*100:.1f}%")

    # 更新VIDEO_DIM配置
    NEW_VIDEO_DIM = int(valid_video_dims.sum())
    print(f"\n新视频维度: {NEW_VIDEO_DIM}")

    # ============ 阶段5: 处理各数据集 ============
    print("\n" + "="*70)
    print("阶段5: 处理并保存清洗后的数据")
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

                # 归一化
                # GloVe: 标准Z-score
                text_feat_norm = (text_feat - glove_mean) / glove_std

                # COVAREP: 鲁棒归一化（中位数+MAD）
                audio_feat_norm = robust_normalize(audio_feat.reshape(1, -1), audio_median.reshape(1, -1), audio_mad.reshape(1, -1))
                audio_feat_norm = audio_feat_norm.flatten()

                # OpenFace: 鲁棒归一化 + 维度过滤
                video_feat_norm = robust_normalize(video_feat_raw.reshape(1, -1), video_median.reshape(1, -1), video_mad.reshape(1, -1))
                video_feat_norm = video_feat_norm.flatten()
                video_feat_filtered = video_feat_norm[valid_video_dims]  # 应用维度过滤

                # 检查有效性
                if np.isnan(text_feat_norm).any() or np.isinf(text_feat_norm).any():
                    continue
                if np.isnan(audio_feat_norm).any() or np.isinf(audio_feat_norm).any():
                    continue
                if np.isnan(video_feat_filtered).any() or np.isinf(video_feat_filtered).any():
                    continue

                # 标签
                label = labels_dict[key]['features']
                label_int = int((label[0, 0] + 3))
                if label_int < 0 or label_int > 6:
                    continue

                text_features_list.append(text_feat_norm)
                audio_features_list.append(audio_feat_norm)
                video_features_list.append(video_feat_filtered)
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
                new_video_dim=NEW_VIDEO_DIM
            )
            print(f"[OK] 已保存到: {output_path}")
        else:
            print(f"[WARN] {split_name.upper()} 没有数据!")

    print("\n" + "=" * 70)
    print("[完成] 数据清洗完成!")
    print("=" * 70)
    print(f"\n数据保存在: {processed_dir}")
    print(f"\n重要变化:")
    print(f"  - GloVe (300维): 标准Z-score归一化")
    print(f"  - COVAREP (74维): 鲁棒归一化（中位数+MAD），处理异常值")
    print(f"  - OpenFace: {VIDEO_DIM}维 → {NEW_VIDEO_DIM}维（过滤低变异维度）")
    print(f"  - 使用软裁剪（±5 MAD）而非硬裁剪，保留特征信息")

    print(f"\n下一步: 修改config.py中的VIDEO_DIM={NEW_VIDEO_DIM}")
    print(f"      然后运行: python scripts/train_model.py --data-dir {processed_dir}")

    return NEW_VIDEO_DIM


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    try:
        new_video_dim = clean_and_normalize_data()
        print(f"\n请手动修改 config.py 中的 VIDEO_DIM = {new_video_dim}")
    except Exception as e:
        print(f"\n[ERROR] 清洗失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
