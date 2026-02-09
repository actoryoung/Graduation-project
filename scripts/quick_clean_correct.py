# -*- coding: utf-8 -*-
"""
CMU-MOSEI 快速数据清洗 - 使用正确的四舍五入离散化
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


def process_audio_features(audio_feat):
    """处理COVAREP特征的-inf值"""
    audio_clean = audio_feat.copy()
    audio_clean[np.isneginf(audio_clean)] = 0.0
    return audio_clean


def clean_data_correct():
    """使用正确的四舍五入离散化方法"""
    print("=" * 70)
    print("CMU-MOSEI 数据清洗（标准四舍五入离散化）")
    print("=" * 70)

    data_dir = os.path.join(project_root, 'data', 'mosei')
    processed_dir = os.path.join(data_dir, 'processed_cleaned_correct')
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

    # ============ 阶段2: 检测并移除OpenFace极端维度 ============
    print("\n" + "="*70)
    print("阶段2: 检测并移除OpenFace极端异常维度")
    print("="*70)

    video_stds = np.std(train_video_features, axis=0)
    extreme_dims = np.where(video_stds > 10000)[0]
    valid_dims = np.where(video_stds <= 10000)[0]

    print(f"\n检测到 {len(extreme_dims)} 个极端异常维度 (std > 10000):")
    for dim in extreme_dims:
        print(f"  维度{dim}: std={video_stds[dim]:.0f}")

    print(f"\n保留 {len(valid_dims)} 个正常维度")
    train_video_clean = train_video_features[:, valid_dims]

    NEW_VIDEO_DIM = len(valid_dims)

    # ============ 阶段3: 计算归一化参数 ============
    print("\n" + "="*70)
    print("阶段3: 计算归一化参数")
    print("="*70)

    text_mean = np.mean(train_text_features, axis=0)
    text_std = np.std(train_text_features, axis=0)
    text_std = np.where(text_std < 1e-8, 1.0, text_std)

    audio_mean = np.mean(train_audio_features, axis=0)
    audio_std = np.std(train_audio_features, axis=0)
    audio_std = np.where(audio_std < 1e-8, 1.0, audio_std)

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

                # 标签 - 使用标准四舍五入方法
                raw_label = labels_dict[key]['features'][0, 0]
                label_int = int(round(raw_label)) + 3
                if label_int < 0:
                    label_int = 0
                elif label_int > 6:
                    label_int = 6

                text_features_list.append(text_feat_norm)
                audio_features_list.append(audio_feat_norm)
                video_features_list.append(video_feat_norm)
                labels_list.append(label_int)

            except Exception as e:
                continue

        # 保存
        if len(text_features_list) > 0:
            text_arr = np.array(text_features_list)
            audio_arr = np.array(audio_features_list)
            video_arr = np.array(video_features_list)
            labels_arr = np.array(labels_list)

            print(f"\n{split_name.upper()} 统计:")
            print(f"  - 样本数: {len(labels_arr)}")
            print(f"  - 文本特征: shape={text_arr.shape}, range=[{text_arr.min():.4f}, {text_arr.max():.4f}]")
            print(f"  - 音频特征: shape={audio_arr.shape}, range=[{audio_arr.min():.4f}, {audio_arr.max():.4f}]")
            print(f"  - 视频特征: shape={video_arr.shape}, range=[{video_arr.min():.4f}, {video_arr.max():.4f}]")

            # 标签分布
            unique, counts = np.unique(labels_arr, return_counts=True)
            label_names_cn = ['强负面', '负面', '弱负面', '中性', '弱正面', '正面', '强正面']
            print(f"  - 标签分布:")
            for idx, count in zip(unique, counts):
                print(f"      {label_names_cn[idx]}: {count} ({count/len(labels_arr)*100:.1f}%)")

            np.savez_compressed(
                output_path,
                text_features=text_arr,
                audio_features=audio_arr,
                video_features=video_arr,
                labels=labels_arr
            )
            print(f"[OK] 已保存到: {output_path}")
        else:
            print(f"[WARN] {split_name.upper()} 没有数据!")

    print("\n" + "=" * 70)
    print("[完成] 数据清洗完成!")
    print("=" * 70)
    print(f"\n数据保存在: {processed_dir}")
    print(f"\n关键改进:")
    print(f"  - 移除了{len(extreme_dims)}个OpenFace极端异常维度 (std > 10000)")
    print(f"  - 视频维度: 713 → {NEW_VIDEO_DIM}")
    print(f"  - 统一的归一化策略: 所有模态Z-score + 裁剪±3σ")
    print(f"  - 音频-inf精确处理: 用0填充")
    print(f"  - 标签离散化: 使用标准四舍五入方法 (int(round(label)) + 3)")

    print(f"\n下一步: 修改config.py中的VIDEO_DIM={NEW_VIDEO_DIM}")
    print(f"      然后运行: python scripts/train_model.py --data-dir {processed_dir}")

    return NEW_VIDEO_DIM


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    try:
        new_video_dim = clean_data_correct()
        print(f"\n请手动修改 config.py 中的 VIDEO_DIM = {new_video_dim}")
    except Exception as e:
        print(f"\n[ERROR] 清洗失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
