# -*- coding: utf-8 -*-
"""
带归一化的数据转换脚本 - 解决数值范围问题
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


def normalize_features(features, axis=0, epsilon=1e-8):
    """Z-score归一化"""
    mean = np.mean(features, axis=axis, keepdims=True)
    std = np.std(features, axis=axis, keepdims=True)

    # 处理标准差为0的情况
    std = np.where(std < epsilon, 1.0, std)

    normalized = (features - mean) / std

    # 替换NaN和Inf
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)

    return normalized


def clip_features(features, min_val=-100, max_val=100):
    """裁剪特征到合理范围"""
    return np.clip(features, min_val, max_val)


def convert_normalized_data():
    """带归一化的数据转换"""
    print("=" * 70)
    print("CMU-MOSEI 数据转换（带归一化）: .csd → .npz")
    print("=" * 70)

    data_dir = os.path.join(project_root, 'data', 'mosei')
    processed_dir = os.path.join(data_dir, 'processed_normalized')
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
    print("[OK] 数据加载完成")

    # 获取fold信息
    from mmsdk.mmdatasdk import cmu_mosei
    train_fold = set(cmu_mosei.standard_folds.standard_train_fold)
    val_fold = set(cmu_mosei.standard_folds.standard_valid_fold)
    test_fold = set(cmu_mosei.standard_folds.standard_test_fold)

    print(f"\n数据集划分:")
    print(f"  - 训练集: {len(train_fold)} 个视频")
    print(f"  - 验证集: {len(val_fold)} 个视频")
    print(f"  - 测试集: {len(test_fold)} 个视频")

    # 获取数据字典
    glove_dict = highlevel_data.computational_sequences['glove_vectors'].data
    audio_dict = highlevel_data.computational_sequences['COVAREP'].data
    video_dict = highlevel_data.computational_sequences['OpenFace_2'].data
    labels_dict = labels_data.computational_sequences['All Labels'].data

    # 收集训练集所有特征用于计算归一化参数
    print("\n收集训练集特征用于归一化...")
    train_text_features = []
    train_audio_features = []
    train_video_features = []

    glove_keys_list = list(glove_dict.keys())
    first_key = glove_keys_list[0]

    if '[' in first_key:
        keys_to_process = [k for k in glove_dict.keys() if k.split('[')[0] in train_fold]
    else:
        keys_to_process = [k for k in glove_dict.keys() if k in train_fold]

    for key in keys_to_process:
        try:
            if key in glove_dict and key in audio_dict and key in video_dict:
                text_feat = glove_dict[key]['features']
                audio_feat = audio_dict[key]['features']
                video_feat = video_dict[key]['features']

                # 平均池化
                text_feat_mean = np.mean(text_feat, axis=0).astype(np.float32)
                audio_feat_mean = np.mean(audio_feat, axis=0).astype(np.float32)
                video_feat_mean = np.mean(video_feat, axis=0).astype(np.float32)

                train_text_features.append(text_feat_mean)
                train_audio_features.append(audio_feat_mean)
                train_video_features.append(video_feat_mean)
        except:
            continue

    # 转换为numpy数组
    train_text_features = np.array(train_text_features)
    train_audio_features = np.array(train_audio_features)
    train_video_features = np.array(train_video_features)

    print(f"收集到 {len(train_text_features)} 个训练样本")

    # 计算归一化参数（基于训练集）
    print("\n计算归一化参数...")
    # 不使用keepdims，得到1D数组
    text_mean = np.mean(train_text_features, axis=0)
    text_std = np.std(train_text_features, axis=0)
    text_std = np.where(text_std < 1e-8, 1.0, text_std)

    audio_mean = np.mean(train_audio_features, axis=0)
    audio_std = np.std(train_audio_features, axis=0)
    # 处理异常值：使用位运算符而不是逻辑运算符
    audio_std = np.where((np.abs(audio_std) < 1e-8) | np.isnan(audio_std) | np.isinf(audio_std), 1.0, audio_std)

    video_mean = np.mean(train_video_features, axis=0)
    video_std = np.std(train_video_features, axis=0)
    video_std = np.where(video_std < 1e-8, 1.0, video_std)

    print(f"文本归一化参数 - mean range: [{text_mean.min():.4f}, {text_mean.max():.4f}], std range: [{text_std.min():.4f}, {text_std.max():.4f}]")
    print(f"音频归一化参数 - mean range: [{audio_mean.min():.4f}, {audio_mean.max():.4f}], std range: [{audio_std.min():.4f}, {audio_std.max():.4f}]")
    print(f"视频归一化参数 - mean range: [{video_mean.min():.4f}, {video_mean.max():.4f}], std range: [{video_std.min():.4f}, {video_std.max():.4f}]")

    # 处理每个数据集
    splits = [
        ('train', train_fold, os.path.join(processed_dir, 'train.npz')),
        ('val', val_fold, os.path.join(processed_dir, 'val.npz')),
        ('test', test_fold, os.path.join(processed_dir, 'test.npz'))
    ]

    for split_name, fold_ids, output_path in splits:
        print(f"\n{'='*70}")
        print(f"处理 {split_name.upper()} 集")
        print(f"{'='*70}")

        text_features_list = []
        audio_features_list = []
        video_features_list = []
        labels_list = []
        sample_ids_list = []

        glove_keys_list = list(glove_dict.keys())
        first_key = glove_keys_list[0]

        if '[' in first_key:
            keys_to_process = [k for k in glove_dict.keys() if k.split('[')[0] in fold_ids]
        else:
            keys_to_process = [k for k in glove_dict.keys() if k in fold_ids]

        print(f"找到 {len(keys_to_process)} 个数据条目")

        for key in tqdm(keys_to_process, desc=f"处理{split_name}"):
            try:
                if key not in glove_dict or key not in audio_dict or key not in video_dict or key not in labels_dict:
                    continue

                # 获取特征
                text_feat = glove_dict[key]['features']
                audio_feat = audio_dict[key]['features']
                video_feat = video_dict[key]['features']
                label = labels_dict[key]['features']

                # 检查维度
                if text_feat.shape[1] != TEXT_DIM or audio_feat.shape[1] != AUDIO_DIM or video_feat.shape[1] != VIDEO_DIM:
                    continue

                # 平均池化
                text_feat_mean = np.mean(text_feat, axis=0).astype(np.float32)
                audio_feat_mean = np.mean(audio_feat, axis=0).astype(np.float32)
                video_feat_mean = np.mean(video_feat, axis=0).astype(np.float32)

                # Z-score归一化（使用训练集的参数）
                text_feat_norm = (text_feat_mean - text_mean) / text_std
                audio_feat_norm = (audio_feat_mean - audio_mean) / audio_std
                video_feat_norm = (video_feat_mean - video_mean) / video_std

                # 替换NaN和Inf
                text_feat_norm = np.nan_to_num(text_feat_norm, nan=0.0, posinf=1.0, neginf=-1.0)
                audio_feat_norm = np.nan_to_num(audio_feat_norm, nan=0.0, posinf=1.0, neginf=-1.0)
                video_feat_norm = np.nan_to_num(video_feat_norm, nan=0.0, posinf=1.0, neginf=-1.0)

                # 裁剪到合理范围
                text_feat_norm = np.clip(text_feat_norm, -10, 10)
                audio_feat_norm = np.clip(audio_feat_norm, -10, 10)
                video_feat_norm = np.clip(video_feat_norm, -10, 10)

                # 标签转换
                label_int = int((label[0, 0] + 3))
                if label_int < 0 or label_int > 6:
                    continue

                text_features_list.append(text_feat_norm)  # 1D array (300,)
                audio_features_list.append(audio_feat_norm)  # 1D array (74,)
                video_features_list.append(video_feat_norm)  # 1D array (713,)
                labels_list.append(label_int)
                sample_ids_list.append(key)

            except Exception as e:
                continue

        # 保存数据
        if len(text_features_list) > 0:
            text_arr = np.array(text_features_list)
            audio_arr = np.array(audio_features_list)
            video_arr = np.array(video_features_list)
            labels_arr = np.array(labels_list)
            sample_ids_arr = np.array(sample_ids_list)

            print(f"\n{split_name.upper()} 统计:")
            print(f"  - 样本数: {len(labels_arr)}")
            print(f"  - 文本特征范围: [{text_arr.min():.4f}, {text_arr.max():.4f}]")
            print(f"  - 音频特征范围: [{audio_arr.min():.4f}, {audio_arr.max():.4f}]")
            print(f"  - 视频特征范围: [{video_arr.min():.4f}, {video_arr.max():.4f}]")
            print(f"  - 标签范围: [{labels_arr.min()}, {labels_arr.max()}]")

            # 标签分布
            unique, counts = np.unique(labels_arr, return_counts=True)
            print(f"  - 标签分布:")
            label_names = ['强负', '负', '弱负', '中性', '弱正', '正', '强正']
            for idx, count in zip(unique, counts):
                print(f"      {label_names[idx]}: {count} ({count/len(labels_arr)*100:.1f}%)")

            np.savez_compressed(
                output_path,
                text_features=text_arr,
                audio_features=audio_arr,
                video_features=video_arr,
                labels=labels_arr,
                sample_ids=sample_ids_arr
            )
            print(f"\n[OK] 已保存到: {output_path}")

    print("\n" + "=" * 70)
    print("[完成] 归一化数据转换完成!")
    print("=" * 70)
    print(f"\n数据保存在: {processed_dir}")


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    try:
        convert_normalized_data()
    except Exception as e:
        print(f"\n[ERROR] 转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
