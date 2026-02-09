# -*- coding: utf-8 -*-
"""
提取未清洗的原始数据 - 用于消融实验
不做任何数据清洗，直接从.csd文件提取原始特征
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


def extract_raw_baseline():
    """提取未清洗的原始数据（真正的基线）"""
    print("=" * 70)
    print("提取未清洗的原始数据 - 消融实验基线")
    print("=" * 70)

    data_dir = os.path.join(project_root, 'data', 'mosei')
    raw_dir = os.path.join(data_dir, 'raw_baseline')
    os.makedirs(raw_dir, exist_ok=True)

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

    splits = [
        ('train', train_fold, os.path.join(raw_dir, 'train.npz')),
        ('val', val_fold, os.path.join(raw_dir, 'val.npz')),
        ('test', test_fold, os.path.join(raw_dir, 'test.npz'))
    ]

    for split_name, fold_ids, output_path in splits:
        print(f"\n处理 {split_name.upper()} 集...")

        # 收集该fold的keys
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

                # 直接平均池化，不做任何清洗
                text_feat = np.mean(glove_dict[key]['features'], axis=0).astype(np.float32)
                audio_feat = np.mean(audio_dict[key]['features'], axis=0).astype(np.float32)
                video_feat = np.mean(video_dict[key]['features'], axis=0).astype(np.float32)

                # 标签离散化
                raw_label = labels_dict[key]['features'][0, 0]
                label_int = int(round(raw_label)) + 3
                if label_int < 0:
                    label_int = 0
                elif label_int > 6:
                    label_int = 6

                # 直接保存，不做NaN/Inf检查
                text_features_list.append(text_feat)
                audio_features_list.append(audio_feat)
                video_features_list.append(video_feat)
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
            print(f"  - 音频特征: shape={audio_arr.shape}, 有-inf={np.isneginf(audio_arr).any()}, 有inf={np.isinf(audio_arr).any()}")
            print(f"  - 视频特征: shape={video_arr.shape}, 有-inf={np.isneginf(video_arr).any()}, 有inf={np.isinf(video_arr).any()}")

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
    print("[完成] 未清洗数据提取完成!")
    print("=" * 70)
    print(f"\n数据保存在: {raw_dir}")
    print(f"\n特点:")
    print(f"  - 无数据清洗")
    print(f"  - 无归一化")
    print(f"  - 保留所有异常值(-inf, +inf)")
    print(f"  - 保留所有维度(包括OpenFace极端维度)")
    print(f"\n下一步: python scripts/train_baseline.py --data-dir {raw_dir}")


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    try:
        extract_raw_baseline()
    except Exception as e:
        print(f"\n[ERROR] 处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
