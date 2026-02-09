# -*- coding: utf-8 -*-
"""
高效数据转换脚本 - 直接从.csd读取并手动对齐

避免使用SDK的内存密集型align操作，手动处理数据对齐和平均池化
"""
import sys
import os

# 添加SDK到路径
SDK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CMU-MultimodalSDK')
sys.path.insert(0, SDK_PATH)

import numpy as np
from mmsdk import mmdatasdk
from tqdm import tqdm

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import Config, TEXT_DIM, AUDIO_DIM, VIDEO_DIM, NUM_CLASSES


def convert_full_data():
    """
    使用真实OpenFace特征转换完整数据
    手动处理对齐，避免SDK align操作的内存问题
    """
    print("=" * 70)
    print("CMU-MOSEI 完整数据转换: .csd → .npz")
    print("=" * 70)

    data_dir = os.path.join(project_root, 'data', 'mosei')
    processed_dir = os.path.join(data_dir, 'processed')
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

    # 检查文件
    print("\n检查数据文件...")
    for name, path in {**highlevel_files, **labels_files}.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✓ {name}: {size_mb:.1f} MB")
        else:
            print(f"  ✗ {name}: 缺失")
            return 1

    # 加载数据（不进行align）
    print("\n加载数据文件...")
    try:
        highlevel_data = mmdatasdk.mmdataset(highlevel_files)
        labels_data = mmdatasdk.mmdataset(labels_files)
        print("[OK] 数据加载完成")
    except Exception as e:
        print(f"[ERROR] 加载失败: {e}")
        return 1

    # 获取fold信息
    from mmsdk.mmdatasdk import cmu_mosei
    train_fold = set(cmu_mosei.standard_folds.standard_train_fold)
    val_fold = set(cmu_mosei.standard_folds.standard_valid_fold)
    test_fold = set(cmu_mosei.standard_folds.standard_test_fold)

    print(f"\n数据集划分:")
    print(f"  - 训练集: {len(train_fold)} 个视频")
    print(f"  - 验证集: {len(val_fold)} 个视频")
    print(f"  - 测试集: {len(test_fold)} 个视频")

    # 获取所有数据的key
    glove_dict = highlevel_data.computational_sequences['glove_vectors'].data
    audio_dict = highlevel_data.computational_sequences['COVAREP'].data
    video_dict = highlevel_data.computational_sequences['OpenFace_2'].data
    labels_dict = labels_data.computational_sequences['All Labels'].data

    print(f"\n数据规模:")
    print(f"  - GloVe条目: {len(glove_dict)}")
    print(f"  - COVAREP条目: {len(audio_dict)}")
    print(f"  - OpenFace条目: {len(video_dict)}")
    print(f"  - Labels条目: {len(labels_dict)}")

    # 手动处理数据对齐和提取
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

        # 收集所有需要处理的key
        print("\n收集数据条目...")

        # 首先检查glove_dict的keys是否直接是video_id（无segment）
        glove_keys_list = list(glove_dict.keys())
        first_key = glove_keys_list[0]

        # 检查是否有segment信息
        if '[' in first_key:
            # key格式: "video_id[segment_id]"
            keys_to_process = [k for k in glove_dict.keys() if k.split('[')[0] in fold_ids]
        else:
            # key直接是video_id
            keys_to_process = [k for k in glove_dict.keys() if k in fold_ids]

        print(f"找到 {len(keys_to_process)} 个数据条目")

        # 调试信息：显示几个样本key
        print(f"\n样本key格式 (前5个):")
        glove_keys = list(glove_dict.keys())
        print(f"  GloVe: {glove_keys[:5]}")
        print(f"  COVAREP: {list(audio_dict.keys())[:5]}")
        print(f"  OpenFace: {list(video_dict.keys())[:5]}")
        print(f"  Labels: {list(labels_dict.keys())[:5]}")

        # 检查是否包含segment信息
        print(f"\n检查key中是否有'['字符:")
        has_segment = any('[' in k for k in glove_keys)
        print(f"  GloVe包含segment: {has_segment}")
        if has_segment:
            segment_keys = [k for k in glove_keys if '[' in k][:3]
            print(f"  示例: {segment_keys}")

        # 检查fold_ids和glove_keys的匹配
        print(f"\n检查fold_ids匹配:")
        fold_ids_list = list(fold_ids)[:3]
        print(f"  fold_ids示例: {fold_ids_list}")
        print(f"  fold_ids[0]在glove_dict中: {fold_ids_list[0] in glove_dict}")

        # 处理每个条目
        print("\n提取特征并平均池化...")

        # 显示第一个样本的特征形状
        if keys_to_process:
            first_key = keys_to_process[0]
            print(f"\n第一个样本特征检查 (key={first_key}):")
            if first_key in glove_dict:
                print(f"  GloVe shape: {glove_dict[first_key]['features'].shape}")
            if first_key in audio_dict:
                print(f"  COVAREP shape: {audio_dict[first_key]['features'].shape}")
            if first_key in video_dict:
                print(f"  OpenFace shape: {video_dict[first_key]['features'].shape}")
            if first_key in labels_dict:
                print(f"  Label shape: {labels_dict[first_key]['features'].shape}")
            print(f"  预期维度: text={TEXT_DIM}, audio={AUDIO_DIM}, video={VIDEO_DIM}")

        skipped_count = 0
        success_count = 0
        skip_reasons = {}

        for key in tqdm(keys_to_process, desc=f"处理{split_name}"):

            try:
                # 检查所有模态的数据是否存在
                if key not in glove_dict:
                    skip_reasons['glove_missing'] = skip_reasons.get('glove_missing', 0) + 1
                    skipped_count += 1
                    continue
                if key not in audio_dict:
                    skip_reasons['audio_missing'] = skip_reasons.get('audio_missing', 0) + 1
                    skipped_count += 1
                    continue
                if key not in video_dict:
                    skip_reasons['video_missing'] = skip_reasons.get('video_missing', 0) + 1
                    skipped_count += 1
                    continue
                if key not in labels_dict:
                    skip_reasons['label_missing'] = skip_reasons.get('label_missing', 0) + 1
                    skipped_count += 1
                    continue

                # 获取特征
                text_feat = glove_dict[key]['features']  # shape: (seq_len, 300)
                audio_feat = audio_dict[key]['features']  # shape: (seq_len, 74)
                video_feat = video_dict[key]['features']  # shape: (seq_len, 25)
                label = labels_dict[key]['features']  # shape: (1, 1)

                # 检查维度
                if text_feat.shape[1] != TEXT_DIM:
                    skip_reasons['text_dim'] = skip_reasons.get('text_dim', 0) + 1
                    continue
                if audio_feat.shape[1] != AUDIO_DIM:
                    skip_reasons['audio_dim'] = skip_reasons.get('audio_dim', 0) + 1
                    continue
                if video_feat.shape[1] != VIDEO_DIM:
                    skip_reasons['video_dim'] = skip_reasons.get('video_dim', 0) + 1
                    continue

                # 平均池化
                text_feat_mean = np.mean(text_feat, axis=0).astype(np.float32)  # (300,)
                audio_feat_mean = np.mean(audio_feat, axis=0).astype(np.float32)  # (74,)
                video_feat_mean = np.mean(video_feat, axis=0).astype(np.float32)  # (25,)

                # 标签转换: [-3, +3] -> [0, 6]
                label_int = int((label[0, 0] + 3))

                # 验证标签范围
                if label_int < 0 or label_int > 6:
                    skip_reasons['label_range'] = skip_reasons.get('label_range', 0) + 1
                    continue

                text_features_list.append(text_feat_mean)
                audio_features_list.append(audio_feat_mean)
                video_features_list.append(video_feat_mean)
                labels_list.append(label_int)
                sample_ids_list.append(key)
                success_count += 1

            except Exception as e:
                skip_reasons['exception'] = skip_reasons.get('exception', 0) + 1
                skipped_count += 1
                continue

        print(f"\n处理结果: 成功={success_count}, 跳过={skipped_count}")
        if skip_reasons:
            print(f"跳过原因: {skip_reasons}")

        # 保存数据
        if len(text_features_list) > 0:
            text_arr = np.array(text_features_list)
            audio_arr = np.array(audio_features_list)
            video_arr = np.array(video_features_list)
            labels_arr = np.array(labels_list)
            sample_ids_arr = np.array(sample_ids_list)

            print(f"\n{split_name.upper()} 统计:")
            print(f"  - 样本数: {len(labels_arr)}")
            print(f"  - 文本特征: {text_arr.shape}")
            print(f"  - 音频特征: {audio_arr.shape}")
            print(f"  - 视频特征: {video_arr.shape}")
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
        else:
            print(f"[WARN] {split_name.upper()} 没有数据!")

    print("\n" + "=" * 70)
    print("[完成] 所有数据转换完成!")
    print("=" * 70)
    print(f"\n数据保存在: {processed_dir}")
    print("\n下一步: 运行训练脚本")
    print("  python scripts/train_model.py --data_dir data/mosei/processed")


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    try:
        convert_full_data()
    except Exception as e:
        print(f"\n[ERROR] 转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
