# -*- coding: utf-8 -*-
"""
将CMU-MultimodalSDK处理的.csd数据转换为PyTorch训练用的.npz格式

这个脚本在download_mosei_data.py完成后运行，将处理好的.csd文件
转换为可以直接用于PyTorch DataLoader的.npz格式。
"""

import sys
import os

# 添加SDK到路径
SDK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CMU-MultimodalSDK')
sys.path.insert(0, SDK_PATH)

import numpy as np
from mmsdk import mmdatasdk
from tqdm import tqdm


def convert_csd_to_npz(
    csd_data_path: str,
    output_path: str,
    split_name: str
):
    """
    将.csd格式的数据转换为.npz格式

    Args:
        csd_data_path: .csd数据文件路径（目录）
        output_path: 输出.npz文件路径
        split_name: 数据划分名称（train/val/test）
    """
    print(f"\n{'='*70}")
    print(f"转换 {split_name} 数据")
    print(f"{'='*70}")

    # 加载.csd数据
    print(f"加载.csd数据: {csd_data_path}")
    dataset = mmdatasdk.mmdataset(csd_data_path)

    # 获取数据
    print("提取特征...")
    text_features_list = []
    audio_features_list = []
    video_features_list = []
    labels_list = []
    sample_ids = []

    # 获取computational sequences
    csd_dict = dataset.computational_sequences

    # 检查必需的特征
    required_features = ['glove_vectors', 'COVAREP', 'OpenFace_2', 'All Labels']
    for feat in required_features:
        if feat not in csd_dict:
            raise ValueError(f"缺少必需的特征: {feat}")

    # 获取所有样本ID（video ID）
    glove_data = csd_dict['glove_vectors'].data
    all_sample_ids = list(set([key.split('[')[0] for key in glove_data.keys()]))

    print(f"找到 {len(all_sample_ids)} 个样本")

    # 遍历每个样本
    for sample_id in tqdm(all_sample_ids, desc=f"处理{split_name}"):
        try:
            # 获取该样本的所有特征
            sample_keys = [key for key in glove_data.keys() if key.startswith(sample_id)]

            if not sample_keys:
                continue

            # 对每个segment进行处理
            for key in sample_keys:
                # 获取特征数据
                text_feat = csd_dict['glove_vectors'].data[key]['features']
                audio_feat = csd_dict['COVAREP'].data[key]['features']
                video_feat = csd_dict['OpenFace_2'].data[key]['features']

                # 获取标签
                label_key = key  # 标签使用相同的key
                if label_key in csd_dict['All Labels'].data:
                    label = csd_dict['All Labels'].data[label_key]['features']
                    # CMU-MOSEI标签范围是-3到+3，转换为0-6
                    label = int((label[0, 0] + 3))  # 从[-3,3]转换到[0,6]
                else:
                    # 如果没有标签，跳过
                    continue

                # 检查维度
                if text_feat.shape[1] != 300:
                    print(f"警告: 样本{sample_id}的文本维度不是300: {text_feat.shape}")
                if audio_feat.shape[1] != 74:
                    print(f"警告: 样本{sample_id}的音频维度不是74: {audio_feat.shape}")
                if video_feat.shape[1] != 25:
                    print(f"警告: 样本{sample_id}的视频维度不是25: {video_feat.shape}")

                # 使用平均池化将序列特征转换为单个向量
                text_feat_mean = np.mean(text_feat, axis=0)  # (seq_len, 300) -> (300,)
                audio_feat_mean = np.mean(audio_feat, axis=0)  # (seq_len, 74) -> (74,)
                video_feat_mean = np.mean(video_feat, axis=0)  # (seq_len, 25) -> (25,)

                # 添加到列表
                text_features_list.append(text_feat_mean.astype(np.float32))
                audio_features_list.append(audio_feat_mean.astype(np.float32))
                video_features_list.append(video_feat_mean.astype(np.float32))
                labels_list.append(label)
                sample_ids.append(f"{sample_id}_{key}")

        except Exception as e:
            print(f"处理样本{sample_id}时出错: {e}")
            continue

    # 转换为numpy数组
    text_features = np.array(text_features_list)
    audio_features = np.array(audio_features_list)
    video_features = np.array(video_features_list)
    labels = np.array(labels_list)
    sample_ids_arr = np.array(sample_ids)

    print(f"\n{split_name.upper()} 数据统计:")
    print(f"  - 样本数: {len(labels)}")
    print(f"  - 文本特征形状: {text_features.shape}")
    print(f"  - 音频特征形状: {audio_features.shape}")
    print(f"  - 视频特征形状: {video_features.shape}")
    print(f"  - 标签范围: {labels.min()} 到 {labels.max()}")

    # 保存为.npz格式
    print(f"\n保存到: {output_path}")
    np.savez_compressed(
        output_path,
        text_features=text_features,
        audio_features=audio_features,
        video_features=video_features,
        labels=labels,
        sample_ids=sample_ids_arr
    )

    print(f"[OK] 转换完成!")
    return len(labels)


def main():
    """主函数"""
    # 设置Windows控制台编码
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print("=" * 70)
    print("CMU-MOSEI数据转换: .csd → .npz")
    print("=" * 70)

    # 数据目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data', 'mosei')
    processed_dir = os.path.join(data_dir, 'processed')

    # 创建输出目录
    os.makedirs(processed_dir, exist_ok=True)

    # 定义数据路径
    # 注意：download_mosei_data.py会生成final_aligned目录
    train_csd = os.path.join(data_dir, 'final_aligned')
    val_csd = os.path.join(data_dir, 'final_aligned')
    test_csd = os.path.join(data_dir, 'final_aligned')

    # 输出路径
    train_output = os.path.join(processed_dir, 'train.npz')
    val_output = os.path.join(processed_dir, 'val.npz')
    test_output = os.path.join(processed_dir, 'test.npz')

    # 检查输入数据是否存在
    if not os.path.exists(train_csd):
        print(f"\n[ERROR] 找不到处理后的.csd数据: {train_csd}")
        print("\n请先运行: python scripts/download_mosei_data.py")
        print("并等待下载和处理完成。")
        return 1

    try:
        # 转换数据（使用SDK的fold信息）
        # 由于SDK已经将数据分好了fold，我们需要分别处理

        # 加载数据集
        print("\n加载最终对齐的数据集...")
        dataset = mmdatasdk.mmdataset(train_csd)

        # 获取standard fold信息
        from mmsdk.mmdatasdk import cmu_mosei
        train_fold = cmu_mosei.standard_folds.standard_train_fold
        val_fold = cmu_mosei.standard_folds.standard_valid_fold
        test_fold = cmu_mosei.standard_folds.standard_test_fold

        # 获取所有样本ID
        glove_data = dataset.computational_sequences['glove_vectors'].data
        all_sample_ids = list(set([key.split('[')[0] for key in glove_data.keys()]))

        # 按fold划分
        train_ids = set(train_fold)
        val_ids = set(val_fold)
        test_ids = set(test_fold)

        print(f"训练集样本数: {len(train_ids)}")
        print(f"验证集样本数: {len(val_ids)}")
        print(f"测试集样本数: {len(test_ids)}")

        # 分别转换每个fold
        splits = [
            ('train', train_ids, train_output),
            ('val', val_ids, val_output),
            ('test', test_ids, test_output)
        ]

        for split_name, fold_ids, output_path in splits:
            print(f"\n处理 {split_name} 集...")

            # 提取属于该fold的数据
            text_features_list = []
            audio_features_list = []
            video_features_list = []
            labels_list = []
            sample_ids_list = []

            for sample_id in tqdm(fold_ids, desc=f"提取{split_name}"):
                if sample_id not in all_sample_ids:
                    continue

                try:
                    # 获取该样本的所有segments
                    sample_keys = [key for key in glove_data.keys() if key.startswith(sample_id)]

                    for key in sample_keys:
                        # 获取特征
                        text_feat = dataset.computational_sequences['glove_vectors'].data[key]['features']
                        audio_feat = dataset.computational_sequences['COVAREP'].data[key]['features']
                        video_feat = dataset.computational_sequences['OpenFace_2'].data[key]['features']

                        # 获取标签
                        if key in dataset.computational_sequences['All Labels'].data:
                            label = dataset.computational_sequences['All Labels'].data[key]['features']
                            label = int((label[0, 0] + 3))  # 从[-3,3]转换到[0,6]
                        else:
                            continue

                        # 平均池化
                        text_feat_mean = np.mean(text_feat, axis=0).astype(np.float32)
                        audio_feat_mean = np.mean(audio_feat, axis=0).astype(np.float32)
                        video_feat_mean = np.mean(video_feat, axis=0).astype(np.float32)

                        text_features_list.append(text_feat_mean)
                        audio_features_list.append(audio_feat_mean)
                        video_features_list.append(video_feat_mean)
                        labels_list.append(label)
                        sample_ids_list.append(key)

                except Exception as e:
                    print(f"处理样本{sample_id}出错: {e}")
                    continue

            # 保存
            if len(text_features_list) > 0:
                np.savez_compressed(
                    output_path,
                    text_features=np.array(text_features_list),
                    audio_features=np.array(audio_features_list),
                    video_features=np.array(video_features_list),
                    labels=np.array(labels_list),
                    sample_ids=np.array(sample_ids_list)
                )
                print(f"[OK] {split_name.upper()} 已保存: {output_path}")
                print(f"     样本数: {len(labels_list)}")
            else:
                print(f"[WARN] {split_name.upper()} 没有数据!")

        print("\n" + "=" * 70)
        print("[完成] 所有数据转换完成!")
        print("=" * 70)
        print(f"\n数据保存在: {processed_dir}")
        print("现在可以运行: python scripts/train_model.py")

        return 0

    except Exception as e:
        print(f"\n[ERROR] 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
