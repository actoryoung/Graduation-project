# -*- coding: utf-8 -*-
"""
下载并处理CMU-MOSEI数据集

使用官方CMU-MultimodalSDK下载和处理CMU-MOSEI数据集
"""
import sys
import os

# 添加SDK到路径
SDK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CMU-MultimodalSDK')
sys.path.insert(0, SDK_PATH)

import mmsdk
from mmsdk import mmdatasdk
from mmsdk.mmdatasdk import log
import numpy as np


def deploy(in_dataset, destination):
    """将数据集部署到磁盘"""
    deploy_files = {x: x for x in in_dataset.keys()}
    in_dataset.deploy(destination, deploy_files)


def download_data():
    """下载CMU-MOSEI原始数据"""
    print("=" * 70)
    print("步骤1: 下载CMU-MOSEI数据集")
    print("=" * 70)

    source = {
        "highlevel": mmdatasdk.cmu_mosei.highlevel,
        "labels": mmdatasdk.cmu_mosei.labels
    }

    cmumosei_dataset = {}
    for key in source:
        print(f"\n下载 {key} 数据...")
        cmumosei_dataset[key] = mmdatasdk.mmdataset(
            source[key],
            f'cmumosei_{key}/'
        )
        print(f"[OK] {key} 数据下载完成")

    return cmumosei_dataset


def process_data(folders=["cmumosei_highlevel", "cmumosei_labels"]):
    """处理CMU-MOSEI数据"""
    print("\n" + "=" * 70)
    print("步骤2: 处理CMU-MOSEI数据")
    print("=" * 70)

    cmumosei_dataset = {}
    for folder in folders:
        cmumosei_dataset[folder.split("_")[1]] = mmdatasdk.mmdataset(folder)

    # 2.1 词级别对齐
    print("\n[2.1] 执行词级别对齐...")
    cmumosei_dataset["highlevel"].align("glove_vectors")
    print("[OK] 词级别对齐完成")

    # 2.2 插补缺失值
    print("\n[2.2] 插补缺失的模态信息...")
    cmumosei_dataset["highlevel"].impute('glove_vectors')
    print("[OK] 缺失值插补完成")

    # 2.3 保存词级别对齐数据
    print("\n[2.3] 保存词级别对齐数据...")
    deploy(cmumosei_dataset["highlevel"], "word_aligned_highlevel")
    print("[OK] 词级别对齐数据已保存")

    # 2.4 添加标签并对齐
    print("\n[2.4] 添加标签并对齐...")
    cmumosei_dataset["highlevel"].computational_sequences["All Labels"] = \
        cmumosei_dataset["labels"]["All Labels"]
    cmumosei_dataset["highlevel"].align("All Labels")
    print("[OK] 标签对齐完成")

    # 2.5 移除不完整样本
    print("\n[2.5] 移除不完整样本（hard_unify）...")
    cmumosei_dataset["highlevel"].hard_unify()
    print("[OK] 不完整样本已移除")

    # 2.6 保存最终数据
    print("\n[2.6] 保存最终对齐数据...")
    deploy(cmumosei_dataset["highlevel"], "final_aligned")
    print("[OK] 最终数据已保存")

    # 2.7 获取张量
    print("\n[2.7] 获取训练张量...")
    tensors = cmumosei_dataset["highlevel"].get_tensors(
        seq_len=50,
        non_sequences=["All Labels"],
        direction=False,
        folds=[
            mmdatasdk.cmu_mosei.standard_folds.standard_train_fold,
            mmdatasdk.cmu_mosei.standard_folds.standard_valid_fold,
            mmdatasdk.cmu_mosei.standard_folds.standard_test_fold
        ]
    )

    fold_names = ["train", "valid", "test"]

    print("\n[数据集信息]")
    print("-" * 70)
    for i in range(3):
        print(f"\n{fold_names[i].upper()} 数据集:")
        for csd in list(cmumosei_dataset["highlevel"].keys()):
            print(f"  - {csd}: {tensors[i][csd].shape}")

    return tensors, cmumosei_dataset


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("CMU-MOSEI 数据集下载与处理")
    print("=" * 70)

    # 切换到数据目录
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'mosei')
    os.makedirs(data_dir, exist_ok=True)
    os.chdir(data_dir)
    print(f"工作目录: {os.getcwd()}")

    try:
        # 下载数据
        cmumosei_dataset = download_data()

        # 处理数据
        tensors, final_dataset = process_data()

        print("\n" + "=" * 70)
        print("[完成] 数据集处理完成！")
        print("=" * 70)
        print(f"\n数据保存在: {data_dir}")
        print("  - cmumosei_highlevel/     (原始高级特征)")
        print("  - cmumosei_labels/        (原始标签)")
        print("  - word_aligned_highlevel/ (词级别对齐)")
        print("  - final_aligned/          (最终对齐数据)")

    except Exception as e:
        print(f"\n[错误] 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    sys.exit(main())
