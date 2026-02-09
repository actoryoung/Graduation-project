# -*- coding: utf-8 -*-
"""
直接使用本地已下载的.csd文件进行数据对齐处理
"""
import sys
import os

# 添加SDK到路径
SDK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CMU-MultimodalSDK')
sys.path.insert(0, SDK_PATH)

from mmsdk import mmdatasdk
import numpy as np


def main():
    """主函数"""
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print("=" * 70)
    print("CMU-MOSEI 本地数据对齐处理")
    print("=" * 70)

    # 切换到数据目录
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'mosei')
    os.chdir(data_dir)
    print(f"工作目录: {os.getcwd()}")

    # 定义本地.csd文件路径
    highlevel_files = {
        'glove_vectors': os.path.join(data_dir, 'cmumosei_highlevel', 'CMU_MOSEI_TimestampedWordVectors.csd'),
        'COVAREP': os.path.join(data_dir, 'cmumosei_highlevel', 'CMU_MOSEI_COVAREP.csd'),
        'OpenFace_2': os.path.join(data_dir, 'cmumosei_highlevel', 'CMU_MOSEI_VisualOpenFace2.csd'),
    }
    labels_files = {
        'All Labels': os.path.join(data_dir, 'cmumosei_labels', 'CMU_MOSEI_Labels.csd')
    }

    # 检查文件是否存在
    print("\n检查本地文件...")
    for name, path in {**highlevel_files, **labels_files}.items():
        if os.path.exists(path):
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: {path} [缺失]")
            return 1

    try:
        # 加载本地数据
        print("\n" + "=" * 70)
        print("步骤1: 加载本地.csd文件")
        print("=" * 70)

        highlevel_dataset = mmdatasdk.mmdataset(highlevel_files)
        labels_dataset = mmdatasdk.mmdataset(labels_files)
        print("[OK] 数据加载完成")

        # 词级别对齐
        print("\n" + "=" * 70)
        print("步骤2: 词级别对齐")
        print("=" * 70)
        highlevel_dataset.align("glove_vectors")
        print("[OK] 词级别对齐完成")

        # 插补缺失值
        print("\n" + "=" * 70)
        print("步骤3: 插补缺失的模态信息")
        print("=" * 70)
        highlevel_dataset.impute('glove_vectors')
        print("[OK] 缺失值插补完成")

        # 保存词级别对齐数据
        print("\n" + "=" * 70)
        print("步骤4: 保存词级别对齐数据")
        print("=" * 70)
        highlevel_dataset.deploy("word_aligned_highlevel", {
            'glove_vectors': 'glove_vectors',
            'COVAREP': 'COVAREP',
            'OpenFace_2': 'OpenFace_2'
        })
        print("[OK] 词级别对齐数据已保存")

        # 添加标签并对齐
        print("\n" + "=" * 70)
        print("步骤5: 添加标签并对齐")
        print("=" * 70)
        highlevel_dataset.computational_sequences["All Labels"] = \
            labels_dataset.computational_sequences["All Labels"]
        highlevel_dataset.align("All Labels")
        print("[OK] 标签对齐完成")

        # 移除不完整样本
        print("\n" + "=" * 70)
        print("步骤6: 移除不完整样本 (hard_unify)")
        print("=" * 70)
        highlevel_dataset.hard_unify()
        print("[OK] 不完整样本已移除")

        # 保存最终数据
        print("\n" + "=" * 70)
        print("步骤7: 保存最终对齐数据")
        print("=" * 70)
        highlevel_dataset.deploy("final_aligned", {
            'glove_vectors': 'glove_vectors',
            'COVAREP': 'COVAREP',
            'OpenFace_2': 'OpenFace_2',
            'All Labels': 'All Labels'
        })
        print("[OK] 最终数据已保存到 final_aligned/")

        # 获取张量（可选，用于验证）
        print("\n" + "=" * 70)
        print("步骤8: 验证数据")
        print("=" * 70)
        tensors = highlevel_dataset.get_tensors(
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
        for i in range(3):
            print(f"\n{fold_names[i].upper()} 数据集:")
            for csd in list(highlevel_dataset.keys()):
                print(f"  - {csd}: {tensors[i][csd].shape}")

        print("\n" + "=" * 70)
        print("[完成] 数据对齐处理完成！")
        print("=" * 70)
        print("\n下一步: 运行 python scripts/convert_csd_to_npz.py 转换数据")

        return 0

    except Exception as e:
        print(f"\n[错误] 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
