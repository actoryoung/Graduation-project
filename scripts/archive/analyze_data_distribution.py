# -*- coding: utf-8 -*-
"""
深度分析CMU-MOSEI原始数据分布
理解每个模态的物理意义和统计特性
"""
import sys
import os

SDK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CMU-MultimodalSDK')
sys.path.insert(0, SDK_PATH)

import numpy as np
from mmsdk import mmdatasdk
from tqdm import tqdm
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def analyze_distribution_in_depth():
    """深度分析各模态的数据分布"""
    print("=" * 70)
    print("CMU-MOSEI 原始数据分布深度分析")
    print("=" * 70)

    data_dir = os.path.join(project_root, 'data', 'mosei')

    # 加载原始数据
    highlevel_files = {
        'glove_vectors': os.path.join(data_dir, 'cmumosei_highlevel', 'CMU_MOSEI_TimestampedWordVectors.csd'),
        'COVAREP': os.path.join(data_dir, 'cmumosei_highlevel', 'CMU_MOSEI_COVAREP.csd'),
        'OpenFace_2': os.path.join(data_dir, 'cmumosei_highlevel', 'CMU_MOSEI_VisualOpenFace2.csd'),
    }

    print("\n加载数据...")
    highlevel_data = mmdatasdk.mmdataset(highlevel_files)

    glove_dict = highlevel_data.computational_sequences['glove_vectors'].data
    audio_dict = highlevel_data.computational_sequences['COVAREP'].data
    video_dict = highlevel_data.computational_sequences['OpenFace_2'].data

    # 获取训练集
    from mmsdk.mmdatasdk import cmu_mosei
    train_fold = set(cmu_mosei.standard_folds.standard_train_fold)

    glove_keys_list = list(glove_dict.keys())
    first_key = glove_keys_list[0]
    if '[' in first_key:
        keys_to_process = [k for k in glove_dict.keys() if k.split('[')[0] in train_fold]
    else:
        keys_to_process = [k for k in glove_dict.keys() if k in train_fold]

    # ============ 收集原始帧级别数据 ============
    print("\n收集原始帧级数据...")

    glove_frames = []
    audio_frames = []
    video_frames = []

    for key in tqdm(keys_to_process[:500], desc="分析前500个视频"):  # 采样以减少内存
        try:
            if key in glove_dict and key in audio_dict and key in video_dict:
                glove_frames.append(glove_dict[key]['features'])  # (seq, 300)
                audio_frames.append(audio_dict[key]['features'])  # (seq, 74)
                video_frames.append(video_dict[key]['features'])  # (seq, 713)
        except:
            pass

    # 合并所有帧
    glove_all = np.vstack(glove_frames)  # (total_frames, 300)
    audio_all = np.vstack(audio_frames)  # (total_frames, 74)
    video_all = np.vstack(video_frames)  # (total_frames, 713)

    print(f"\n数据量:")
    print(f"  GloVe总帧数: {glove_all.shape[0]:,}")
    print(f"  COVAREP总帧数: {audio_all.shape[0]:,}")
    print(f"  OpenFace总帧数: {video_all.shape[0]:,}")

    # ============ GloVe分析 ============
    print("\n" + "=" * 70)
    print("1. GloVe (词向量) 分析")
    print("=" * 70)

    glove_report = {
        'shape': glove_all.shape,
        'physical_meaning': '300维预训练词向量，捕捉语义相似性',
        'expected_range': '通常在[-2, 2]之间，大部分在[-1, 1]',
        'statistics': {}
    }

    for dim in [0, 50, 100, 150, 200, 250, 299]:
        data = glove_all[:, dim]
        glove_report['statistics'][f'dim_{dim}'] = {
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'median': float(np.median(data)),
            'p1': float(np.percentile(data, 1)),
            'p5': float(np.percentile(data, 5)),
            'p95': float(np.percentile(data, 95)),
            'p99': float(np.percentile(data, 99)),
        }

    print(f"\n随机7个维度的统计:")
    for dim, stats in glove_report['statistics'].items():
        print(f"  {dim}: min={stats['min']:.3f}, max={stats['max']:.3f}, "
              f"mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        print(f"        百分位: P1={stats['p1']:.3f}, P5={stats['p5']:.3f}, "
              f"P95={stats['p95']:.3f}, P99={stats['p99']:.3f}")

    # 全局统计
    print(f"\nGloVe全局统计:")
    print(f"  整体最小值: {np.min(glove_all):.3f}")
    print(f"  整体最大值: {np.max(glove_all):.3f}")
    print(f"  超过±2的比例: {((np.abs(glove_all) > 2).sum() / glove_all.size * 100):.2f}%")
    print(f"  超过±3的比例: {((np.abs(glove_all) > 3).sum() / glove_all.size * 100):.2f}%")

    # ============ COVAREP分析 ============
    print("\n" + "=" * 70)
    print("2. COVAREP (声学特征) 分析")
    print("=" * 70)

    glove_report = {
        'shape': audio_all.shape,
        'physical_meaning': '74维声学特征：pitch(1-12), formant(13-20), voicing(21-30), MFCC(31-54)等',
        'statistics': {}
    }

    # 按特征组分析
    print(f"\n特征分组:")
    print(f"  维度1-12:   F0及谐波 (基频)")
    print(f"  维度13-20:  Formant频率 (共振峰)")
    print(f"  维度21-30:  Voicing相关特征")
    print(f"  维度31-54:  MFCC (梅尔频率倒谱系数)")
    print(f"  维度55-74:  其他声学特征")

    # 分析每个维度
    for dim in range(audio_all.shape[1]):
        data = audio_all[:, dim]
        glove_report['statistics'][f'dim_{dim}'] = {
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'has_inf': bool(np.isinf(data).any()),
            'has_neg_inf': bool(np.isneginf(data).any()),
            'has_pos_inf': bool(np.isposinf(data).any()),
            'neg_inf_ratio': float((np.isneginf(data).sum() / len(data))) if np.isneginf(data).any() else 0.0,
        }

    # 找出有问题的维度
    problem_dims = []
    for dim, stats in glove_report['statistics'].items():
        if stats['has_inf']:
            problem_dims.append((dim, stats))

    print(f"\n包含-inf的维度: {len(problem_dims)}个")
    for dim_name, stats in problem_dims[:10]:  # 只显示前10个
        dim_idx = int(dim_name.split('_')[1])
        print(f"  维度{dim_idx}: -inf比例={stats['neg_inf_ratio']*100:.2f}%, "
              f"有效值范围=[{stats['min']:.2f}, {stats['max']:.2f}]")

    # 统计没有-inf的维度
    clean_dims = [d for d in range(audio_all.shape[1]) if not glove_report['statistics'][f'dim_{d}']['has_inf']]
    print(f"\n干净的维度数: {len(clean_dims)}/{audio_all.shape[1]}")

    if clean_dims:
        print(f"\n干净维度的统计示例 (前5个):")
        for dim in clean_dims[:5]:
            data = audio_all[:, dim]
            print(f"  维度{dim}: min={np.min(data):.2f}, max={np.max(data):.2f}, "
                  f"mean={np.mean(data):.2f}, std={np.std(data):.2f}")

    # ============ OpenFace分析 ============
    print("\n" + "=" * 70)
    print("3. OpenFace (面部特征) 分析")
    print("=" * 70)

    print(f"\n特征分组 (OpenFace2):")
    print(f"  维度0-2:     姿态 (pitch, yaw, roll)")
    print(f"  维度3-12:    面部landmarks (2D/3D坐标)")
    print(f"  维度13-16:   眼球gaze方向")
    print(f"  维度17-67:   Action Units (AU强度, 17个AU × 3 = 51维)")
    print(f"  维度68-712:  其他面部特征")

    # 找出极端值维度
    video_means = np.mean(video_all, axis=0)
    video_stds = np.std(video_all, axis=0)
    video_maxs = np.max(video_all, axis=0)
    video_mins = np.min(video_all, axis=0)

    # 找出std特别大的维度
    high_std_dims = np.where(video_stds > 1000)[0]
    print(f"\n标准差>1000的维度数: {len(high_std_dims)}")

    if len(high_std_dims) > 0:
        print(f"  示例 (前10个):")
        for dim in high_std_dims[:10]:
            print(f"    维度{dim}: std={video_stds[dim]:.0f}, "
                  f"range=[{video_mins[dim]:.0f}, {video_maxs[dim]:.0f}]")

    # 找出正常范围的维度
    normal_dims = np.where(video_stds < 100)[0]
    print(f"\n标准差<100的维度数: {len(normal_dims)}")
    if len(normal_dims) > 0:
        print(f"  示例 (前10个):")
        for dim in normal_dims[:10]:
            print(f"    维度{dim}: std={video_stds[dim]:.2f}, "
                  f"range=[{video_mins[dim]:.2f}, {video_maxs[dim]:.2f}]")

    # ============ 建议的归一化策略 ============
    print("\n" + "=" * 70)
    print("基于分析的建议归一化策略")
    print("=" * 70)

    print("""
1. GloVe (词向量):
   - 数据分布良好，范围约[-2, 2]
   - 策略: 标准Z-score归一化
   - 理由: GloVe本身就有合理的分布，无需特殊处理

2. COVAREP (声学特征):
   - 部分维度包含-inf (静音帧)
   - 策略:
     a) 对于有-inf的维度: 用中位数填充-inf，然后鲁棒归一化
     b) 对于干净的维度: 可以用Z-score或鲁棒归一化
   - 理由: -inf不是数据错误，而是物理上的"静音"，需要特殊处理

3. OpenFace (面部特征):
   - 包含极端大值 (某些维度std > 1000)
   - 策略:
     a) 分组归一化: 姿态、AU、landmarks分别处理
     b) 对高std维度使用对数变换或更强的裁剪
     c) 考虑物理意义: AU在[0, 5]范围，pose可以很大
   - 理由: 不同特征组有不同的物理单位和范围
    """)

    # 保存详细报告
    report = {
        'glove': glove_report,
        'covarep': glove_report,
        'openface': {
            'high_std_dims': int(len(high_std_dims)),
            'normal_dims': int(len(normal_dims)),
            'total_dims': video_all.shape[1]
        }
    }

    report_path = os.path.join(project_root, 'docs', 'data_distribution_analysis.json')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n详细报告已保存: {report_path}")


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    try:
        analyze_distribution_in_depth()
    except Exception as e:
        print(f"\n[ERROR] 分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
