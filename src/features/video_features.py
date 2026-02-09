# -*- coding: utf-8 -*-
"""
OpenFace视频特征提取器模块

本模块实现了基于OpenFace的视频特征提取器，用于从视频文件中提取
面部动作单元(Action Units)特征。OpenFace能够检测面部关键点并计算
17个动作单元的强度，结合头部姿态和视线方向形成25维特征向量。

主要功能:
    - 使用OpenFace工具提取面部动作单元特征
    - 自动提取视频帧并处理
    - 输出25维特征向量（17维AU + 6维姿态 + 2维注视）
    - 支持降级方案（OpenFace失败时返回零向量）
    - 完整异常处理

依赖:
    - opencv-python: 视频帧提取
    - OpenFace: 面部特征提取工具（外部CLI）
    - numpy: 数组计算

OpenFace安装:
    请访问 https://github.com/TadasBaltrusaitis/OpenFace 下载并安装

示例:
    >>> from src.features.video_features import OpenFaceFeatureExtractor
    >>> extractor = OpenFaceFeatureExtractor()
    >>> features = extractor.extract_from_raw('path/to/video.mp4')
    >>> print(features.shape)
    (25,)
"""

# -*- coding: utf-8 -*-

import os
import subprocess
import tempfile
import warnings
from typing import Any, List, Optional

import numpy as np

from config import Config, VIDEO_MODEL, VIDEO_DIM, NUM_VIDEO_FRAMES, IMAGE_SIZE
from src.features.base import FeatureExtractor


class OpenFaceFeatureExtractor(FeatureExtractor):
    """
    OpenFace视频特征提取器

    使用OpenFace工具从视频中提取面部动作单元(Action Units)特征。
    特征包括17个AU的强度、6维头部姿态（3个位置+3个旋转）和2维视线方向。

    类属性:
        model_name (str): 模型名称 ('openface')
        feature_dim (int): 输出特征维度（25）
        num_frames (int): 提取的帧数
        image_size (int): 图像尺寸

    AU特征说明:
        AU01, AU02, AU04, AU05, AU06, AU07, AU09, AU10, AU12,
        AU14, AU15, AU17, AU20, AU23, AU25, AU26, AU45 (共17个)

    示例:
        >>> extractor = OpenFaceFeatureExtractor()
        >>> features = extractor.extract_from_raw('video.mp4')
        >>> print(features.shape)
        (25,)
    """

    def __init__(self, device: str = None, openface_path: str = None):
        """
        初始化OpenFace特征提取器

        设置OpenFace可执行文件路径。OpenFace需要单独安装。

        Args:
            device: 计算设备（视频特征提取仅支持CPU）
            openface_path: OpenFace可执行文件路径，默认为None时自动查找

        Raises:
            ValueError: 当device不是'cpu'时（OpenFace仅支持CPU）
            RuntimeError: 当OpenFace未安装或无法访问时
        """
        # OpenFace仅支持CPU
        if device is not None and device != 'cpu':
            warnings.warn("OpenFace仅支持CPU运行，将忽略device参数")

        super().__init__(device='cpu')

        self.model_name = VIDEO_MODEL
        self.feature_dim = VIDEO_DIM
        self.num_frames = NUM_VIDEO_FRAMES
        self.image_size = IMAGE_SIZE

        # 动态导入cv2
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            raise ImportError(
                "需要安装opencv-python库。请运行: pip install opencv-python"
            )

        # 设置OpenFace路径
        self.openface_path = openface_path or self._find_openface()

    def _find_openface(self) -> Optional[str]:
        """
        查找系统中的OpenFace可执行文件

        按优先级在以下位置查找：
        1. 环境变量 OPENFACE_PATH
        2. 常见安装路径

        Returns:
            找到的OpenFace路径，如果未找到则返回None
        """
        # 检查环境变量
        env_path = os.environ.get('OPENFACE_PATH')
        if env_path and os.path.exists(env_path):
            return env_path

        # 常见安装路径
        common_paths = [
            os.path.join(os.path.expanduser('~'), 'OpenFace', 'build', 'bin', 'FeatureExtraction'),
            '/usr/local/bin/FeatureExtraction',
            'C:\\OpenFace\\build\\bin\\Release\\FeatureExtraction.exe',
        ]

        for path in common_paths:
            if os.path.exists(path):
                return path

        return None

    def preprocess(self, raw_input: str) -> str:
        """
        预处理视频输入

        验证视频文件路径并检查文件格式。

        Args:
            raw_input: 视频文件路径

        Returns:
            str: 验证后的视频文件路径

        Raises:
            FileNotFoundError: 当视频文件不存在时
            ValueError: 当输入不是字符串时

        示例:
            >>> extractor = OpenFaceFeatureExtractor()
            >>> path = extractor.preprocess('video.mp4')
            >>> print(path)
            video.mp4
        """
        # 验证输入类型
        if not isinstance(raw_input, str):
            raise TypeError(
                f"输入必须是视频文件路径（字符串），得到: {type(raw_input)}"
            )

        # 验证文件存在
        if not os.path.exists(raw_input):
            raise FileNotFoundError(f"视频文件不存在: {raw_input}")

        # 验证文件扩展名
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        _, ext = os.path.splitext(raw_input)
        if ext.lower() not in valid_extensions:
            warnings.warn(
                f"文件扩展名 '{ext}' 不在常见视频格式列表中，"
                f"但仍会尝试处理"
            )

        return raw_input

    def extract(self, video_path: str) -> np.ndarray:
        """
        从视频中提取OpenFace特征

        执行以下步骤：
        1. 使用OpenFace处理视频，提取AU特征
        2. 读取生成的CSV文件
        3. 提取17个AU强度 + 6维姿态 + 2维注视
        4. 对所有帧进行平均池化

        Args:
            video_path: 视频文件路径

        Returns:
            np.ndarray: 25维特征向量，dtype为float32
                        如果OpenFace失败，返回零向量

        示例:
            >>> extractor = OpenFaceFeatureExtractor()
            >>> features = extractor.extract('video.mp4')
            >>> print(features.shape)
            (25,)
        """
        # 降级方案：直接返回零向量
        return self._extract_with_fallback(video_path)

    def _extract_with_fallback(self, video_path: str) -> np.ndarray:
        """
        使用降级方案提取特征

        如果OpenFace可用则正常提取，否则返回零向量。

        Args:
            video_path: 视频文件路径

        Returns:
            np.ndarray: 25维特征向量
        """
        try:
            return self._extract_openface_features(video_path)
        except Exception as e:
            warnings.warn(
                f"OpenFace特征提取失败: {e}，使用零向量作为降级方案"
            )
            return np.zeros(self.feature_dim, dtype=np.float32)

    def _extract_openface_features(self, video_path: str) -> np.ndarray:
        """
        使用OpenFace提取特征的内部方法

        Args:
            video_path: 视频文件路径

        Returns:
            np.ndarray: 25维特征向量

        Raises:
            RuntimeError: 当OpenFace执行失败时
        """
        # 检查OpenFace是否可用
        if self.openface_path is None:
            raise RuntimeError(
                "OpenFace未找到。请安装OpenFace并设置OPENFACE_PATH环境变量，"
                "或在初始化时传入openface_path参数。"
            )

        # 创建临时输出目录
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, 'output')
            os.makedirs(output_dir, exist_ok=True)

            # 构建OpenFace命令
            cmd = [
                self.openface_path,
                '-f', video_path,
                '-out_dir', output_dir,
                '-2Dfp', '',  # 不输出2D landmarks
                '-3Dfp', '',  # 不输出3D landmarks
                '-pdmparams', '',  # 不输出PDM参数
                '-pose', '',  # 不输出姿态
                '-aus',  # 输出动作单元
                '-gaze', '',  # 不输出视线
                '-format', 'csv',  # CSV格式输出
            ]

            # 执行OpenFace
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=300,  # 5分钟超时
                    check=False
                )

                if result.returncode != 0:
                    raise RuntimeError(
                        f"OpenFace执行失败，返回码: {result.returncode}\n"
                        f"stderr: {result.stderr.decode('utf-8', errors='ignore')}"
                    )

            except subprocess.TimeoutExpired:
                raise RuntimeError("OpenFace执行超时（超过5分钟）")
            except FileNotFoundError:
                raise RuntimeError(f"OpenFace可执行文件未找到: {self.openface_path}")

            # 读取输出CSV
            csv_path = os.path.join(output_dir, os.path.basename(video_path) + '.csv')
            if not os.path.exists(csv_path):
                raise RuntimeError(f"OpenFace未生成输出文件: {csv_path}")

            # 解析CSV并提取特征
            return self._parse_openface_csv(csv_path)

    def _parse_openface_csv(self, csv_path: str) -> np.ndarray:
        """
        解析OpenFace输出的CSV文件

        提取以下特征：
        - 17个AU强度 (AU01, AU02, AU04, ..., AU45)
        - 6维头部姿态 (pose_Tx, pose_Ty, pose_Tz, pose_Rx, pose_Ry, pose_Rz)
        - 2维视线方向 (gaze_angle_x, gaze_angle_y)

        Args:
            csv_path: CSV文件路径

        Returns:
            np.ndarray: 平均池化后的25维特征向量

        Raises:
            RuntimeError: 当CSV解析失败时
        """
        try:
            import pandas as pd
        except ImportError:
            # 如果没有pandas，使用标准库csv
            import csv
            return self._parse_csv_with_csv_module(csv_path)

        # 使用pandas读取CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise RuntimeError(f"读取CSV文件失败: {e}")

        # AU特征列名
        au_columns = [
            ' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r',
            ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r',
            ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r',
            ' AU26_r', ' AU45_r'
        ]

        # 姿态特征列名
        pose_columns = [
            ' pose_Tx', ' pose_Ty', ' pose_Tz',
            ' pose_Rx', ' pose_Ry', ' pose_Rz'
        ]

        # 视线特征列名
        gaze_columns = [
            ' gaze_angle_x', ' gaze_angle_y'
        ]

        # 组合所有特征列
        feature_columns = au_columns + pose_columns + gaze_columns

        # 检查列是否存在
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            warnings.warn(
                f"CSV中缺少以下列: {missing_cols}，"
                f"将使用可用特征"
            )
            feature_columns = [col for col in feature_columns if col in df.columns]

        if not feature_columns:
            raise RuntimeError("CSV中未找到任何特征列")

        # 提取特征并对所有帧取平均
        features = df[feature_columns].values  # (num_frames, num_features)

        # 处理NaN值（用均值填充）
        if np.isnan(features).any():
            col_means = np.nanmean(features, axis=0)
            for i in range(features.shape[1]):
                features[np.isnan(features[:, i]), i] = col_means[i]

        # 平均池化
        avg_features = features.mean(axis=0)  # (num_features,)

        # 确保输出维度正确（填充或截断到25维）
        if len(avg_features) < self.feature_dim:
            # 填充零
            padded = np.zeros(self.feature_dim, dtype=np.float32)
            padded[:len(avg_features)] = avg_features.astype(np.float32)
            return padded
        elif len(avg_features) > self.feature_dim:
            # 截断
            return avg_features[:self.feature_dim].astype(np.float32)
        else:
            return avg_features.astype(np.float32)

    def _parse_csv_with_csv_module(self, csv_path: str) -> np.ndarray:
        """
        使用标准csv模块解析CSV（当pandas不可用时）

        Args:
            csv_path: CSV文件路径

        Returns:
            np.ndarray: 25维特征向量
        """
        import csv

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise RuntimeError("CSV文件为空")

        # 提取特征
        features_list = []
        for row in rows:
            frame_features = []
            # AU特征
            for i in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]:
                key = f' AU{i}_r'
                if key in row:
                    try:
                        frame_features.append(float(row[key]))
                    except ValueError:
                        frame_features.append(0.0)
            # 简化版本：只返回25维零向量（因为没有pandas时解析复杂）
            if len(frame_features) < 25:
                frame_features.extend([0.0] * (25 - len(frame_features)))
            features_list.append(frame_features[:25])

        if not features_list:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # 平均池化
        features_array = np.array(features_list, dtype=np.float32)
        return features_array.mean(axis=0)

    def extract_frames_opencv(self, video_path: str, num_frames: int = None) -> List[np.ndarray]:
        """
        使用OpenCV从视频中提取帧

        本方法作为OpenFace的替代方案，使用OpenCV直接提取视频帧。
        主要用于：
        1. 视频预览
        2. 降级方案的一部分
        3. 其他需要帧数据的功能

        Args:
            video_path: 视频文件路径
            num_frames: 提取的帧数，默认为self.num_frames

        Returns:
            List[np.ndarray]: 提取的帧列表，每个元素为HxWxC的numpy数组

        Raises:
            RuntimeError: 当视频打开失败时

        示例:
            >>> extractor = OpenFaceFeatureExtractor()
            >>> frames = extractor.extract_frames_opencv('video.mp4', num_frames=10)
            >>> print(len(frames))
            10
        """
        num_frames = num_frames or self.num_frames

        # 打开视频
        cap = self.cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")

        try:
            # 获取视频信息
            total_frames = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(self.cv2.CAP_PROP_FPS)

            # 计算采样间隔
            if total_frames <= num_frames:
                # 如果帧数不足，提取所有帧
                frame_indices = range(total_frames)
            else:
                # 均匀采样
                interval = total_frames // num_frames
                frame_indices = range(0, total_frames, interval)[:num_frames]

            # 提取帧
            frames = []
            for idx in frame_indices:
                cap.set(self.cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)

            if len(frames) < num_frames:
                warnings.warn(
                    f"实际提取的帧数({len(frames)})少于请求的帧数({num_frames})"
                )

        finally:
            cap.release()

        return frames


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("OpenFace视频特征提取器测试")
    print("=" * 60)

    # 测试1: 初始化
    print("\n[测试1] 初始化特征提取器")
    print("-" * 40)
    try:
        extractor = OpenFaceFeatureExtractor()
        print(f"✓ 初始化成功")
        print(f"  - 设备: {extractor.device}")
        print(f"  - 模型: {extractor.model_name}")
        print(f"  - 特征维度: {extractor.feature_dim}")
        print(f"  - 提取帧数: {extractor.num_frames}")
        print(f"  - OpenFace路径: {extractor.openface_path or '未找到（将使用降级方案）'}")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        exit(1)

    # 测试2: 降级方案测试
    print("\n[测试2] 降级方案测试")
    print("-" * 40)
    print("注意: 由于OpenFace可能未安装，此测试主要验证降级方案")
    try:
        # 使用不存在的文件路径
        features = extractor.extract_from_raw("nonexistent_video.mp4")
        print(f"✓ 降级方案返回特征")
        print(f"  - 特征形状: {features.shape}")
        assert features.shape == (25,), f"特征维度应为(25,)，得到{features.shape}"
        print(f"  ✓ 维度正确: (25,)")
        assert features.dtype == np.float32
        print(f"  ✓ 数据类型正确: float32")
        assert np.allclose(features, np.zeros(25))
        print(f"  ✓ 降级方案返回零向量（符合预期）")

    except FileNotFoundError:
        print(f"✓ 正确捕获FileNotFoundError（文件不存在）")
    except Exception as e:
        print(f"✗ 降级方案测试失败: {e}")

    # 测试3: 创建测试视频（如果可能）
    print("\n[测试3] 创建测试视频")
    print("-" * 40)
    import tempfile

    try:
        # 创建一个简单的测试视频
        temp_dir = tempfile.gettempdir()
        test_video_path = os.path.join(temp_dir, "test_video.mp4")

        # 使用OpenCV创建测试视频
        fourcc = self.cv2.VideoWriter_fourcc(*'mp4v')
        out = self.cv2.VideoWriter(
            test_video_path,
            fourcc,
            10.0,  # fps
            (320, 240)  # 尺寸
        )

        # 写入30帧（随机噪声）
        for _ in range(30):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            out.write(frame)

        out.release()
        print(f"✓ 测试视频创建成功: {test_video_path}")

    except Exception as e:
        print(f"✗ 创建测试视频失败: {e}")
        test_video_path = None

    # 测试4: 预处理测试
    print("\n[测试4] 预处理测试")
    print("-" * 40)
    if test_video_path:
        try:
            path = extractor.preprocess(test_video_path)
            print(f"✓ 预处理成功: {path}")
        except Exception as e:
            print(f"✗ 预处理失败: {e}")
    else:
        print("跳过（测试视频未创建）")

    # 测试5: OpenCV帧提取测试
    print("\n[测试5] OpenCV帧提取测试")
    print("-" * 40)
    if test_video_path:
        try:
            frames = extractor.extract_frames_opencv(test_video_path, num_frames=10)
            print(f"✓ 帧提取成功")
            print(f"  - 提取帧数: {len(frames)}")
            if frames:
                print(f"  - 第一帧形状: {frames[0].shape}")
                print(f"  ✓ 帧格式正确 (H, W, C)")
        except Exception as e:
            print(f"✗ 帧提取失败: {e}")
    else:
        print("跳过（测试视频未创建）")

    # 测试6: 特征提取（使用降级方案）
    print("\n[测试6] 特征提取测试（降级方案）")
    print("-" * 40)
    if test_video_path:
        try:
            features = extractor.extract_from_raw(test_video_path)
            print(f"✓ 特征提取成功")
            print(f"  - 特征形状: {features.shape}")
            assert features.shape == (25,)
            print(f"  ✓ 维度正确: (25,)")
            print(f"  - 数据类型: {features.dtype}")
            print(f"  - 均值: {features.mean():.6f}")
            print(f"  - 标准差: {features.std():.6f}")
            # 降级方案应该返回零向量
            if np.allclose(features, np.zeros(25)):
                print(f"  ℹ 使用降级方案（OpenFace未配置或失败）")
        except Exception as e:
            print(f"✗ 特征提取失败: {e}")
    else:
        print("跳过（测试视频未创建）")

    # 测试7: 异常处理
    print("\n[测试7] 异常处理测试")
    print("-" * 40)

    # 错误的输入类型
    print("测试错误的输入类型:")
    try:
        extractor.preprocess(12345)
        print("  ✗ 应该抛出TypeError但没有")
    except TypeError as e:
        print(f"  ✓ 正确捕获TypeError")

    # 不存在的文件
    print("\n测试不存在的文件:")
    try:
        extractor.preprocess('nonexistent_file.mp4')
        print("  ✗ 应该抛出FileNotFoundError但没有")
    except FileNotFoundError as e:
        print(f"  ✓ 正确捕获FileNotFoundError")

    # 清理临时文件
    if test_video_path and os.path.exists(test_video_path):
        try:
            os.remove(test_video_path)
            print(f"\n✓ 临时文件已清理")
        except:
            pass

    # 测试总结
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print("\n使用方法:")
    print("  from src.features.video_features import OpenFaceFeatureExtractor")
    print("  extractor = OpenFaceFeatureExtractor()")
    print("  features = extractor.extract_from_raw('path/to/video.mp4')")
    print(f"  # features.shape = (25,)")
    print("\n注意事项:")
    print("  - OpenFace需要单独安装")
    print("  - 设置环境变量 OPENFACE_PATH 指向OpenFace可执行文件")
    print("  - 如果OpenFace不可用，将自动返回零向量")
    print("  - 可使用 extract_frames_opencv() 方法仅提取视频帧")
