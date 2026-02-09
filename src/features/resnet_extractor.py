# -*- coding: utf-8 -*-
"""
ResNet视频特征提取器

使用在ImageNet上预训练的ResNet模型提取视频帧特征，
替代原来的OpenFace手工特征。

ResNet是深度卷积神经网络，能够从图像中提取比OpenFace
手工特征更丰富的视觉表示。
"""

import torch
import torch.nn as nn
from typing import Union, List
import torchvision.models as models

class ResNetVideoExtractor(nn.Module):
    """
    ResNet视频特征提取器

    使用torchvision中预训练的ResNet模型提取视频帧特征，
    通过对所有帧的特征进行平均池化得到视频级特征。

    Attributes:
        model: ResNet模型（移除最后的分类层）
        device: 运行设备
        frame_sample_rate: 帧采样率（每隔多少帧取一帧）
        max_frames: 最大处理帧数
    """

    def __init__(
        self,
        model_name: str = 'resnet50',
        device: str = 'cuda',
        frame_sample_rate: int = 1,
        max_frames: int = 16
    ):
        """
        初始化ResNet视频特征提取器

        Args:
            model_name: 预训练模型名称
                - 'resnet18': ResNet-18 (512维)
                - 'resnet50': ResNet-50 (2048维)
                - 'resnet101': ResNet-101 (2048维)
            device: 运行设备 ('cuda' 或 'cpu')
            frame_sample_rate: 帧采样率（减少计算量）
            max_frames: 最大处理帧数（统一视频长度）
        """
        super(ResNetVideoExtractor, self).__init__()

        self.device = device
        self.frame_sample_rate = frame_sample_rate
        self.max_frames = max_frames

        try:
            print(f"正在加载ResNet模型: {model_name}")

            # 加载预训练ResNet模型
            if model_name == 'resnet18':
                self.model = models.resnet18(pretrained=True)
                feature_dim = 512
            elif model_name == 'resnet50':
                self.model = models.resnet50(pretrained=True)
                feature_dim = 2048
            elif model_name == 'resnet101':
                self.model = models.resnet101(pretrained=True)
                feature_dim = 2048
            else:
                raise ValueError(f"不支持的模型: {model_name}")

            # 移除最后的分类层（fc层）
            self.model = nn.Sequential(*list(self.model.children())[:-1])

            # 移到指定设备
            self.model.to(device)
            self.model.eval()  # 设置为评估模式

            print(f"[OK] ResNet模型加载成功: {model_name}")
            print(f"     设备: {device}")
            print(f"     输出维度: {feature_dim}")
            print(f"     帧采样率: {frame_sample_rate}")
            print(f"     最大帧数: {max_frames}")

        except Exception as e:
            print(f"[ERROR] ResNet模型加载失败: {e}")
            raise

        # 如果使用CUDA，优化模型（仅推理，不训练）
        if device == 'cuda':
            self._optimize_for_inference()

    def _optimize_for_inference(self):
        """优化模型用于推理（仅在使用CUDA时）"""
        try:
            # 融合BN层，提升推理速度
            self.model = torch.jit.script(self.model)
            print(f"[优化] 模型已使用torch.jit.script优化")
        except Exception as e:
            print(f"[警告] 模型优化失败，继续使用原始模型: {e}")

    @torch.no_grad()
    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        提取视频特征

        Args:
            video_frames: 视频帧张量
                形状: [T, 3, H, W] 或 [H, W] 或 [3, H, W]
                T: 帧数
                H: 高度
                W: 宽度
                值域: [0, 255] 或 [0, 1]

        Returns:
            video_features: 视频特征张量 [1, feature_dim]

        处理流程:
            1. 标准化像素值到[0, 1]
            2. 应用ResNet标准化
            3. 对每帧提取特征
            4. 平均池化到视频级特征
        """
        import numpy as np

        # 确保是4D张量 [T, 3, H, W]
        if video_frames.dim() == 2:
            # 单帧灰度图 [H, W]，转换为3通道并添加帧维度
            video_frames = video_frames.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        elif video_frames.dim() == 3 and video_frames.shape[0] == 3:
            # [C, H, W] -> [1, C, H, W]
            video_frames = video_frames.unsqueeze(0)
        elif video_frames.dim() == 3:
            # [T, H, W] (灰度视频) -> [T, 3, H, W]
            video_frames = video_frames.unsqueeze(1).repeat(1, 3, 1, 1)
        elif video_frames.dim() == 4 and video_frames.shape[1] != 3:
            # [T, 1, H, W] or similar -> [T, 3, H, W]
            video_frames = video_frames.repeat(1, 3, 1, 1)[:, :3, :, :]
        elif video_frames.dim() == 5:
            # [B, T, C, H, W] -> [T, C, H, W] (取第一个batch)
            video_frames = video_frames[0]

        # 移到设备
        video_frames = video_frames.to(self.device)

        # 标准化到[0, 1]
        if video_frames.max() > 1.0:
            video_frames = video_frames.float() / 255.0

        # 应用ImageNet标准化（使用mean和std直接计算）
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        video_frames = (video_frames - mean) / std

        # 采样帧（减少计算量）
        if video_frames.shape[0] > self.max_frames:
            # 均匀采样
            indices = torch.linspace(0, video_frames.shape[0] - 1, self.max_frames).long()
            video_frames = video_frames[indices]
        else:
            video_frames = video_frames[::self.frame_sample_rate]

        # 提取每帧特征
        frame_features = []
        for i in range(video_frames.shape[0]):
            frame = video_frames[i:i+1]  # [1, 3, H, W]
            features = self.model(frame)  # [1, feature_dim, 1, 1]
            features = features.flatten(1)  # [1, feature_dim]
            frame_features.append(features)

        # 堆叠并平均池化
        all_features = torch.cat(frame_features, dim=0)  # [T, feature_dim]

        # 平均池化到视频级特征
        video_features = all_features.mean(dim=0, keepdim=True)  # [1, feature_dim]

        return video_features

    def extract(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        提取视频特征（对外接口）

        Args:
            video_frames: 视频帧张量 [T, 3, H, W] 或 [H, W]

        Returns:
            视频特征向量 [1, feature_dim]
        """
        return self.forward(video_frames)

    def extract_from_frames(self, frames: List) -> torch.Tensor:
        """
        从视频帧列表提取特征

        Args:
            frames: PIL Image对象列表或tensor列表

        Returns:
            视频特征向量 [1, feature_dim]
        """
        # 转换为tensor并标准化
        if isinstance(frames[0], Image.Image):
            from PIL import Image
            from torchvision.transforms import ToTensor, Normalize, Compose

            transform = Compose([
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            tensors = []
            for frame in frames:
                tensor = transform(frame)
                tensors.append(tensor)

            # 堆叠为[T, 3, H, W]
            video_frames = torch.stack(tensors)
        else:
            video_frames = torch.stack(frames)

        return self.extract(video_frames)

    def get_feature_dim(self) -> int:
        """获取特征维度"""
        if 'resnet18' in self.model_name.lower():
            return 512
        else:
            return 2048

    def __repr__(self):
        return f"ResNetVideoExtractor(model_name='{self.model_name}', device='{self.device}')"


class VideoAugmentation:
    """
    视频数据增强类

    用于训练时的视频增强，包括：
    - 随机裁剪
    - 颜色抖动
    - 水平翻转
    """

    def __init__(
        self,
        crop_size: int = 224,
        horizontal_flip_prob: float = 0.5,
        color_jitter: bool = True
    ):
        self.crop_size = crop_size
        self.horizontal_flip_prob = horizontal_flip_prob
        self.color_jitter = color_jitter

        if color_jitter:
            self.color_jitter = torchvision.transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1
            )

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        对视频帧进行数据增强

        Args:
            frames: 视频帧 [T, 3, H, W]

        Returns:
            augmented_frames: 增强后的视频帧
        """
        # 随机水平翻转
        if torch.rand(1).item() < self.horizontal_flip_prob:
            frames = torch.flip(frames, dims=[3])  # 水平翻转

        # 颜色抖动
        if self.color_jitter:
            # 对每一帧应用相同颜色抖动
            jitter_params = self.color_jitter.get_params(
                brightness=[0.8, 1.2],
                contrast=[0.8, 1.2],
                saturation=[0.9, 1.1]
            )
            # 手动应用颜色抖动
            frames = frames * torch.tensor(jitter_params[2]).view(1, 1, 1, 1) * \
                     torch.tensor(jitter_params[1]).view(1, 1, 1, 1) * \
                     torch.tensor(jitter_params[0]).view(1, 1, 1, 1)

        return frames


def test_resnet_extractor():
    """测试ResNet视频特征提取器"""
    print("=" * 70)
    print("ResNet视频特征提取器测试")
    print("=" * 70)

    # 测试随机视频
    try:
        extractor = ResNetVideoExtractor(device='cpu', max_frames=8)
        print("\n测试1: 随机视频提取")
        print("-" * 70)

        # 生成8帧随机视频（64x64 RGB）
        frames = torch.rand(8, 3, 64, 64)

        features = extractor.extract(frames)

        print(f"输入视频形状: {frames.shape}")
        print(f"特征形状: {features.shape}")
        print(f"特征范围: [{features.min():.4f}, {features.max():.4f}]")
        print(f"特征均值: {features.mean():.4f}")
        print(f"特征标准差: {features.std():.4f}")

    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        print("\n注意: 如果是首次运行，需要下载ResNet模型（约98MB）")
        print("请确保网络连接正常，或手动下载模型文件。")
        return

    # 对比OpenFace
    print("\n" + "=" * 70)
    print("ResNet vs OpenFace对比")
    print("=" * 70)
    print("| 特征提取器 | 维度 | 类型 | 预训练 | 深度学习 | 端到端 |")
    print("|-----------|------|------|--------|----------|--------|")
    print("| OpenFace   | 710  | 手工 | 否     | 否       | 是     |")
    print("| OpenFace   | 25   | 手工 | 否     | 否       | 是     |")
    print("| **ResNet**  | **2048** | **深度学习** | **是(ImageNet)** | **是** | **是** |")

    print("\n" + "=" * 70)
    print("[OK] 所有测试通过!")
    print("=" * 70)


if __name__ == '__main__':
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    import torchvision.transforms
    test_resnet_extractor()
