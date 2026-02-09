# -*- coding: utf-8 -*-
"""
训练基线多模态情感分析模型

使用现有的手工特征（GloVe 300维 + COVAREP 74维 + OpenFace 710维）
作为基线对比模型。
"""

import sys
import os
import argparse

SDK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CMU-MultimodalSDK')
sys.path.insert(0, SDK_PATH)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import Config, NUM_CLASSES, EMOTIONS, DEVICE
import torch.nn.functional as F


# =============================================================================
# Focal Loss（用于处理类别不平衡）
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance

    Args:
        alpha: Weighting factor (default: 1)
        gamma: Focusing parameter, default 2
    """

    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


# =============================================================================
# 数据集类
# =============================================================================

class MOSEIBaselineDataset(Dataset):
    """CMU-MOSEI基线数据集（使用现有特征）"""

    def __init__(self, data_path: str, split: str = 'train', handle_inf=True):
        self.data_path = data_path
        self.split = split
        self.handle_inf = handle_inf

        # 加载数据
        data = np.load(data_path, allow_pickle=True)

        self.text_features = data['text_features'].astype(np.float32)  # [N, 300]
        self.audio_features = data['audio_features'].astype(np.float32)  # [N, 74]
        self.video_features = data['video_features'].astype(np.float32)  # [N, 710]
        self.labels = data['labels'].astype(np.int64)

        # 处理-inf值（仅当handle_inf=True时）
        if self.handle_inf:
            self.audio_features[np.isneginf(self.audio_features)] = 0.0
            self.audio_features[np.isposinf(self.audio_features)] = 0.0
            self.video_features[np.isneginf(self.video_features)] = 0.0
            self.video_features[np.isposinf(self.video_features)] = 0.0

        print(f"[{split.upper()}] 加载了 {len(self.labels)} 个样本")
        print(f"  文本特征: {self.text_features.shape}")
        print(f"  音频特征: {self.audio_features.shape}")
        print(f"  视频特征: {self.video_features.shape}")
        if self.handle_inf:
            print(f"  已处理: ±inf替换为0")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'text': torch.from_numpy(self.text_features[idx]),
            'audio': torch.from_numpy(self.audio_features[idx]),
            'video': torch.from_numpy(self.video_features[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# =============================================================================
# 基线融合模型
# =============================================================================

class BaselineFusionModel(nn.Module):
    """基线多模态融合模型（使用手工特征）"""

    def __init__(
        self,
        text_dim=300,
        audio_dim=74,
        video_dim=710,
        fusion_dim=256,
        num_classes=7,
        dropout_rate=0.3
    ):
        super(BaselineFusionModel, self).__init__()

        self.input_dim = text_dim + audio_dim + video_dim

        # 模态特定编码器（用于降维）
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 融合层
        total_dim = 128 + 64 + 128
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.65)
        )

        # 分类器
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

    def forward(self, text, audio, video):
        # 模态编码
        t_feat = self.text_encoder(text)
        a_feat = self.audio_encoder(audio)
        v_feat = self.video_encoder(video)

        # 拼接并融合
        fused = torch.cat([t_feat, a_feat, v_feat], dim=1)
        features = self.fusion(fused)
        logits = self.classifier(features)

        return logits


class AttentionFusionModel(nn.Module):
    """S3: 跨模态注意力融合模型"""

    def __init__(
        self,
        text_dim=300,
        audio_dim=74,
        video_dim=710,
        fusion_dim=256,
        num_classes=7,
        dropout_rate=0.3,
        num_heads=4
    ):
        super(AttentionFusionModel, self).__init__()

        # 模态编码器（与基线相同）
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=num_heads,
            batch_first=True
        )

        # 投影到64维（注意力维度）
        self.text_proj = nn.Linear(128, 64)
        self.audio_proj = nn.Linear(64, 64)
        self.video_proj = nn.Linear(128, 64)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(64 * 3, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.65)
        )

        # 分类器
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

    def forward(self, text, audio, video):
        # 编码
        t = self.text_encoder(text)
        a = self.audio_encoder(audio)
        v = self.video_encoder(video)

        # 投影到注意力维度
        t_proj = self.text_proj(t)
        a_proj = self.audio_proj(a)
        v_proj = self.video_proj(v)

        # 堆叠为序列 (batch, 3, 64)
        features = torch.stack([t_proj, a_proj, v_proj], dim=1)

        # 跨模态注意力
        attended, _ = self.cross_attention(features, features, features)

        # 展平并融合
        fused = attended.flatten(1)
        features = self.fusion(fused)
        logits = self.classifier(features)

        return logits


class TransformerFusionModel(nn.Module):
    """S5: Transformer融合模型"""

    def __init__(
        self,
        text_dim=300,
        audio_dim=74,
        video_dim=710,
        fusion_dim=256,
        num_classes=7,
        dropout_rate=0.3,
        num_heads=4,
        num_layers=2
    ):
        super(TransformerFusionModel, self).__init__()

        # 模态编码器（与基线相同）
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 投影到统一维度（64维）
        self.text_proj = nn.Linear(128, 64)
        self.audio_proj = nn.Linear(64, 64)
        self.video_proj = nn.Linear(128, 64)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(64 * 3, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.65)
        )

        # 分类器
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

    def forward(self, text, audio, video):
        # 编码
        t = self.text_encoder(text)
        a = self.audio_encoder(audio)
        v = self.video_encoder(video)

        # 投影到统一维度
        t_proj = self.text_proj(t)
        a_proj = self.audio_proj(a)
        v_proj = self.video_proj(v)

        # 堆叠为序列 (batch, 3, 64)
        features = torch.stack([t_proj, a_proj, v_proj], dim=1)

        # Transformer编码
        transformed = self.transformer(features)

        # 展平并融合
        fused = transformed.flatten(1)
        features = self.fusion(fused)
        logits = self.classifier(features)

        return logits


# =============================================================================
# 训练和验证函数
# =============================================================================

def create_data_loaders(data_dir, batch_size=32):
    """创建数据加载器"""
    train_path = os.path.join(data_dir, 'train.npz')
    val_path = os.path.join(data_dir, 'val.npz')
    test_path = os.path.join(data_dir, 'test.npz')

    train_dataset = MOSEIBaselineDataset(train_path, split='train')
    val_dataset = MOSEIBaselineDataset(val_path, split='val')
    test_dataset = MOSEIBaselineDataset(test_path, split='test')

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        text = batch['text'].to(device)
        audio = batch['audio'].to(device)
        video = batch['video'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(text, audio, video)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            labels = batch['label'].to(device)

            outputs = model(text, audio, video)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='训练基线多模态情感分析模型')
    parser.add_argument('--data-dir', type=str, default='data/mosei/processed_cleaned_correct',
                       help='数据目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--fusion-dim', type=int, default=256, help='融合维度')
    parser.add_argument('--video-dim', type=int, default=None, help='视频维度(自动检测)')
    parser.add_argument('--num-classes', type=int, default=None, help='类别数(默认使用config)')
    parser.add_argument('--use-focal-loss', action='store_true', help='使用Focal Loss')
    parser.add_argument('--fusion-method', type=str, default='baseline',
                       choices=['baseline', 'attention', 'transformer'],
                       help='融合方法: baseline(拼接), attention(S3), transformer(S5)')
    parser.add_argument('--use-class-weights', action='store_true', help='使用类别权重处理不平衡')
    parser.add_argument('--weight-strategy', type=str, default='sqrt',
                       choices=['sqrt', 'linear', 'square', 'log'],
                       help='类别权重策略: sqrt(平方根), linear(线性), square(平方), log(对数)')

    args = parser.parse_args()

    # 自动检测视频维度
    if args.video_dim is None:
        train_path = os.path.join(args.data_dir, 'train.npz')
        if os.path.exists(train_path):
            data = np.load(train_path, allow_pickle=True)
            args.video_dim = data['video_features'].shape[1]
        else:
            args.video_dim = 710  # 默认值

    # 确定类别数
    num_classes = args.num_classes if args.num_classes is not None else NUM_CLASSES

    # 确定损失函数
    loss_type = "Focal Loss" if args.use_focal_loss else "CrossEntropy"

    # 确定融合方法名称
    fusion_names = {
        'baseline': '特征拼接',
        'attention': '跨模态注意力 (S3)',
        'transformer': 'Transformer融合 (S5)'
    }

    print("=" * 70)
    print("多模态情感分析模型训练")
    print(f"使用特征: GloVe(300) + COVAREP(74) + OpenFace({args.video_dim}) = {300+74+args.video_dim}维")
    print(f"融合方法: {fusion_names[args.fusion_method]}")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  数据目录: {args.data_dir}")
    print(f"  类别数: {num_classes}")
    print(f"  融合方法: {args.fusion_method}")
    print(f"  损失函数: {loss_type}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  设备: {args.device}")
    print(f"  融合维度: {args.fusion_dim}")
    print(f"  视频维度: {args.video_dim}")

    # 创建数据加载器
    print("\n加载数据...")
    train_loader, val_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)

    # 计算类别权重（如果启用）
    class_weights = None
    if args.use_class_weights:
        print(f"\n计算类别权重 (策略: {args.weight_strategy})...")
        # 从训练集统计类别分布
        train_labels = []
        for batch in train_loader:
            train_labels.extend(batch['label'].numpy())
        train_labels = np.array(train_labels)

        # 统计每个类别的样本数
        unique, counts = np.unique(train_labels, return_counts=True)
        class_counts = dict(zip(unique, counts))

        print(f"训练集类别分布: {class_counts}")

        # 根据策略计算权重
        counts_array = np.array([class_counts[i] for i in range(num_classes)])

        if args.weight_strategy == 'sqrt':
            # 平方根权重（温和调整，避免过度补偿）
            class_weights = 1.0 / np.sqrt(counts_array)
            strategy_name = "平方根权重 (1/√count)"
        elif args.weight_strategy == 'linear':
            # 线性权重（更激进）
            class_weights = 1.0 / counts_array
            strategy_name = "线性权重 (1/count)"
        elif args.weight_strategy == 'square':
            # 平方权重（最激进）
            class_weights = 1.0 / (counts_array ** 2)
            strategy_name = "平方权重 (1/count²)"
        elif args.weight_strategy == 'log':
            # 对数权重（最温和）
            class_weights = 1.0 / np.log(counts_array + 1)
            strategy_name = "对数权重 (1/log(count+1))"
        else:
            raise ValueError(f"Unknown weight strategy: {args.weight_strategy}")

        # 归一化到平均为1
        class_weights = class_weights / class_weights.mean()

        print(f"{strategy_name}")
        print(f"类别权重: {class_weights}")
        print(f"权重解释: 较少类别获得更高权重")

        # 暂存为numpy，稍后转换为tensor
        class_weights = class_weights
    else:
        print("\n不使用类别权重")

    # 创建模型
    print("\n创建模型...")

    if args.fusion_method == 'baseline':
        model = BaselineFusionModel(
            text_dim=300,
            audio_dim=74,
            video_dim=args.video_dim,
            fusion_dim=args.fusion_dim,
            num_classes=num_classes
        ).to(args.device)
    elif args.fusion_method == 'attention':
        model = AttentionFusionModel(
            text_dim=300,
            audio_dim=74,
            video_dim=args.video_dim,
            fusion_dim=args.fusion_dim,
            num_classes=num_classes
        ).to(args.device)
    elif args.fusion_method == 'transformer':
        model = TransformerFusionModel(
            text_dim=300,
            audio_dim=74,
            video_dim=args.video_dim,
            fusion_dim=args.fusion_dim,
            num_classes=num_classes
        ).to(args.device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 将类别权重转换为tensor（如果需要）
    if args.use_class_weights and class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(args.device)
    elif not args.use_class_weights:
        class_weights = None

    # 损失函数和优化器
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=1, gamma=2)
    elif args.use_class_weights and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # 训练循环
    best_val_acc = 0
    best_epoch = 0

    # 确定模型保存路径
    fusion_suffix = f"_{args.fusion_method}"
    class_suffix = f"_{num_classes}class"
    loss_suffix = "_focal" if args.use_focal_loss else "_ce"
    if args.use_class_weights:
        weight_suffix = f"_weighted_{args.weight_strategy}"
    else:
        weight_suffix = "_ce"
    model_name = f"baseline{fusion_suffix}{class_suffix}{weight_suffix}_best_model.pth"

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device
        )

        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, args.device
        )

        # 学习率调度
        scheduler.step(val_loss)

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

            os.makedirs('checkpoints', exist_ok=True)
            model_path = os.path.join('checkpoints', model_name)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, model_path)
            print(f"[OK] 保存最佳模型到 {model_name} (Epoch {epoch+1}, Val Acc: {val_acc:.4f})")

    # 测试
    print("\n" + "=" * 70)
    print("在测试集上评估...")
    print("=" * 70)

    # 加载最佳模型
    model_path = os.path.join('checkpoints', model_name)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc = validate(model, test_loader, criterion, args.device)

    print(f"\n测试集结果:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"\n最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")

    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"\n模型已保存到: checkpoints/baseline_best_model.pth")
    print(f"测试准确率: {test_acc:.4f}")


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    main()
