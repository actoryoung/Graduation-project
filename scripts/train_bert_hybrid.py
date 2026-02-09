# -*- coding: utf-8 -*-
"""
训练BERT混合多模态情感分析模型

使用BERT(768) + COVAREP(74) + OpenFace(710) = 1552维特征
相比基线模型(GloVe 300维)，仅升级文本编码器
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


# =============================================================================
# 数据集类
# =============================================================================

class BERTHybridDataset(Dataset):
    """BERT混合特征数据集"""

    def __init__(self, data_path: str, split: str = 'train'):
        self.data_path = data_path
        self.split = split

        # 加载数据
        data = np.load(data_path, allow_pickle=True)

        self.text_features = data['text_features'].astype(np.float32)  # [N, 768] BERT
        self.audio_features = data['audio_features'].astype(np.float32)  # [N, 74] COVAREP
        self.video_features = data['video_features'].astype(np.float32)  # [N, 710] OpenFace
        self.labels = data['labels'].astype(np.int64)

        print(f"[{split.upper()}] 加载了 {len(self.labels)} 个样本")
        print(f"  BERT文本特征: {self.text_features.shape}")
        print(f"  COVAREP音频特征: {self.audio_features.shape}")
        print(f"  OpenFace视频特征: {self.video_features.shape}")

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
# BERT混合融合模型
# =============================================================================

class BERTHybridModel(nn.Module):
    """BERT混合多模态融合模型"""

    def __init__(
        self,
        text_dim=768,      # BERT特征
        audio_dim=74,      # COVAREP特征
        video_dim=710,     # OpenFace特征
        fusion_dim=512,
        num_classes=7,
        dropout_rate=0.3
    ):
        super(BERTHybridModel, self).__init__()

        self.input_dim = text_dim + audio_dim + video_dim

        # 模态特定编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 融合层
        total_dim = 256 + 128 + 256
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
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


# =============================================================================
# 训练和验证函数
# =============================================================================

def create_data_loaders(data_dir, batch_size=32):
    """创建数据加载器"""
    train_path = os.path.join(data_dir, 'train.npz')
    val_path = os.path.join(data_dir, 'val.npz')
    test_path = os.path.join(data_dir, 'test.npz')

    train_dataset = BERTHybridDataset(train_path, split='train')
    val_dataset = BERTHybridDataset(val_path, split='val')
    test_dataset = BERTHybridDataset(test_path, split='test')

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
    parser = argparse.ArgumentParser(description='训练BERT混合多模态情感分析模型')
    parser.add_argument('--data-dir', type=str, default='data/mosei/bert_hybrid_fixed',
                       help='数据目录 (使用修复后的正确标签)')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--fusion-dim', type=int, default=512, help='融合维度')

    args = parser.parse_args()

    print("=" * 70)
    print("BERT混合多模态情感分析模型训练")
    print("特征: BERT(768) + COVAREP(74) + OpenFace(710) = 1552维")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  数据目录: {args.data_dir}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  设备: {args.device}")
    print(f"  融合维度: {args.fusion_dim}")

    # 创建数据加载器
    print("\n加载数据...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)
    except FileNotFoundError as e:
        print(f"\n[错误] 数据文件不存在!")
        print(f"请先运行: python scripts/extract_bert_hybrid.py")
        print(f"\n错误详情: {e}")
        return

    # 创建模型
    print("\n创建模型...")
    model = BERTHybridModel(
        text_dim=768,
        audio_dim=74,
        video_dim=710,
        fusion_dim=args.fusion_dim,
        num_classes=NUM_CLASSES
    ).to(args.device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # 训练循环
    best_val_acc = 0
    best_epoch = 0

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
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, 'checkpoints/bert_hybrid_best_model.pth')
            print(f"[OK] 保存最佳模型 (Epoch {epoch+1}, Val Acc: {val_acc:.4f})")

    # 测试
    print("\n" + "=" * 70)
    print("在测试集上评估...")
    print("=" * 70)

    # 加载最佳模型
    checkpoint = torch.load('checkpoints/bert_hybrid_best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc = validate(model, test_loader, criterion, args.device)

    print(f"\n测试集结果:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"\n最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")

    # 与基线对比
    print("\n" + "=" * 70)
    print("与基线模型对比")
    print("=" * 70)
    print(f"基线模型 (GloVe+COVAREP+OpenFace): 53.11%")
    print(f"BERT混合模型 (BERT+COVAREP+OpenFace): {test_acc:.2%}")
    improvement = (test_acc - 0.5311) * 100
    print(f"提升: {improvement:+.2f}%")

    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"\n模型已保存到: checkpoints/bert_hybrid_best_model.pth")
    print(f"测试准确率: {test_acc:.4f}")


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    main()
