# -*- coding: utf-8 -*-
"""
训练BERT混合多模态情感分析模型 - 使用类别权重处理不平衡
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


class BERTHybridDataset(Dataset):
    """BERT混合特征数据集"""
    def __init__(self, data_path: str, split: str = 'train'):
        self.data_path = data_path
        self.split = split
        data = np.load(data_path, allow_pickle=True)

        self.text_features = data['text_features'].astype(np.float32)
        self.audio_features = data['audio_features'].astype(np.float32)
        self.video_features = data['video_features'].astype(np.float32)
        self.labels = data['labels'].astype(np.int64)

        print(f"[{split.upper()}] 加载了 {len(self.labels)} 个样本")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'text': torch.from_numpy(self.text_features[idx]),
            'audio': torch.from_numpy(self.audio_features[idx]),
            'video': torch.from_numpy(self.video_features[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class BERTHybridModel(nn.Module):
    """BERT混合多模态融合模型"""
    def __init__(self, text_dim=768, audio_dim=74, video_dim=710,
                 fusion_dim=512, num_classes=7, dropout_rate=0.3):
        super(BERTHybridModel, self).__init__()

        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2)
        )
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2)
        )
        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2)
        )

        total_dim = 256 + 128 + 256
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2), nn.LayerNorm(fusion_dim // 2), nn.ReLU(),
            nn.Dropout(dropout_rate * 0.65)
        )
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

    def forward(self, text, audio, video):
        t_feat = self.text_encoder(text)
        a_feat = self.audio_encoder(audio)
        v_feat = self.video_encoder(video)
        fused = torch.cat([t_feat, a_feat, v_feat], dim=1)
        features = self.fusion(fused)
        logits = self.classifier(features)
        return logits


def compute_class_weights(labels, num_classes=7):
    """计算类别权重（反比于类别频率）"""
    from collections import Counter
    class_counts = Counter(labels)
    total = len(labels)

    # 计算权重：weight = total / (num_classes * count)
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weight = total / (num_classes * count)
        weights.append(weight)

    weights = torch.FloatTensor(weights).to(DEVICE)
    return weights


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training"):
        text = batch['text'].to(device)
        audio = batch['audio'].to(device)
        video = batch['video'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(text, audio, video)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
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


def main():
    parser = argparse.ArgumentParser(description='训练BERT混合模型（带类别权重）')
    parser.add_argument('--data-dir', type=str, default='data/mosei/bert_hybrid_fixed')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fusion-dim', type=int, default=512)

    args = parser.parse_args()

    print("=" * 70)
    print("BERT混合模型训练 - 使用类别权重处理不平衡")
    print("=" * 70)

    # 创建数据加载器
    train_path = os.path.join(args.data_dir, 'train.npz')
    val_path = os.path.join(args.data_dir, 'val.npz')
    test_path = os.path.join(args.data_dir, 'test.npz')

    train_dataset = BERTHybridDataset(train_path, 'train')
    val_dataset = BERTHybridDataset(val_path, 'val')
    test_dataset = BERTHybridDataset(test_path, 'test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 计算类别权重
    print("\n计算类别权重...")
    train_labels = train_dataset.labels
    class_weights = compute_class_weights(train_labels, NUM_CLASSES)

    emotion_names = ['strong_negative', 'negative', 'weak_negative', 'neutral',
                     'weak_positive', 'positive', 'strong_positive']
    print("\n类别权重:")
    for i, w in enumerate(class_weights):
        print(f"  类别{i} ({emotion_names[i]}): {w:.4f}")

    # 创建模型
    print("\n创建模型...")
    model = BERTHybridModel(
        text_dim=768, audio_dim=74, video_dim=710,
        fusion_dim=args.fusion_dim, num_classes=NUM_CLASSES
    ).to(args.device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 使用加权损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # 训练循环
    best_val_acc = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, args.device)

        scheduler.step(val_loss)

        print(f"\nTrain - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'checkpoints/bert_hybrid_weighted_best_model.pth')
            print(f"[OK] 保存最佳模型 (Epoch {epoch+1}, Val Acc: {val_acc:.4f})")

    # 测试
    print(f"\n最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")

    # 加载最佳模型并测试
    checkpoint = torch.load('checkpoints/bert_hybrid_weighted_best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc = validate(model, test_loader, criterion, args.device)
    print(f"\n测试准确率: {test_acc:.4f}")

    print("\n训练完成!")


if __name__ == '__main__':
    main()
