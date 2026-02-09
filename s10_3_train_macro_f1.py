# -*- coding: utf-8 -*-
"""
S10-3: 损失函数优化 - 使用Macro F1 Loss训练模型

任务:
1. 实现Macro F1 Loss（可微分近似）
2. 使用Macro F1 Loss训练3类模型
3. 评估Macro F1提升

目标:
- 通过Macro F1 Loss优化类别平衡
- 提升Macro F1分数
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import json
from tqdm import tqdm
import time

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import Config, DEVICE


# =============================================================================
# Macro F1 Loss（可微分近似）
# =============================================================================

class MacroF1Loss(nn.Module):
    """
    Macro F1 Loss - 可微分近似

    使用softmax近似来优化F1分数，解决不可微问题。

    参考: "F1 Loss for Highly Imbalanced Classification" 等相关工作
    """

    def __init__(self, num_classes=3, epsilon=1e-8):
        super(MacroF1Loss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, logits, targets):
        """
        计算Macro F1 Loss

        Args:
            logits: 模型输出 (N, num_classes)
            targets: 真实标签 (N,)

        Returns:
            loss: Macro F1 Loss（负值，用于最小化）
        """
        # 获取预测概率
        probs = torch.softmax(logits, dim=1)  # (N, C)

        # One-hot编码标签
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)  # (N, C)

        # 计算每个类别的precision和recall（软版本）
        # Precision = TP / (TP + FP)
        tp = (probs * targets_one_hot).sum(dim=0)  # (C,)
        fp = (probs * (1 - targets_one_hot)).sum(dim=0)  # (C,)
        fn = ((1 - probs) * targets_one_hot).sum(dim=0)  # (C,)

        precision = tp / (tp + fp + self.epsilon)  # (C,)
        recall = tp / (tp + fn + self.epsilon)  # (C,)

        # F1 = 2 * Precision * Recall / (Precision + Recall)
        f1_per_class = 2 * precision * recall / (precision + recall + self.epsilon)  # (C,)

        # Macro F1
        macro_f1 = f1_per_class.mean()

        # 返回负值（因为我们要最大化F1，但优化器是最小化）
        loss = -macro_f1

        return loss


class FocalMacroF1Loss(nn.Module):
    """
    Focal + Macro F1 组合损失

    结合Focal Loss处理类别不平衡和Macro F1 Loss优化F1分数
    """

    def __init__(self, num_classes=3, alpha=0.5, gamma=2.0, epsilon=1e-8):
        super(FocalMacroF1Loss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # Focal Loss权重
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, logits, targets):
        # CE Loss
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')

        # Focal Loss项
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        # Macro F1 Loss项
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        tp = (probs * targets_one_hot).sum(dim=0)
        fp = (probs * (1 - targets_one_hot)).sum(dim=0)
        fn = ((1 - probs) * targets_one_hot).sum(dim=0)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        f1_per_class = 2 * precision * recall / (precision + recall + self.epsilon)
        macro_f1 = f1_per_class.mean()

        # 组合损失（Macro F1用负值）
        loss = self.alpha * focal_loss.mean() - (1 - self.alpha) * macro_f1

        return loss


# =============================================================================
# 数据集类（复用S10-1的转换逻辑）
# =============================================================================

class MOSEI3ClassDataset(Dataset):
    """CMU-MOSEI 3类数据集"""

    def __init__(self, data_path: str):
        # 加载数据
        data = np.load(data_path)

        self.text_features = data['text_features'].astype(np.float32)
        self.audio_features = data['audio_features'].astype(np.float32)
        self.video_features = data['video_features'].astype(np.float32)
        labels_7class = data['labels'].astype(np.int64)

        # 7类到3类的映射
        label_map_7to3 = {
            0: 0, 1: 0, 2: 0,  # Negative
            3: 1,              # Neutral
            4: 2, 5: 2, 6: 2   # Positive
        }

        # 转换标签
        self.labels = np.array([label_map_7to3[int(l)] for l in labels_7class], dtype=np.int64)

        print(f"[Dataset] 加载了 {len(self.labels)} 个样本")
        print(f"  标签分布: {np.bincount(self.labels)}")

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
# 模型（复用S10-1的结构）
# =============================================================================

class Attention3ClassModel(nn.Module):
    """3类多模态融合模型（带注意力机制）"""

    def __init__(self, text_dim=300, audio_dim=74, video_dim=710, num_classes=3):
        super(Attention3ClassModel, self).__init__()

        # 编码器维度
        text_encoded_dim = 128
        audio_encoded_dim = 64
        video_encoded_dim = 128

        # 编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, text_encoded_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, audio_encoded_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, video_encoded_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 投影层
        self.text_proj = nn.Linear(text_encoded_dim, 64)
        self.audio_proj = nn.Linear(audio_encoded_dim, 64)
        self.video_proj = nn.Linear(video_encoded_dim, 64)

        # 交叉注意力
        self.cross_attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        # 融合层
        fused_dim = 64 * 3
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 分类器
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, text_feat, audio_feat, video_feat):
        # 编码
        text_encoded = self.text_encoder(text_feat)
        audio_encoded = self.audio_encoder(audio_feat)
        video_encoded = self.video_encoder(video_feat)

        # 投影
        text_proj = self.text_proj(text_encoded)
        audio_proj = self.audio_proj(audio_encoded)
        video_proj = self.video_proj(video_encoded)

        # 堆叠为序列
        seq = torch.stack([text_proj, audio_proj, video_proj], dim=1)

        # 注意力
        attn_out, _ = self.cross_attention(seq, seq, seq)

        # 展平
        fused = attn_out.flatten(1)

        # 融合
        fused = self.fusion(fused)

        # 分类
        logits = self.classifier(fused)

        return logits


# =============================================================================
# 训练和评估函数
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        text = batch['text'].to(device)
        audio = batch['audio'].to(device)
        video = batch['video'].to(device)
        labels = batch['label'].to(device)

        # 前向传播
        logits = model(text, audio, video)
        loss = criterion(logits, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            labels = batch['label'].to(device)

            logits = model(text, audio, video)

            # CE loss for reporting
            loss = nn.functional.cross_entropy(logits, labels)
            total_loss += loss.item()

            _, predicted = torch.max(logits, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    f1_per_class = f1_score(all_labels, all_predictions, average=None)
    weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')

    avg_loss = total_loss / len(dataloader)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'f1_per_class': f1_per_class,
        'weighted_f1': weighted_f1,
        'predictions': all_predictions,
        'labels': all_labels
    }


# =============================================================================
# 主训练函数
# =============================================================================

def train_with_macro_f1_loss(
    train_data_path,
    val_data_path,
    test_data_path,
    save_dir,
    num_epochs=30,
    batch_size=32,
    learning_rate=0.001,
    loss_type='macro_f1'  # 'macro_f1' or 'focal_macro_f1' or 'ce'
):
    """
    使用Macro F1 Loss训练模型

    Args:
        loss_type: 损失函数类型
            - 'macro_f1': 纯Macro F1 Loss
            - 'focal_macro_f1': Focal + Macro F1组合
            - 'ce': 标准交叉熵（对照）
    """

    print(f"[S10-3] 开始S10-3训练")
    print(f"[S10-3] 损失函数: {loss_type}")
    print(f"=" * 50)

    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    print("[S10-3] 加载数据...")
    train_dataset = MOSEI3ClassDataset(train_data_path)
    val_dataset = MOSEI3ClassDataset(val_data_path)
    test_dataset = MOSEI3ClassDataset(test_data_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    print("[S10-3] 创建模型...")
    model = Attention3ClassModel(num_classes=3).to(DEVICE)

    # 选择损失函数
    if loss_type == 'macro_f1':
        criterion = MacroF1Loss(num_classes=3)
    elif loss_type == 'focal_macro_f1':
        criterion = FocalMacroF1Loss(num_classes=3, alpha=0.5)
    else:  # 'ce'
        criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # 训练循环
    print("[S10-3] 开始训练...")
    best_val_f1 = 0
    best_epoch = 0
    history = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # 验证
        val_metrics = evaluate(model, val_loader, DEVICE)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"Val F1 per class: {val_metrics['f1_per_class']}")

        # 记录历史
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_macro_f1': val_metrics['macro_f1'],
            'val_f1_per_class': val_metrics['f1_per_class'].tolist()
        })

        # 学习率调度
        scheduler.step(val_metrics['macro_f1'])

        # 保存最佳模型
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_epoch = epoch + 1

            # 保存模型
            model_path = os.path.join(save_dir, f's10_3_{loss_type}_best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metrics['macro_f1'],
                'val_acc': val_metrics['accuracy'],
                'loss_type': loss_type
            }, model_path)

            print(f"[S10-3] 保存最佳模型 (Epoch {epoch+1}, Val F1: {val_metrics['macro_f1']:.4f})")

    # 测试最佳模型
    print("\n[S10-3] 训练完成，加载最佳模型进行测试...")
    checkpoint = torch.load(os.path.join(save_dir, f's10_3_{loss_type}_best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, DEVICE)

    print(f"\n[S10-3] 测试集结果:")
    print(f"  准确率: {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"  各类别F1: {test_metrics['f1_per_class']}")

    # 保存结果
    results = {
        'loss_type': loss_type,
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'test_results': {
            'accuracy': test_metrics['accuracy'],
            'macro_f1': test_metrics['macro_f1'],
            'weighted_f1': test_metrics['weighted_f1'],
            'f1_per_class': test_metrics['f1_per_class'].tolist()
        },
        'history': history
    }

    results_path = os.path.join(save_dir, f's10_3_{loss_type}_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[S10-3] 结果已保存: {results_path}")
    print(f"[S10-3] 训练完成!")
    print(f"=" * 50)

    return results


def main():
    """主函数"""
    # 数据路径
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'mosei', 'processed_cleaned_correct')
    train_data_path = os.path.join(data_dir, 'train.npz')
    val_data_path = os.path.join(data_dir, 'val.npz')
    test_data_path = os.path.join(data_dir, 'test.npz')

    # 保存目录
    save_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    results_dir = os.path.join(PROJECT_ROOT, 'results', 's10')

    # 训练参数
    num_epochs = 30
    batch_size = 32
    learning_rate = 0.001

    # 尝试不同的损失函数
    loss_types = ['focal_macro_f1', 'macro_f1', 'ce']

    for loss_type in loss_types:
        print(f"\n\n{'='*60}")
        print(f"训练配置: {loss_type}")
        print(f"{'='*60}\n")

        results = train_with_macro_f1_loss(
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            test_data_path=test_data_path,
            save_dir=save_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_type=loss_type
        )

        # 打印对比
        print(f"\n[S10-3] {loss_type} 最终结果:")
        print(f"  测试Macro F1: {results['test_results']['macro_f1']:.4f}")
        print(f"  测试准确率: {results['test_results']['accuracy']:.4f}")
        print(f"  Negative F1: {results['test_results']['f1_per_class'][0]:.4f}")


if __name__ == '__main__':
    main()
