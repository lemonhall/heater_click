"""
基于Wav2Vec2的热水器开关声音检测器
使用预训练的Wav2Vec2模型进行特征提取，然后训练一个简单的分类器
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random

# 检查是否有transformers库
try:
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers库未安装")
    print("   安装命令: pip install transformers")

class SwitchSoundDataset(Dataset):
    """开关声音数据集"""
    
    def __init__(self, audio_files, labels, feature_extractor, max_length=16000*5):
        self.audio_files = audio_files
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # 加载音频文件
        audio_path = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 重采样到16kHz（如果需要）
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # 调整长度
        waveform = waveform.squeeze()
        if len(waveform) > self.max_length:
            # 随机裁剪
            start = random.randint(0, len(waveform) - self.max_length)
            waveform = waveform[start:start + self.max_length]
        else:
            # 零填充
            padding = self.max_length - len(waveform)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # 使用feature_extractor预处理
        inputs = self.feature_extractor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        return {
            'input_values': inputs.input_values.squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class Wav2Vec2Classifier(nn.Module):
    """基于Wav2Vec2的分类器"""
    
    def __init__(self, num_classes=2, freeze_wav2vec=True):
        super().__init__()
        
        # 加载预训练的Wav2Vec2模型
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # 是否冻结Wav2Vec2参数
        if freeze_wav2vec:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, input_values):
        # 提取Wav2Vec2特征
        with torch.no_grad() if hasattr(self, '_freeze_wav2vec') else torch.enable_grad():
            outputs = self.wav2vec2(input_values)
            features = outputs.last_hidden_state  # [batch, time, 768]
        
        # 全局平均池化: [batch, time, 768] -> [batch, 768]
        pooled = features.mean(dim=1)
        
        # 分类
        logits = self.classifier(pooled)
        
        return logits

def generate_negative_samples(positive_files, output_dir="samples_wav", num_negatives=6):
    """
    生成负样本（背景噪音）
    通过对正样本进行变换来创建负样本
    """
    
    print(f"🔄 生成 {num_negatives} 个负样本...")
    
    negative_files = []
    
    for i in range(num_negatives):
        # 随机选择一个正样本作为基础
        base_file = random.choice(positive_files)
        
        try:
            # 加载音频
            waveform, sample_rate = torchaudio.load(base_file)
            waveform = waveform.squeeze()
            
            # 应用变换生成负样本
            if i % 3 == 0:
                # 方法1: 添加高斯噪音
                noise = torch.randn_like(waveform) * 0.1
                modified_waveform = noise
            elif i % 3 == 1:
                # 方法2: 低通滤波（模拟远距离声音）
                modified_waveform = waveform * 0.1 + torch.randn_like(waveform) * 0.05
            else:
                # 方法3: 静音 + 轻微噪音
                modified_waveform = torch.randn_like(waveform) * 0.02
            
            # 保存负样本
            negative_filename = f"background_{i+1:02d}.wav"
            negative_path = os.path.join(output_dir, negative_filename)
            
            torchaudio.save(negative_path, modified_waveform.unsqueeze(0), sample_rate)
            negative_files.append(negative_path)
            
            print(f"✅ 生成负样本: {negative_filename}")
            
        except Exception as e:
            print(f"❌ 生成负样本失败: {e}")
    
    return negative_files

def prepare_dataset():
    """准备训练数据集"""
    
    # 查找正样本文件
    positive_files = glob.glob("samples_wav/switch_on_*.wav")
    
    if not positive_files:
        print("❌ 没有找到正样本文件")
        return None, None, None, None
    
    print(f"📊 找到 {len(positive_files)} 个正样本")
    
    # 生成负样本
    negative_files = generate_negative_samples(positive_files)
    
    # 合并所有文件和标签
    all_files = positive_files + negative_files
    all_labels = [1] * len(positive_files) + [0] * len(negative_files)
    
    print(f"📊 数据集统计:")
    print(f"   正样本: {len(positive_files)} 个")
    print(f"   负样本: {len(negative_files)} 个")
    print(f"   总计: {len(all_files)} 个")
    
    # 划分训练集和测试集
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    
    return train_files, test_files, train_labels, test_labels

def train_model(train_dataset, test_dataset, num_epochs=20):
    """训练模型"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # 创建模型
    model = Wav2Vec2Classifier(num_classes=2)
    model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    # 训练历史
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"\n🚀 开始训练...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            input_values = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # 测试阶段
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_values = batch['input_values'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_values)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * test_correct / test_total
        
        # 记录历史
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, "
              f"Test Acc: {test_accuracy:.2f}%")
    
    return model, train_losses, train_accuracies, test_accuracies

def evaluate_model(model, test_dataset):
    """评估模型性能"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_values = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_values)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    
    print(f"\n📊 模型评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"\n分类报告:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=['无开关', '有开关']))
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['无开关', '有开关'],
                yticklabels=['无开关', '有开关'])
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy

def save_model(model, filepath="switch_detector_model.pth"):
    """保存训练好的模型"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': 'Wav2Vec2Classifier'
    }, filepath)
    print(f"💾 模型已保存到: {filepath}")

def main():
    """主函数"""
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ 需要安装transformers库才能使用Wav2Vec2")
        return
    
    print("🎯 热水器开关声音检测器")
    print("=" * 50)
    
    # 准备数据集
    train_files, test_files, train_labels, test_labels = prepare_dataset()
    
    if train_files is None:
        return
    
    # 创建feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    
    # 创建数据集
    train_dataset = SwitchSoundDataset(train_files, train_labels, feature_extractor)
    test_dataset = SwitchSoundDataset(test_files, test_labels, feature_extractor)
    
    print(f"\n📊 数据集划分:")
    print(f"   训练集: {len(train_dataset)} 个样本")
    print(f"   测试集: {len(test_dataset)} 个样本")
    
    # 训练模型
    model, train_losses, train_accuracies, test_accuracies = train_model(
        train_dataset, test_dataset, num_epochs=15
    )
    
    # 评估模型
    accuracy = evaluate_model(model, test_dataset)
    
    # 保存模型
    save_model(model)
    
    print(f"\n🎉 训练完成！")
    print(f"   最终测试准确率: {accuracy:.4f}")
    print(f"   模型已保存，可用于实时检测")

if __name__ == "__main__":
    main() 