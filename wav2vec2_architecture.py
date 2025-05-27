"""
Wav2Vec2 架构原理详解
这个文件展示了Wav2Vec2的核心组件和工作原理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class Wav2Vec2Architecture:
    """
    Wav2Vec2架构的简化实现，用于理解其工作原理
    """
    
    def __init__(self):
        self.setup_components()
    
    def setup_components(self):
        """设置Wav2Vec2的各个组件"""
        
        # 1. 特征编码器 (Feature Encoder)
        self.feature_encoder = self.create_feature_encoder()
        
        # 2. 上下文网络 (Context Network) - Transformer
        self.context_network = self.create_context_network()
        
        # 3. 量化模块 (Quantization Module)
        self.quantizer = self.create_quantizer()
        
        # 4. 投影层
        self.projection_head = nn.Linear(768, 256)
    
    def create_feature_encoder(self):
        """
        特征编码器：将原始音频波形转换为局部特征
        使用多层1D卷积，逐步降采样
        """
        layers = []
        
        # 第一层：stride=5, 将16kHz降到3.2kHz
        layers.append(nn.Conv1d(1, 512, kernel_size=10, stride=5, bias=False))
        layers.append(nn.GroupNorm(512, 512))
        layers.append(nn.GELU())
        
        # 后续层：进一步降采样和特征提取
        for i in range(6):  # 总共7层卷积
            layers.append(nn.Conv1d(512, 512, kernel_size=3, stride=2, bias=False))
            layers.append(nn.GroupNorm(512, 512))
            layers.append(nn.GELU())
        
        return nn.Sequential(*layers)
    
    def create_context_network(self):
        """
        上下文网络：Transformer编码器
        用于捕获长距离依赖关系
        """
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=3072,
            dropout=0.1,
            activation='gelu'
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=12)
    
    def create_quantizer(self):
        """
        量化模块：将连续特征转换为离散表示
        使用Gumbel-Softmax进行可微分的离散化
        """
        return VectorQuantizer(
            num_vars=320,      # 码本大小
            temp=2.0,          # 温度参数
            groups=2,          # 分组数量
            dim=768           # 特征维度
        )

class VectorQuantizer(nn.Module):
    """向量量化器的简化实现"""
    
    def __init__(self, num_vars, temp, groups, dim):
        super().__init__()
        self.num_vars = num_vars
        self.temp = temp
        self.groups = groups
        self.dim = dim
        
        # 简化的码本 (Codebook) - 直接使用全维度
        self.codebook = nn.Parameter(torch.randn(num_vars, dim))
    
    def forward(self, x):
        """前向传播：量化输入特征"""
        # x: [batch, time, dim]
        batch_size, time_steps, dim = x.shape
        
        # 重塑输入以便计算距离 - 使用reshape确保连续性
        x_flat = x.reshape(-1, dim)  # [batch*time, dim]
        
        # 计算与码本的距离
        distances = torch.cdist(x_flat, self.codebook)  # [batch*time, num_vars]
        
        # 使用Gumbel-Softmax进行软量化
        prob = F.gumbel_softmax(-distances, tau=self.temp, hard=False, dim=-1)
        
        # 获取量化后的表示
        quantized_flat = torch.matmul(prob, self.codebook)  # [batch*time, dim]
        quantized = quantized_flat.reshape(batch_size, time_steps, dim)
        
        # 重塑概率矩阵
        prob_reshaped = prob.reshape(batch_size, time_steps, self.num_vars)
        
        return quantized, prob_reshaped

def demonstrate_wav2vec2_process():
    """演示Wav2Vec2的完整处理流程"""
    
    print("=== Wav2Vec2 音频编码过程演示 ===\n")
    
    # 1. 模拟原始音频输入
    batch_size = 2
    audio_length = 16000 * 3  # 3秒音频，16kHz采样率
    raw_audio = torch.randn(batch_size, audio_length)
    
    print(f"1. 原始音频输入:")
    print(f"   形状: {raw_audio.shape}")
    print(f"   含义: [batch_size, audio_samples]")
    print(f"   示例: {audio_length}个采样点 = 3秒 × 16000Hz\n")
    
    # 2. 特征编码器处理
    feature_encoder = nn.Sequential(
        nn.Conv1d(1, 512, kernel_size=10, stride=5),  # 降采样5倍
        nn.GELU(),
        nn.Conv1d(512, 512, kernel_size=3, stride=2), # 降采样2倍
        nn.GELU(),
        nn.Conv1d(512, 512, kernel_size=3, stride=2), # 降采样2倍
        nn.GELU(),
        nn.Conv1d(512, 768, kernel_size=3, stride=2), # 最终特征维度
        nn.GELU()
    )
    
    # 添加通道维度
    audio_input = raw_audio.unsqueeze(1)  # [batch, 1, samples]
    
    with torch.no_grad():
        encoded_features = feature_encoder(audio_input)
        encoded_features = encoded_features.transpose(1, 2)  # [batch, time, dim]
    
    print(f"2. 特征编码器输出:")
    print(f"   形状: {encoded_features.shape}")
    print(f"   含义: [batch_size, time_steps, feature_dim]")
    print(f"   降采样比例: {audio_length // encoded_features.shape[1]}:1")
    print(f"   时间分辨率: {1000 * encoded_features.shape[1] / (audio_length/16000):.1f}ms per frame\n")
    
    # 3. 上下文网络处理
    context_network = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072),
        num_layers=3  # 简化版本
    )
    
    with torch.no_grad():
        # Transformer需要 [seq_len, batch, dim] 格式
        context_input = encoded_features.transpose(0, 1)
        context_output = context_network(context_input)
        context_output = context_output.transpose(0, 1)  # 转回 [batch, seq, dim]
    
    print(f"3. 上下文网络输出:")
    print(f"   形状: {context_output.shape}")
    print(f"   含义: 包含长距离依赖关系的上下文特征")
    print(f"   每个时间步都能'看到'整个序列的信息\n")
    
    # 4. 量化过程
    quantizer = VectorQuantizer(num_vars=320, temp=2.0, groups=2, dim=768)
    
    with torch.no_grad():
        quantized_features, quantization_probs = quantizer(context_output)
    
    print(f"4. 量化模块输出:")
    print(f"   量化特征形状: {quantized_features.shape}")
    print(f"   量化概率形状: {quantization_probs.shape}")
    print(f"   含义: 将连续特征离散化为有限的'音频单元'\n")
    
    return {
        'raw_audio': raw_audio,
        'encoded_features': encoded_features,
        'context_output': context_output,
        'quantized_features': quantized_features
    }

def explain_self_supervised_training():
    """解释Wav2Vec2的自监督训练过程"""
    
    print("=== Wav2Vec2 自监督训练原理 ===\n")
    
    print("训练目标：对比学习 (Contrastive Learning)")
    print("核心思想：学习区分真实的未来音频片段和虚假的干扰项\n")
    
    print("训练步骤：")
    print("1. 输入：原始音频波形")
    print("2. 掩码：随机掩盖一些时间步（类似BERT的[MASK]）")
    print("3. 编码：通过特征编码器和上下文网络")
    print("4. 量化：将目标时间步量化为离散表示")
    print("5. 对比：预测被掩盖位置的正确量化表示\n")
    
    print("损失函数组成：")
    print("- 对比损失：区分正样本和负样本")
    print("- 多样性损失：鼓励使用码本中的所有条目")
    print("- 总损失 = 对比损失 + α × 多样性损失\n")

def visualize_feature_extraction():
    """可视化特征提取过程"""
    
    print("=== 特征提取可视化 ===\n")
    
    # 模拟不同阶段的特征
    time_steps = [1000, 200, 100, 50, 25]  # 不同阶段的时间步数
    feature_dims = [1, 512, 512, 512, 768]  # 对应的特征维度
    stage_names = ['原始音频', '卷积层1', '卷积层2', '卷积层3', '最终特征']
    
    print("特征提取的降采样过程：")
    for i, (steps, dim, name) in enumerate(zip(time_steps, feature_dims, stage_names)):
        print(f"{i+1}. {name:8s}: {steps:4d} 时间步 × {dim:3d} 维特征")
        if i > 0:
            ratio = time_steps[i-1] / steps
            print(f"   降采样比例: {ratio:.1f}:1")
        print()

def compare_with_traditional_methods():
    """与传统方法的对比"""
    
    print("=== Wav2Vec2 vs 传统方法 ===\n")
    
    comparison = {
        '特征类型': {
            'MFCC': '手工设计的频域特征',
            'Wav2Vec2': '端到端学习的时域特征'
        },
        '依赖性': {
            'MFCC': '依赖于傅里叶变换和梅尔滤波器',
            'Wav2Vec2': '直接从原始波形学习'
        },
        '上下文信息': {
            'MFCC': '局部帧级特征，缺乏长距离依赖',
            'Wav2Vec2': 'Transformer捕获全局上下文'
        },
        '鲁棒性': {
            'MFCC': '对噪声和录音条件敏感',
            'Wav2Vec2': '通过大规模预训练获得鲁棒性'
        },
        '适应性': {
            'MFCC': '固定的特征提取方式',
            'Wav2Vec2': '可以通过微调适应特定任务'
        }
    }
    
    for aspect, methods in comparison.items():
        print(f"{aspect}:")
        for method, description in methods.items():
            print(f"  {method:12s}: {description}")
        print()

if __name__ == "__main__":
    # 运行演示
    results = demonstrate_wav2vec2_process()
    explain_self_supervised_training()
    visualize_feature_extraction()
    compare_with_traditional_methods()
    
    print("=== 总结 ===")
    print("Wav2Vec2通过以下步骤将音频编码为向量：")
    print("1. 卷积编码器：原始波形 → 局部特征表示")
    print("2. Transformer：局部特征 → 全局上下文特征") 
    print("3. 量化模块：连续特征 → 离散音频单元")
    print("4. 自监督训练：通过对比学习优化表示质量") 