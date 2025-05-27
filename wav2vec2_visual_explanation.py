"""
Wav2Vec2 可视化原理解释
通过图表和示例来理解Wav2Vec2的工作机制
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.patches as patches

def create_wav2vec2_architecture_diagram():
    """创建Wav2Vec2架构图"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # 定义各个组件的位置和大小
    components = [
        {"name": "原始音频波形", "pos": (1, 6), "size": (2, 1), "color": "lightblue"},
        {"name": "特征编码器\n(7层1D卷积)", "pos": (4, 6), "size": (2, 1), "color": "lightgreen"},
        {"name": "上下文网络\n(12层Transformer)", "pos": (7, 6), "size": (2, 1), "color": "lightcoral"},
        {"name": "量化模块\n(Vector Quantizer)", "pos": (10, 6), "size": (2, 1), "color": "lightyellow"},
        {"name": "投影层", "pos": (7, 4), "size": (2, 1), "color": "lightgray"},
        {"name": "对比学习目标", "pos": (7, 2), "size": (2, 1), "color": "lightpink"}
    ]
    
    # 绘制组件
    for comp in components:
        rect = patches.Rectangle(comp["pos"], comp["size"][0], comp["size"][1], 
                               linewidth=2, edgecolor='black', facecolor=comp["color"])
        ax.add_patch(rect)
        ax.text(comp["pos"][0] + comp["size"][0]/2, comp["pos"][1] + comp["size"][1]/2, 
               comp["name"], ha='center', va='center', fontsize=10, weight='bold')
    
    # 绘制箭头连接
    arrows = [
        ((3, 6.5), (4, 6.5)),  # 原始音频 -> 特征编码器
        ((6, 6.5), (7, 6.5)),  # 特征编码器 -> 上下文网络
        ((9, 6.5), (10, 6.5)), # 上下文网络 -> 量化模块
        ((8, 6), (8, 5)),      # 上下文网络 -> 投影层
        ((8, 4), (8, 3)),      # 投影层 -> 对比学习
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # 添加维度信息
    dimensions = [
        {"pos": (2, 5.5), "text": "[B, 48000]"},
        {"pos": (5, 5.5), "text": "[B, 1199, 768]"},
        {"pos": (8, 5.5), "text": "[B, 1199, 768]"},
        {"pos": (11, 5.5), "text": "[B, 1199, 320]"},
        {"pos": (8, 3.5), "text": "[B, 1199, 256]"},
    ]
    
    for dim in dimensions:
        ax.text(dim["pos"][0], dim["pos"][1], dim["text"], 
               ha='center', va='center', fontsize=9, style='italic', color='blue')
    
    ax.set_xlim(0, 13)
    ax.set_ylim(1, 8)
    ax.set_title('Wav2Vec2 架构图', fontsize=16, weight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('wav2vec2_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_audio_processing_steps():
    """可视化音频处理的各个步骤"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 原始音频波形
    t = np.linspace(0, 1, 1000)
    audio_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    
    axes[0, 0].plot(t, audio_signal)
    axes[0, 0].set_title('1. 原始音频波形 (16kHz采样)', fontsize=12, weight='bold')
    axes[0, 0].set_xlabel('时间 (秒)')
    axes[0, 0].set_ylabel('幅度')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 特征编码器输出（模拟）
    time_steps = np.arange(0, 50)
    features = np.random.randn(50, 10)  # 简化的特征表示
    
    im1 = axes[0, 1].imshow(features.T, aspect='auto', cmap='viridis')
    axes[0, 1].set_title('2. 特征编码器输出\n(降采样后的局部特征)', fontsize=12, weight='bold')
    axes[0, 1].set_xlabel('时间步')
    axes[0, 1].set_ylabel('特征维度')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 3. 上下文网络输出（模拟）
    context_features = np.random.randn(50, 10)
    # 添加一些全局相关性
    for i in range(1, 50):
        context_features[i] = 0.7 * context_features[i] + 0.3 * context_features[i-1]
    
    im2 = axes[1, 0].imshow(context_features.T, aspect='auto', cmap='plasma')
    axes[1, 0].set_title('3. 上下文网络输出\n(包含全局上下文信息)', fontsize=12, weight='bold')
    axes[1, 0].set_xlabel('时间步')
    axes[1, 0].set_ylabel('特征维度')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # 4. 量化结果（模拟）
    quantized_ids = np.random.randint(0, 20, 50)
    
    axes[1, 1].scatter(range(50), quantized_ids, c=quantized_ids, cmap='tab20', s=50)
    axes[1, 1].set_title('4. 量化结果\n(离散的音频单元ID)', fontsize=12, weight='bold')
    axes[1, 1].set_xlabel('时间步')
    axes[1, 1].set_ylabel('量化ID')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('wav2vec2_processing_steps.png', dpi=300, bbox_inches='tight')
    plt.show()

def explain_contrastive_learning():
    """解释对比学习的原理"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 创建一个简化的对比学习示意图
    
    # 输入序列
    sequence_length = 10
    x_positions = np.arange(sequence_length)
    
    # 绘制输入序列
    for i, x in enumerate(x_positions):
        if i == 5:  # 被掩盖的位置
            color = 'red'
            label = 'MASK' if i == 5 else ''
        else:
            color = 'lightblue'
            label = ''
        
        rect = patches.Rectangle((x, 5), 0.8, 1, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        
        if label:
            ax.text(x + 0.4, 5.5, label, ha='center', va='center', fontsize=10, weight='bold')
        else:
            ax.text(x + 0.4, 5.5, f't{i}', ha='center', va='center', fontsize=9)
    
    # 绘制候选项
    candidates = ['正确', '错误1', '错误2', '错误3']
    colors = ['green', 'gray', 'gray', 'gray']
    
    for i, (cand, color) in enumerate(zip(candidates, colors)):
        rect = patches.Rectangle((2 + i * 2, 2), 1.5, 1, facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.text(2.75 + i * 2, 2.5, cand, ha='center', va='center', fontsize=10, weight='bold')
    
    # 绘制箭头
    ax.annotate('', xy=(2.75, 3), xytext=(5.4, 5),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    for i in range(1, 4):
        ax.annotate('', xy=(2.75 + i * 2, 3), xytext=(5.4, 5),
                   arrowprops=dict(arrowstyle='->', lw=1, color='red', linestyle='--'))
    
    # 添加说明文字
    ax.text(5, 7, '对比学习目标：学习区分正确和错误的音频表示', 
           ha='center', va='center', fontsize=14, weight='bold')
    
    ax.text(1, 0.5, '模型需要学习：\n1. 被掩盖位置的正确表示\n2. 区分真实和虚假的候选项', 
           ha='left', va='center', fontsize=11, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(0, 8)
    ax.set_title('Wav2Vec2 对比学习原理', fontsize=16, weight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('wav2vec2_contrastive_learning.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_feature_extraction_methods():
    """比较不同特征提取方法"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 模拟音频信号
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 5 * t) + 0.3 * np.sin(2 * np.pi * 15 * t) + 0.1 * np.random.randn(1000)
    
    # 1. 原始音频
    axes[0, 0].plot(t, signal)
    axes[0, 0].set_title('原始音频信号', fontsize=12, weight='bold')
    axes[0, 0].set_xlabel('时间 (秒)')
    axes[0, 0].set_ylabel('幅度')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. MFCC特征（模拟）
    mfcc_features = np.random.randn(13, 100)  # 13个MFCC系数，100个时间帧
    
    im1 = axes[0, 1].imshow(mfcc_features, aspect='auto', cmap='coolwarm')
    axes[0, 1].set_title('MFCC特征\n(手工设计的频域特征)', fontsize=12, weight='bold')
    axes[0, 1].set_xlabel('时间帧')
    axes[0, 1].set_ylabel('MFCC系数')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 3. Wav2Vec2特征（模拟）
    wav2vec_features = np.random.randn(768, 50)  # 768维特征，50个时间步
    
    im2 = axes[1, 0].imshow(wav2vec_features[:50, :], aspect='auto', cmap='viridis')  # 只显示前50维
    axes[1, 0].set_title('Wav2Vec2特征\n(学习得到的表示)', fontsize=12, weight='bold')
    axes[1, 0].set_xlabel('时间步')
    axes[1, 0].set_ylabel('特征维度 (前50维)')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # 4. 特征质量对比
    methods = ['MFCC', 'Wav2Vec2']
    metrics = ['鲁棒性', '上下文信息', '适应性', '性能']
    mfcc_scores = [3, 2, 2, 6]
    wav2vec_scores = [8, 9, 9, 9]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, mfcc_scores, width, label='MFCC', color='lightcoral')
    axes[1, 1].bar(x + width/2, wav2vec_scores, width, label='Wav2Vec2', color='lightblue')
    
    axes[1, 1].set_title('特征质量对比', fontsize=12, weight='bold')
    axes[1, 1].set_xlabel('评估指标')
    axes[1, 1].set_ylabel('评分 (1-10)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_extraction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """运行所有可视化"""
    
    print("🎯 生成Wav2Vec2可视化图表...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    try:
        print("1. 创建架构图...")
        create_wav2vec2_architecture_diagram()
        
        print("2. 可视化处理步骤...")
        visualize_audio_processing_steps()
        
        print("3. 解释对比学习...")
        explain_contrastive_learning()
        
        print("4. 比较特征提取方法...")
        compare_feature_extraction_methods()
        
        print("✅ 所有图表已生成完成！")
        
    except Exception as e:
        print(f"❌ 生成图表时出错: {e}")
        print("💡 提示：如果字体问题，图表仍会生成，只是中文可能显示为方块")

if __name__ == "__main__":
    main() 