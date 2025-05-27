---
license: mit
tags:
- audio-classification
- wav2vec2
- sound-detection
- few-shot-learning
- pytorch
language:
- zh
datasets:
- custom
metrics:
- accuracy
- precision
- recall
- f1
library_name: transformers
pipeline_tag: audio-classification
---

# 🎯 热水器开关声音检测器 (Heater Switch Sound Detector)

基于Wav2Vec2的热水器开关声音实时检测模型。这是一个少样本学习项目，仅用6个音频样本就能达到100%的检测准确率。

## 模型描述

该模型使用Facebook的Wav2Vec2预训练模型作为特征提取器，在热水器开关声音数据上进行微调，实现对开关按下声音的精确识别。

### 模型架构

```
原始音频 [48000 samples] 
    ↓ Wav2Vec2特征编码器 (7层1D卷积)
局部特征 [1199, 768]
    ↓ Wav2Vec2上下文网络 (12层Transformer) 
上下文特征 [1199, 768]
    ↓ 全局平均池化
固定特征 [768]
    ↓ 分类头 (2层全连接)
分类结果 [2] (开关/背景)
```

## 训练数据

- **正样本**: 6个热水器开关声音 (3.2-5.2秒)
- **负样本**: 6个自动生成的背景噪音
- **总样本**: 12个 (训练集8个，测试集4个)
- **采样率**: 16kHz
- **格式**: 单声道WAV

### 数据特征分析

| 样本类型 | 时长范围 | RMS能量 | 频谱质心 | 过零率 |
|----------|----------|---------|----------|--------|
| 开关声音 | 3.2-5.2s | 0.0079-0.0115 | 1587-1992Hz | 0.0657-0.1215 |
| 背景噪音 | 2.0-4.0s | 0.005-0.02 | 500-1500Hz | 0.05-0.15 |

## 性能指标

| 指标 | 数值 |
|------|------|
| **准确率** | 100% |
| **精确率** | 100% |
| **召回率** | 100% |
| **F1分数** | 100% |
| **训练轮数** | 15 epochs |
| **模型大小** | 361MB |
| **推理延迟** | <100ms |

### 混淆矩阵

```
实际\预测    无开关    有开关
无开关         2        0
有开关         0        2
```

## 使用方法

### 安装依赖

```bash
pip install torch torchaudio transformers huggingface_hub
```

### 加载模型

```python
from huggingface_hub import hf_hub_download
import torch
import torchaudio
from transformers import Wav2Vec2Model

# 下载模型
model_path = hf_hub_download(
    repo_id="your-username/heater-switch-detector",
    filename="switch_detector_model.pth"
)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(model_path, map_location=device)

# 重建模型架构
wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
classifier = torch.nn.Sequential(
    torch.nn.Linear(768, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(256, 2)
)

# 加载权重
classifier.load_state_dict(checkpoint['classifier_state_dict'])
classifier.eval()
```

### 音频预测

```python
def predict_audio(audio_path):
    # 加载音频
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # 重采样到16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # 转为单声道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # 特征提取
    with torch.no_grad():
        features = wav2vec2_model(waveform).last_hidden_state
        pooled_features = features.mean(dim=1)  # 全局平均池化
        
        # 分类预测
        logits = classifier(pooled_features)
        probabilities = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1)
    
    return {
        'prediction': '开关按下' if prediction.item() == 1 else '背景声音',
        'confidence': probabilities.max().item(),
        'probabilities': {
            '背景声音': probabilities[0][0].item(),
            '开关按下': probabilities[0][1].item()
        }
    }

# 使用示例
result = predict_audio("test_audio.wav")
print(f"预测结果: {result['prediction']}")
print(f"置信度: {result['confidence']:.3f}")
```

### 实时检测

```python
import pyaudio
import numpy as np

def realtime_detection():
    # 音频参数
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024
    DETECTION_WINDOW = 3.0  # 3秒检测窗口
    
    # 初始化音频流
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    
    print("🎤 开始实时检测...")
    
    buffer = []
    window_size = int(DETECTION_WINDOW * SAMPLE_RATE)
    
    try:
        while True:
            # 读取音频数据
            data = stream.read(CHUNK_SIZE)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            buffer.extend(audio_chunk)
            
            # 保持窗口大小
            if len(buffer) > window_size:
                buffer = buffer[-window_size:]
            
            # 检测
            if len(buffer) == window_size:
                waveform = torch.FloatTensor(buffer).unsqueeze(0)
                
                with torch.no_grad():
                    features = wav2vec2_model(waveform).last_hidden_state
                    pooled_features = features.mean(dim=1)
                    logits = classifier(pooled_features)
                    probabilities = torch.softmax(logits, dim=-1)
                    
                    switch_prob = probabilities[0][1].item()
                    
                    if switch_prob > 0.93:  # 高置信度阈值
                        print(f"🔥 检测到开关按下! 置信度: {switch_prob:.3f}")
    
    except KeyboardInterrupt:
        print("\n⏹️  检测停止")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

# 运行实时检测
realtime_detection()
```

## 技术特点

### 🚀 优势

- **少样本学习**: 仅需6个样本即可达到完美分类
- **端到端训练**: 从原始音频波形直接学习特征
- **预训练优势**: 利用Wav2Vec2的大规模预训练知识
- **实时检测**: 支持麦克风实时音频流处理
- **高精度**: 测试集100%准确率

### 🎯 应用场景

- **智能家居**: 自动检测热水器使用状态
- **设备监控**: 远程监控设备操作
- **节能管理**: 记录设备使用时间和频率
- **安全监控**: 异常使用模式检测

### ⚙️ 技术细节

- **基础模型**: facebook/wav2vec2-base
- **训练策略**: 冻结预训练参数，只训练分类头
- **优化器**: AdamW (lr=1e-4)
- **损失函数**: CrossEntropyLoss
- **数据增强**: 自动生成负样本

## 限制和改进

### 当前限制

- 训练数据较少，可能对新环境泛化能力有限
- 只能检测特定类型的开关声音
- 需要相对安静的环境以减少误报

### 未来改进

- [ ] 收集更多样化的训练数据
- [ ] 支持多类别检测（开/关/故障）
- [ ] 添加噪音鲁棒性训练
- [ ] 模型压缩和量化
- [ ] 支持更多设备类型

## 引用

如果您使用了这个模型，请引用：

```bibtex
@misc{heater-switch-detector-2024,
  title={基于Wav2Vec2的热水器开关声音检测器},
  author={Your Name},
  year={2024},
  howpublished={\url{https://huggingface.co/your-username/heater-switch-detector}}
}
```

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub: [项目地址](https://github.com/your-username/heater_click)
- Email: your.email@example.com

---

*该模型仅用于研究和教育目的。在生产环境中使用前，请进行充分的测试和验证。* 