# 🎯 基于Wav2Vec2的热水器开关声音检测器

使用Facebook的Wav2Vec2预训练模型进行热水器开关声音的实时检测。这是一个少样本学习项目，仅用6个音频样本就能达到100%的检测准确率。

## 🚀 项目特点

- **少样本学习**: 仅需6个开关声音样本即可训练
- **高精度检测**: 测试准确率达到100%
- **实时检测**: 支持麦克风实时音频流检测
- **端到端**: 从原始音频波形直接学习特征
- **预训练模型**: 基于Facebook Wav2Vec2-base模型

## 📁 项目结构

```
heater_click/
├── 📊 数据处理
│   ├── convert_audio.py          # m4a转wav格式转换
│   ├── rename_audio_files.py     # 音频文件重命名
│   └── analyze_heater_sounds.py  # 音频特征分析
├── 🤖 模型训练
│   ├── wav2vec2_switch_detector.py  # 主训练脚本
│   └── switch_detector_model.pth    # 训练好的模型(361MB)
├── 🎤 实时检测
│   └── realtime_mic_detector.py     # 实时麦克风检测
├── 📚 技术说明
│   ├── wav2vec2_explanation.md      # Wav2Vec2原理详解
│   ├── wav2vec2_architecture.py     # 架构演示代码
│   └── wav2vec2_visual_explanation.py # 可视化说明
├── 🎵 音频数据
│   └── samples_wav/                 # 转换后的wav文件
│       ├── switch_on_01.wav ~ switch_on_06.wav  # 开关声音
│       └── background_01.wav ~ background_06.wav # 背景噪音
└── 📈 分析结果
    ├── confusion_matrix.png         # 混淆矩阵
    ├── analysis_plots/              # 音频分析图表
    └── wav2vec2_*.png              # 架构说明图
```

## 🛠️ 安装依赖

```bash
pip install torch torchaudio transformers scikit-learn matplotlib seaborn pyaudio requests
```

## ⚠️ 重要说明

**模型文件处理**: 由于训练好的模型文件(`switch_detector_model.pth`)大小为361MB，超过GitHub的100MB限制，因此未包含在Git仓库中。

## 🎯 快速开始

### 1. 获取模型文件

**选项A: 下载预训练模型**
```bash
# 运行模型下载脚本 (如果可用)
python download_model.py
```

**选项B: 训练新模型**
```bash
# 准备音频数据 (将6个m4a文件放在项目根目录)
python convert_audio.py

# 训练模型
python wav2vec2_switch_detector.py
```

### 2. 实时检测

```bash
python realtime_mic_detector.py
```

训练过程：
- 自动加载6个开关声音样本
- 生成6个背景噪音负样本
- 使用Wav2Vec2进行特征提取
- 训练二分类器(开关 vs 背景)
- 保存模型到 `switch_detector_model.pth`

### 2. 实时检测

```bash
python realtime_mic_detector.py
```

功能特点：
- 实时麦克风音频流处理
- 3秒滑动窗口检测
- 置信度阈值过滤
- 防重复检测机制

## 🧠 技术原理

### Wav2Vec2架构

```
原始音频 [48000] 
    ↓ 特征编码器(7层1D卷积)
局部特征 [1199, 768]
    ↓ 上下文网络(12层Transformer) 
上下文特征 [1199, 768]
    ↓ 全局平均池化
固定特征 [768]
    ↓ 分类头(2层全连接)
分类结果 [2] (开关/背景)
```

### 自监督学习

- **掩码预测**: 随机掩盖音频片段，预测被掩盖内容
- **对比学习**: 区分真实音频片段和随机干扰项
- **量化表示**: 将连续特征离散化为有限音频单元

### 训练策略

- **冻结预训练参数**: 只训练分类头，避免过拟合
- **数据增强**: 自动生成背景噪音负样本
- **小批量训练**: 适合少样本场景

## 📊 实验结果

| 指标 | 数值 |
|------|------|
| 训练样本 | 12个 (6正+6负) |
| 测试准确率 | 100% |
| 训练轮数 | 15 epochs |
| 模型大小 | 361MB |
| 检测延迟 | <100ms |

### 混淆矩阵
```
实际\预测  无开关  有开关
无开关      2      0
有开关      0      2
```

## 🎵 音频数据分析

| 文件 | 时长 | RMS能量 | 频谱质心 | 过零率 |
|------|------|---------|----------|--------|
| switch_on_01 | 3.20s | 0.0102 | 1715Hz | 0.0787 |
| switch_on_02 | 3.63s | 0.0102 | 1587Hz | 0.0693 |
| switch_on_03 | 3.82s | 0.0115 | 1723Hz | 0.0849 |
| switch_on_04 | 3.48s | 0.0085 | 1849Hz | 0.1091 |
| switch_on_05 | 3.39s | 0.0080 | 1992Hz | 0.1215 |
| switch_on_06 | 5.21s | 0.0079 | 1591Hz | 0.0657 |

## 🔧 参数调优

### 检测参数
```python
RealtimeSwitchDetector(
    model_path="switch_detector_model.pth",
    sample_rate=16000,           # 采样率
    chunk_size=1024,             # 音频块大小
    detection_window=3.0,        # 检测窗口(秒)
    confidence_threshold=0.93    # 置信度阈值(高精度检测)
)
```

### 训练参数
```python
# 学习率: 1e-4
# 批大小: 4
# 优化器: AdamW
# 损失函数: CrossEntropyLoss
```

## 🎯 使用场景

- **智能家居**: 检测热水器开关状态
- **设备监控**: 远程监控设备使用情况
- **节能管理**: 自动记录设备使用时间
- **安全监控**: 异常使用检测

## 🔍 技术优势

### vs 传统MFCC方法
- **端到端学习**: 无需手工特征工程
- **全局上下文**: Transformer捕获长距离依赖
- **鲁棒性强**: 大规模预训练提供泛化能力
- **少样本友好**: 预训练特征减少数据需求

### vs 其他深度学习方法
- **预训练优势**: 利用大规模无标签音频数据
- **计算效率**: 冻结预训练参数，只训练分类头
- **稳定性好**: 避免从头训练的不稳定性

## 📈 性能监控

实时检测器提供以下监控信息：
- 检测历史记录
- 置信度统计
- 检测频率分析
- 音频设备状态

## 🚀 未来改进

- [ ] 支持多类别检测(开/关/故障)
- [ ] 添加音频预处理滤波
- [ ] 优化模型大小(知识蒸馏)
- [ ] 支持边缘设备部署
- [ ] 增加Web界面监控

## 📝 技术细节

详细的技术原理和架构说明请参考：
- `wav2vec2_explanation.md` - Wav2Vec2详细原理
- `wav2vec2_architecture.py` - 架构演示代码
- `wav2vec2_visual_explanation.py` - 可视化说明

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## �� 许可证

MIT License 