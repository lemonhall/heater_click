"""
热水器开关声音检测器 - 使用示例
"""

from huggingface_hub import hf_hub_download
import torch
import torchaudio
from transformers import Wav2Vec2Model

def load_model(repo_id="your-username/heater-switch-detector"):
    """加载模型"""
    
    print("📥 下载模型...")
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="switch_detector_model.pth"
    )
    
    print("🔧 加载模型...")
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
    wav2vec2_model.eval()
    
    return wav2vec2_model, classifier

def predict_audio(audio_path, wav2vec2_model, classifier):
    """预测音频"""
    
    # 加载音频
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # 重采样到16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # 转为单声道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # 特征提取和预测
    with torch.no_grad():
        features = wav2vec2_model(waveform).last_hidden_state
        pooled_features = features.mean(dim=1)
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

if __name__ == "__main__":
    # 加载模型
    wav2vec2_model, classifier = load_model()
    
    # 测试音频文件
    test_file = input("请输入音频文件路径: ").strip()
    
    if test_file and os.path.exists(test_file):
        result = predict_audio(test_file, wav2vec2_model, classifier)
        print(f"\n🎯 预测结果: {result['prediction']}")
        print(f"📊 置信度: {result['confidence']:.3f}")
        print(f"📈 详细概率:")
        for label, prob in result['probabilities'].items():
            print(f"   {label}: {prob:.3f}")
    else:
        print("❌ 文件不存在")
