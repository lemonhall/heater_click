"""
实时麦克风热水器开关声音检测器
使用训练好的Wav2Vec2模型进行实时检测
"""

import torch
import torch.nn as nn
import numpy as np
import pyaudio
import threading
import time
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import queue
import warnings
warnings.filterwarnings("ignore")

# 检查依赖
try:
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("❌ 需要安装transformers库")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("❌ 需要安装pyaudio库")
    print("   安装命令: pip install pyaudio")

class Wav2Vec2Classifier(nn.Module):
    """基于Wav2Vec2的分类器（与训练时相同的架构）"""
    
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
        with torch.no_grad():
            outputs = self.wav2vec2(input_values)
            features = outputs.last_hidden_state  # [batch, time, 768]
        
        # 全局平均池化: [batch, time, 768] -> [batch, 768]
        pooled = features.mean(dim=1)
        
        # 分类
        logits = self.classifier(pooled)
        
        return logits

class RealtimeSwitchDetector:
    """实时开关声音检测器"""
    
    def __init__(self, model_path="switch_detector_model.pth", 
                 sample_rate=16000, chunk_size=1024, 
                 detection_window=3.0, confidence_threshold=0.93):
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.detection_window = detection_window
        self.confidence_threshold = confidence_threshold
        
        # 音频缓冲区
        self.audio_buffer = deque(maxlen=int(sample_rate * detection_window))
        
        # 检测结果队列
        self.detection_queue = queue.Queue()
        
        # 控制标志
        self.is_running = False
        self.is_detecting = False
        
        # 加载模型
        self.load_model(model_path)
        
        # 初始化音频
        self.init_audio()
        
        # 检测历史
        self.detection_history = deque(maxlen=100)
        self.last_detection_time = 0
        
        print("🎯 实时开关声音检测器已初始化")
        print(f"   采样率: {sample_rate} Hz")
        print(f"   检测窗口: {detection_window} 秒")
        print(f"   置信度阈值: {confidence_threshold}")
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        try:
            # 加载模型
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model = Wav2Vec2Classifier(num_classes=2)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # 加载特征提取器
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
            
            print(f"✅ 模型加载成功: {model_path}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def init_audio(self):
        """初始化音频输入"""
        if not PYAUDIO_AVAILABLE:
            raise ImportError("需要安装pyaudio库")
        
        self.audio = pyaudio.PyAudio()
        
        # 查找可用的音频设备
        self.list_audio_devices()
        
        # 打开音频流
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        
        print("🎤 音频输入已初始化")
    
    def list_audio_devices(self):
        """列出可用的音频设备"""
        print("\n🎤 可用音频设备:")
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"   设备 {i}: {info['name']} (输入通道: {info['maxInputChannels']})")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数"""
        if self.is_running:
            # 转换音频数据
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # 添加到缓冲区
            self.audio_buffer.extend(audio_data)
            
            # 如果缓冲区满了，进行检测
            if len(self.audio_buffer) >= int(self.sample_rate * self.detection_window):
                if not self.is_detecting:
                    # 启动检测线程
                    detection_thread = threading.Thread(target=self.detect_switch)
                    detection_thread.daemon = True
                    detection_thread.start()
        
        return (in_data, pyaudio.paContinue)
    
    def detect_switch(self):
        """检测开关声音"""
        if self.is_detecting:
            return
        
        self.is_detecting = True
        
        try:
            # 获取音频数据
            audio_data = np.array(list(self.audio_buffer))
            
            # 确保长度正确
            target_length = int(self.sample_rate * self.detection_window)
            if len(audio_data) > target_length:
                audio_data = audio_data[-target_length:]
            elif len(audio_data) < target_length:
                # 零填充
                padding = target_length - len(audio_data)
                audio_data = np.pad(audio_data, (0, padding), 'constant')
            
            # 预处理音频
            inputs = self.feature_extractor(
                audio_data,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model(inputs.input_values)
                probabilities = torch.softmax(outputs, dim=1)
                confidence = probabilities[0][1].item()  # 开关声音的概率
                prediction = 1 if confidence > self.confidence_threshold else 0
            
            # 记录检测结果
            current_time = time.time()
            result = {
                'timestamp': current_time,
                'prediction': prediction,
                'confidence': confidence,
                'datetime': datetime.now().strftime("%H:%M:%S")
            }
            
            # 添加到历史记录
            self.detection_history.append(result)
            
            # 如果检测到开关声音
            if prediction == 1:
                # 避免重复检测（2秒内只报告一次）
                if current_time - self.last_detection_time > 2.0:
                    self.last_detection_time = current_time
                    print(f"🔔 [{result['datetime']}] 检测到开关声音! 置信度: {confidence:.3f}")
                    
                    # 添加到检测队列
                    self.detection_queue.put(result)
            
        except Exception as e:
            print(f"❌ 检测过程出错: {e}")
        
        finally:
            self.is_detecting = False
    
    def start_detection(self):
        """开始检测"""
        print("\n🚀 开始实时检测...")
        print("按 Ctrl+C 停止检测")
        
        self.is_running = True
        self.stream.start_stream()
        
        try:
            while self.is_running:
                time.sleep(0.1)
                
                # 处理检测结果队列
                try:
                    while not self.detection_queue.empty():
                        result = self.detection_queue.get_nowait()
                        # 这里可以添加其他处理逻辑，比如发送通知等
                        pass
                except queue.Empty:
                    pass
                    
        except KeyboardInterrupt:
            print("\n⏹️  检测已停止")
        finally:
            self.stop_detection()
    
    def stop_detection(self):
        """停止检测"""
        self.is_running = False
        
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        
        if hasattr(self, 'audio'):
            self.audio.terminate()
        
        print("🔚 检测器已关闭")
    
    def get_detection_stats(self):
        """获取检测统计信息"""
        if not self.detection_history:
            return "暂无检测数据"
        
        total_detections = len(self.detection_history)
        switch_detections = sum(1 for d in self.detection_history if d['prediction'] == 1)
        
        recent_detections = [d for d in self.detection_history 
                           if time.time() - d['timestamp'] < 60]  # 最近1分钟
        recent_switch_count = sum(1 for d in recent_detections if d['prediction'] == 1)
        
        stats = f"""
📊 检测统计 (最近1分钟):
   总检测次数: {len(recent_detections)}
   开关检测次数: {recent_switch_count}
   平均置信度: {np.mean([d['confidence'] for d in recent_detections]):.3f}
        """
        
        return stats

def main():
    """主函数"""
    
    if not TRANSFORMERS_AVAILABLE or not PYAUDIO_AVAILABLE:
        print("❌ 缺少必要的依赖库")
        return
    
    print("🎯 热水器开关声音实时检测器")
    print("=" * 50)
    
    # 检查模型文件
    model_path = "switch_detector_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("   请先运行训练脚本生成模型")
        return
    
    try:
        # 创建检测器
        detector = RealtimeSwitchDetector(
            model_path=model_path,
            confidence_threshold=0.93  # 调整置信度阈值，只有高置信度才认为是开关声音
        )
        
        # 开始检测
        detector.start_detection()
        
    except Exception as e:
        print(f"❌ 检测器启动失败: {e}")
    
    print("\n👋 再见!")

if __name__ == "__main__":
    import os
    main() 