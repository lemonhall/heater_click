"""
改进的数据增强脚本
用于生成更多样化和真实的负样本，提高模型鲁棒性
"""

import os
import glob
import numpy as np
import torch
import torchaudio
import random
from pathlib import Path

class AdvancedDataAugmentation:
    """高级数据增强器"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def add_gaussian_noise(self, waveform, noise_level=0.01):
        """添加高斯噪音"""
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def add_background_hum(self, waveform, freq=50, amplitude=0.005):
        """添加电器背景嗡嗡声（50Hz/60Hz）"""
        t = torch.linspace(0, len(waveform)/self.sample_rate, len(waveform))
        hum = amplitude * torch.sin(2 * np.pi * freq * t)
        return waveform + hum
    
    def simulate_room_reverb(self, waveform, decay=0.3, delay_samples=800):
        """模拟房间混响"""
        reverb = torch.zeros_like(waveform)
        reverb[delay_samples:] = waveform[:-delay_samples] * decay
        return waveform + reverb
    
    def simulate_distant_sound(self, waveform, attenuation=0.3, lpf_cutoff=0.3):
        """模拟远距离声音（衰减+低通滤波）"""
        # 简单的低通滤波效果
        filtered = waveform * attenuation
        # 添加轻微的混响
        return self.simulate_room_reverb(filtered, decay=0.2, delay_samples=400)
    
    def time_stretch(self, waveform, stretch_factor=1.2):
        """时间拉伸（改变语速但不改变音调）"""
        # 简单的重采样实现
        original_length = len(waveform)
        new_length = int(original_length * stretch_factor)
        indices = torch.linspace(0, original_length-1, new_length)
        return torch.nn.functional.interpolate(
            waveform.unsqueeze(0).unsqueeze(0), 
            size=new_length, 
            mode='linear'
        ).squeeze()
    
    def pitch_shift(self, waveform, shift_semitones=2):
        """音调变换"""
        # 简单的重采样实现音调变换
        shift_factor = 2 ** (shift_semitones / 12.0)
        stretched = self.time_stretch(waveform, 1/shift_factor)
        # 裁剪或填充到原始长度
        if len(stretched) > len(waveform):
            return stretched[:len(waveform)]
        else:
            padding = len(waveform) - len(stretched)
            return torch.nn.functional.pad(stretched, (0, padding))
    
    def generate_synthetic_negative_samples(self, output_dir="samples_wav", num_samples=20):
        """生成合成负样本"""
        
        print(f"🔄 生成 {num_samples} 个合成负样本...")
        
        negative_files = []
        
        for i in range(num_samples):
            # 生成不同类型的负样本
            sample_type = i % 8
            
            if sample_type == 0:
                # 纯噪音
                duration = random.uniform(2.0, 5.0)
                samples = int(duration * self.sample_rate)
                waveform = torch.randn(samples) * 0.02
                filename = f"synthetic_noise_{i+1:02d}.wav"
                
            elif sample_type == 1:
                # 电器嗡嗡声
                duration = random.uniform(3.0, 6.0)
                samples = int(duration * self.sample_rate)
                base_noise = torch.randn(samples) * 0.01
                waveform = self.add_background_hum(base_noise, freq=random.choice([50, 60, 120]))
                filename = f"synthetic_hum_{i+1:02d}.wav"
                
            elif sample_type == 2:
                # 模拟敲击声
                duration = random.uniform(1.0, 3.0)
                samples = int(duration * self.sample_rate)
                # 生成短促的脉冲
                waveform = torch.zeros(samples)
                num_taps = random.randint(1, 5)
                for _ in range(num_taps):
                    pos = random.randint(0, samples-1000)
                    pulse = torch.exp(-torch.linspace(0, 5, 1000)) * random.uniform(0.1, 0.3)
                    waveform[pos:pos+1000] += pulse
                filename = f"synthetic_tap_{i+1:02d}.wav"
                
            elif sample_type == 3:
                # 模拟风声
                duration = random.uniform(4.0, 8.0)
                samples = int(duration * self.sample_rate)
                # 低频噪音模拟风声
                waveform = torch.randn(samples) * 0.03
                # 添加低频成分
                t = torch.linspace(0, duration, samples)
                low_freq = 0.01 * torch.sin(2 * np.pi * random.uniform(0.5, 2.0) * t)
                waveform += low_freq
                filename = f"synthetic_wind_{i+1:02d}.wav"
                
            elif sample_type == 4:
                # 模拟机械声
                duration = random.uniform(2.0, 4.0)
                samples = int(duration * self.sample_rate)
                base_freq = random.uniform(100, 500)
                t = torch.linspace(0, duration, samples)
                waveform = 0.05 * torch.sin(2 * np.pi * base_freq * t)
                # 添加谐波
                waveform += 0.02 * torch.sin(2 * np.pi * base_freq * 2 * t)
                waveform += torch.randn(samples) * 0.01
                filename = f"synthetic_mechanical_{i+1:02d}.wav"
                
            elif sample_type == 5:
                # 模拟远距离声音
                duration = random.uniform(3.0, 6.0)
                samples = int(duration * self.sample_rate)
                base_sound = torch.randn(samples) * 0.1
                waveform = self.simulate_distant_sound(base_sound)
                filename = f"synthetic_distant_{i+1:02d}.wav"
                
            elif sample_type == 6:
                # 模拟间歇性声音
                duration = random.uniform(4.0, 7.0)
                samples = int(duration * self.sample_rate)
                waveform = torch.zeros(samples)
                # 添加随机的短促声音
                num_bursts = random.randint(3, 8)
                for _ in range(num_bursts):
                    start = random.randint(0, samples-2000)
                    burst_len = random.randint(500, 2000)
                    burst = torch.randn(burst_len) * random.uniform(0.02, 0.08)
                    waveform[start:start+burst_len] += burst
                filename = f"synthetic_intermittent_{i+1:02d}.wav"
                
            else:
                # 混合多种噪音
                duration = random.uniform(3.0, 5.0)
                samples = int(duration * self.sample_rate)
                waveform = torch.randn(samples) * 0.01
                waveform = self.add_background_hum(waveform, freq=random.choice([50, 60]))
                waveform = self.add_gaussian_noise(waveform, noise_level=0.005)
                filename = f"synthetic_mixed_{i+1:02d}.wav"
            
            # 保存文件
            output_path = os.path.join(output_dir, filename)
            torchaudio.save(output_path, waveform.unsqueeze(0), self.sample_rate)
            negative_files.append(output_path)
            
            print(f"✅ 生成负样本: {filename}")
        
        return negative_files
    
    def augment_existing_samples(self, input_dir="samples_wav", output_dir="samples_wav_augmented"):
        """对现有样本进行数据增强"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 找到所有正样本
        positive_files = glob.glob(os.path.join(input_dir, "switch_on_*.wav"))
        
        print(f"🔄 对 {len(positive_files)} 个正样本进行数据增强...")
        
        augmented_files = []
        
        for i, file_path in enumerate(positive_files):
            # 加载原始文件
            waveform, sample_rate = torchaudio.load(file_path)
            waveform = waveform.squeeze()
            
            # 重采样到目标采样率
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # 保存原始文件
            original_name = f"switch_on_{i+1:02d}_original.wav"
            original_path = os.path.join(output_dir, original_name)
            torchaudio.save(original_path, waveform.unsqueeze(0), self.sample_rate)
            augmented_files.append(original_path)
            
            # 生成增强版本
            augmentations = [
                ("noise", lambda x: self.add_gaussian_noise(x, 0.01)),
                ("hum", lambda x: self.add_background_hum(x, freq=50)),
                ("reverb", lambda x: self.simulate_room_reverb(x)),
                ("distant", lambda x: self.simulate_distant_sound(x)),
                ("pitch_up", lambda x: self.pitch_shift(x, 1)),
                ("pitch_down", lambda x: self.pitch_shift(x, -1)),
            ]
            
            for aug_name, aug_func in augmentations:
                try:
                    augmented = aug_func(waveform)
                    aug_filename = f"switch_on_{i+1:02d}_{aug_name}.wav"
                    aug_path = os.path.join(output_dir, aug_filename)
                    torchaudio.save(aug_path, augmented.unsqueeze(0), self.sample_rate)
                    augmented_files.append(aug_path)
                    print(f"✅ 生成增强样本: {aug_filename}")
                except Exception as e:
                    print(f"❌ 增强失败 {aug_name}: {e}")
        
        return augmented_files

def main():
    """主函数"""
    
    print("🎯 高级数据增强工具")
    print("=" * 50)
    
    augmenter = AdvancedDataAugmentation()
    
    # 生成合成负样本
    synthetic_negatives = augmenter.generate_synthetic_negative_samples(num_samples=30)
    
    # 对现有正样本进行增强
    augmented_positives = augmenter.augment_existing_samples()
    
    print(f"\n📊 数据增强完成:")
    print(f"   合成负样本: {len(synthetic_negatives)} 个")
    print(f"   增强正样本: {len(augmented_positives)} 个")
    print(f"\n💡 建议:")
    print(f"   1. 收集真实的环境声音作为负样本")
    print(f"   2. 录制不同环境下的开关声音作为正样本")
    print(f"   3. 使用增强后的数据重新训练模型")

if __name__ == "__main__":
    main() 