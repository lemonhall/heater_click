"""
热水器声音分析脚本
分析转换后的wav文件，为后续的分类任务做准备
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️  librosa未安装，将使用基础分析功能")
    print("   安装命令: pip install librosa")

def analyze_audio_basic(wav_file):
    """基础音频分析（不依赖librosa）"""
    try:
        import wave
        
        with wave.open(wav_file, 'rb') as wav:
            frames = wav.getnframes()
            sample_rate = wav.getframerate()
            duration = frames / sample_rate
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            
            return {
                'duration': duration,
                'sample_rate': sample_rate,
                'frames': frames,
                'channels': channels,
                'sample_width': sample_width
            }
    except Exception as e:
        print(f"❌ 基础分析失败: {e}")
        return None

def analyze_audio_advanced(wav_file):
    """高级音频分析（使用librosa）"""
    if not LIBROSA_AVAILABLE:
        return analyze_audio_basic(wav_file)
    
    try:
        # 加载音频文件
        y, sr = librosa.load(wav_file, sr=16000)
        
        # 基础统计
        duration = len(y) / sr
        rms_energy = np.sqrt(np.mean(y**2))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # 频谱特征
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # MFCC特征（用于对比）
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        return {
            'duration': duration,
            'sample_rate': sr,
            'samples': len(y),
            'rms_energy': rms_energy,
            'zero_crossing_rate': zero_crossing_rate,
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'mfcc_mean': np.mean(mfccs, axis=1),
            'mfcc_std': np.std(mfccs, axis=1),
            'audio_data': y,
            'mfccs': mfccs
        }
        
    except Exception as e:
        print(f"❌ 高级分析失败: {e}")
        return analyze_audio_basic(wav_file)

def visualize_audio_features(wav_files, output_dir="analysis_plots"):
    """可视化音频特征"""
    if not LIBROSA_AVAILABLE:
        print("⚠️  需要librosa进行可视化分析")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('热水器声音特征分析', fontsize=16, weight='bold')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, wav_file in enumerate(wav_files[:6]):  # 最多显示6个文件
        file_name = Path(wav_file).stem
        color = colors[i % len(colors)]
        
        try:
            # 加载音频
            y, sr = librosa.load(wav_file, sr=16000)
            
            # 1. 波形图
            ax = axes[0, 0]
            time = np.linspace(0, len(y)/sr, len(y))
            ax.plot(time, y, alpha=0.7, color=color, label=f'文件{i+1}')
            ax.set_title('音频波形对比')
            ax.set_xlabel('时间 (秒)')
            ax.set_ylabel('幅度')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. 频谱图
            ax = axes[0, 1]
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            if i == 0:  # 只显示第一个文件的频谱图
                librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax)
                ax.set_title(f'频谱图 - {file_name}')
                plt.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
            
            # 3. MFCC特征
            ax = axes[0, 2]
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            if i == 0:  # 只显示第一个文件的MFCC
                librosa.display.specshow(mfccs, x_axis='time', ax=ax)
                ax.set_title(f'MFCC特征 - {file_name}')
                plt.colorbar(ax.collections[0], ax=ax)
            
            # 4. 频谱质心
            ax = axes[1, 0]
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            time_frames = librosa.frames_to_time(np.arange(len(spectral_centroids)), sr=sr)
            ax.plot(time_frames, spectral_centroids, alpha=0.7, color=color, label=f'文件{i+1}')
            ax.set_title('频谱质心')
            ax.set_xlabel('时间 (秒)')
            ax.set_ylabel('频率 (Hz)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 5. 过零率
            ax = axes[1, 1]
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            time_frames = librosa.frames_to_time(np.arange(len(zcr)), sr=sr)
            ax.plot(time_frames, zcr, alpha=0.7, color=color, label=f'文件{i+1}')
            ax.set_title('过零率')
            ax.set_xlabel('时间 (秒)')
            ax.set_ylabel('过零率')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 6. RMS能量
            ax = axes[1, 2]
            rms = librosa.feature.rms(y=y)[0]
            time_frames = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
            ax.plot(time_frames, rms, alpha=0.7, color=color, label=f'文件{i+1}')
            ax.set_title('RMS能量')
            ax.set_xlabel('时间 (秒)')
            ax.set_ylabel('RMS')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"❌ 可视化失败 {file_name}: {e}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heater_audio_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 分析图表已保存到: {output_dir}/heater_audio_analysis.png")

def compare_audio_features(wav_files):
    """比较不同音频文件的特征"""
    print("\n🔍 音频特征对比分析:")
    print("=" * 80)
    
    features_list = []
    
    for wav_file in wav_files:
        file_name = Path(wav_file).stem
        features = analyze_audio_advanced(wav_file)
        
        if features:
            features_list.append((file_name, features))
            
            print(f"\n📊 {file_name}:")
            print(f"   时长: {features['duration']:.2f} 秒")
            print(f"   采样率: {features['sample_rate']} Hz")
            
            if 'rms_energy' in features:
                print(f"   RMS能量: {features['rms_energy']:.4f}")
                print(f"   过零率: {features['zero_crossing_rate']:.4f}")
                print(f"   频谱质心: {features['spectral_centroid_mean']:.1f} ± {features['spectral_centroid_std']:.1f} Hz")
                print(f"   频谱滚降: {features['spectral_rolloff_mean']:.1f} Hz")
    
    # 特征相似性分析
    if len(features_list) > 1 and LIBROSA_AVAILABLE:
        print(f"\n🎯 特征相似性分析:")
        print("-" * 40)
        
        # 计算MFCC特征的相似性
        mfcc_features = []
        names = []
        
        for name, features in features_list:
            if 'mfcc_mean' in features:
                mfcc_features.append(features['mfcc_mean'])
                names.append(name)
        
        if len(mfcc_features) > 1:
            mfcc_features = np.array(mfcc_features)
            
            # 计算欧几里得距离矩阵
            from scipy.spatial.distance import pdist, squareform
            distances = pdist(mfcc_features, metric='euclidean')
            distance_matrix = squareform(distances)
            
            print("MFCC特征距离矩阵 (数值越小越相似):")
            print("文件编号:", [f"{i+1}" for i in range(len(names))])
            
            for i, name in enumerate(names):
                row_str = f"文件{i+1}: "
                for j in range(len(names)):
                    if i != j:
                        row_str += f"{distance_matrix[i,j]:.2f}  "
                    else:
                        row_str += "0.00  "
                print(row_str)

def main():
    """主函数"""
    print("🎵 热水器声音分析工具")
    print("=" * 50)
    
    # 查找wav文件
    wav_files = glob.glob("samples_wav/*.wav")
    
    if not wav_files:
        print("❌ 在samples_wav目录中没有找到wav文件")
        print("   请先运行convert_audio.py转换音频文件")
        return
    
    wav_files = sorted(wav_files)
    print(f"🎵 找到 {len(wav_files)} 个wav文件")
    
    # 基础分析
    compare_audio_features(wav_files)
    
    # 可视化分析
    if LIBROSA_AVAILABLE:
        print(f"\n📊 生成可视化分析图表...")
        visualize_audio_features(wav_files)
    
    print(f"\n💡 分析建议:")
    print("1. 观察不同文件的时长和能量特征")
    print("2. 比较频谱质心和过零率的差异")
    print("3. 查看MFCC特征的相似性，判断是否为同类声音")
    print("4. 这些特征将用于训练Wav2Vec2分类器")

if __name__ == "__main__":
    main() 