"""
音频格式转换脚本
将m4a文件批量转换为wav格式，用于后续的音频分析
"""

import os
import subprocess
import glob
from pathlib import Path

def convert_m4a_to_wav(input_dir="samples", output_dir="samples_wav"):
    """
    批量将m4a文件转换为wav格式
    
    Args:
        input_dir: 输入目录（包含m4a文件）
        output_dir: 输出目录（存放wav文件）
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有m4a文件
    m4a_files = glob.glob(os.path.join(input_dir, "*.m4a"))
    
    if not m4a_files:
        print(f"❌ 在 {input_dir} 目录中没有找到m4a文件")
        return
    
    print(f"🎵 找到 {len(m4a_files)} 个m4a文件")
    print("开始转换...\n")
    
    success_count = 0
    
    for m4a_file in m4a_files:
        # 获取文件名（不含扩展名）
        file_name = Path(m4a_file).stem
        output_file = os.path.join(output_dir, f"{file_name}.wav")
        
        print(f"🔄 转换: {os.path.basename(m4a_file)} -> {os.path.basename(output_file)}")
        
        try:
            # 使用ffmpeg转换
            # -i: 输入文件
            # -ar 16000: 设置采样率为16kHz（Wav2Vec2的标准采样率）
            # -ac 1: 转换为单声道
            # -y: 覆盖输出文件
            cmd = [
                "ffmpeg",
                "-i", m4a_file,
                "-ar", "16000",  # 16kHz采样率
                "-ac", "1",      # 单声道
                "-y",            # 覆盖已存在的文件
                output_file
            ]
            
            # 执行转换命令（隐藏输出）
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            print(f"✅ 成功转换: {os.path.basename(output_file)}")
            success_count += 1
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 转换失败: {os.path.basename(m4a_file)}")
            print(f"   错误信息: {e.stderr}")
        except FileNotFoundError:
            print("❌ 错误: 找不到ffmpeg命令")
            print("   请确保ffmpeg已正确安装并添加到PATH环境变量中")
            break
        except Exception as e:
            print(f"❌ 未知错误: {e}")
    
    print(f"\n🎯 转换完成！成功转换 {success_count}/{len(m4a_files)} 个文件")
    
    if success_count > 0:
        print(f"📁 转换后的wav文件保存在: {output_dir}")
        
        # 显示转换后的文件信息
        wav_files = glob.glob(os.path.join(output_dir, "*.wav"))
        print("\n📋 转换后的文件列表:")
        for wav_file in sorted(wav_files):
            file_size = os.path.getsize(wav_file) / 1024  # KB
            print(f"   {os.path.basename(wav_file)} ({file_size:.1f} KB)")

def check_ffmpeg():
    """检查ffmpeg是否可用"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print("✅ ffmpeg 可用")
        # 提取版本信息
        version_line = result.stdout.split('\n')[0]
        print(f"   版本: {version_line}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg 不可用")
        print("   请确保ffmpeg已安装并添加到PATH环境变量中")
        return False

def analyze_audio_files(directory="samples_wav"):
    """分析转换后的音频文件"""
    try:
        import librosa
        import numpy as np
        
        wav_files = glob.glob(os.path.join(directory, "*.wav"))
        
        if not wav_files:
            print(f"❌ 在 {directory} 目录中没有找到wav文件")
            return
        
        print(f"\n🔍 分析 {len(wav_files)} 个wav文件:")
        
        for wav_file in sorted(wav_files):
            try:
                # 加载音频文件
                y, sr = librosa.load(wav_file, sr=None)
                duration = len(y) / sr
                
                print(f"📊 {os.path.basename(wav_file)}:")
                print(f"   采样率: {sr} Hz")
                print(f"   时长: {duration:.2f} 秒")
                print(f"   样本数: {len(y)}")
                print(f"   最大幅度: {np.max(np.abs(y)):.3f}")
                print()
                
            except Exception as e:
                print(f"❌ 分析失败 {os.path.basename(wav_file)}: {e}")
                
    except ImportError:
        print("💡 提示: 安装librosa可以获得更详细的音频分析")
        print("   pip install librosa")

if __name__ == "__main__":
    print("🎵 音频格式转换工具")
    print("=" * 50)
    
    # 检查ffmpeg
    if not check_ffmpeg():
        exit(1)
    
    print()
    
    # 执行转换
    convert_m4a_to_wav()
    
    # 分析转换后的文件
    analyze_audio_files()
    
    print("\n🎯 转换完成！现在可以使用wav文件进行音频分析了。") 