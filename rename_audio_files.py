"""
重命名音频文件脚本
将热水器开关声音文件重命名为更简洁的格式
"""

import os
import glob
from pathlib import Path

def rename_audio_files(input_dir="samples_wav", prefix="switch_on"):
    """
    重命名音频文件为更简洁的格式
    
    Args:
        input_dir: 输入目录
        prefix: 文件名前缀
    """
    
    # 查找所有wav文件
    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    
    if not wav_files:
        print(f"❌ 在 {input_dir} 目录中没有找到wav文件")
        return
    
    # 按文件名排序
    wav_files = sorted(wav_files)
    
    print(f"🎵 找到 {len(wav_files)} 个wav文件")
    print("开始重命名...\n")
    
    success_count = 0
    
    for i, old_file in enumerate(wav_files, 1):
        # 生成新文件名
        new_filename = f"{prefix}_{i:02d}.wav"
        new_file = os.path.join(input_dir, new_filename)
        
        old_basename = os.path.basename(old_file)
        
        try:
            # 重命名文件
            os.rename(old_file, new_file)
            print(f"✅ {old_basename} -> {new_filename}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 重命名失败: {old_basename}")
            print(f"   错误信息: {e}")
    
    print(f"\n🎯 重命名完成！成功重命名 {success_count}/{len(wav_files)} 个文件")
    
    # 显示重命名后的文件列表
    if success_count > 0:
        print(f"\n📋 重命名后的文件列表:")
        new_wav_files = glob.glob(os.path.join(input_dir, f"{prefix}_*.wav"))
        for wav_file in sorted(new_wav_files):
            file_size = os.path.getsize(wav_file) / 1024  # KB
            print(f"   {os.path.basename(wav_file)} ({file_size:.1f} KB)")

def create_labels_file(input_dir="samples_wav", prefix="switch_on", output_file="labels.txt"):
    """
    创建标签文件，标记所有开关声音为正样本
    
    Args:
        input_dir: 音频文件目录
        prefix: 文件名前缀
        output_file: 输出标签文件
    """
    
    # 查找重命名后的文件
    wav_files = glob.glob(os.path.join(input_dir, f"{prefix}_*.wav"))
    wav_files = sorted(wav_files)
    
    if not wav_files:
        print(f"❌ 没有找到 {prefix}_*.wav 文件")
        return
    
    # 创建标签文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 热水器开关声音标签文件\n")
        f.write("# 格式: 文件名,标签 (1=有开关按下, 0=无开关按下)\n")
        f.write("filename,label\n")
        
        for wav_file in wav_files:
            filename = os.path.basename(wav_file)
            # 所有文件都标记为1（有开关按下）
            f.write(f"{filename},1\n")
    
    print(f"📝 标签文件已创建: {output_file}")
    print(f"   包含 {len(wav_files)} 个正样本（开关按下声音）")
    
    # 显示标签文件内容
    print(f"\n📋 标签文件内容:")
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith('#') and line.strip():
                print(f"   {line.strip()}")

def main():
    """主函数"""
    print("🏷️  音频文件重命名工具")
    print("=" * 50)
    
    # 重命名文件
    rename_audio_files()
    
    print()
    
    # 创建标签文件
    create_labels_file()
    
    print(f"\n💡 说明:")
    print("1. 所有文件已重命名为 switch_on_XX.wav 格式")
    print("2. 创建了标签文件，标记所有文件为正样本")
    print("3. 接下来需要收集负样本（背景噪音）用于训练")
    print("4. 或者使用数据增强技术生成负样本")

if __name__ == "__main__":
    main() 