"""
模型下载脚本
用于下载预训练的热水器开关声音检测模型
"""

import os
import requests
from pathlib import Path
import hashlib

def download_file(url, filename, expected_size=None, chunk_size=8192):
    """下载文件并显示进度"""
    
    print(f"📥 开始下载: {filename}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r📊 下载进度: {progress:.1f}% ({downloaded_size}/{total_size} bytes)", end='')
        
        print(f"\n✅ 下载完成: {filename}")
        
        # 验证文件大小
        if expected_size and os.path.getsize(filename) != expected_size:
            print(f"⚠️  文件大小不匹配，可能下载不完整")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def verify_model_file(filepath, expected_hash=None):
    """验证模型文件完整性"""
    
    if not os.path.exists(filepath):
        return False
    
    # 检查文件大小（大约361MB）
    file_size = os.path.getsize(filepath)
    if file_size < 300_000_000 or file_size > 400_000_000:
        print(f"⚠️  模型文件大小异常: {file_size} bytes")
        return False
    
    # 如果提供了哈希值，验证文件完整性
    if expected_hash:
        print("🔍 验证文件完整性...")
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        if sha256_hash.hexdigest() != expected_hash:
            print("❌ 文件哈希值不匹配")
            return False
    
    print("✅ 模型文件验证通过")
    return True

def download_from_huggingface(repo_id):
    """从Hugging Face Hub下载模型"""
    
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"📥 从Hugging Face下载模型: {repo_id}")
        
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="switch_detector_model.pth",
            local_dir=".",
            local_dir_use_symlinks=False
        )
        
        print(f"✅ 下载完成: {model_path}")
        return True
        
    except ImportError:
        print("❌ 需要安装huggingface_hub:")
        print("   pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def download_pretrained_model():
    """下载预训练模型"""
    
    model_filename = "switch_detector_model.pth"
    
    # 检查模型是否已存在
    if os.path.exists(model_filename):
        if verify_model_file(model_filename):
            print(f"✅ 模型文件已存在且有效: {model_filename}")
            return True
        else:
            print(f"⚠️  现有模型文件无效，重新下载...")
            os.remove(model_filename)
    
    print("🎯 热水器开关声音检测模型下载器")
    print("=" * 50)
    
    # 模型下载选项
    download_options = [
        {
            "name": "Hugging Face Hub (推荐)",
            "repo_id": "lemonhall/heater-switch-detector",
            "description": "从Hugging Face模型库下载，支持大文件 - 已上传成功！"
        },
        {
            "name": "自定义Hugging Face仓库",
            "repo_id": "custom",
            "description": "输入自定义的Hugging Face仓库ID"
        }
    ]
    
    print("📦 可用下载源:")
    for i, option in enumerate(download_options, 1):
        print(f"   {i}. {option['name']}: {option['description']}")
    
    print(f"\n💡 提示:")
    print(f"   - 模型文件大小约361MB")
    print(f"   - 需要稳定的网络连接")
    print(f"   - 下载完成后会自动验证文件完整性")
    
    # 选择下载源
    choice = input(f"\n请选择下载源 (1-{len(download_options)}): ").strip()
    
    try:
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(download_options):
            option = download_options[choice_idx]
            
            if option["repo_id"] == "custom":
                repo_id = input("请输入Hugging Face仓库ID (格式: username/repo-name): ").strip()
                if not repo_id or '/' not in repo_id:
                    print("❌ 无效的仓库ID格式")
                    return False
            else:
                repo_id = option["repo_id"]
            
            print(f"\n🚀 开始从 {repo_id} 下载...")
            return download_from_huggingface(repo_id)
        else:
            print("❌ 无效选择")
            return False
            
    except ValueError:
        print("❌ 请输入有效数字")
        return False

def setup_model():
    """设置模型文件"""
    
    model_filename = "switch_detector_model.pth"
    
    if os.path.exists(model_filename):
        if verify_model_file(model_filename):
            print("✅ 模型已准备就绪")
            return True
    
    print("🔍 未找到有效的模型文件")
    
    # 提供选项
    print("\n📋 获取模型的方式:")
    print("   1. 下载预训练模型 (如果可用)")
    print("   2. 训练新模型")
    
    choice = input("\n请选择 (1/2): ").strip()
    
    if choice == "1":
        return download_pretrained_model()
    elif choice == "2":
        print("\n🚀 开始训练新模型...")
        print("   运行命令: python wav2vec2_switch_detector.py")
        return False
    else:
        print("❌ 无效选择")
        return False

def main():
    """主函数"""
    
    print("🎯 热水器开关声音检测器 - 模型设置")
    print("=" * 50)
    
    if setup_model():
        print("\n🎉 模型设置完成！")
        print("   现在可以运行实时检测:")
        print("   python realtime_mic_detector.py")
    else:
        print("\n💡 模型设置未完成")
        print("   请按照提示完成模型获取")

if __name__ == "__main__":
    main() 