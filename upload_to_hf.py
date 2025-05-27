"""
上传模型到Hugging Face Hub
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
import torch

def setup_hf_repo():
    """设置Hugging Face仓库"""
    
    print("🤗 Hugging Face模型上传工具")
    print("=" * 50)
    
    # 获取用户信息
    username = input("请输入你的Hugging Face用户名: ").strip()
    if not username:
        print("❌ 用户名不能为空")
        return None, None
    
    repo_name = input("请输入仓库名称 (默认: heater-switch-detector): ").strip()
    if not repo_name:
        repo_name = "heater-switch-detector"
    
    repo_id = f"{username}/{repo_name}"
    
    print(f"\n📋 仓库信息:")
    print(f"   用户名: {username}")
    print(f"   仓库名: {repo_name}")
    print(f"   完整ID: {repo_id}")
    
    return repo_id, repo_name

def verify_model_file():
    """验证模型文件是否存在"""
    
    model_path = "switch_detector_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("   请先运行训练脚本生成模型:")
        print("   python wav2vec2_switch_detector.py")
        return False
    
    # 检查文件大小
    file_size = os.path.getsize(model_path)
    size_mb = file_size / (1024 * 1024)
    
    print(f"✅ 找到模型文件: {model_path}")
    print(f"   文件大小: {size_mb:.1f} MB")
    
    if size_mb < 300 or size_mb > 400:
        print(f"⚠️  文件大小异常，预期约361MB")
        confirm = input("是否继续上传? (y/N): ").strip().lower()
        if confirm != 'y':
            return False
    
    return True

def create_model_config():
    """创建模型配置文件"""
    
    config_content = """{
  "model_type": "wav2vec2_classifier",
  "base_model": "facebook/wav2vec2-base",
  "num_classes": 2,
  "class_names": ["background", "switch"],
  "sample_rate": 16000,
  "max_length": 80000,
  "feature_size": 768,
  "classifier_config": {
    "hidden_size": 256,
    "dropout": 0.3,
    "num_layers": 2
  },
  "training_config": {
    "epochs": 15,
    "learning_rate": 1e-4,
    "batch_size": 4,
    "optimizer": "AdamW",
    "loss_function": "CrossEntropyLoss"
  },
  "performance": {
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1_score": 1.0
  }
}"""
    
    with open("config.json", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print("✅ 创建配置文件: config.json")

def create_model_card(repo_id):
    """创建模型卡片"""
    
    # 直接使用README_HF.md作为模型卡片
    if os.path.exists("README_HF.md"):
        # 读取并更新README_HF.md中的用户名
        with open("README_HF.md", "r", encoding="utf-8") as f:
            content = f.read()
        
        # 更新占位符
        content = content.replace("your-username/heater-switch-detector", repo_id)
        content = content.replace("your-username/heater_click", repo_id.split('/')[0] + "/heater_click")
        content = content.replace("Your Name", repo_id.split('/')[0])
        content = content.replace("your.email@example.com", f"{repo_id.split('/')[0]}@example.com")
        
        # 保存为README.md (Hugging Face要求的文件名)
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(content)
        
        print("✅ 创建模型卡片: README.md (基于README_HF.md)")
    else:
        print("❌ 未找到README_HF.md模板")
        return False
    
    return True



def upload_to_huggingface(repo_id):
    """上传到Hugging Face"""
    
    try:
        # 初始化API
        api = HfApi()
        
        print(f"🚀 开始上传到 {repo_id}...")
        
        # 创建仓库
        try:
            create_repo(repo_id, exist_ok=True, private=False)
            print(f"✅ 仓库创建成功: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"⚠️  仓库可能已存在: {e}")
        
        # 上传文件列表
        files_to_upload = [
            ("switch_detector_model.pth", "模型文件"),
            ("README.md", "模型卡片"),
            ("config.json", "配置文件")
        ]
        
        for filename, description in files_to_upload:
            if os.path.exists(filename):
                print(f"📤 上传{description}: {filename}")
                upload_file(
                    path_or_fileobj=filename,
                    path_in_repo=filename,
                    repo_id=repo_id,
                    commit_message=f"Add {description}"
                )
                print(f"✅ {filename} 上传成功")
            else:
                print(f"⚠️  跳过不存在的文件: {filename}")
        
        print(f"\n🎉 上传完成!")
        print(f"🔗 模型地址: https://huggingface.co/{repo_id}")
        print(f"📖 使用方法:")
        print(f"   from huggingface_hub import hf_hub_download")
        print(f"   model_path = hf_hub_download('{repo_id}', 'switch_detector_model.pth')")
        
        return True
        
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        print("💡 请检查:")
        print("   1. 是否已登录 Hugging Face: huggingface-cli login")
        print("   2. 网络连接是否正常")
        print("   3. 仓库名称是否有效")
        return False

def main():
    """主函数"""
    
    print("🎯 热水器开关声音检测器 - Hugging Face上传工具")
    print("=" * 60)
    
    # 1. 验证模型文件
    if not verify_model_file():
        return
    
    # 2. 设置仓库信息
    repo_id, repo_name = setup_hf_repo()
    if not repo_id:
        return
    
    # 3. 创建必要文件
    print(f"\n📝 准备上传文件...")
    create_model_config()
    
    if not create_model_card(repo_id):
        return
    
    # 4. 确认上传
    print(f"\n📋 准备上传以下文件:")
    files = ["switch_detector_model.pth", "README.md", "config.json"]
    for f in files:
        if os.path.exists(f):
            size = os.path.getsize(f) / (1024*1024)
            print(f"   ✅ {f} ({size:.1f} MB)")
        else:
            print(f"   ❌ {f} (不存在)")
    
    confirm = input(f"\n确认上传到 {repo_id}? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ 取消上传")
        return
    
    # 5. 上传到Hugging Face
    if upload_to_huggingface(repo_id):
        print(f"\n🎊 恭喜！模型已成功上传到Hugging Face!")
        print(f"   现在其他人可以通过以下方式使用你的模型:")
        print(f"   hf_hub_download('{repo_id}', 'switch_detector_model.pth')")
    
    # 6. 清理临时文件
    temp_files = ["config.json"]  # 不删除README.md，因为它是项目的主要文档
    cleanup = input("\n是否清理临时文件? (y/N): ").strip().lower()
    if cleanup == 'y':
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
                print(f"🗑️  删除: {f}")
        print("💡 保留README.md作为项目主要文档")

if __name__ == "__main__":
    main() 