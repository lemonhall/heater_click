# 🚀 部署指南

## 📦 大文件处理策略

### 问题说明
- 训练好的模型文件 `switch_detector_model.pth` 大小为 **361MB**
- GitHub单文件限制为 **100MB**
- 音频文件和生成的图片也会占用较多空间

### 解决方案

#### 1. Git忽略大文件
已配置 `.gitignore` 排除以下文件：
```
# 模型文件
*.pth
*.pkl
*.h5

# 音频文件
*.wav
*.mp3
*.m4a
samples_wav/

# 生成的图片
*.png
*.jpg
analysis_plots/
```

#### 2. 模型分发选项

**选项A: GitHub Releases**
```bash
# 创建Release并上传大文件
gh release create v1.0 switch_detector_model.pth --title "热水器开关检测器 v1.0"
```

**选项B: Git LFS (Large File Storage)**
```bash
# 安装Git LFS
git lfs install

# 跟踪大文件
git lfs track "*.pth"
git lfs track "*.wav"

# 提交.gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"

# 正常提交大文件
git add switch_detector_model.pth
git commit -m "Add trained model"
git push
```

**选项C: 云存储**
- Google Drive
- OneDrive
- 百度网盘
- 阿里云OSS

**选项D: 模型托管平台**
- Hugging Face Hub
- ModelScope
- TensorFlow Hub

#### 3. 自动化部署脚本

**setup.py** - 一键环境设置
```python
import subprocess
import os

def setup_environment():
    """设置项目环境"""
    
    # 安装依赖
    subprocess.run(["pip", "install", "-r", "requirements.txt"])
    
    # 检查模型文件
    if not os.path.exists("switch_detector_model.pth"):
        print("⚠️  模型文件不存在，请运行:")
        print("   python download_model.py  # 下载预训练模型")
        print("   或")
        print("   python wav2vec2_switch_detector.py  # 训练新模型")
    
    # 检查音频数据
    if not os.path.exists("samples_wav"):
        print("⚠️  音频数据不存在，请运行:")
        print("   python convert_audio.py  # 转换音频格式")

if __name__ == "__main__":
    setup_environment()
```

## 🔄 CI/CD 配置

### GitHub Actions
```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Download model (if available)
      run: |
        python download_model.py || echo "Model download failed, will train new model"
    
    - name: Train model if needed
      run: |
        if [ ! -f "switch_detector_model.pth" ]; then
          echo "Training new model..."
          python wav2vec2_switch_detector.py
        fi
    
    - name: Run tests
      run: |
        python -m pytest tests/
```

## 📋 用户获取指南

### 新用户快速开始

1. **克隆仓库**
```bash
git clone https://github.com/your-username/heater_click.git
cd heater_click
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **获取模型**
```bash
# 方式1: 下载预训练模型
python download_model.py

# 方式2: 训练新模型
python wav2vec2_switch_detector.py
```

4. **开始使用**
```bash
python realtime_mic_detector.py
```

### 开发者贡献指南

1. **Fork项目**
2. **创建功能分支**
```bash
git checkout -b feature/new-feature
```

3. **开发和测试**
```bash
# 不要提交大文件
git add *.py *.md requirements.txt
git commit -m "Add new feature"
```

4. **提交PR**
```bash
git push origin feature/new-feature
```

## 🔧 生产环境部署

### Docker部署
```dockerfile
# Dockerfile
FROM python:3.8-slim

WORKDIR /app

# 复制代码
COPY *.py requirements.txt ./
COPY *.md ./

# 安装依赖
RUN pip install -r requirements.txt

# 下载或训练模型
RUN python download_model.py || python wav2vec2_switch_detector.py

# 暴露端口 (如果有Web界面)
EXPOSE 8000

# 启动应用
CMD ["python", "realtime_mic_detector.py"]
```

### 云服务器部署
```bash
# 1. 上传代码
scp -r heater_click/ user@server:/opt/

# 2. 服务器端设置
ssh user@server
cd /opt/heater_click
pip install -r requirements.txt

# 3. 获取模型
python download_model.py

# 4. 设置系统服务
sudo systemctl enable heater-detector.service
sudo systemctl start heater-detector.service
```

## 💡 最佳实践

1. **版本控制**
   - 只提交源代码和配置文件
   - 使用语义化版本号
   - 在Release中分发大文件

2. **模型管理**
   - 记录模型训练参数
   - 保存模型元数据
   - 定期备份重要模型

3. **文档维护**
   - 更新部署说明
   - 记录已知问题
   - 提供故障排除指南

4. **安全考虑**
   - 不要在代码中硬编码密钥
   - 使用环境变量管理配置
   - 定期更新依赖包

通过合理的大文件处理策略，可以确保项目的可维护性和可分发性！ 