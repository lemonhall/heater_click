# 🎯 音频AI模型技术对比演示

这是一个PPT风格的HTML网页，用于展示热水器开关声音检测器项目中的技术选型对比。

## 📁 文件结构

```
docs/tech-comparison/
├── index.html          # 主HTML文件
├── styles.css          # CSS样式文件
├── script.js           # JavaScript交互逻辑
└── README.md           # 使用说明
```

## 🚀 快速开始

### 本地运行

1. **直接打开**：
   ```bash
   # 在浏览器中打开
   open docs/tech-comparison/index.html
   ```

2. **使用本地服务器**（推荐）：
   ```bash
   # Python 3
   cd docs/tech-comparison
   python -m http.server 8000
   
   # 然后访问 http://localhost:8000
   ```

3. **使用Live Server**（VS Code扩展）：
   - 安装Live Server扩展
   - 右键点击`index.html` → "Open with Live Server"

## 🎮 操作指南

### 🖱️ 鼠标操作
- **导航按钮**：点击左右箭头按钮切换幻灯片
- **全屏按钮**：点击展开图标进入全屏模式
- **分享按钮**：点击分享图标复制链接

### ⌨️ 键盘快捷键
- `→` 或 `空格键`：下一张幻灯片
- `←`：上一张幻灯片
- `1-8`：快速跳转到指定幻灯片
- `Home`：跳转到首页
- `End`：跳转到最后一页
- `Esc`：切换全屏模式

### 📱 触摸手势
- **左滑**：下一张幻灯片
- **右滑**：上一张幻灯片

## 📊 幻灯片内容

### 第1页：标题页
- 项目介绍
- 四种模型预览

### 第2页：候选模型概览
- Wav2Vec2（我们的选择）
- VGGish（传统方案）
- CLAP（通用方案）
- Soundwave（前沿方案）

### 第3页：详细技术对比
- 输入格式对比
- 特征维度对比
- 预训练方式对比
- 能力特性对比

### 第4页：Wav2Vec2优势
- 完美匹配需求
- 少样本学习优势
- 计算效率高
- 端到端优化

### 第5页：其他模型局限性
- VGGish的问题
- CLAP的问题
- Soundwave的问题

### 第6页：实际性能对比
- 准确率对比
- 训练样本需求
- 推理时间对比
- 模型大小对比

### 第7页：技术演进趋势
- 2017-2025年技术发展时间线
- 各阶段技术特点

### 第8页：总结与展望
- 当前最佳选择
- 未来扩展场景
- 关键洞察
- 项目成果

## 🎨 设计特色

### 视觉设计
- **现代化UI**：毛玻璃效果、渐变背景
- **响应式布局**：适配桌面和移动设备
- **动画效果**：平滑过渡和进入动画
- **色彩编码**：不同模型使用不同颜色主题

### 交互体验
- **流畅导航**：支持多种操作方式
- **进度指示**：底部进度条显示当前位置
- **状态保持**：URL hash保存当前页面
- **防误操作**：离开时确认提示

### 特殊动画
- **性能对比页**：指标条动画展示
- **时间线页**：逐步显示技术演进
- **卡片悬停**：3D提升效果

## 🔧 自定义配置

### 修改幻灯片内容
编辑`index.html`中对应的幻灯片div：
```html
<div class="slide" id="slide1">
    <div class="slide-content">
        <!-- 修改这里的内容 -->
    </div>
</div>
```

### 调整样式
编辑`styles.css`中的相关样式：
```css
/* 修改主题色 */
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
}
```

### 添加新功能
在`script.js`中扩展功能：
```javascript
// 添加新的键盘快捷键
case 'KeyP':
    PresentationUtils.exportToPDF();
    break;
```

## 📱 移动端优化

- **触摸友好**：大按钮、手势支持
- **响应式字体**：自动调整文字大小
- **简化布局**：移动端优化的网格布局
- **性能优化**：减少动画复杂度

## 🌐 浏览器兼容性

### 支持的浏览器
- ✅ Chrome 80+
- ✅ Firefox 75+
- ✅ Safari 13+
- ✅ Edge 80+

### 需要的功能
- CSS Grid Layout
- CSS Flexbox
- ES6 Classes
- Fetch API
- Fullscreen API

## 🚀 部署建议

### GitHub Pages
```bash
# 推送到GitHub后，在Settings中启用Pages
# 访问：https://username.github.io/repo-name/docs/tech-comparison/
```

### Netlify
```bash
# 拖拽docs/tech-comparison文件夹到Netlify
# 或连接GitHub仓库自动部署
```

### 自定义域名
在`index.html`的`<head>`中添加：
```html
<link rel="canonical" href="https://yourdomain.com/tech-comparison/">
```

## 🎯 使用场景

### 技术分享
- 团队内部技术分享
- 会议演示
- 客户汇报

### 教学培训
- 音频AI技术培训
- 模型选型教学
- 技术对比分析

### 文档展示
- 项目技术文档
- 决策过程记录
- 知识库内容

## 🔍 SEO优化

已包含的SEO元素：
- 语义化HTML结构
- 合适的标题层级
- Meta描述和关键词
- 结构化数据标记

## 📈 性能优化

- **懒加载**：按需加载幻灯片内容
- **压缩资源**：CSS/JS文件压缩
- **CDN加速**：字体和图标使用CDN
- **缓存策略**：合理的缓存头设置

## 🤝 贡献指南

欢迎提交改进建议：
1. Fork项目
2. 创建功能分支
3. 提交更改
4. 发起Pull Request

## 📄 许可证

MIT License - 详见项目根目录LICENSE文件

---

**💡 提示**：打开浏览器控制台可以看到键盘快捷键说明和演讲者备注！ 