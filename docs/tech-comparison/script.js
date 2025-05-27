// 幻灯片控制器
class SlideController {
    constructor() {
        this.currentSlide = 1;
        this.totalSlides = 8;
        this.slides = document.querySelectorAll('.slide');
        this.prevBtn = document.getElementById('prevBtn');
        this.nextBtn = document.getElementById('nextBtn');
        this.slideCounter = document.getElementById('slideCounter');
        this.progressFill = document.getElementById('progressFill');
        
        this.init();
    }
    
    init() {
        // 绑定事件监听器
        this.prevBtn.addEventListener('click', () => this.previousSlide());
        this.nextBtn.addEventListener('click', () => this.nextSlide());
        
        // 键盘导航
        document.addEventListener('keydown', (e) => this.handleKeyPress(e));
        
        // 触摸手势支持
        this.initTouchGestures();
        
        // 初始化显示
        this.updateDisplay();
        
        // 自动播放功能（可选）
        // this.startAutoPlay();
    }
    
    // 下一张幻灯片
    nextSlide() {
        if (this.currentSlide < this.totalSlides) {
            this.currentSlide++;
            this.updateDisplay();
        }
    }
    
    // 上一张幻灯片
    previousSlide() {
        if (this.currentSlide > 1) {
            this.currentSlide--;
            this.updateDisplay();
        }
    }
    
    // 跳转到指定幻灯片
    goToSlide(slideNumber) {
        if (slideNumber >= 1 && slideNumber <= this.totalSlides) {
            this.currentSlide = slideNumber;
            this.updateDisplay();
        }
    }
    
    // 更新显示
    updateDisplay() {
        // 更新幻灯片显示
        this.slides.forEach((slide, index) => {
            if (index + 1 === this.currentSlide) {
                slide.classList.add('active');
                // 触发动画
                this.triggerSlideAnimations(slide);
            } else {
                slide.classList.remove('active');
            }
        });
        
        // 更新计数器
        this.slideCounter.textContent = `${this.currentSlide} / ${this.totalSlides}`;
        
        // 更新按钮状态
        this.prevBtn.disabled = this.currentSlide === 1;
        this.nextBtn.disabled = this.currentSlide === this.totalSlides;
        
        // 更新进度条
        const progress = (this.currentSlide / this.totalSlides) * 100;
        this.progressFill.style.width = `${progress}%`;
        
        // 更新URL hash（可选）
        window.location.hash = `slide-${this.currentSlide}`;
    }
    
    // 触发幻灯片动画
    triggerSlideAnimations(slide) {
        // 重置动画
        const animatedElements = slide.querySelectorAll('.slide-content > *');
        animatedElements.forEach(el => {
            el.style.animation = 'none';
            el.offsetHeight; // 触发重排
            el.style.animation = null;
        });
        
        // 特殊动画处理
        this.handleSpecialAnimations(slide);
    }
    
    // 处理特殊动画
    handleSpecialAnimations(slide) {
        const slideId = slide.id;
        
        switch(slideId) {
            case 'slide6': // 性能对比页面
                this.animateMetricBars(slide);
                break;
            case 'slide7': // 时间线页面
                this.animateTimeline(slide);
                break;
        }
    }
    
    // 动画化指标条
    animateMetricBars(slide) {
        const metricBars = slide.querySelectorAll('.metric-fill');
        metricBars.forEach((bar, index) => {
            setTimeout(() => {
                const width = bar.style.width;
                bar.style.width = '0%';
                setTimeout(() => {
                    bar.style.width = width;
                }, 100);
            }, index * 200);
        });
    }
    
    // 动画化时间线
    animateTimeline(slide) {
        const timelineItems = slide.querySelectorAll('.timeline-item');
        timelineItems.forEach((item, index) => {
            item.style.opacity = '0';
            item.style.transform = 'translateY(30px)';
            
            setTimeout(() => {
                item.style.transition = 'all 0.6s ease';
                item.style.opacity = '1';
                item.style.transform = 'translateY(0)';
            }, index * 300);
        });
    }
    
    // 键盘事件处理
    handleKeyPress(e) {
        switch(e.key) {
            case 'ArrowRight':
            case ' ': // 空格键
                e.preventDefault();
                this.nextSlide();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                this.previousSlide();
                break;
            case 'Home':
                e.preventDefault();
                this.goToSlide(1);
                break;
            case 'End':
                e.preventDefault();
                this.goToSlide(this.totalSlides);
                break;
            case 'Escape':
                e.preventDefault();
                this.toggleFullscreen();
                break;
        }
        
        // 数字键快速跳转
        if (e.key >= '1' && e.key <= '8') {
            e.preventDefault();
            this.goToSlide(parseInt(e.key));
        }
    }
    
    // 触摸手势初始化
    initTouchGestures() {
        let startX = 0;
        let startY = 0;
        let endX = 0;
        let endY = 0;
        
        document.addEventListener('touchstart', (e) => {
            startX = e.touches[0].clientX;
            startY = e.touches[0].clientY;
        });
        
        document.addEventListener('touchend', (e) => {
            endX = e.changedTouches[0].clientX;
            endY = e.changedTouches[0].clientY;
            
            const deltaX = endX - startX;
            const deltaY = endY - startY;
            
            // 检查是否为水平滑动
            if (Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > 50) {
                if (deltaX > 0) {
                    this.previousSlide(); // 向右滑动，上一张
                } else {
                    this.nextSlide(); // 向左滑动，下一张
                }
            }
        });
    }
    
    // 全屏切换
    toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch(err => {
                console.log(`Error attempting to enable fullscreen: ${err.message}`);
            });
        } else {
            document.exitFullscreen();
        }
    }
    
    // 自动播放功能
    startAutoPlay(interval = 10000) {
        this.autoPlayInterval = setInterval(() => {
            if (this.currentSlide < this.totalSlides) {
                this.nextSlide();
            } else {
                this.stopAutoPlay();
            }
        }, interval);
    }
    
    stopAutoPlay() {
        if (this.autoPlayInterval) {
            clearInterval(this.autoPlayInterval);
            this.autoPlayInterval = null;
        }
    }
    
    // 从URL hash恢复状态
    restoreFromHash() {
        const hash = window.location.hash;
        if (hash.startsWith('#slide-')) {
            const slideNumber = parseInt(hash.replace('#slide-', ''));
            if (slideNumber >= 1 && slideNumber <= this.totalSlides) {
                this.currentSlide = slideNumber;
                this.updateDisplay();
            }
        }
    }
}

// 工具函数
class PresentationUtils {
    // 添加演讲者备注功能
    static showSpeakerNotes() {
        const notes = {
            1: "欢迎大家！今天我们要深入分析四种主流音频AI模型的技术特点。",
            2: "首先看看我们评估的四个候选模型，每个都有其独特的优势。",
            3: "这是详细的技术对比表，可以看到各模型在不同维度的表现。",
            4: "让我们深入了解为什么选择Wav2Vec2作为我们的解决方案。",
            5: "当然，其他模型也有各自的局限性，我们来分析一下。",
            6: "从实际性能数据来看，Wav2Vec2确实表现最佳。",
            7: "这是音频AI技术的演进历程，可以看到技术发展的趋势。",
            8: "最后总结一下我们的技术选型和项目成果。"
        };
        
        console.log("演讲者备注:", notes);
    }
    
    // 导出为PDF功能
    static exportToPDF() {
        if (window.print) {
            // 添加打印样式
            const printStyles = `
                @media print {
                    .nav-bar, .progress-bar { display: none !important; }
                    .slide { position: static !important; opacity: 1 !important; 
                             transform: none !important; page-break-after: always; }
                    body { background: white !important; }
                }
            `;
            
            const styleSheet = document.createElement('style');
            styleSheet.textContent = printStyles;
            document.head.appendChild(styleSheet);
            
            // 显示所有幻灯片
            document.querySelectorAll('.slide').forEach(slide => {
                slide.classList.add('active');
            });
            
            window.print();
            
            // 恢复原状态
            setTimeout(() => {
                document.head.removeChild(styleSheet);
                window.slideController.updateDisplay();
            }, 1000);
        }
    }
    
    // 分享功能
    static sharePresentation() {
        if (navigator.share) {
            navigator.share({
                title: '音频AI模型技术对比',
                text: '深入分析四种主流音频AI模型的技术特点、优劣势和适用场景',
                url: window.location.href
            });
        } else {
            // 复制链接到剪贴板
            navigator.clipboard.writeText(window.location.href).then(() => {
                alert('链接已复制到剪贴板！');
            });
        }
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    // 创建幻灯片控制器
    window.slideController = new SlideController();
    
    // 从URL hash恢复状态
    window.slideController.restoreFromHash();
    
    // 添加额外的控制按钮（可选）
    const navContent = document.querySelector('.nav-content');
    
    // 全屏按钮
    const fullscreenBtn = document.createElement('button');
    fullscreenBtn.className = 'nav-btn';
    fullscreenBtn.innerHTML = '<i class="fas fa-expand"></i>';
    fullscreenBtn.title = '全屏模式 (Esc)';
    fullscreenBtn.addEventListener('click', () => window.slideController.toggleFullscreen());
    
    // 分享按钮
    const shareBtn = document.createElement('button');
    shareBtn.className = 'nav-btn';
    shareBtn.innerHTML = '<i class="fas fa-share-alt"></i>';
    shareBtn.title = '分享演示';
    shareBtn.addEventListener('click', () => PresentationUtils.sharePresentation());
    
    // 添加到导航栏
    const navControls = document.querySelector('.nav-controls');
    navControls.appendChild(fullscreenBtn);
    navControls.appendChild(shareBtn);
    
    // 添加键盘快捷键提示
    console.log(`
    🎯 键盘快捷键:
    ← → 或 空格键: 切换幻灯片
    1-8: 快速跳转到指定幻灯片
    Home/End: 跳转到首页/末页
    Esc: 切换全屏模式
    
    📱 触摸手势:
    左右滑动: 切换幻灯片
    `);
    
    // 显示演讲者备注
    PresentationUtils.showSpeakerNotes();
});

// 监听hash变化
window.addEventListener('hashchange', () => {
    if (window.slideController) {
        window.slideController.restoreFromHash();
    }
});

// 监听全屏变化
document.addEventListener('fullscreenchange', () => {
    const fullscreenBtn = document.querySelector('.nav-btn[title*="全屏"]');
    if (fullscreenBtn) {
        if (document.fullscreenElement) {
            fullscreenBtn.innerHTML = '<i class="fas fa-compress"></i>';
            fullscreenBtn.title = '退出全屏 (Esc)';
        } else {
            fullscreenBtn.innerHTML = '<i class="fas fa-expand"></i>';
            fullscreenBtn.title = '全屏模式 (Esc)';
        }
    }
});

// 防止意外刷新
window.addEventListener('beforeunload', (e) => {
    if (window.slideController && window.slideController.currentSlide > 1) {
        e.preventDefault();
        e.returnValue = '确定要离开演示吗？';
    }
}); 