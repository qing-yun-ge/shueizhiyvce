import streamlit as st
import base64

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="污水处理智能系统",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义CSS样式 ====================
st.markdown("""
<style>
    /* 主标题样式 */
    .main-title {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1E88E5 0%, #00BCD4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0 1rem 0;
        animation: fadeInDown 1s;
    }
    
    /* 副标题样式 */
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1.5rem;
        margin-bottom: 3rem;
        animation: fadeInUp 1s;
    }
    
    /* Hero区域 */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        animation: fadeIn 1.5s;
    }
    
    .hero-section h1 {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .hero-section p {
        font-size: 1.3rem;
        opacity: 0.95;
    }
    
    /* 特性卡片 */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s, box-shadow 0.3s;
        height: 100%;
        border-left: 5px solid #2196F3;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.2);
    }
    
    .feature-card h3 {
        color: #2196F3;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-card p {
        color: #666;
        line-height: 1.8;
    }
    
    /* 个人信息卡片 */
    .profile-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .profile-card h2 {
        margin-top: 1rem;
        font-size: 2rem;
    }
    
    .profile-card p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* 链接按钮 */
    .link-button {
        display: inline-block;
        background: white;
        color: #2196F3;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        text-decoration: none;
        font-weight: bold;
        margin: 0.5rem;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .link-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        color: #1976D2;
    }
    
    /* 技术栈标签 */
    .tech-tag {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-size: 0.9rem;
        font-weight: bold;
    }
    
    /* 时间线 */
    .timeline-item {
        border-left: 3px solid #2196F3;
        padding-left: 2rem;
        margin-bottom: 2rem;
        position: relative;
    }
    
    .timeline-item::before {
        content: '';
        position: absolute;
        left: -8px;
        top: 0;
        width: 15px;
        height: 15px;
        border-radius: 50%;
        background: #2196F3;
    }
    
    .timeline-item h4 {
        color: #2196F3;
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
    }
    
    .timeline-item p {
        color: #666;
        line-height: 1.8;
    }
    
    /* 统计卡片 */
    .stat-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .stat-card h2 {
        font-size: 3rem;
        margin: 0;
    }
    
    .stat-card p {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* 动画 */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* 分隔线 */
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #2196F3, transparent);
        margin: 3rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 侧边栏导航 ====================
st.sidebar.title("🌊 导航菜单")
st.sidebar.markdown("---")

st.sidebar.markdown("""
### 📖 系统功能
- 🏠 **首页**: 项目介绍与个人简介
- 💧 **水质预测**: 出水水质智能预测
- 🎯 **多目标优化**: NSGA-II优化系统

### 🔗 快速链接
""")

st.sidebar.markdown("""
<a href="https://blog.csdn.net/zzqingyun?type=blog" target="_blank">
    <div style="background: linear-gradient(135deg, #FC466B 0%, #3F5EFB 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white; margin: 1rem 0; text-decoration: none;">
        📝 CSDN博客
    </div>
</a>

<a href="https://github.com/qing-yun-ge/qing-yun-ge" target="_blank">
    <div style="background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white; margin: 1rem 0; text-decoration: none;">
        💻 GitHub主页
    </div>
</a>

<a href="https://envdama.top/" target="_blank">
    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white; margin: 1rem 0; text-decoration: none;">
        🌐 ENVDAMA 知识分享站
    </div>
</a>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.info("💡 **提示**: 使用上方菜单切换不同功能页面")

# ==================== 主页内容 ====================

# 标题
st.markdown('<h1 class="main-title">🌊 污水处理智能优化系统</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">基于机器学习与多目标优化的智能决策平台</p>', unsafe_allow_html=True)

# Hero区域
st.markdown("""
<div class="hero-section">
    <h1>🚀 智能水处理，科技守护蓝天碧水</h1>
    <p>融合人工智能、大数据分析与多目标优化算法</p>
    <p>为污水处理厂提供精准预测与最优控制方案</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ==================== 项目介绍 ====================
st.header("📌 项目简介")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="stat-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <h2>🎯</h2>
        <p><strong>2大核心功能</strong></p>
        <p>水质预测 + 多目标优化</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
        <h2>📊</h2>
        <p><strong>7项预测指标</strong></p>
        <p>SNH, TSS, TN, COD, BOD5等</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
        <h2>🧬</h2>
        <p><strong>先进算法</strong></p>
        <p>NSGA-II + TOPSIS决策</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 项目特色
st.subheader("✨ 项目特色")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>💧 智能水质预测</h3>
        <p>
        基于机器学习模型，输入进水参数和控制参数，即可精准预测出水水质指标。
        </p>
        <ul>
            <li>🔵 预测出水SNH、TSS、总氮等7项关键指标</li>
            <li>📈 实时计算污染物去除效率</li>
            <li>📊 可视化进出水对比分析</li>
            <li>💾 支持结果导出和报告生成</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 多目标智能优化</h3>
        <p>
        采用NSGA-II遗传算法进行多目标优化，自动寻找能耗与水质的最佳平衡点。
        </p>
        <ul>
            <li>🔋 最小化总能耗</li>
            <li>💧 最小化出水水质指数</li>
            <li>⚖️ 熵权TOPSIS智能决策</li>
            <li>📊 Pareto前沿可视化</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>🔬 核心技术栈</h3>
        <p>
        <span class="tech-tag">Python</span>
        <span class="tech-tag">Streamlit</span>
        <span class="tech-tag">Scikit-learn</span>
        <span class="tech-tag">Pymoo</span>
        <span class="tech-tag">Plotly</span>
        <span class="tech-tag">Pandas</span>
        <span class="tech-tag">NumPy</span>
        </p>
        <p style="margin-top: 1rem;">
        采用现代化的Python生态系统，构建高效、易用的Web应用。
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>📱 系统优势</h3>
        <p>
        <ul>
            <li>🎨 <strong>美观界面</strong>: 现代化UI设计</li>
            <li>⚡ <strong>高效预测</strong>: 秒级响应速度</li>
            <li>🎯 <strong>精准优化</strong>: 多目标全局最优</li>
            <li>📊 <strong>丰富可视化</strong>: 交互式图表</li>
            <li>💾 <strong>结果导出</strong>: 支持CSV/TXT格式</li>
            <li>📱 <strong>响应式设计</strong>: 适配各种屏幕</li>
        </ul>
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ==================== 技术路线 ====================
st.header("🛣️ 技术路线")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="timeline-item">
        <h4>1️⃣ 数据预处理与模型训练</h4>
        <p>
        • 收集污水处理厂历史运行数据<br>
        • 数据清洗、特征工程与标准化<br>
        • 训练多个机器学习回归模型<br>
        • 模型评估与参数调优
        </p>
    </div>
    
    <div class="timeline-item">
        <h4>2️⃣ 水质预测系统开发</h4>
        <p>
        • 加载预训练模型<br>
        • 构建用户交互界面<br>
        • 实现7项指标实时预测<br>
        • 开发可视化分析模块
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="timeline-item">
        <h4>3️⃣ 多目标优化算法实现</h4>
        <p>
        • 定义优化问题（能耗+水质）<br>
        • 实现NSGA-II遗传算法<br>
        • 应用熵权TOPSIS决策方法<br>
        • 生成Pareto最优解集
        </p>
    </div>
    
    <div class="timeline-item">
        <h4>4️⃣ 系统集成与部署</h4>
        <p>
        • 多页面架构设计<br>
        • 响应式UI美化<br>
        • 性能优化与测试<br>
        • 在线部署与维护
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ==================== 个人介绍 ====================
st.header("👨‍💻 开发者介绍")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("""
    <div class="profile-card">
        <div style="font-size: 5rem; margin-bottom: 1rem;">👨‍🔬</div>
        <h2>卿云阁</h2>
        <p>人工智能| 数据科学爱好者</p>
        <p style="margin-top: 1.5rem;">
            <a href="https://blog.csdn.net/zzqingyun?type=blog" target="_blank" class="link-button">
                📝 CSDN博客
            </a>
        </p>
        <p>
            <a href="https://github.com/qing-yun-ge/qing-yun-ge" target="_blank" class="link-button">
                💻 GitHub
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>🎓 个人简介</h3>
        <p style="line-height: 2; font-size: 1.05rem;">
        热衷于将<strong>机器学习</strong>、<strong>深度学习</strong>和<strong>优化算法</strong>应用于环境工程领域，
        致力于通过数据驱动的方法提升污水处理效率，推动智慧水务发展。
        </p>
        
        <h4 style="color: #2196F3; margin-top: 2rem; margin-bottom: 1rem;">🔍 研究方向</h4>
        <ul style="line-height: 2;">
            <li>🌊 污水处理过程建模与优化</li>
            <li>🤖 机器学习在环境工程中的应用</li>
            <li>📊 数据挖掘与可视化分析</li>
            <li>🎯 多目标优化算法研究</li>
        </ul>
        
        <h4 style="color: #2196F3; margin-top: 2rem; margin-bottom: 1rem;">💼 技能特长</h4>
        <p>
        <span class="tech-tag">Python</span>
        <span class="tech-tag">机器学习</span>
        <span class="tech-tag">深度学习</span>
        <span class="tech-tag">数据分析</span>
        <span class="tech-tag">优化算法</span>
        <span class="tech-tag">Web开发</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ==================== 快速开始 ====================
st.header("🚀 快速开始")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card" style="text-align: center;">
        <h3>1️⃣ 水质预测</h3>
        <p>输入进水参数和控制参数，立即获得出水水质预测结果</p>
        <br>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 前往水质预测页面", use_container_width=True, key="goto_prediction"):
        st.switch_page("pages/page1.py")

with col2:
    st.markdown("""
    <div class="feature-card" style="text-align: center;">
        <h3>2️⃣ 多目标优化</h3>
        <p>设置优化参数，自动寻找能耗与水质的最佳平衡点</p>
        <br>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🎯 前往多目标优化页面", use_container_width=True, key="goto_optimization"):
        st.switch_page("pages/page2.py")

with col3:
    st.markdown("""
    <div class="feature-card" style="text-align: center;">
        <h3>3️⃣ 知识分享</h3>
        <p>访问ENVDAMA知识分享站，获取更多环境数据分析资源</p>
        <br>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🌐 访问 ENVDAMA 知识分享站", use_container_width=True, key="goto_envdama"):
        st.markdown("""
        <meta http-equiv="refresh" content="0; url=https://envdama.top/">
        <script>window.open('https://envdama.top/', '_blank');</script>
        """, unsafe_allow_html=True)
        st.success("✅ 正在打开 ENVDAMA 知识分享站...")
        st.info("💡 如未自动跳转，请点击: [https://envdama.top/](https://envdama.top/)")

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ==================== 联系方式 ====================
st.header("📮 联系方式")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #FC466B 0%, #3F5EFB 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white;">
        <h3>📝 CSDN博客</h3>
        <p>技术文章与项目分享</p>
        <a href="https://blog.csdn.net/zzqingyun?type=blog" target="_blank" style="color: white; text-decoration: underline;">
            访问博客 →
        </a>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0F2027 0%, #2C5364 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white;">
        <h3>💻 GitHub</h3>
        <p>开源项目与代码仓库</p>
        <a href="https://github.com/qing-yun-ge/qing-yun-ge" target="_blank" style="color: white; text-decoration: underline;">
            访问主页 →
        </a>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white;">
        <h3>🌐 知识分享站</h3>
        <p>ENVDAMA 环境数据分析</p>
        <a href="https://envdama.top/" target="_blank" style="color: white; text-decoration: underline;">
            访问网站 →
        </a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ==================== 页脚 ====================
st.markdown("""
<div style='text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white; margin-top: 3rem;'>
    <h2 style='margin-bottom: 1rem;'>🌊 污水处理智能优化系统</h2>
    <p style='font-size: 1.1rem; opacity: 0.9; margin-bottom: 2rem;'>让科技赋能环保，用智慧守护碧水蓝天</p>
    <div style='border-top: 2px solid rgba(255,255,255,0.3); padding-top: 2rem; margin-top: 2rem;'>
        <p>© 2025 卿云阁 | 基于 Streamlit 开发</p>
        <p style='margin-top: 0.5rem;'>
            <a href="https://blog.csdn.net/zzqingyun?type=blog" target="_blank" style="color: white; margin: 0 1rem;">CSDN</a> |
            <a href="https://github.com/qing-yun-ge/qing-yun-ge" target="_blank" style="color: white; margin: 0 1rem;">GitHub</a>
        </p>
    </div>
</div>
""", unsafe_allow_html=True)