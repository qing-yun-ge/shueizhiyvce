import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymcdm.methods import TOPSIS
from pymcdm import weights
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'energy_quality_models.pkl')
# ==================== 中文字体配置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="污水多目标优化系统",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== 自定义CSS样式 ====================
st.markdown("""
<style>
    [data-testid="stSidebar"] {display: none;}
    .main-title {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1E88E5 0%, #00BCD4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        text-align: center;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    .metric-card h2 {
        margin: 0.5rem 0 0 0;
        font-size: 2rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .weight-box {
        background-color: #F3E5F5;
        border-left: 5px solid #9C27B0;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
    }
    .help-tip {
        background-color: #F5F7FA;
        border: 2px solid #E0E0E0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 页面标题 ====================
st.markdown('<h1 class="main-title">💧 污水处理多目标优化系统（手动权重版）</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">基于 NSGA-II + 自定义权重TOPSIS 的智能决策平台</p>', unsafe_allow_html=True)

# ==================== 系统说明 ====================
with st.expander("📘 系统说明与参数解释", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 优化目标**
        - 🔋 最小化总能耗
        - 💧 最小化出水水质指数
        
        **🎛️ 优化变量**
        - R2_NO2: 缺氧区硝态氮 (0.5-10.0 mg/L)
        - R5_DO: 好氧区溶解氧 (1.5-4.0 mg/L)
        """)
    
    with col2:
        st.markdown("""
        **⚖️ 权重设置说明**
        - **手动模式**: 自定义能耗和水质的重要性
        - **熵权模式**: 由算法自动确定权重
        - 权重和必须等于1.0
        - 权重越大表示该目标越重要
        """)
    
    with col3:
        st.markdown("""
        **💡 权重推荐**
        - **节能优先**: 能耗0.7, 水质0.3
        - **水质优先**: 能耗0.3, 水质0.7
        - **均衡模式**: 能耗0.5, 水质0.5
        - **自动模式**: 使用熵权法
        """)

st.markdown("---")

# ==================== 模型加载 ====================
st.header("1️⃣ 模型加载")

col1, col2 = st.columns([3, 1])
with col1:
    model_path1 = st.text_input("模型文件路径", value="energy_quality_models.pkl", label_visibility="collapsed", placeholder="请输入模型文件路径")
with col2:
    load_btn = st.button("🔄 加载模型", use_container_width=True)

if load_btn or 'models' not in st.session_state:
    try:
        models = joblib.load(model_path,'./energy_quality_models.pkl')
        st.session_state.models = models
        st.markdown('<div class="success-box">✅ 模型加载成功！包含模型: ' + 
                   ', '.join(list(models.keys())) + '</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="warning-box">❌ 模型加载失败: {e}</div>', unsafe_allow_html=True)
        st.stop()

if 'models' not in st.session_state:
    st.markdown('<div class="info-box">ℹ️ 请先加载模型文件</div>', unsafe_allow_html=True)
    st.stop()

models = st.session_state.models

st.markdown("---")

# ==================== 入水数据输入 ====================
st.header("2️⃣ 进水水质参数配置")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    SNH_in = st.number_input("**SNH_in** (mg/L)", value=30.0, step=1.0, min_value=0.0)
with col2:
    TSS_in = st.number_input("**TSS_in** (mg/L)", value=150.0, step=1.0, min_value=0.0)
with col3:
    TotalN_in = st.number_input("**TotalN_in** (mg/L)", value=50.0, step=1.0, min_value=0.0)
with col4:
    COD_in = st.number_input("**COD_in** (mg/L)", value=300.0, step=10.0, min_value=0.0)
with col5:
    BOD5_in = st.number_input("**BOD5_in** (mg/L)", value=150.0, step=5.0, min_value=0.0)

inlet_data = {
    "SNH_in": SNH_in,
    "TSS_in": TSS_in,
    "TotalN_in": TotalN_in,
    "COD_in": COD_in,
    "BOD5_in": BOD5_in
}

st.markdown("**📋 当前进水数据:**")
inlet_df = pd.DataFrame([inlet_data])
st.dataframe(inlet_df, use_container_width=True)

st.markdown("---")

# ==================== 优化参数配置 ====================
st.header("3️⃣ 优化参数配置")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🎛️ 决策变量范围")
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        r2_min = st.number_input("R2_NO2 最小值 (mg/L)", value=0.5, min_value=0.0, max_value=10.0)
        r5_min = st.number_input("R5_DO 最小值 (mg/L)", value=1.5, min_value=0.0, max_value=10.0)
    with col1_2:
        r2_max = st.number_input("R2_NO2 最大值 (mg/L)", value=10.0, min_value=0.0, max_value=10.0)
        r5_max = st.number_input("R5_DO 最大值 (mg/L)", value=4.0, min_value=0.0, max_value=10.0)
    
    st.subheader("🧬 NSGA-II 算法参数")
    col2_1, col2_2 = st.columns(2)
    with col2_1:
        pop_size = st.number_input("种群大小", value=50, step=10, min_value=10, max_value=200)
    with col2_2:
        n_gen = st.number_input("迭代代数", value=100, step=10, min_value=10, max_value=500)

with col_right:
    st.subheader("⚖️ TOPSIS权重配置")
    
    # 权重模式选择
    weight_mode = st.radio(
        "选择权重确定方式:",
        ["🤖 自动模式（熵权法）", "✋ 手动模式（自定义）"],
        horizontal=True
    )
    
    if weight_mode == "✋ 手动模式（自定义）":
        st.markdown("**手动设置目标权重（权重和 = 1.0）:**")
        
        # 预设方案
        preset = st.selectbox(
            "快速选择预设方案:",
            ["自定义", "节能优先 (0.7, 0.3)", "水质优先 (0.3, 0.7)", "均衡模式 (0.5, 0.5)"]
        )
        
        if preset == "节能优先 (0.7, 0.3)":
            default_w1, default_w2 = 0.7, 0.3
        elif preset == "水质优先 (0.3, 0.7)":
            default_w1, default_w2 = 0.3, 0.7
        elif preset == "均衡模式 (0.5, 0.5)":
            default_w1, default_w2 = 0.5, 0.5
        else:
            default_w1, default_w2 = 0.5, 0.5
        
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            w_energy = st.number_input(
                "🔋 能耗权重", 
                min_value=0.0, 
                max_value=1.0, 
                value=default_w1, 
                step=0.05,
                help="能耗目标的重要性，范围0-1"
            )
        with col_w2:
            w_quality = st.number_input(
                "💧 水质权重", 
                min_value=0.0, 
                max_value=1.0, 
                value=default_w2, 
                step=0.05,
                help="水质目标的重要性，范围0-1"
            )
        
        # 权重和验证
        weight_sum = w_energy + w_quality
        if abs(weight_sum - 1.0) > 0.001:
            st.markdown(f"""
            <div class="warning-box">
            ⚠️ <strong>权重和 = {weight_sum:.3f}</strong>，必须等于1.0！<br>
            请调整权重值。
            </div>
            """, unsafe_allow_html=True)
            manual_weights = None
        else:
            st.markdown(f"""
            <div class="weight-box">
            ✅ <strong>权重配置正确</strong><br>
            • 能耗权重: {w_energy:.2f} ({w_energy*100:.0f}%)<br>
            • 水质权重: {w_quality:.2f} ({w_quality*100:.0f}%)<br>
            权重和 = {weight_sum:.3f}
            </div>
            """, unsafe_allow_html=True)
            manual_weights = np.array([w_energy, w_quality])
    else:
        st.markdown("""
        <div class="info-box">
        🤖 <strong>熵权法自动模式</strong><br>
        系统将根据Pareto解集的数据分布自动计算最优权重。<br>
        熵权法能够客观反映各目标的差异程度。
        </div>
        """, unsafe_allow_html=True)
        manual_weights = None
    
    st.markdown("""
    <div class="help-tip">
    <strong>💡 权重选择建议:</strong><br>
    • <strong>节能优先</strong>: 运行成本敏感场景<br>
    • <strong>水质优先</strong>: 环保要求严格场景<br>
    • <strong>均衡模式</strong>: 综合考虑两者<br>
    • <strong>熵权法</strong>: 不确定优先级时
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================== 定义优化问题类 ====================
class WastewaterOptimization(Problem):
    def __init__(self, inlet_data, models, r2_range, r5_range):
        self.inlet_data = inlet_data
        self.models = models
        super().__init__(
            n_var=2, n_obj=2, n_ieq_constr=0,
            xl=np.array([r2_range[0], r5_range[0]]),
            xu=np.array([r2_range[1], r5_range[1]])
        )

    def _evaluate(self, x, out, *args, **kwargs):
        objs = []
        for i in range(x.shape[0]):
            features = self.inlet_data.copy()
            features['R2_NO2'] = float(x[i, 0])
            features['R5_DO'] = float(x[i, 1])
            df = pd.DataFrame([features])
            energy = float(self.models['total_energy'].predict(df)[0])
            eq = float(self.models['EQ_contrib'].predict(df)[0])
            objs.append([energy, eq])
        out["F"] = np.array(objs)

# ==================== 运行优化 ====================
st.header("4️⃣ 开始优化")

# 检查手动模式下权重是否有效
can_optimize = True
if weight_mode == "✋ 手动模式（自定义）" and manual_weights is None:
    can_optimize = False
    st.warning("⚠️ 请先正确设置权重（权重和必须等于1.0）")

if st.button("🚀 运行 NSGA-II 多目标优化", use_container_width=True, disabled=not can_optimize):
    
    # 进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("🔄 正在运行优化算法..."):
        status_text.text("⚙️ 初始化优化问题...")
        progress_bar.progress(10)
        
        problem = WastewaterOptimization(inlet_data, models, (r2_min, r2_max), (r5_min, r5_max))
        
        status_text.text(f"🧬 配置NSGA-II算法 (种群={pop_size}, 代数={n_gen})...")
        progress_bar.progress(20)
        
        algorithm = NSGA2(
            pop_size=int(pop_size),
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20)
        )
        
        status_text.text("🚀 执行多目标优化...")
        progress_bar.progress(30)
        
        res = minimize(problem, algorithm, ('n_gen', int(n_gen)), verbose=False)
        progress_bar.progress(70)

    f = res.F  # 目标值
    x = res.X  # 决策变量
    
    status_text.text("🎯 应用TOPSIS决策...")
    progress_bar.progress(80)

    # ==================== TOPSIS决策 ====================
    types = np.array([-1, -1])  # 两个目标都是越小越好
    
    # 根据模式选择权重
    if weight_mode == "🤖 自动模式（熵权法）":
        w = weights.entropy_weights(f)
        weight_method = "熵权法（自动）"
    else:
        w = manual_weights
        weight_method = f"手动设置（能耗={w[0]:.2f}, 水质={w[1]:.2f}）"
    
    topsis = TOPSIS()
    scores = topsis(f, w, types)
    best_idx = np.argmax(scores)
    best_x = x[best_idx]
    best_f = f[best_idx]
    
    progress_bar.progress(90)

    # ==================== 预测最优解下的指标 ====================
    best_features = inlet_data.copy()
    best_features['R2_NO2'] = best_x[0]
    best_features['R5_DO'] = best_x[1]
    df_best = pd.DataFrame([best_features])

    predictions = {}
    for target in models.keys():
        predictions[target] = float(models[target].predict(df_best)[0])
    
    progress_bar.progress(100)
    status_text.text("✅ 优化完成！")
    
    st.balloons()
    
    st.markdown("---")

    # ==================== 结果展示 ====================
    st.header("📊 优化结果")
    
    # 显示权重信息
    st.markdown(f"""
    <div class="weight-box">
    <strong>⚖️ 使用的权重方法: {weight_method}</strong><br>
    • 能耗权重: <strong>{w[0]:.4f}</strong> ({w[0]*100:.1f}%)<br>
    • 水质权重: <strong>{w[1]:.4f}</strong> ({w[1]*100:.1f}%)
    </div>
    """, unsafe_allow_html=True)
    
    # 关键指标卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h3>R2_NO2</h3>
            <h2>{best_x[0]:.2f}</h2>
            <p>mg/L</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3>R5_DO</h3>
            <h2>{best_x[1]:.2f}</h2>
            <p>mg/L</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3>总能耗</h3>
            <h2>{predictions['total_energy']:.2f}</h2>
            <p>kWh</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <h3>水质指数</h3>
            <h2>{predictions['EQ_contrib']:.2f}</h2>
            <p>点</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 出水水质指标
    st.subheader("💧 出水水质指标")
    outlet_targets = ['SNH', 'TSS', 'TotalN', 'COD', 'BOD5']
    
    cols = st.columns(5)
    for i, target in enumerate(outlet_targets):
        if target in predictions:
            inlet_val = inlet_data.get(f'{target}_in', 0)
            outlet_val = predictions[target]
            removal = ((inlet_val - outlet_val) / inlet_val * 100) if inlet_val > 0 else 0
            
            with cols[i]:
                st.metric(
                    label=target,
                    value=f"{outlet_val:.2f} mg/L",
                    delta=f"↓{removal:.1f}%",
                    delta_color="inverse"
                )
    
    st.markdown("---")
    
    # ==================== 可视化标签页 ====================
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Pareto前沿", "🏆 Top 10 最优解", "📊 水质对比", "🎯 综合分析"])
    
    with tab1:
        st.subheader("Pareto前沿分布")
        
        fig = go.Figure()
        
        # Pareto解集
        fig.add_trace(go.Scatter(
            x=f[:, 0],
            y=f[:, 1],
            mode='markers',
            name='Pareto解集',
            marker=dict(
                size=8,
                color=scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="TOPSIS<br>分数"),
                line=dict(width=1, color='white')
            ),
            text=[f'Score: {s:.4f}' for s in scores],
            hovertemplate='<b>能耗:</b> %{x:.2f} kWh<br>' +
                         '<b>水质指数:</b> %{y:.2f}<br>' +
                         '%{text}<extra></extra>'
        ))
        
        # 最优解
        fig.add_trace(go.Scatter(
            x=[best_f[0]],
            y=[best_f[1]],
            mode='markers',
            name='TOPSIS最优解',
            marker=dict(
                size=20,
                color='red',
                symbol='star',
                line=dict(width=2, color='darkred')
            ),
            hovertemplate='<b>最优解</b><br>' +
                         '<b>能耗:</b> %{x:.2f} kWh<br>' +
                         '<b>水质指数:</b> %{y:.2f}<br>' +
                         f'<b>TOPSIS分数:</b> {scores[best_idx]:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Pareto前沿分布图 (共 {len(f)} 个非支配解)',
            xaxis_title='总能耗 (kWh) - 越小越好',
            yaxis_title='出水水质指数 (点) - 越小越好',
            hovermode='closest',
            height=500,
            showlegend=True,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示权重信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info(f"⚖️ **权重方法**\n\n{weight_method}")
        with col2:
            st.info(f"🔋 **能耗权重**\n\n{w[0]:.4f}")
        with col3:
            st.info(f"💧 **水质权重**\n\n{w[1]:.4f}")
        with col4:
            st.info(f"⭐ **最优分数**\n\n{scores[best_idx]:.4f}")
    
    with tab2:
        st.subheader("🏆 TOPSIS分数排名 Top 10")
        
        # 获取Top 10索引
        top10_indices = np.argsort(scores)[::-1][:10]
        
        # 构建Top 10数据框
        top10_data = []
        for rank, idx in enumerate(top10_indices, 1):
            row = {
                '排名': rank,
                'R2_NO2 (mg/L)': f"{x[idx, 0]:.3f}",
                'R5_DO (mg/L)': f"{x[idx, 1]:.3f}",
                '总能耗 (kWh)': f"{f[idx, 0]:.2f}",
                '水质指数 (点)': f"{f[idx, 1]:.2f}",
                'TOPSIS分数': f"{scores[idx]:.4f}"
            }
            
            # 添加出水指标
            temp_features = inlet_data.copy()
            temp_features['R2_NO2'] = x[idx, 0]
            temp_features['R5_DO'] = x[idx, 1]
            df_temp = pd.DataFrame([temp_features])
            
            for target in outlet_targets:
                if target in models:
                    row[f'{target} (mg/L)'] = f"{models[target].predict(df_temp)[0]:.2f}"
            
            top10_data.append(row)
        
        top10_df = pd.DataFrame(top10_data)
        
        # 高亮显示
        def highlight_first(row):
            if row['排名'] == 1:
                return ['background-color: #FFD700; font-weight: bold'] * len(row)
            elif row['排名'] <= 3:
                return ['background-color: #E8F5E9'] * len(row)
            else:
                return [''] * len(row)
        
        styled_df = top10_df.style.apply(highlight_first, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # 下载按钮
        csv = top10_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载 Top 10 解 (CSV)",
            data=csv,
            file_name="top10_solutions.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with tab3:
        st.subheader("进出水水质对比分析")
        
        # 进出水对比图
        parameters = ['SNH', 'TSS', 'TotalN', 'COD', 'BOD5']
        inlet_vals = [inlet_data[f'{p}_in'] for p in parameters]
        outlet_vals = [predictions.get(p, 0) for p in parameters]
        removal_rates = [(inlet_vals[i] - outlet_vals[i]) / inlet_vals[i] * 100 
                        if inlet_vals[i] > 0 else 0 
                        for i in range(len(parameters))]
        
        fig2 = make_subplots(
            rows=1, cols=2,
            subplot_titles=("进出水浓度对比", "污染物去除效率"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 进出水对比
        fig2.add_trace(
            go.Bar(
                name='进水',
                x=parameters,
                y=inlet_vals,
                marker_color='#FF6B6B',
                text=[f'{v:.1f}' for v in inlet_vals],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        fig2.add_trace(
            go.Bar(
                name='出水',
                x=parameters,
                y=outlet_vals,
                marker_color='#4ECDC4',
                text=[f'{v:.1f}' for v in outlet_vals],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 去除率
        fig2.add_trace(
            go.Bar(
                x=parameters,
                y=removal_rates,
                marker_color='#95E1D3',
                text=[f'{r:.1f}%' for r in removal_rates],
                textposition='outside',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig2.update_xaxes(title_text="水质指标", row=1, col=1)
        fig2.update_xaxes(title_text="水质指标", row=1, col=2)
        fig2.update_yaxes(title_text="浓度 (mg/L)", row=1, col=1)
        fig2.update_yaxes(title_text="去除率 (%)", row=1, col=2)
        fig2.update_layout(height=500, showlegend=True, template='plotly_white')
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # 详细数据表
        comparison_df = pd.DataFrame({
            '指标': parameters,
            '进水 (mg/L)': [f"{v:.2f}" for v in inlet_vals],
            '出水 (mg/L)': [f"{v:.2f}" for v in outlet_vals],
            '去除量 (mg/L)': [f"{inlet_vals[i] - outlet_vals[i]:.2f}" for i in range(len(parameters))],
            '去除率 (%)': [f"{r:.2f}" for r in removal_rates]
        })
        st.dataframe(comparison_df, use_container_width=True)
    
    with tab4:
        st.subheader("综合性能分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 性能雷达图
            categories = ['能耗效率', '水质达标', 'COD去除', 'TN去除', 'NH去除']
            values = [
                100 - (predictions['total_energy'] / f[:, 0].max() * 100) if f[:, 0].max() > 0 else 0,
                100 - (predictions['EQ_contrib'] / f[:, 1].max() * 100) if f[:, 1].max() > 0 else 0,
                removal_rates[3],
                removal_rates[2],
                removal_rates[0]
            ]
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='最优解性能',
                line=dict(color='#6C5CE7', width=2),
                fillcolor='rgba(108, 92, 231, 0.3)'
            ))
            
            fig3.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                title='综合性能评估雷达图',
                height=400
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # TOPSIS分数分布
            fig4 = go.Figure()
            fig4.add_trace(go.Histogram(
                x=scores,
                nbinsx=30,
                marker_color='#FFD93D',
                marker_line_color='black',
                marker_line_width=1,
                name='TOPSIS分数分布'
            ))
            
            fig4.add_vline(
                x=scores[best_idx],
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"最优解: {scores[best_idx]:.4f}",
                annotation_position="top right"
            )
            
            fig4.update_layout(
                title='TOPSIS分数分布直方图',
                xaxis_title='TOPSIS分数',
                yaxis_title='频数',
                height=400,
                template='plotly_white',
                showlegend=True
            )
            
            st.plotly_chart(fig4, use_container_width=True)
        
        # 统计信息
        st.markdown("### 📊 统计信息")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Pareto解数量", f"{len(f)}")
        with col2:
            st.metric("平均TOPSIS分数", f"{np.mean(scores):.4f}")
        with col3:
            st.metric("最高TOPSIS分数", f"{np.max(scores):.4f}")
        with col4:
            st.metric("分数标准差", f"{np.std(scores):.4f}")
    
    st.markdown("---")
    
    # ==================== 导出所有结果 ====================
    st.header("💾 导出优化结果")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Pareto解集
        df_pareto = pd.DataFrame({
            'R2_NO2 (mg/L)': x[:, 0],
            'R5_DO (mg/L)': x[:, 1],
            '总能耗 (kWh)': f[:, 0],
            '水质指数 (点)': f[:, 1],
            'TOPSIS分数': scores
        })
        csv_pareto = df_pareto.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载完整Pareto解集",
            data=csv_pareto,
            file_name="pareto_solutions.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # 最优解详细信息
        best_solution_data = {
            '参数/指标': ['权重方法', '能耗权重', '水质权重', 'R2_NO2', 'R5_DO', '总能耗', '水质指数', 'TOPSIS分数'] + 
                        [f'{t}出水' for t in outlet_targets],
            '数值': [
                weight_method,
                f"{w[0]:.4f}",
                f"{w[1]:.4f}",
                f"{best_x[0]:.3f} mg/L",
                f"{best_x[1]:.3f} mg/L",
                f"{predictions['total_energy']:.2f} kWh",
                f"{predictions['EQ_contrib']:.2f} 点",
                f"{scores[best_idx]:.4f}"
            ] + [f"{predictions.get(t, 0):.2f} mg/L" for t in outlet_targets]
        }
        df_best = pd.DataFrame(best_solution_data)
        csv_best = df_best.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载最优解详情",
            data=csv_best,
            file_name="best_solution.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        # Top 10解集
        csv_top10 = top10_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载Top 10解集",
            data=csv_top10,
            file_name="top10_solutions.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # 成功提示
    st.markdown("""
    <div class="success-box">
        <h3>✅ 优化成功完成！</h3>
        <p><strong>权重配置: {}</strong></p>
        <p><strong>最优操作建议：</strong></p>
        <ul>
            <li>设置 R2_NO2 (缺氧区硝态氮) 为 <strong>{:.2f} mg/L</strong></li>
            <li>设置 R5_DO (好氧区溶解氧) 为 <strong>{:.2f} mg/L</strong></li>
            <li>预期总能耗: <strong>{:.2f} kWh</strong></li>
            <li>预期水质指数: <strong>{:.2f} 点</strong></li>
        </ul>
    </div>
    """.format(weight_method, best_x[0], best_x[1], predictions['total_energy'], predictions['EQ_contrib']), 
    unsafe_allow_html=True)

else:
    # 未开始优化时的提示
    st.markdown("""
    <div class="info-box">
        <h3>ℹ️ 准备开始优化</h3>
        <p>请按照以下步骤操作：</p>
        <ol>
            <li>✅ 确认模型已加载</li>
            <li>⚙️ 配置进水水质参数</li>
            <li>🎛️ 设置决策变量范围</li>
            <li>⚖️ <strong>选择权重模式（手动或自动）</strong></li>
            <li>🧬 调整NSGA-II算法参数</li>
            <li>🚀 点击"运行优化"按钮</li>
        </ol>
        <p><strong>提示：</strong>首次运行建议使用自动模式（熵权法），熟悉后可尝试手动模式。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 显示示例结果
    st.subheader("📌 示例参考")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **典型进水条件：**
        - SNH_in: 30 mg/L
        - TSS_in: 150 mg/L
        - TotalN_in: 50 mg/L
        - COD_in: 300 mg/L
        - BOD5_in: 150 mg/L
        """)
    
    with col2:
        st.markdown("""
        **优化参数范围：**
        - R2_NO2: 0.5-10.0 mg/L
        - R5_DO: 1.5-4.0 mg/L
        - 种群大小: 50
        - 迭代代数: 100
        """)
    
    with col3:
        st.markdown("""
        **权重设置示例：**
        - 🤖 自动模式: 熵权法
        - 节能优先: (0.7, 0.3)
        - 水质优先: (0.3, 0.7)
        - 均衡模式: (0.5, 0.5)
        """)

# ==================== 页脚信息 ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background-color: #F5F7FA; border-radius: 10px; margin-top: 2rem;'>
    <h3 style='color: #1E88E5; margin-bottom: 1rem;'>💧 污水处理多目标优化系统（手动权重版）</h3>
    <p style='color: #666; margin: 0.5rem 0;'><strong>优化算法:</strong> NSGA-II (非支配排序遗传算法-II)</p>
    <p style='color: #666; margin: 0.5rem 0;'><strong>决策方法:</strong> TOPSIS (逼近理想解排序法)</p>
    <p style='color: #666; margin: 0.5rem 0;'><strong>权重模式:</strong> 🤖 自动（熵权法）| ✋ 手动（自定义）</p>
    <p style='color: #666; margin: 0.5rem 0;'><strong>优化目标:</strong> 最小化能耗 & 最小化出水水质指数</p>
    <p style='color: #666; margin: 0.5rem 0;'><strong>控制参数:</strong> R2_NO2 (缺氧区硝态氮) & R5_DO (好氧区溶解氧)</p>
    <hr style='margin: 1rem 0; border: none; border-top: 1px solid #ddd;'>
    <p style='color: #999; font-size: 0.9rem;'>© 2025 污水处理智能优化系统 | Powered by NSGA-II & TOPSIS</p>
</div>

""", unsafe_allow_html=True)

