import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="污水处理厂预测系统", 
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== 自定义CSS样式 ====================
st.markdown("""
<style>
    /* 隐藏侧边栏 */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* 主标题样式 */
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #2196F3 0%, #00BCD4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    
    /* 副标题样式 */
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* 卡片样式 */
    .metric-card {
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        text-align: center;
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .metric-card h2 {
        margin: 0.5rem 0 0 0;
        font-size: 2rem;
        font-weight: bold;
    }
    
    .metric-card p {
        margin: 0.3rem 0 0 0;
        font-size: 0.85rem;
        opacity: 0.8;
    }
    
    /* 信息框样式 */
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
    
    /* 按钮样式 */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2196F3 0%, #00BCD4 100%);
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 1rem 2rem;
        border: none;
        font-size: 1.2rem;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(33, 150, 243, 0.5);
    }
    
    /* 输入框组样式 */
    .input-group {
        background-color: #F5F7FA;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* 分隔线 */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 页面标题 ====================
st.markdown('<h1 class="main-title">🌊 污水处理厂智能预测系统</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">基于机器学习的出水水质与能耗预测平台</p>', unsafe_allow_html=True)

# ==================== 加载模型 ====================
@st.cache_resource
def load_models():
    try:
        return joblib.load('energy_quality_models.pkl')
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        st.stop()

models = load_models()
st.markdown('<div class="success-box">✅ 模型加载成功！共包含 {} 个预测模型</div>'.format(len(models)), unsafe_allow_html=True)

# ==================== 系统说明 ====================
with st.expander("📖 系统使用说明", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📥 输入参数说明**
        - **入水参数**: SNH, TSS, TotalN, COD, BOD5
        - **控制参数**: R2_NO2 (第2反应池硝态氮)
        - **控制参数**: R5_DO (第5反应池溶解氧)
        """)
    
    with col2:
        st.markdown("""
        **📊 预测指标**
        - 出水水质: SNH, TSS, TotalN, COD, BOD5
        - 总能耗 (kWh)
        - 出水水质指数
        """)
    
    with col3:
        st.markdown("""
        **💡 使用提示**
        1. 输入当前进水参数
        2. 设置控制参数
        3. 点击预测按钮
        4. 查看预测结果和可视化
        """)

st.markdown("---")

# ==================== 定义特征和标签信息 ====================
features_info = {
    'SNH_in': {'name': '入水SNH浓度', 'unit': 'mg/L', 'range': (0, 100), 'default': 30, 'icon': '🔵'},
    'TSS_in': {'name': '入水TSS浓度', 'unit': 'mg/L', 'range': (0, 500), 'default': 150, 'icon': '🟤'},
    'TotalN_in': {'name': '入水总氮', 'unit': 'mg/L', 'range': (0, 100), 'default': 50, 'icon': '🟢'},
    'COD_in': {'name': '入水COD浓度', 'unit': 'mg/L', 'range': (0, 1000), 'default': 300, 'icon': '🔴'},
    'BOD5_in': {'name': '入水BOD5浓度', 'unit': 'mg/L', 'range': (0, 500), 'default': 150, 'icon': '🟡'},
    'R2_NO2': {'name': '第2反应池硝态氮', 'unit': 'mg/L', 'range': (0, 50), 'default': 10, 'icon': '⚙️'},
    'R5_DO': {'name': '第5反应池溶解氧', 'unit': 'mg/L', 'range': (0, 10), 'default': 3, 'icon': '⚙️'}
}

targets_info = {
    'SNH': {'name': '出水SNH', 'unit': 'mg/L', 'lower_better': True, 'color': '#2196F3'},
    'TSS': {'name': '出水TSS', 'unit': 'mg/L', 'lower_better': True, 'color': '#FF9800'},
    'TotalN': {'name': '出水总氮', 'unit': 'mg/L', 'lower_better': True, 'color': '#4CAF50'},
    'COD': {'name': '出水COD', 'unit': 'mg/L', 'lower_better': True, 'color': '#F44336'},
    'BOD5': {'name': '出水BOD5', 'unit': 'mg/L', 'lower_better': True, 'color': '#FFC107'},
    'total_energy': {'name': '总能耗', 'unit': 'kWh', 'lower_better': True, 'color': '#9C27B0'},
    'EQ_contrib': {'name': '出水水质指数', 'unit': '点', 'lower_better': True, 'color': '#00BCD4'}
}

# ==================== 输入特征参数 ====================
st.header("📝 输入参数配置")

# 进水参数
st.subheader("💧 进水水质参数")
input_features = {}

col1, col2, col3, col4, col5 = st.columns(5)
inlet_features = ['SNH_in', 'TSS_in', 'TotalN_in', 'COD_in', 'BOD5_in']

for idx, feature in enumerate(inlet_features):
    info = features_info[feature]
    col = [col1, col2, col3, col4, col5][idx]
    with col:
        input_features[feature] = st.number_input(
            label=f"{info['icon']} **{info['name']}**\n({info['unit']})",
            min_value=float(info['range'][0]),
            max_value=float(info['range'][1]),
            value=float(info['default']),
            step=1.0,
            format="%.1f",
            key=feature
        )

st.markdown("<br>", unsafe_allow_html=True)

# 控制参数
st.subheader("⚙️ 工艺控制参数")
col1, col2 = st.columns(2)

control_features = ['R2_NO2', 'R5_DO']
for idx, feature in enumerate(control_features):
    info = features_info[feature]
    col = col1 if idx == 0 else col2
    with col:
        input_features[feature] = st.number_input(
            label=f"{info['icon']} **{info['name']}**\n({info['unit']})",
            min_value=float(info['range'][0]),
            max_value=float(info['range'][1]),
            value=float(info['default']),
            step=0.1,
            format="%.2f",
            key=feature,
            help=f"第{2 if idx == 0 else 5}反应池的{'硝态氮浓度' if idx == 0 else '溶解氧浓度'}"
        )

# 显示当前输入参数摘要
with st.expander("📋 查看当前输入参数", expanded=False):
    input_df = pd.DataFrame([{
        '参数': features_info[k]['name'],
        '数值': f"{v:.2f}",
        '单位': features_info[k]['unit']
    } for k, v in input_features.items()])
    st.dataframe(input_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ==================== 预测按钮 ====================
st.header("🔮 模型预测")

predict_button = st.button("🚀 开始预测", use_container_width=True, type="primary")

if predict_button:
    with st.spinner("🔄 正在预测中..."):
        # 进行预测
        predictions = {}
        input_df_pred = pd.DataFrame([input_features])
        
        for target in models.keys():
            pred_value = models[target].predict(input_df_pred)[0]
            predictions[target] = pred_value
    
    st.success("✅ 预测完成！")
    
    st.markdown("---")
    
    # ==================== 预测结果展示 ====================
    st.header("📊 预测结果")
    
    # 关键指标卡片
    st.subheader("🎯 核心指标")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #9C27B0 0%, #E91E63 100%);">
            <h3>总能耗</h3>
            <h2>{predictions['total_energy']:.2f}</h2>
            <p>kWh</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #00BCD4 0%, #2196F3 100%);">
            <h3>出水水质指数</h3>
            <h2>{predictions['EQ_contrib']:.2f}</h2>
            <p>点（越低越好）</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 出水水质指标
    st.subheader("💧 出水水质指标")
    
    water_quality_targets = ['SNH', 'TSS', 'TotalN', 'COD', 'BOD5']
    cols = st.columns(5)
    
    gradients = [
        "linear-gradient(135deg, #2196F3 0%, #21CBF3 100%)",
        "linear-gradient(135deg, #FF9800 0%, #FFC107 100%)",
        "linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%)",
        "linear-gradient(135deg, #F44336 0%, #FF5722 100%)",
        "linear-gradient(135deg, #FFC107 0%, #FFEB3B 100%)"
    ]
    
    for idx, target in enumerate(water_quality_targets):
        with cols[idx]:
            info = targets_info[target]
            inlet_val = input_features.get(f'{target}_in', 0)
            outlet_val = predictions[target]
            removal = ((inlet_val - outlet_val) / inlet_val * 100) if inlet_val > 0 else 0
            
            st.markdown(f"""
            <div class="metric-card" style="background: {gradients[idx]};">
                <h3>{info['name']}</h3>
                <h2>{outlet_val:.2f}</h2>
                <p>{info['unit']} | 去除率: {removal:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ==================== 详细数据表格 ====================
    st.subheader("📋 详细预测数据")
    
    result_data = []
    for target, value in predictions.items():
        info = targets_info[target]
        
        # 计算去除率（仅对水质指标）
        if target in water_quality_targets:
            inlet_key = f'{target}_in'
            if inlet_key in input_features:
                inlet_val = input_features[inlet_key]
                removal_rate = ((inlet_val - value) / inlet_val * 100) if inlet_val > 0 else 0
                removal_str = f"{removal_rate:.2f}%"
            else:
                removal_str = "N/A"
        else:
            removal_str = "N/A"
        
        result_data.append({
            '指标': info['name'],
            '预测值': f"{value:.2f}",
            '单位': info['unit'],
            '去除率': removal_str,
            '趋势': '↓ 越低越好' if info['lower_better'] else '↑ 越高越好'
        })
    
    result_df = pd.DataFrame(result_data)
    st.dataframe(result_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ==================== 可视化分析 ====================
    st.header("📈 可视化分析")
    
    tab1, tab2, tab3 = st.tabs(["📊 进出水对比", "🎯 去除效率", "📉 预测结果总览"])
    
    with tab1:
        st.subheader("进出水水质对比")
        
        # 进出水对比图
        inlet_vals = [input_features.get(f'{t}_in', 0) for t in water_quality_targets]
        outlet_vals = [predictions[t] for t in water_quality_targets]
        
        fig1 = go.Figure()
        
        fig1.add_trace(go.Bar(
            name='进水',
            x=[targets_info[t]['name'] for t in water_quality_targets],
            y=inlet_vals,
            marker_color='#FF6B6B',
            text=[f'{v:.1f}' for v in inlet_vals],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>进水: %{y:.2f} mg/L<extra></extra>'
        ))
        
        fig1.add_trace(go.Bar(
            name='出水',
            x=[targets_info[t]['name'] for t in water_quality_targets],
            y=outlet_vals,
            marker_color='#4ECDC4',
            text=[f'{v:.1f}' for v in outlet_vals],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>出水: %{y:.2f} mg/L<extra></extra>'
        ))
        
        fig1.update_layout(
            title='进出水浓度对比',
            xaxis_title='水质指标',
            yaxis_title='浓度 (mg/L)',
            barmode='group',
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        st.subheader("污染物去除效率分析")
        
        # 去除率图
        removal_rates = [
            ((inlet_vals[i] - outlet_vals[i]) / inlet_vals[i] * 100) if inlet_vals[i] > 0 else 0
            for i in range(len(water_quality_targets))
        ]
        
        fig2 = go.Figure()
        
        colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#FFC107']
        
        fig2.add_trace(go.Bar(
            x=[targets_info[t]['name'] for t in water_quality_targets],
            y=removal_rates,
            marker_color=colors,
            text=[f'{r:.1f}%' for r in removal_rates],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>去除率: %{y:.2f}%<extra></extra>'
        ))
        
        fig2.update_layout(
            title='污染物去除效率',
            xaxis_title='水质指标',
            yaxis_title='去除率 (%)',
            height=500,
            template='plotly_white',
            showlegend=False
        )
        
        # 添加参考线
        fig2.add_hline(y=90, line_dash="dash", line_color="green", 
                      annotation_text="优秀线 (90%)", annotation_position="right")
        fig2.add_hline(y=80, line_dash="dash", line_color="orange", 
                      annotation_text="良好线 (80%)", annotation_position="right")
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # 去除效率评价
        avg_removal = np.mean(removal_rates)
        if avg_removal >= 90:
            assessment = "🌟 优秀"
            color = "success"
        elif avg_removal >= 80:
            assessment = "✅ 良好"
            color = "info"
        elif avg_removal >= 70:
            assessment = "⚠️ 一般"
            color = "warning"
        else:
            assessment = "❌ 需改进"
            color = "error"
        
        st.markdown(f"""
        <div class="{color}-box">
        <strong>综合去除效率评价:</strong> {assessment}<br>
        <strong>平均去除率:</strong> {avg_removal:.2f}%
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.subheader("预测结果综合总览")
        
        # 雷达图
        categories = [targets_info[t]['name'] for t in water_quality_targets]
        
        # 归一化处理（假设最大值为进水值）
        normalized_inlet = [100 for _ in range(len(water_quality_targets))]
        normalized_outlet = [
            (outlet_vals[i] / inlet_vals[i] * 100) if inlet_vals[i] > 0 else 0
            for i in range(len(water_quality_targets))
        ]
        
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatterpolar(
            r=normalized_inlet,
            theta=categories,
            fill='toself',
            name='进水（基准）',
            line=dict(color='#FF6B6B', width=2),
            fillcolor='rgba(255, 107, 107, 0.2)'
        ))
        
        fig3.add_trace(go.Scatterpolar(
            r=normalized_outlet,
            theta=categories,
            fill='toself',
            name='出水（预测）',
            line=dict(color='#4ECDC4', width=2),
            fillcolor='rgba(78, 205, 196, 0.2)'
        ))
        
        fig3.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 120]
                )
            ),
            showlegend=True,
            title='进出水水质雷达对比图（归一化）',
            height=500
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # 关键指标汇总
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="总能耗",
                value=f"{predictions['total_energy']:.2f} kWh",
                delta=None
            )
        
        with col2:
            st.metric(
                label="出水水质指数",
                value=f"{predictions['EQ_contrib']:.2f} 点",
                delta=None,
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                label="平均去除率",
                value=f"{avg_removal:.2f}%",
                delta=None
            )
    
    st.markdown("---")
    
    # ==================== 导出结果 ====================
    st.header("💾 导出结果")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 导出预测结果
        export_data = {
            '参数类型': ['输入参数'] * len(input_features) + ['预测结果'] * len(predictions),
            '参数名称': list(input_features.keys()) + list(predictions.keys()),
            '数值': [f"{v:.2f}" for v in input_features.values()] + [f"{v:.2f}" for v in predictions.values()],
            '单位': [features_info[k]['unit'] for k in input_features.keys()] + 
                    [targets_info[k]['unit'] for k in predictions.keys()]
        }
        export_df = pd.DataFrame(export_data)
        csv = export_df.to_csv(index=False, encoding='utf-8-sig')
        
        st.download_button(
            label="📥 下载预测结果 (CSV)",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # 导出详细报告
        report_lines = [
            "="*50,
            "污水处理厂预测报告",
            "="*50,
            "",
            "【输入参数】",
            *[f"  {features_info[k]['name']}: {v:.2f} {features_info[k]['unit']}" 
              for k, v in input_features.items()],
            "",
            "【预测结果】",
            *[f"  {targets_info[k]['name']}: {v:.2f} {targets_info[k]['unit']}" 
              for k, v in predictions.items()],
            "",
            f"【综合评价】",
            f"  平均去除率: {avg_removal:.2f}%",
            f"  综合评价: {assessment}",
            "",
            "="*50
        ]
        report_text = "\n".join(report_lines)
        
        st.download_button(
            label="📄 下载文本报告 (TXT)",
            data=report_text,
            file_name="prediction_report.txt",
            mime="text/plain",
            use_container_width=True
        )

else:
    # 未预测时的提示
    st.markdown("""
    <div class="info-box">
        <h3>ℹ️ 使用指南</h3>
        <p><strong>操作步骤：</strong></p>
        <ol>
            <li>📥 输入进水水质参数（SNH_in, TSS_in, TotalN_in, COD_in, BOD5_in）</li>
            <li>⚙️ 设置工艺控制参数（R2_NO2, R5_DO）</li>
            <li>🚀 点击"开始预测"按钮</li>
            <li>📊 查看预测结果和可视化分析</li>
            <li>💾 下载预测报告</li>
        </ol>
        <p><strong>提示：</strong>所有参数都有默认值，您可以直接点击预测查看示例结果。</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== 页脚 ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background-color: #F5F7FA; border-radius: 10px;'>
    <h3 style='color: #2196F3; margin-bottom: 1rem;'>🌊 污水处理厂智能预测系统</h3>
    <p style='color: #666; margin: 0.5rem 0;'><strong>预测模型:</strong> 基于机器学习的多目标回归</p>
    <p style='color: #666; margin: 0.5rem 0;'><strong>预测指标:</strong> 出水水质 (SNH, TSS, TotalN, COD, BOD5) + 总能耗 + 水质指数</p>
    <p style='color: #666; margin: 0.5rem 0;'><strong>控制参数:</strong> R2_NO2 (第2反应池硝态氮) & R5_DO (第5反应池溶解氧)</p>
    <hr style='margin: 1rem 0; border: none; border-top: 1px solid #ddd;'>
    <p style='color: #999; font-size: 0.9rem;'>© 2025 污水处理智能预测系统 | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)