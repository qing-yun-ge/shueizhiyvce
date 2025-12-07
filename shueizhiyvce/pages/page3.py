import streamlit as st
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem import Descriptors
import joblib
import plotly.graph_objects as go
from PIL import Image
import io

st.set_page_config(page_title="抑制剂预测", layout="wide", initial_sidebar_state="collapsed")

# 自定义CSS美化
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
    }
    .stTitle {
        color: white;
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-box {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
        color: white;
    }
    .result-success {
        background: linear-gradient(135deg, #00d084 0%, #00d084 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 1.5em;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-danger {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff6b6b 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 1.5em;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .input-section {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .molecule-section {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# 标题
st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <h1 style="color: white; font-size: 3em; margin: 0;">🧪 分子抑制剂预测系统</h1>
        <p style="color: rgba(255,255,255,0.8); font-size: 1.1em;">使用Machine Learning快速预测分子的抑制剂活性</p>
    </div>
""", unsafe_allow_html=True)

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load(model_path,'.inhibitor_model.pkl')

model = load_model()

# 生成分子SMILES结构图（使用在线API）
def get_molecule_image(smiles):
    try:
        # 使用PubChem API获取分子结构图
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/PNG?image_size=350x350"
        import requests
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
    except:
        pass
    return None

# 输入区域
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.subheader("📝 输入分子SMILES")
smiles_input = st.text_area("请输入SMILES字符串:", height=80, placeholder="例如: CC(C)Cc1ccc(cc1)C(C)C(O)=O")

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    predict_button = st.button("🔍 预测", use_container_width=True, key="predict_btn")

st.markdown('</div>', unsafe_allow_html=True)

# 处理预测
if predict_button and smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    
    if mol is None:
        st.error("❌ 无效的SMILES，请检查输入")
    else:
        # 生成指纹和预测
        fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        prediction = model.predict([fp])[0]
        probability = model.predict_proba([fp])[0]
        
        # 分子结构和信息
        col1, col2 = st.columns([1, 1.2], gap="large")
        
        with col1:
            st.markdown('<div class="molecule-section">', unsafe_allow_html=True)
            st.subheader("🧬 分子结构")
            
            # 尝试获取分子图像
            mol_img = get_molecule_image(smiles_input)
            if mol_img:
                st.image(mol_img, use_column_width=True)
            else:
                st.info("📌 分子结构图加载中...\n(使用在线API渲染)")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="molecule-section">', unsafe_allow_html=True)
            st.subheader("📊 分子性质")
            
            # 创建信息卡片
            properties = [
                ("分子量", f"{Descriptors.MolWt(mol):.2f} g/mol"),
                ("LogP", f"{Descriptors.MolLogP(mol):.2f}"),
                ("H供体", f"{Descriptors.NumHDonors(mol)}"),
                ("H受体", f"{Descriptors.NumHAcceptors(mol)}"),
                ("可旋转键", f"{Descriptors.NumRotatableBonds(mol)}"),
                ("TPSA", f"{Descriptors.TPSA(mol):.2f} Ų"),
                ("环数", f"{Descriptors.RingCount(mol)}")
            ]
            
            for label, value in properties:
                st.markdown(f"""
                    <div class="metric-box">
                        <strong>{label}</strong><br>
                        <span style="font-size: 1.2em;">{value}</span>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 预测结果
        st.markdown("---")
        st.subheader("🎯 预测结果")
        
        col_result1, col_result2, col_result3 = st.columns([1.5, 1, 1.5], gap="large")
        
        with col_result1:
            if prediction == 1:
                st.markdown('<div class="result-success">✅ 是抑制剂</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-danger">❌ 非抑制剂</div>', unsafe_allow_html=True)
        
        with col_result2:
            confidence = max(probability)
            st.metric("置信度", f"{confidence:.2%}")
        
        with col_result3:
            st.metric("预测概率", f"{probability[1]:.2%}")
        
        # 概率可视化
        st.markdown("---")
        st.subheader("📈 预测概率分布")
        
        # 使用Plotly绘制概率图
        fig = go.Figure(data=[
            go.Bar(
                x=['非抑制剂', '是抑制剂'],
                y=[probability[0], probability[1]],
                marker=dict(
                    color=['#ff6b6b', '#00d084'],
                    opacity=0.8
                ),
                text=[f'{probability[0]:.2%}', f'{probability[1]:.2%}'],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>概率: %{y:.4f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="预测概率",
            xaxis_title="类别",
            yaxis_title="概率",
            height=400,
            showlegend=False,
            template="plotly_white",
            font=dict(size=12),
            plot_bgcolor='rgba(240,240,240,0.5)',
            paper_bgcolor='rgba(255,255,255,0.95)',
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif predict_button:
    st.warning("⚠️ 请先输入SMILES")

# 底部示例
st.markdown("---")
st.subheader("💡 示例SMILES")
col_ex1, col_ex2, col_ex3 = st.columns(3, gap="large")

with col_ex1:
    st.markdown("""
    <div style="background: rgba(102,126,234,0.1); padding: 15px; border-radius: 10px; text-align: center;">
        <code style="background: rgba(102,126,234,0.2); padding: 8px; border-radius: 5px;">CC(C)Cc1ccc(cc1)</code>
    </div>
    """, unsafe_allow_html=True)

with col_ex2:
    st.markdown("""
    <div style="background: rgba(102,126,234,0.1); padding: 15px; border-radius: 10px; text-align: center;">
        <code style="background: rgba(102,126,234,0.2); padding: 8px; border-radius: 5px;">c1ccccc1</code>
    </div>
    """, unsafe_allow_html=True)

with col_ex3:
    st.markdown("""
    <div style="background: rgba(102,126,234,0.1); padding: 15px; border-radius: 10px; text-align: center;">
        <code style="background: rgba(102,126,234,0.2); padding: 8px; border-radius: 5px;">CCO</code>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 20px; color: rgba(255,255,255,0.8);">
        <p>Made with ❤️ using Streamlit & RDKit</p>
    </div>
""", unsafe_allow_html=True)

