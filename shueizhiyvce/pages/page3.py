import streamlit as st
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem import Descriptors, Draw
import joblib

st.set_page_config(page_title="抑制剂预测", layout="wide")

# 标题
st.title("🧪 分子抑制剂预测系统")

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load('inhibitor_model.pkl')

model = load_model()

# 输入SMILES
st.subheader("输入SMILES")
smiles_input = st.text_area("请输入SMILES字符串:", height=100)

col_predict = st.columns(1)[0]
predict_button = col_predict.button("🔍 预测", use_container_width=True)

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
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("分子结构")
            img = Draw.MolToImage(mol, size=(300, 300))
            st.image(img)
        
        with col2:
            st.subheader("📊 分子信息")
            st.write(f"**分子量**: {Descriptors.MolWt(mol):.2f}")
            st.write(f"**LogP**: {Descriptors.MolLogP(mol):.2f}")
            st.write(f"**H供体**: {Descriptors.NumHDonors(mol)}")
            st.write(f"**H受体**: {Descriptors.NumHAcceptors(mol)}")
            st.write(f"**可旋转键**: {Descriptors.NumRotatableBonds(mol)}")
            st.write(f"**TPSA**: {Descriptors.TPSA(mol):.2f}")
            st.write(f"**环数**: {Descriptors.RingCount(mol)}")
        
        # 预测结果
        st.subheader("🎯 预测结果")
        
        col_result1, col_result2 = st.columns([1, 1])
        
        with col_result1:
            if prediction == 1:
                st.success("✅ **是抑制剂**", icon="✅")
            else:
                st.info("❌ **非抑制剂**", icon="ℹ️")
        
        with col_result2:
            confidence = max(probability)
            st.metric("置信度", f"{confidence:.4f}")
        
        # 概率分布
        st.subheader("📈 预测概率")
        
        col_prob1, col_prob2 = st.columns(2)
        with col_prob1:
            st.write(f"非抑制剂概率: **{probability[0]:.4f}**")
        with col_prob2:
            st.write(f"是抑制剂概率: **{probability[1]:.4f}**")
        
        st.progress(probability[1])

elif predict_button:
    st.warning("⚠️ 请先输入SMILES")

# 底部示例
st.markdown("---")
st.subheader("💡 示例SMILES")
col_ex1, col_ex2, col_ex3 = st.columns(3)
with col_ex1:
    st.code("CC(C)Cc1ccc(cc1)", language="")
with col_ex2:
    st.code("c1ccccc1", language="")
with col_ex3:
    st.code("CCO", language="")
