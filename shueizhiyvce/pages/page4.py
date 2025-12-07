import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
import plotly.graph_objects as go
from PIL import Image
import io

st.set_page_config(
    page_title="分子生成系统",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== 样式 ====================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
    }
    .stTitle {
        color: white;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .molecule-card {
        background: rgba(255,255,255,0.95);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-badge {
        background: #00d084;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .invalid-badge {
        background: #ff6b6b;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 参数 ====================
BOND_DIM = 5
ATOM_DIM = 11
NUM_ATOMS = 120
LATENT_DIM = 435

SMILE_CHARSET = ["C", "B", "F", "I", "H", "O", "N", "S", "P", "Cl", "Br"]
SMILE_to_index = {c: i for i, c in enumerate(SMILE_CHARSET)}
index_to_SMILE = {i: c for i, c in enumerate(SMILE_CHARSET)}

bond_mapping = {
    "SINGLE": 0, 0: Chem.BondType.SINGLE,
    "DOUBLE": 1, 1: Chem.BondType.DOUBLE,
    "TRIPLE": 2, 2: Chem.BondType.TRIPLE,
    "AROMATIC": 3, 3: Chem.BondType.AROMATIC,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 模型定义 ====================
class RelationalGraphConvLayer(nn.Module):
    def __init__(self, atom_dim, bond_dim, units=32):
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(bond_dim, atom_dim, units) * 0.01)
    
    def forward(self, adjacency, features):
        x = torch.einsum('bdij,bjk->bdik', adjacency, features)
        x = torch.einsum('bdij,djk->bik', x, self.kernel)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gconv = RelationalGraphConvLayer(ATOM_DIM, BOND_DIM, units=32)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mean = nn.Linear(64, LATENT_DIM)
        self.fc_logvar = nn.Linear(64, LATENT_DIM)
    
    def forward(self, adjacency, features):
        x = self.gconv(adjacency, features)
        x = torch.mean(x, dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z_mean = self.fc_mean(x)
        z_logvar = self.fc_logvar(x)
        return z_mean, z_logvar

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(LATENT_DIM, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.adjacency_out = nn.Linear(128, BOND_DIM * NUM_ATOMS * NUM_ATOMS)
        self.features_out = nn.Linear(128, ATOM_DIM * NUM_ATOMS)
    
    def forward(self, z):
        x = F.tanh(self.fc1(z))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        
        adjacency = self.adjacency_out(x)
        adjacency = adjacency.view(-1, BOND_DIM, NUM_ATOMS, NUM_ATOMS)
        adjacency = (adjacency + adjacency.transpose(2, 3)) / 2
        adjacency = F.softmax(adjacency, dim=1)
        
        features = self.features_out(x)
        features = features.view(-1, NUM_ATOMS, ATOM_DIM)
        features = F.softmax(features, dim=2)
        
        return adjacency, features

class MoleculeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.property_pred = nn.Linear(LATENT_DIM, 1)
    
    def forward(self, adjacency, features):
        z_mean, z_logvar = self.encoder(adjacency, features)
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z = z_mean + eps * std
        
        recon_adjacency, recon_features = self.decoder(z)
        property_pred = self.property_pred(z_mean)
        
        return z_mean, z_logvar, property_pred, recon_adjacency, recon_features

# ==================== 工具函数 ====================
@st.cache_resource
def load_model(model_path="molecule_vae_model.pth"):
    """加载模型"""
    try:
        model = MoleculeVAE().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except:
        return None

def graph_to_molecule(adjacency, features, threshold=0.3):
    """从图重建分子"""
    try:
        molecule = Chem.RWMol()
        
        keep_idx = np.where(np.argmax(features, axis=1) != ATOM_DIM - 1)[0]
        if len(keep_idx) == 0:
            return None
        
        features = features[keep_idx]
        adjacency = adjacency[:, keep_idx, :][:, keep_idx, :]
        
        for atom_type_idx in np.argmax(features, axis=1):
            atom_symbol = index_to_SMILE.get(atom_type_idx, "C")
            molecule.AddAtom(Chem.Atom(atom_symbol))
        
        added_bonds = set()
        indices = np.triu_indices(adjacency.shape[1], k=1)
        bond_count = 0
        
        for i, j in zip(indices[0], indices[1]):
            bond_probs = adjacency[:, i, j]
            bond_type_idx = np.argmax(bond_probs)
            bond_prob = bond_probs[bond_type_idx]
            
            if bond_type_idx == BOND_DIM - 1 or bond_prob < threshold:
                continue
            
            bond_key = tuple(sorted([i, j]))
            if bond_key in added_bonds:
                continue
            
            try:
                bond_type = bond_mapping[bond_type_idx]
                molecule.AddBond(int(i), int(j), bond_type)
                added_bonds.add(bond_key)
                bond_count += 1
            except:
                continue
        
        if bond_count == 0:
            return None
        
        try:
            Chem.SanitizeMol(
                molecule, 
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_AROMATIZE
            )
        except:
            try:
                Chem.SanitizeMol(
                    molecule,
                    sanitizeOps=(
                        Chem.SanitizeFlags.SANITIZE_PROPERTIES |
                        Chem.SanitizeFlags.SANITIZE_SYMMRINGS
                    )
                )
            except:
                return None
        
        return molecule if molecule.GetNumAtoms() > 0 else None
    except:
        return None

def mol_to_image(mol, size=(300, 300)):
    """分子转图像"""
    if mol is None:
        return None
    try:
        img = Draw.MolToImage(mol, size=size)
        return img
    except:
        return None

def get_molecule_properties(mol):
    """获取分子性质"""
    if mol is None:
        return None
    
    try:
        return {
            "分子量": f"{Descriptors.MolWt(mol):.2f}",
            "LogP": f"{Descriptors.MolLogP(mol):.2f}",
            "H供体": f"{Descriptors.NumHDonors(mol)}",
            "H受体": f"{Descriptors.NumHAcceptors(mol)}",
            "TPSA": f"{Descriptors.TPSA(mol):.2f}",
            "环数": f"{Descriptors.RingCount(mol)}",
        }
    except:
        return None

# ==================== 页面布局 ====================
st.markdown("""
<div style="text-align: center; margin-bottom: 40px;">
    <h1 style="color: white; font-size: 3em; margin: 0;">🧪 分子生成系统</h1>
    <p style="color: rgba(255,255,255,0.8); font-size: 1.1em;">基于VAE的分子结构生成与可视化</p>
</div>
""", unsafe_allow_html=True)

# 左侧配置
with st.sidebar:
    st.header("⚙️ 配置")
    
    mode = st.radio(
        "选择模式",
        ["🎲 随机生成", "🧬 批量生成", "📊 分子分析"]
    )
    
    threshold = st.slider(
        "键概率阈值",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.1,
        help="更高的值会生成更少但更确定的键"
    )
    
    random_seed = st.checkbox("使用固定随机种子", value=False)
    if random_seed:
        seed_value = st.number_input("随机种子", value=42, min_value=0)
    else:
        seed_value = None

# 主内容区
if mode == "🎲 随机生成":
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; color: white;">
        <h3>生成控制</h3>
        </div>
        """, unsafe_allow_html=True)
        
        num_molecules = st.number_input(
            "生成分子数",
            min_value=1,
            max_value=50,
            value=5,
            step=1
        )
        
        if st.button("🚀 生成分子", use_container_width=True):
            model = load_model()
            
            if model is None:
                st.error("❌ 模型加载失败，请检查 molecule_vae_model.pth 文件")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                molecules = []
                smiles_list = []
                valid_count = 0
                
                if seed_value is not None:
                    torch.manual_seed(seed_value)
                    np.random.seed(seed_value)
                
                with torch.no_grad():
                    z = torch.randn(num_molecules, LATENT_DIM).to(device)
                    recon_adjacency, recon_features = model.decoder(z)
                    
                    for i in range(num_molecules):
                        adjacency = recon_adjacency[i].cpu().numpy()
                        features = recon_features[i].cpu().numpy()
                        
                        mol = graph_to_molecule(adjacency, features, threshold=threshold)
                        molecules.append(mol)
                        
                        if mol is not None:
                            smiles = Chem.MolToSmiles(mol)
                            smiles_list.append(smiles)
                            valid_count += 1
                        else:
                            smiles_list.append("INVALID")
                        
                        progress_bar.progress((i + 1) / num_molecules)
                        status_text.text(f"进度: {i + 1}/{num_molecules}")
                
                st.success(f"✅ 生成完成! {valid_count}/{num_molecules} 个有效分子")
    
    with col2:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; color: white;">
        <h3>生成结果</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if 'molecules' in locals():
            cols = st.columns(3)
            
            for idx, (mol, smiles) in enumerate(zip(molecules, smiles_list)):
                with cols[idx % 3]:
                    if mol is not None:
                        img = mol_to_image(mol)
                        if img:
                            st.image(img, caption=f"分子 {idx+1}", use_column_width=True)
                        
                        st.markdown(f"""
                        <div class="molecule-card">
                        <span class="success-badge">✅ 有效</span><br>
                        <code>{smiles}</code>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        props = get_molecule_properties(mol)
                        if props:
                            with st.expander("📊 分子性质"):
                                for key, value in props.items():
                                    st.write(f"**{key}**: {value}")
                    else:
                        st.markdown("""
                        <div class="molecule-card">
                        <span class="invalid-badge">❌ 无效</span>
                        </div>
                        """, unsafe_allow_html=True)

elif mode == "🧬 批量生成":
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
    <h3>批量生成并导出</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_batches = st.number_input("批次数", min_value=1, max_value=10, value=3)
        molecules_per_batch = st.number_input("每批分子数", min_value=1, max_value=20, value=10)
    
    with col2:
        if st.button("🔄 开始批量生成", use_container_width=True):
            model = load_model()
            
            if model is None:
                st.error("❌ 模型加载失败")
            else:
                all_smiles = []
                total_valid = 0
                
                progress_container = st.container()
                
                for batch_num in range(num_batches):
                    with progress_container:
                        st.write(f"**批次 {batch_num + 1}/{num_batches}**")
                        progress_bar = st.progress(0)
                    
                    if seed_value is not None:
                        torch.manual_seed(seed_value + batch_num)
                    
                    with torch.no_grad():
                        z = torch.randn(molecules_per_batch, LATENT_DIM).to(device)
                        recon_adjacency, recon_features = model.decoder(z)
                        
                        for i in range(molecules_per_batch):
                            adjacency = recon_adjacency[i].cpu().numpy()
                            features = recon_features[i].cpu().numpy()
                            
                            mol = graph_to_molecule(adjacency, features, threshold=threshold)
                            
                            if mol is not None:
                                smiles = Chem.MolToSmiles(mol)
                                all_smiles.append(smiles)
                                total_valid += 1
                            
                            progress_bar.progress((i + 1) / molecules_per_batch)
                
                st.success(f"✅ 批量生成完成! 共 {total_valid}/{num_batches * molecules_per_batch} 个有效分子")
                
                # 显示结果
                st.subheader("📋 生成结果")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.dataframe(
                        {"SMILES": all_smiles},
                        use_container_width=True
                    )
                
                with col2:
                    # 下载按钮
                    smiles_text = "\n".join(all_smiles)
                    st.download_button(
                        label="📥 下载 SMILES",
                        data=smiles_text,
                        file_name="generated_molecules.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

elif mode == "📊 分子分析":
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
    <h3>生成统计分析</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📈 生成统计数据", use_container_width=True):
        model = load_model()
        
        if model is None:
            st.error("❌ 模型加载失败")
        else:
            num_samples = 100
            all_molecules = []
            mw_list = []
            logp_list = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if seed_value is not None:
                torch.manual_seed(seed_value)
            
            with torch.no_grad():
                z = torch.randn(num_samples, LATENT_DIM).to(device)
                recon_adjacency, recon_features = model.decoder(z)
                
                for i in range(num_samples):
                    adjacency = recon_adjacency[i].cpu().numpy()
                    features = recon_features[i].cpu().numpy()
                    
                    mol = graph_to_molecule(adjacency, features, threshold=threshold)
                    
                    if mol is not None:
                        all_molecules.append(mol)
                        mw_list.append(Descriptors.MolWt(mol))
                        logp_list.append(Descriptors.MolLogP(mol))
                    
                    progress_bar.progress((i + 1) / num_samples)
                    status_text.text(f"进度: {i + 1}/{num_samples}")
            
            valid_count = len(all_molecules)
            st.success(f"✅ 分析完成! {valid_count}/{num_samples} 个有效分子")
            
            # 显示统计
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("有效率", f"{valid_count/num_samples*100:.1f}%")
            with col2:
                st.metric("平均分子量", f"{np.mean(mw_list):.2f}" if mw_list else "N/A")
            with col3:
                st.metric("平均LogP", f"{np.mean(logp_list):.2f}" if logp_list else "N/A")
            
            # 绘制分布
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=mw_list, name="分子量", nbinsx=30))
            fig.update_layout(
                title="分子量分布",
                xaxis_title="分子量",
                yaxis_title="频数",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=logp_list, name="LogP", nbinsx=30))
            fig2.update_layout(
                title="LogP分布",
                xaxis_title="LogP",
                yaxis_title="频数",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)

# 底部信息
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.8);">
    <p>🧬 分子生成系统 | 基于变分自编码器 (VAE)</p>
</div>
""", unsafe_allow_html=True)