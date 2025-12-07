import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
import numpy as np
from PIL import Image
import io
import os
import requests

# 禁用 RDKit 日志
RDLogger.DisableLog("rdApp.*")

# ==================== 配置参数 (与训练时保持一致) ====================
LATENT_DIM = 256
BOND_DIM = 5
ATOM_DIM = 11
NUM_ATOMS = 120

SMILE_CHARSET = ["C", "B", "F", "I", "H", "O", "N", "S", "P", "Cl", "Br"]
index_to_SMILE = {i: c for i, c in enumerate(SMILE_CHARSET)}

bond_mapping = {
    "SINGLE": 0, 0: Chem.BondType.SINGLE,
    "DOUBLE": 1, 1: Chem.BondType.DOUBLE,
    "TRIPLE": 2, 2: Chem.BondType.TRIPLE,
    "AROMATIC": 3, 3: Chem.BondType.AROMATIC,
}

# 使用 CPU 或 GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'molecule_vae_model_f16.pth')


# ==================== 数据处理函数 (绕过 Sanitize 校验) ====================
def graph_to_molecule(adjacency, features):
    """从图重建分子，绕过 RDKit Sanitize 校验"""
    try:
        molecule = Chem.RWMol()
        
        # 1. 确定保留的原子索引 (非占位符原子且有连接)
        keep_idx = np.where(
            (np.argmax(features, axis=1) != ATOM_DIM - 1)
            & (np.sum(adjacency[:-1], axis=(0, 1)) != 0)
        )[0]
        
        features = features[keep_idx]
        adjacency = adjacency[:, keep_idx, :][:, :, keep_idx]
        
        # 2. 添加原子
        for atom_type_idx in np.argmax(features, axis=1):
            atom = Chem.Atom(index_to_SMILE.get(atom_type_idx, "C"))
            molecule.AddAtom(atom)
        
        # 3. 添加键
        (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
        for bond_ij, atom_i, atom_j in zip(bonds_ij, atoms_i, atoms_j):
            if atom_i == atom_j or bond_ij == BOND_DIM - 1:
                continue
            bond_type = bond_mapping[bond_ij]
            molecule.AddBond(int(atom_i), int(atom_j), bond_type)
        
        # 4. 绕过 RDKit Sanitize 校验，直接尝试转换
        try:
             # 尝试 Kekulize 以便生成 SMILES
             Chem.Kekulize(molecule) 
             return molecule.GetMol()
        except Exception:
             # 如果 Kekulize 失败，返回 RWMol 对象 (可能无法生成 SMILES)
             return molecule 
             
    except Exception:
        # 如果在 RDKit 操作中出现任何底层错误，返回 None
        return None

# ==================== 模型定义 (结构与训练代码一致) ====================

class RelationalGraphConvLayer(nn.Module):
    def __init__(self, atom_dim, bond_dim, units=32):
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(bond_dim, atom_dim, units) * 0.01) 
    def forward(self, adjacency, features, debug=False):
        x = torch.einsum('bdij,bjk->bdik', adjacency, features)
        x = torch.einsum('bdij,djk->bik', x, self.kernel)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, bond_dim, atom_dim, latent_dim=LATENT_DIM): 
        super().__init__()
        self.gconv = RelationalGraphConvLayer(atom_dim, bond_dim, units=32)
        self.fc1 = nn.Linear(32, 64)   
        self.fc2 = nn.Linear(64, 32)   
        self.fc_mean = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.dropout = nn.Dropout(0.1)
    def forward(self, adjacency, features, debug=False):
        x = self.gconv(adjacency, features)
        x = torch.mean(x, dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc_mean(x), self.fc_logvar(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, bond_dim, atom_dim, num_atoms):
        super().__init__()
        self.latent_dim = latent_dim
        self.bond_dim = bond_dim
        self.atom_dim = atom_dim
        self.num_atoms = num_atoms
        self.fc1 = nn.Linear(latent_dim, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.adjacency_out = nn.Linear(64, bond_dim * num_atoms * num_atoms)
        self.features_out = nn.Linear(64, atom_dim * num_atoms)
        self.dropout = nn.Dropout(0.2)
    def forward(self, z, debug=False):
        x = F.tanh(self.fc1(z))
        x = self.dropout(x)
        x = F.tanh(self.fc2(x))
        x = self.dropout(x)
        x = F.tanh(self.fc3(x))
        adjacency = self.adjacency_out(x).view(-1, self.bond_dim, self.num_atoms, self.num_atoms)
        adjacency = (adjacency + adjacency.transpose(2, 3)) / 2
        adjacency = F.softmax(adjacency, dim=1)
        features = self.features_out(x).view(-1, self.num_atoms, self.atom_dim)
        features = F.softmax(features, dim=2)
        return adjacency, features

class MoleculeVAE(nn.Module):
    def __init__(self, bond_dim, atom_dim, num_atoms, latent_dim=LATENT_DIM): 
        super().__init__()
        self.encoder = Encoder(bond_dim, atom_dim, latent_dim)
        self.decoder = Decoder(latent_dim, bond_dim, atom_dim, num_atoms)
        self.property_pred = nn.Linear(latent_dim, 1)
        self.latent_dim = latent_dim
    def reparameterize(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mean + eps * std
    def forward(self, adjacency, features, debug=False):
        z_mean, z_logvar = self.encoder(adjacency, features)
        z = self.reparameterize(z_mean, z_logvar)
        recon_adjacency, recon_features = self.decoder(z)
        property_pred = self.property_pred(z_mean)
        return z_mean, z_logvar, property_pred, recon_adjacency, recon_features
    
    def inference(self, batch_size):
        with torch.no_grad():
            # 使用 half() 对应 float16
            z = torch.randn(batch_size, self.latent_dim).to(DEVICE).half()
            recon_adj, recon_feat = self.decoder(z)
            
            adjacency_idx = torch.argmax(recon_adj, dim=1)
            features_idx = torch.argmax(recon_feat, dim=2)
            
            # 转换成 numpy 和 float32 以便 RDKit 处理
            adjacency_np = F.one_hot(adjacency_idx, num_classes=BOND_DIM).permute(0, 3, 1, 2).cpu().float().numpy()
            features_np = F.one_hot(features_idx, num_classes=ATOM_DIM).cpu().float().numpy()
            
            molecules = []
            for i in range(batch_size):
                mol = graph_to_molecule(adjacency_np[i], features_np[i])
                molecules.append(mol)
            
            return molecules

# ==================== Streamlit CSS 美化 ====================
st.set_page_config(page_title="分子生成系统", layout="wide")

st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #4B0082; 
        padding: 10px;
        margin-bottom: 20px;
        border-bottom: 3px solid #663399;
    }
    .generation-settings {
        background: #F0F8FF;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 30px;
    }
    .generation-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        min-height: 400px;
        text-align: center;
    }
    .warning-box {
        background: #fff8e1; /* 浅黄色 */
        color: #ffa000;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #ffc107;
        text-align: center;
        font-weight: bold;
    }
    .error-box {
        background: #ffe6e6; /* 浅红色 */
        color: #cc0000;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #cc0000;
        text-align: center;
        font-weight: bold;
    }
    .success-summary {
        background: #e6ffe6;
        color: #008000;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== Streamlit 资源加载 ====================
@st.cache_resource
def load_vae_model():
    """加载模型结构，并载入 float16 参数"""
    
    model = MoleculeVAE(BOND_DIM, ATOM_DIM, NUM_ATOMS, LATENT_DIM)
    model = model.to(torch.float16) 
    
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"❌ 错误：模型文件未找到！请确保 '{MODEL_PATH}' 存在。")
        st.stop()
    except Exception as e:
        st.error(f"❌ 模型加载或参数不匹配错误：{e}")
        st.info("提示：请确保模型结构参数 (LATENT_DIM, FC 层宽度) 与保存的模型文件完全一致。")
        st.stop()

# 生成分子结构图 (使用在线 API)
def get_molecule_image(smiles):
    """通过 PubChem API 获取分子结构图"""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/PNG?image_size=300x300"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
    except:
        pass
    return None

# ==================== Streamlit 应用主体 ====================

st.markdown('<h1 class="main-header">🧬 分子生成 VAE 系统</h1>', unsafe_allow_html=True)
st.markdown("从模型的潜在空间中采样，并解码生成新的分子结构。")

# 加载模型
vae_model = load_vae_model()
st.success("✅ VAE 模型加载成功 (结构: GraphVAE, 精度: float16)")

# --- 设置区域 ---
st.markdown('<div class="generation-settings">', unsafe_allow_html=True)
st.subheader("⚙️ 分子生成设置")

col_slider, col_button = st.columns([3, 1])

with col_slider:
    num_to_generate = st.slider("选择生成分子数量", 1, 15, 6)

with col_button:
    st.markdown("##")
    generate_button = st.button("🚀 开始生成", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# --- 结果显示区域 ---
if generate_button:
    st.subheader(f"✨ 生成结果 (共 {num_to_generate} 个采样)")
    
    with st.spinner(f"正在从 {vae_model.latent_dim} 维潜在空间中解码..."):
        generated_mols = vae_model.inference(batch_size=num_to_generate)
    
    valid_count = 0
    cols = st.columns(3)
    
    for i, mol in enumerate(generated_mols):
        col = cols[i % 3]
        
        with col:
            st.markdown(f'<div class="generation-card">', unsafe_allow_html=True)
            st.markdown(f"**分子 #{i+1}**")
            
            smiles = None
            if mol is not None:
                try:
                    smiles = Chem.MolToSmiles(mol)
                except Exception:
                    smiles = None # 如果无法生成 SMILES

            if smiles is not None and len(smiles) > 0:
                valid_count += 1
                
                mol_img = get_molecule_image(smiles)
                
                if mol_img:
                    st.image(mol_img, caption=smiles, width=250)
                else:
                    st.code(smiles)
                    st.warning("结构图加载失败 (API)")
                    
                # 警告：未进行校验
                st.markdown('<div class="warning-box">⚠️ **未进行 RDKit 校验**</div>', unsafe_allow_html=True)
                
            else:
                # 无法重建或生成 SMILES 的分子
                st.markdown('<div class="error-box">💀 解码失败：结构无法转换为有效 SMILES</div>', unsafe_allow_html=True)
                st.markdown("---")
                st.caption("提示：模型输出的图结构在 RDKit 中解析失败。")
                
            st.markdown('</div>', unsafe_allow_html=True)

    # 结果总结
    st.markdown("---")
    summary_cols = st.columns([1, 2, 1])
    with summary_cols[1]:
        st.markdown(f"""
            <div class="success-summary">
                **生成总结**：
                <br>成功生成有效 SMILES 的分子数量: **{valid_count}** / {num_to_generate}
                <br>SMILES 生成率: **{valid_count/num_to_generate:.1%}**
            </div>
        """, unsafe_allow_html=True)
