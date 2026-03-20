# graphr1/hyper_attention.py
import os
import json
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 🌟 1. 放入纯正学术版：端到端超图神经网络 (HGNN)
class EndToEndHypergraphNetwork(nn.Module):
    def __init__(self, embedding_dim, num_heads=8, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # ================= 核心修复 1 =================
        # 缩小角色的初始化权重，防止起步阶段掩盖文本特征
        self.role_embeddings = nn.Embedding(6, embedding_dim)
        nn.init.normal_(self.role_embeddings.weight, std=0.01) # 强制极小初始化
        
        # 引入一个可学习的缩放标量，初始给予极小影响
        self.role_scale = nn.Parameter(torch.tensor(0.05)) 
        # ============================================

        self.node_norm = nn.LayerNorm(embedding_dim)
        
        # 真正的图消息传递层 (Message Passing)
        self.msg_passing = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=8, 
            batch_first=True,
            dropout=0.1
        )
        self.post_msg_norm = nn.LayerNorm(embedding_dim)
        
        self.W_q = nn.Linear(embedding_dim, num_heads * head_dim)
        self.W_k = nn.Linear(embedding_dim, num_heads * head_dim)
        self.dropout = nn.Dropout(0.2)
        
        # ================= 核心修复 2 =================
        # 深度交叉评分网络 (注意：输入维度从 4*head_dim 变成了 5*head_dim)
        self.interaction_scorer = nn.Sequential(
            nn.Linear(5 * head_dim, 2 * head_dim),
            nn.GELU(),
            nn.LayerNorm(2 * head_dim),
            nn.Dropout(0.15),
            
            nn.Linear(2 * head_dim, head_dim),
            nn.GELU(),
            nn.LayerNorm(head_dim),
            nn.Dropout(0.1),
            
            nn.Linear(head_dim, 1) 
        )

    def forward(self, q_emb, target_data):
        batch_size = q_emb.size(0)
        
        # 强制走真正的图学习前向传播（彻底废弃散装实体兼容）
        ent_embs, roles, mask = target_data
        max_ent = ent_embs.size(1)
        
        # Step 1: 节点特征融合 (实体文本特征 + 极度缩放的角色特征)
        r_embs = self.role_embeddings(roles)
        # 核心修复 3：逼迫网络主要依赖 ent_embs，角色只是一个微弱的 Bias
        H_0 = self.node_norm(ent_embs + r_embs * self.role_scale) # [Batch, MaxEnt, EmbDim]
        
        # Step 2: 真正的超图消息传递 (Message Passing)
        key_padding_mask = (mask == 0.0) 
        msg_out, _ = self.msg_passing(H_0, H_0, H_0, key_padding_mask=key_padding_mask)
        H_1 = self.post_msg_norm(H_0 + msg_out) # 此时 H_1 包含了超边内所有实体的拓扑上下文
        
        # Step 3: Query 与节点表达的深度交叉
        q = self.dropout(F.gelu(self.W_q(q_emb))) 
        
        # 核心修复 4：保留原始纯文本的 K，防止拓扑融合后特征坍缩
        k_msg = self.dropout(F.gelu(self.W_k(H_1)))       # 含有拓扑和角色信息的 K
        k_raw = self.dropout(F.gelu(self.W_k(ent_embs)))  # 纯净的原始文本 K 
        
        q = q.view(batch_size, 1, self.num_heads, self.head_dim)
        k_msg = k_msg.view(batch_size, max_ent, self.num_heads, self.head_dim)
        k_raw = k_raw.view(batch_size, max_ent, self.num_heads, self.head_dim)
        
        q_expanded = q.expand_as(k_msg) 
        
        # 核心修复 5：将原始文本差异与拓扑差异合并，逼迫 MLP 必须看文本特征！
        interaction = torch.cat([
            q_expanded, 
            k_msg, 
            torch.abs(q_expanded - k_msg),  # 拓扑语义差距
            torch.abs(q_expanded - k_raw),  # 纯文本语义差距 (强制引入)
            q_expanded * k_msg
        ], dim=-1)
        
        cross_scores = self.interaction_scorer(interaction).squeeze(-1) # [Batch, MaxEnt]
        
        # Step 4: 屏蔽无效节点并进行均值池化 (Mean Pooling) 替代 Sum
        mask_expanded = mask.unsqueeze(-1).float()
        valid_node_scores = cross_scores * mask_expanded
        
        # 核心修复 6：按实际有效节点数取平均，消除长超边偏见
        valid_counts = mask_expanded.sum(dim=1).clamp(min=1.0) 
        hyperedge_scores = valid_node_scores.sum(dim=1) / valid_counts
        
        final_score = hyperedge_scores.mean(dim=-1)
        weights = torch.sigmoid(final_score)
        
        if self.training:
            return weights, q.squeeze(1)
        return weights

# 2. 全局单例与缓存
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ATTENTION_MODEL = None
GLOBAL_ENTITY_CACHE = {}

def init_attention_system(model_path: str, vdb_path: str, embedding_dim: int):
    global ATTENTION_MODEL, GLOBAL_ENTITY_CACHE
    if ATTENTION_MODEL is not None: return 
        
    print(">>> 🧠 正在唤醒 End-to-End Hypergraph Neural Network (HGNN) 临床直觉模块...")
    
    if os.path.exists(vdb_path):
        with open(vdb_path, "r", encoding="utf-8") as f:
            vdb_data = json.load(f)
        matrix_bytes = base64.b64decode(vdb_data["matrix"])
        matrix_np = np.frombuffer(matrix_bytes, dtype=np.float32).reshape(-1, vdb_data["embedding_dim"])
        tensor_matrix = torch.tensor(matrix_np)
        
        for idx, item in enumerate(vdb_data["data"]):
            entity_name = item["entity_name"].upper()
            GLOBAL_ENTITY_CACHE[entity_name] = tensor_matrix[idx]
        print(f"  ✔ 成功加载了 {len(GLOBAL_ENTITY_CACHE)} 个实体特征的离线向量！")
    else:
        print(f"  ❌ 警告: 找不到实体向量库 {vdb_path}")

    # 🌟 实例化新版 HGNN 模型
    ATTENTION_MODEL = EndToEndHypergraphNetwork(
        embedding_dim=embedding_dim, 
        num_heads=8, 
        head_dim=128
    )
    
    ATTENTION_MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
    ATTENTION_MODEL.to(DEVICE)
    ATTENTION_MODEL.eval() 
    print(">>> ✔ 纯正学术版 (HGNN) 临床直觉网络初始化完成并已进入推理模式！")

# 替换原有的 compute_dynamic_weights_sync
def compute_hyperedge_scores_sync(query_tensor, hyperedges_data):
    """
    同步计算超边的整体推荐度
    hyperedges_data: (ent_embs_batch, roles_batch, mask_batch)
    """
    with torch.no_grad():
        ent_embs_batch, roles_batch, mask_batch = hyperedges_data
        
        # 将 Query 扩张到与 Batch Size (超边数量) 一致
        batch_size = ent_embs_batch.size(0)
        q_emb_batch = query_tensor.unsqueeze(0).expand(batch_size, -1).to(DEVICE)
        
        ent_embs_batch = ent_embs_batch.to(DEVICE)
        roles_batch = roles_batch.to(DEVICE)
        mask_batch = mask_batch.to(DEVICE)
        
        # 送入 HGNN 进行端到端打分
        weights = ATTENTION_MODEL(q_emb_batch, (ent_embs_batch, roles_batch, mask_batch))
        
        return weights.cpu().numpy() # 返回每个超边的最终概率 [Batch_size]
