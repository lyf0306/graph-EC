# graphr1/hyper_attention.py
import os
import json
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 放入我们最新训练的重装学术版模型架构
class DeepHeuristicHypergraphAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=8, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # 角色注入与层归一化
        self.role_embeddings = nn.Embedding(6, embedding_dim)
        self.fusion_norm = nn.LayerNorm(embedding_dim)
        
        self.W_q = nn.Linear(embedding_dim, num_heads * head_dim)
        self.W_k = nn.Linear(embedding_dim, num_heads * head_dim)
        self.dropout = nn.Dropout(0.2)
        
        # 多层深度交互网络 (Deep FFN)
        self.interaction_scorer = nn.Sequential(
            nn.Linear(4 * head_dim, 2 * head_dim),
            nn.GELU(),
            nn.LayerNorm(2 * head_dim),
            nn.Dropout(0.15),
            
            nn.Linear(2 * head_dim, head_dim),
            nn.GELU(),
            nn.LayerNorm(head_dim),
            nn.Dropout(0.1),
            
            nn.Linear(head_dim, 1) 
        )
        
        self.idf_gate_net = nn.Linear(1, 1)

    def forward(self, q_emb, target_data):
        batch_size = q_emb.size(0)
        
        # ================= 场景 A：无缝兼容 GraphR1 检索管线 =================
        if not isinstance(target_data, (tuple, list)):
            target_emb = target_data
            # 默认赋予 CONTEXT 角色 (index 1)
            default_role = torch.tensor([1], dtype=torch.long, device=target_emb.device) 
            target_emb = self.fusion_norm(target_emb + self.role_embeddings(default_role).squeeze(0))
            
            q = self.dropout(F.gelu(self.W_q(q_emb))).view(-1, self.num_heads, self.head_dim)
            k = self.dropout(F.gelu(self.W_k(target_emb))).view(-1, self.num_heads, self.head_dim)
            
            # 🌟 修复点：将患者 Query 的维度 (1, 8, 128) 广播复制为与实体数量一致的 (424, 8, 128)
            q_expanded = q.expand_as(k)
            
            diff = torch.abs(q_expanded - k)
            mult = q_expanded * k
            # 此时所有的 tensor 第0维都是 N (例如424)，可以完美拼接！
            interaction = torch.cat([q_expanded, k, diff, mult], dim=-1)
            
            # 经过深度 MLP 得到非线性基础得分
            scores = self.interaction_scorer(interaction).squeeze(-1)
            
            # 输出对齐：恢复为 0.0 ~ 1.0 的平滑临床推荐度
            weights = torch.sigmoid(scores.mean(dim=-1))
            
            if self.training: return weights, q
            return weights

        # ================= 场景 B：超图结构化对齐（训练级） =================
        ent_embs, roles, idfs, mask = target_data
        max_ent = ent_embs.size(1)
        
        r_embs = self.role_embeddings(roles)
        node_features = self.fusion_norm(ent_embs + r_embs)
        
        q = self.dropout(F.gelu(self.W_q(q_emb))) 
        k = self.dropout(F.gelu(self.W_k(node_features))) 
        
        q = q.view(batch_size, 1, self.num_heads, self.head_dim)
        k = k.view(batch_size, max_ent, self.num_heads, self.head_dim)
        
        q_expanded = q.expand_as(k) 
        diff = torch.abs(q_expanded - k)
        mult = q_expanded * k
        interaction = torch.cat([q_expanded, k, diff, mult], dim=-1)
        
        base_scores = self.interaction_scorer(interaction).squeeze(-1)
        
        idf_log = torch.log1p(idfs).unsqueeze(-1) 
        gate_weights = torch.sigmoid(self.idf_gate_net(idf_log)) 
        
        gated_scores = base_scores * gate_weights
        
        mask_expanded = mask.unsqueeze(-1)
        hyperedge_scores = (gated_scores * mask_expanded).sum(dim=1) 
        
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
    """
    在系统主程序启动时调用一次即可。
    注意：这里的 model_path 请传入新的 clinical_attention_v2.pth
    """
    global ATTENTION_MODEL, GLOBAL_ENTITY_CACHE
    
    if ATTENTION_MODEL is not None:
        return # 防止重复加载
        
    print(">>> 🧠 正在唤醒 Deep Heuristic Hypergraph Attention 临床直觉模块...")
    
    # 1. 加载图谱实体特征 Base64 矩阵
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

    # 2. 加载全新架构的 PyTorch 权重 [关键修改]
    ATTENTION_MODEL = DeepHeuristicHypergraphAttention(
        embedding_dim=embedding_dim, 
        num_heads=8,      # 适配训练时的 8 头
        head_dim=128      # 适配训练时的 128 维
    )
    
    # 加载权重
    ATTENTION_MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
    ATTENTION_MODEL.to(DEVICE)
    ATTENTION_MODEL.eval() # 开启推理模式
    print(">>> ✔ 临床直觉网络(V2重装版)初始化完成并已进入推理模式！")

def compute_dynamic_weights_sync(query_tensor, entity_tensors):
    """同步纯计算函数，将被 asyncio.to_thread 放入独立线程池执行，绝不阻塞主异步循环"""
    with torch.no_grad():
        query_tensor = query_tensor.to(DEVICE)
        entity_tensors = entity_tensors.to(DEVICE)
        # 模型会自动走进 fallback 兼容层，完成深层交互打分
        weights = ATTENTION_MODEL(query_tensor.unsqueeze(0), entity_tensors)
        return weights.cpu().numpy()
