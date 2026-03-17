# graphr1/hyper_attention.py
import os
import json
import base64
import numpy as np
import torch
import torch.nn as nn

# 1. 结构必须与训练时完全一致
class QueryAwareHypergraphAttention(nn.Module):
    def __init__(self, embedding_dim, attention_dim=256):
        super(QueryAwareHypergraphAttention, self).__init__()
        self.W_q = nn.Linear(embedding_dim, attention_dim)
        self.W_k = nn.Linear(embedding_dim, attention_dim)
        self.dropout = nn.Dropout(0.2)
        self.scale = attention_dim ** 0.5
        
    def forward(self, query_emb, target_emb):
        q = self.dropout(torch.relu(self.W_q(query_emb)))
        k = self.dropout(torch.relu(self.W_k(target_emb)))
        scores = (q * k).sum(dim=1) / self.scale
        weights = torch.sigmoid(scores) * 1.9 + 0.1 
        return weights

# 2. 全局单例与缓存
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ATTENTION_MODEL = None
GLOBAL_ENTITY_CACHE = {}

def init_attention_system(model_path: str, vdb_path: str, embedding_dim: int):
    """
    在系统主程序启动时调用一次即可。
    model_path: 你的 clinical_attention_v1.pth 路径
    vdb_path: 你的 vdb_entities.json 路径
    """
    global ATTENTION_MODEL, GLOBAL_ENTITY_CACHE
    
    if ATTENTION_MODEL is not None:
        return # 防止重复加载
        
    print(">>> 🧠 正在唤醒 Query-Aware Hypergraph Attention 临床直觉模块...")
    
    # 1. 加载通过 neo4j_to_vdb 压制的离线图谱实体特征 Base64 矩阵
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

    # 2. 加载你刚训练好的 PyTorch 权重
    ATTENTION_MODEL = QueryAwareHypergraphAttention(embedding_dim=embedding_dim)
    ATTENTION_MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
    ATTENTION_MODEL.to(DEVICE)
    ATTENTION_MODEL.eval()
    print(">>> ✔ 临床直觉网络初始化完成并已进入推理模式！")

def compute_dynamic_weights_sync(query_tensor, entity_tensors):
    """同步纯计算函数，将被 asyncio.to_thread 放入独立线程池执行，绝不阻塞主异步循环"""
    with torch.no_grad():
        query_tensor = query_tensor.to(DEVICE)
        entity_tensors = entity_tensors.to(DEVICE)
        # 注意: 训练时 query_emb_batch 是一批，这里是一个患者，需要 unsqueeze(0) 补齐 Batch 维度，并配合 broadcast
        weights = ATTENTION_MODEL(query_tensor.unsqueeze(0), entity_tensors)
        return weights.cpu().numpy()
