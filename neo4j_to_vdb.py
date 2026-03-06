import os
import json
import base64
import hashlib
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# ================= 1. 配置区域 =================
NEO4J_URI = "neo4j://localhost:7688"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "12345678"
WORKING_DIR = "/root/Graph-R1/expr/DeepSeek_QwenEmbed_Graph" 

# 新增：你的 Embedding 模型路径
EMBEDDING_MODEL_PATH = "/root/Model/Qwen3-Embedding-4B"
BATCH_SIZE = 64  # 根据你的显存大小调整（如果你报 OOM 显存不足，请调小到 32 或 16）

# ================= 2. 加载 Embedding 模型 =================
print(f">>> [初始化] 正在加载 Embedding 模型: {EMBEDDING_MODEL_PATH} ...")
try:
    embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH, trust_remote_code=True)
    EMBEDDING_DIM = embed_model.get_sentence_embedding_dimension()
    print(f"✔ 模型加载成功！检测到向量维度为: {EMBEDDING_DIM}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit(1)

# ================= 3. 核心：构建 Base64 向量矩阵 =================
def build_vdb_json(data_list, matrix_list, file_name):
    """将数据和向量列表打包成 nano-vectordb 的标准 JSON"""
    if not data_list:
        print(f"没有数据需要写入 {file_name}。")
        return

    # 1. 将 list 转为 numpy 的 float32 二维矩阵
    matrix_np = np.array(matrix_list, dtype=np.float32)
    
    # 2. 转换为字节流后进行 Base64 编码
    matrix_b64 = base64.b64encode(matrix_np.tobytes()).decode('utf-8')
    
    # 3. 构建完整的 JSON 结构
    vdb_content = {
        "embedding_dim": EMBEDDING_DIM,
        "data": data_list,
        "matrix": matrix_b64
    }
    
    file_path = os.path.join(WORKING_DIR, file_name)
    os.makedirs(WORKING_DIR, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(vdb_content, f, ensure_ascii=False)
    print(f"✅ 成功写入 {file_name}，共 {len(data_list)} 条数据！")

# ================= 4. 重构 Entities 向量库 =================
def rebuild_entities():
    print("\n--- 正在重构 Entities 向量库 (vdb_entities.json) ---")
    data_list = []
    matrix_list = []
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    # 仅查询 Entity 节点及其属性
    query = """
    MATCH (n:Entity) 
    RETURN n.name AS name, n.description AS description, elementId(n) AS id
    """
    
    with driver.session() as session:
        result = list(session.run(query))
        total = len(result)
        print(f"找到 {total} 个 Entity 节点，开始批量生成 Embedding...")
        
        batch_texts = []
        batch_items = []
        
        for record in result:
            name = record["name"] or ""
            desc = record["description"] or ""
            
            # 使用 md5(entity_name) 保持与 graphR1 源生代码 ID 计算逻辑一致
            entity_id = f"ent-{hashlib.md5(name.encode('utf-8')).hexdigest()}"
            content = name + desc  # Entity的向量化文本
            
            item = {
                "__id__": entity_id,
                "entity_name": name,
                "content": content
            }
            if record["id"]: 
                item["id"] = record["id"]
            
            batch_texts.append(content)
            batch_items.append(item)
            
            # 批量处理逻辑
            if len(batch_texts) >= BATCH_SIZE:
                embeddings = embed_model.encode(batch_texts, normalize_embeddings=True)
                matrix_list.extend(embeddings.tolist())
                data_list.extend(batch_items)
                batch_texts, batch_items = [], []
                print(f"  进度: {len(data_list)} / {total}")
                
        # 处理剩余批次
        if batch_texts:
            embeddings = embed_model.encode(batch_texts, normalize_embeddings=True)
            matrix_list.extend(embeddings.tolist())
            data_list.extend(batch_items)
            print(f"  进度: {len(data_list)} / {total}")
            
    driver.close()
    build_vdb_json(data_list, matrix_list, "vdb_entities.json")

# ================= 5. 重构 Hyperedges 向量库 =================
def rebuild_hyperedges():
    print("\n--- 正在重构 Hyperedges 向量库 (vdb_hyperedges.json) ---")
    data_list = []
    matrix_list = []
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    # 关键修改：
    # 1. 将 Hyperedge 作为节点 (Node) 匹配
    # 2. 关系没有写方向箭头 ()-[]-()，完美兼容你在 Neo4j 里翻转过边方向的改动
    query = """
    MATCH (h:Hyperedge)
    OPTIONAL MATCH (p:Paper)-[:BELONG_TO]-(h)
    OPTIONAL MATCH (h)-[:RELATES_TO]-(e:Entity)
    RETURN h.name AS name, 
           h.weight AS weight, 
           collect(DISTINCT p.name) AS papers, 
           collect(DISTINCT e.name) AS entities,
           elementId(h) AS id
    """
    
    with driver.session() as session:
        result = list(session.run(query))
        total = len(result)
        print(f"找到 {total} 个 Hyperedge 节点，开始批量生成 Embedding...")
        
        batch_texts = []
        batch_items = []
        
        for record in result:
            name = record["name"] or ""
            weight = record["weight"] or 1.0
            papers = [p for p in record["papers"] if p is not None]
            entities = [e for e in record["entities"] if e is not None]
            
            # 使用 md5(hyperedge_name) 保持与 graphR1 源码 ID 计算逻辑一致
            edge_id = f"rel-{hashlib.md5(name.encode('utf-8')).hexdigest()}"
            
            # 拼接丰富的上下文用于检索计算：超边名称 + 其归属的 Paper + 其关联的实体
            entities_str = ", ".join(entities) if entities else "None"
            papers_str = ", ".join(papers) if papers else "None"
            text_to_embed = f"Hyperedge: {name} | Linked Entities: {entities_str} | Source Papers: {papers_str}"
            
            item = {
                "__id__": edge_id,
                "hyperedge_name": name,
                "weight": weight,
                "content": text_to_embed
            }
            if record["id"]: 
                item["id"] = record["id"]
            
            batch_texts.append(text_to_embed)
            batch_items.append(item)
            
            # 批量处理逻辑
            if len(batch_texts) >= BATCH_SIZE:
                embeddings = embed_model.encode(batch_texts, normalize_embeddings=True)
                matrix_list.extend(embeddings.tolist())
                data_list.extend(batch_items)
                batch_texts, batch_items = [], []
                print(f"  进度: {len(data_list)} / {total}")
                
        # 处理剩余批次
        if batch_texts:
            embeddings = embed_model.encode(batch_texts, normalize_embeddings=True)
            matrix_list.extend(embeddings.tolist())
            data_list.extend(batch_items)
            print(f"  进度: {len(data_list)} / {total}")
                
    driver.close()
    build_vdb_json(data_list, matrix_list, "vdb_hyperedges.json")

def main():
    print("🚀 启动全局向量库重构任务！")
    rebuild_entities()
    rebuild_hyperedges()
    print("\n🎯 重构彻底完成！现在你的 JSON 向量库已经与 Neo4j 完美同步！")

if __name__ == "__main__":
    main()