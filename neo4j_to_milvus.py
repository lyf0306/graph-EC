import os
import hashlib
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

# ================= 1. 配置区域 =================
NEO4J_URI = "neo4j://localhost:7688"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "12345678"

# 新增：Milvus 数据库配置
MILVUS_URI = "http://localhost:19530"

# 【注意】：GraphR1 默认的命名空间通常是 entities 和 hyperedges。
# 如果后续运行 test.py 时报错说找不到某张表，你可以将这里的名字改为报错提示的名字（例如 "vdb_entities"）
COLLECTION_ENTITIES = "entities"       
COLLECTION_HYPEREDGES = "hyperedges"   

EMBEDDING_MODEL_PATH = "/root/Model/Qwen3-Embedding-4B"
BATCH_SIZE = 64

# ================= 2. 加载 Embedding 模型 =================
print(f">>> [初始化] 正在加载 Embedding 模型: {EMBEDDING_MODEL_PATH} ...")
try:
    embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH, trust_remote_code=True)
    EMBEDDING_DIM = embed_model.get_sentence_embedding_dimension()
    print(f"✔ 模型加载成功！检测到向量维度为: {EMBEDDING_DIM}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit(1)

# ================= 3. 连接 Milvus 并初始化集合 =================
print(f">>> [初始化] 正在连接 Milvus: {MILVUS_URI} ...")
try:
    milvus_client = MilvusClient(uri=MILVUS_URI)
except Exception as e:
    print(f"❌ 连接 Milvus 失败，请检查 Docker 容器是否正常运行: {e}")
    exit(1)

def init_milvus_collection(collection_name):
    """创建 Milvus 集合并配置动态 Schema 与 HNSW 索引"""
    if milvus_client.has_collection(collection_name):
        print(f"⚠️ 发现已存在的集合 [{collection_name}]，正在将其清空重构...")
        milvus_client.drop_collection(collection_name)
    
    print(f"📦 正在创建 Milvus 集合: {collection_name} ...")
    
    # 1. 先准备好 HNSW 索引参数
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type="COSINE",
        index_type="HNSW",
        index_name="vector_index",
        params={"M": 16, "efConstruction": 256}
    )
    
    # 2. 在创建集合时，直接把 index_params 传进去
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=EMBEDDING_DIM,
        primary_field_name="id",       
        vector_field_name="vector",    
        id_type="string",
        max_length=65535,
        metric_type="COSINE",
        enable_dynamic_field=True,
        index_params=index_params      # 🔴 关键修改：直接在这里传入索引配置
    )
    
    # 3. 删除了原来独立的 milvus_client.create_index(...) 行，直接加载集合
    milvus_client.load_collection(collection_name)
    print(f"✔ 集合 [{collection_name}] 创建并加载完毕！\n")

# ================= 4. 重构 Entities 向量库 =================
def rebuild_entities():
    print("--- 正在重构 Entities 向量库 ---")
    init_milvus_collection(COLLECTION_ENTITIES)
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    query = """
    MATCH (n:Entity) 
    RETURN n.name AS name, n.description AS description, elementId(n) AS neo4j_id
    """
    
    with driver.session() as session:
        result = list(session.run(query))
        total = len(result)
        print(f"找到 {total} 个 Entity 节点，开始批量生成并插入 Milvus...")
        
        batch_texts = []
        batch_data = []
        inserted_count = 0
        
        for record in result:
            name = record["name"] or ""
            desc = record["description"] or ""
            
            # 使用 md5 保持与 GraphR1 源生代码 ID 计算逻辑一致
            entity_id = f"ent-{hashlib.md5(name.encode('utf-8')).hexdigest()}"
            content = name + desc
            
            item = {
                "id": entity_id,            # Milvus 主键
                "content": content,         # 文本内容
                "entity_name": name,        # 动态元数据
            }
            if record["neo4j_id"]: 
                item["neo4j_id"] = record["neo4j_id"]
            
            batch_texts.append(content)
            batch_data.append(item)
            
            # 批量插入逻辑
            if len(batch_texts) >= BATCH_SIZE:
                embeddings = embed_model.encode(batch_texts, normalize_embeddings=True)
                for i, emb in enumerate(embeddings):
                    batch_data[i]["vector"] = emb.tolist()
                
                milvus_client.insert(collection_name=COLLECTION_ENTITIES, data=batch_data)
                inserted_count += len(batch_data)
                batch_texts, batch_data = [], []
                print(f"  进度: {inserted_count} / {total}")
                
        # 处理剩余尾部数据
        if batch_texts:
            embeddings = embed_model.encode(batch_texts, normalize_embeddings=True)
            for i, emb in enumerate(embeddings):
                batch_data[i]["vector"] = emb.tolist()
            milvus_client.insert(collection_name=COLLECTION_ENTITIES, data=batch_data)
            inserted_count += len(batch_data)
            print(f"  进度: {inserted_count} / {total}")
            
    driver.close()
    print(f"✅ 成功将 {inserted_count} 条 Entity 数据写入 Milvus！\n")

# ================= 5. 重构 Hyperedges 向量库 =================
def rebuild_hyperedges():
    print("--- 正在重构 Hyperedges 向量库 ---")
    init_milvus_collection(COLLECTION_HYPEREDGES)
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    query = """
    MATCH (h:Hyperedge)
    OPTIONAL MATCH (p:Paper)-[:BELONG_TO]-(h)
    OPTIONAL MATCH (h)-[:RELATES_TO]-(e:Entity)
    RETURN h.name AS name, 
           h.weight AS weight, 
           collect(DISTINCT p.name) AS papers, 
           collect(DISTINCT e.name) AS entities,
           elementId(h) AS neo4j_id
    """
    
    with driver.session() as session:
        result = list(session.run(query))
        total = len(result)
        print(f"找到 {total} 个 Hyperedge 节点，开始批量生成并插入 Milvus...")
        
        batch_texts = []
        batch_data = []
        inserted_count = 0
        
        for record in result:
            name = record["name"] or ""
            weight = record["weight"] or 1.0
            papers = [p for p in record["papers"] if p is not None]
            entities = [e for e in record["entities"] if e is not None]
            
            edge_id = f"rel-{hashlib.md5(name.encode('utf-8')).hexdigest()}"
            
            entities_str = ", ".join(entities) if entities else "None"
            papers_str = ", ".join(papers) if papers else "None"
            text_to_embed = f"Hyperedge: {name} | Linked Entities: {entities_str} | Source Papers: {papers_str}"
            
            item = {
                "id": edge_id,
                "content": text_to_embed,
                "hyperedge_name": name,
                "weight": weight
            }
            if record["neo4j_id"]: 
                item["neo4j_id"] = record["neo4j_id"]
            
            batch_texts.append(text_to_embed)
            batch_data.append(item)
            
            if len(batch_texts) >= BATCH_SIZE:
                embeddings = embed_model.encode(batch_texts, normalize_embeddings=True)
                for i, emb in enumerate(embeddings):
                    batch_data[i]["vector"] = emb.tolist()
                
                milvus_client.insert(collection_name=COLLECTION_HYPEREDGES, data=batch_data)
                inserted_count += len(batch_data)
                batch_texts, batch_data = [], []
                print(f"  进度: {inserted_count} / {total}")
                
        if batch_texts:
            embeddings = embed_model.encode(batch_texts, normalize_embeddings=True)
            for i, emb in enumerate(embeddings):
                batch_data[i]["vector"] = emb.tolist()
            milvus_client.insert(collection_name=COLLECTION_HYPEREDGES, data=batch_data)
            inserted_count += len(batch_data)
            print(f"  进度: {inserted_count} / {total}")
                
    driver.close()
    print(f"✅ 成功将 {inserted_count} 条 Hyperedge 数据写入 Milvus！\n")

def main():
    print("🚀 启动全局向量库重构任务 (Neo4j -> Milvus)！")
    rebuild_entities()
    rebuild_hyperedges()
    print("🎯 重构彻底完成！图谱数据已全量灌入 Milvus 高性能向量引擎！")

if __name__ == "__main__":
    main()