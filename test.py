import os
import asyncio
import numpy as np
from openai import AsyncOpenAI
from graphr1 import GraphR1, QueryParam
from graphr1.utils import wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer

# ================= 配置区域 =================

VLLM_API_URL = "http://localhost:8000/v1" 
VLLM_API_KEY = "EMPTY" 
LLM_MODEL_NAME = "OriClinical" 

WORKING_DIR = "/root/Graph-R1/expr/DeepSeek_QwenEmbed_Graph" 
EMBEDDING_MODEL_PATH = "/root/Model/Qwen3-Embedding-4B"

os.environ["NEO4J_URI"] = "neo4j://localhost:7688"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "12345678"
os.environ["NEO4J_DATABASE"] = "neo4j"

# ===========================================

async def get_source_details(graph_engine, knowledge_text):
    sources_info = []
    try:
        node_id = f"<hyperedge>{knowledge_text}"
        storage = graph_engine.chunk_entity_relation_graph
        
        # 1. 针对 Neo4JStorage 的逻辑
        if hasattr(storage, 'driver'):
            # 关键修改：去除了箭头方向 -[r]->，改为无向匹配 -[r]-，并添加了 :Paper 标签过滤
            cypher_query = """
            MATCH (paper:Paper)-[r]-(target)
            WHERE target.name = $node_id AND (type(r) = 'BELONG_TO' OR r.role = 'EVIDENCE')
            RETURN paper.name AS src_id, paper.pmid AS pmid, paper.title AS title, paper.guidelines AS guidelines
            """
            async with storage.driver.session() as session:
                result = await session.run(cypher_query, node_id=node_id)
                records = await result.data()
            
            for record in records:
                src_id = record.get("src_id", "Unknown").replace('"', '')
                raw_pmid = record.get("pmid")
                
                # --- [智能甄别] 过滤掉被填入哈希值的伪 PMID ---
                final_pmid = "Unknown"
                if raw_pmid and len(str(raw_pmid)) < 20: 
                    # 正常的 PMID 是较短的数字，不可能是 64 位哈希
                    final_pmid = str(raw_pmid)
                elif "paper::" in src_id:
                    final_pmid = src_id.replace("paper::", "")
                    
                raw_gl = record.get("guidelines")
                gl_str = ", ".join(raw_gl) if isinstance(raw_gl, list) else str(raw_gl or "General Evidence")
                
                sources_info.append({
                    "id": src_id,
                    "pmid": final_pmid, 
                    "title": record.get("title") or "No Title",
                    "guidelines": gl_str
                })
                
        # 2. 兼容 NetworkX (本地缓存) 模式
        elif hasattr(storage, '_graph'):
            nx_graph = storage._graph
            if node_id in nx_graph:
                predecessors = list(nx_graph.predecessors(node_id)) if hasattr(nx_graph, 'predecessors') else list(nx_graph.neighbors(node_id))
                for pred in predecessors:
                    edge_data = nx_graph.get_edge_data(pred, node_id)
                    if isinstance(edge_data, dict) and 0 in edge_data: edge_data = edge_data[0] 
                    
                    if edge_data and (edge_data.get('role') == 'EVIDENCE' or edge_data.get('description') == 'BELONG_TO'):
                        paper_node = await storage.get_node(pred)
                        if paper_node:
                            src_id = pred.replace('"', '')
                            raw_pmid = paper_node.get('pmid')
                            
                            # --- [智能甄别] 过滤本地缓存中的伪 PMID ---
                            final_pmid = "Unknown"
                            if raw_pmid and len(str(raw_pmid)) < 20:
                                final_pmid = str(raw_pmid)
                            elif "paper::" in src_id:
                                final_pmid = src_id.replace("paper::", "")
                            
                            raw_gl = paper_node.get('guidelines')
                            gl_str = ", ".join(raw_gl) if isinstance(raw_gl, list) else str(raw_gl or "General Evidence")
                            
                            sources_info.append({
                                "id": src_id,
                                "pmid": final_pmid,
                                "title": paper_node.get('title', 'No Title'),
                                "guidelines": gl_str
                            })

    except Exception as e:
        print(f"[Warning] Source lookup failed: {e}")
    
    return sources_info

# ===========================================

print(f">>> [1/5] 正在加载 Embedding 模型: {EMBEDDING_MODEL_PATH} ...")
try:
    embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH, trust_remote_code=True)
    
    @wrap_embedding_func_with_attrs(embedding_dim=embed_model.get_sentence_embedding_dimension(), max_token_size=8192)
    async def embedding_func(texts):
        if isinstance(texts, str): texts = [texts]
        embeddings = embed_model.encode(texts, normalize_embeddings=True)
        return embeddings
    print("Embedding 模型加载成功。")
except Exception as e:
    print(f"Embedding 模型加载失败: {e}")
    exit(1)

async def main():
    print(f">>> [2/5] 正在初始化 GraphR1 检索引擎...")
    try:
        graph_engine = GraphR1(
            working_dir=WORKING_DIR,
            embedding_func=embedding_func,
            kv_storage="JsonKVStorage",
            vector_storage="NanoVectorDBStorage", 
            graph_storage="Neo4JStorage", 
        )
    except Exception as e:
        print(f"GraphR1 初始化失败: {e}")
        exit(1)

    patient_case = (
        "患者女，58岁。绝经后阴道流血2月余。MRI提示子宫内膜增厚，侵犯肌层深度 > 1/2。 "
        "术后病理提示：子宫内膜样腺癌，G3（低分化）。 "
        "手术分期为 FIGO IB期。淋巴结检测阴性。 "
        "既往有高血压病史，无生育要求。请制定详细的术后辅助治疗方案。"
    )
    print(f"\n>>> [3/5] 输入病例:\n{patient_case}")

    print("\n>>> [4/5] 正在执行混合检索并构建文献库...")
    
    bibliography = {}
    llm_context_list = []

    try:
        param = QueryParam(mode="hybrid", top_k=5, max_token_for_text_unit=4000)
        retrieved_results = await graph_engine.aquery(patient_case, param)

        print("\n" + "="*20 + " 检索到的证据片段 " + "="*20)
        
        if isinstance(retrieved_results, list):
            for i, item in enumerate(retrieved_results):
                if isinstance(item, dict):
                    content = item.get('<knowledge>', str(item))
                    
                    sources = await get_source_details(graph_engine, content)
                    
                    source_ids = []
                    for src in sources:
                        bibliography[src['id']] = src
                        
                        # --- 在 LLM 提示上下文中，区分是否有真实的 PMID ---
                        if src['pmid'] != "Unknown":
                            display_id = f"PMID:{src['pmid']}"
                        else:
                            display_id = f"DocID:{src['id'][:8]}" # 只取前8位作为指代
                        source_ids.append(display_id)
                    
                    source_display = ", ".join(source_ids) if source_ids else "Unknown Source"
                    
                    print(f"[{i+1}] 来源: {source_display}")
                    print(f"    内容: {content[:80]}...") 
                    print("-" * 30)
                    
                    ref_tag = f"[Ref: {', '.join(source_ids)}]" if source_ids else "[Ref: Unknown]"
                    llm_context_list.append(f"{ref_tag} {content}")
        
        context_str = "\n\n".join(llm_context_list)
        print("="*50)

    except Exception as e:
        print(f"检索出错: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n>>> [5/5] 正在生成治疗方案...")
    
    system_prompt = (
        "你是一个专业的妇科肿瘤临床辅助决策系统。\n"
        "请根据【参考证据】制定治疗方案。\n"
        "要求：\n"
        "1. 使用 <think>...</think> 进行分析。\n"
        "2. 输出 <answer>...</answer> 结论。\n"
        "3. 在建议中引用证据时，请直接使用原文中的 [Ref: ID] 标签，例如 '根据 [Ref: PMID:33078978] 的研究...'"
    )

    user_prompt = f"""【患者信息】
{patient_case}

【参考证据】
{context_str}

请制定术后辅助治疗方案。"""

    client = AsyncOpenAI(base_url=VLLM_API_URL, api_key=VLLM_API_KEY)

    try:
        response = await client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=True,
            temperature=0.1,
            max_tokens=4096
        )

        print("\n>>> 模型回复:\n")
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
        
        print("\n\n" + "="*20 + " 参考文献 (References) " + "="*20)
        if bibliography:
            index = 1
            for paper_id, details in bibliography.items():
                pmid_val = details.get('pmid', 'Unknown')
                
                # --- 最终打印：明确区分标准文献与中文指南 ---
                if pmid_val != 'Unknown':
                    print(f"[{index}] PMID: {pmid_val}")
                else:
                    print(f"[{index}] DocID: {paper_id[:8]}... (无标准PMID文献)")
                    
                print(f"    Title: {details['title']}")
                print(f"    Guidelines: {details['guidelines']}")
                print("-" * 10)
                index += 1
        else:
            print("（本次检索未关联到具体文献节点）")
        print("="*50)

    except Exception as e:
        print(f"LLM 调用失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())