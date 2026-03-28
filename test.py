import os
import asyncio
import numpy as np
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import AsyncOpenAI
from graphr1 import GraphR1, QueryParam
from graphr1.utils import wrap_embedding_func_with_attrs
from graphr1.hyper_attention import init_attention_system

# ================= 配置区域 =================

# LLM 服务配置
VLLM_API_URL = "http://localhost:8000/v1" 
VLLM_API_KEY = "EMPTY" 
LLM_MODEL_NAME = "OriClinical" 

# Rerank 服务配置
RERANK_API_URL = "http://localhost:8001/v1"
RERANK_API_KEY = "EMPTY"
RERANK_MODEL_NAME = "QwenReranker"

# Embedding 服务配置
EMBEDDING_API_URL = "http://localhost:8002/v1"
EMBEDDING_API_KEY = "EMPTY"
EMBEDDING_MODEL_NAME = "QwenEmbedding" 
EMBEDDING_DIM = 2560

WORKING_DIR = "/root/Graph-R1/expr/DeepSeek_QwenEmbed_Graph" 
MOE_MODEL_PATH = "/root/Model/moe_router.pth"

# Neo4j 数据库配置
os.environ["NEO4J_URI"] = "neo4j://localhost:7688"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "12345678"
os.environ["NEO4J_DATABASE"] = "neo4j"

init_attention_system(
    model_path="/root/Model/clinical_attention_v3.pth",
    vdb_path="/root/Graph-R1/expr/DeepSeek_QwenEmbed_Graph/vdb_entities.json",
    embedding_dim=2560 
)

# ================= MoE 门控路由器网络 =================
class MoERouter(nn.Module):
    def __init__(self, input_dim):
        super(MoERouter, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # 输出 0 到 1 之间的门控权重 g
        g = torch.sigmoid(self.fc3(x))
        return g

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 指南权威层级判定 =================
def get_guideline_tier(guideline_str):
    if not guideline_str: return 4
    g_str = str(guideline_str).upper()
    if "ESGO" in g_str: return 1
    elif "FIGO" in g_str: return 2
    elif "NCCN" in g_str: return 3
    else: return 4

TIER_NAMES = {
    1: "ESGO指南 (首选)", 2: "FIGO推荐 (次选)", 
    3: "NCCN指南 (推荐)", 4: "其他指南参考"
}

# ================= 获取图谱文献来源 =================
async def get_source_details(graph_engine, knowledge_text):
    sources_info = []
    try:
        storage = graph_engine.chunk_entity_relation_graph
        if not hasattr(storage, 'driver'): return sources_info
            
        async with storage.driver.session() as session:
            # 兼容宏观聚合片段与单条语义片段
            if knowledge_text.startswith("【权威循证溯源："):
                match = re.search(r'【权威循证溯源：(.*?)】', knowledge_text)
                if match:
                    paper_name = match.group(1).strip()
                    cypher_query = """
                    MATCH (paper:Paper)
                    WHERE paper.name = $paper_name OR paper.pmid = $paper_name
                    RETURN paper.name AS src_id, paper.pmid AS pmid, paper.title AS title, paper.guidelines AS guidelines
                    """
                    result = await session.run(cypher_query, paper_name=paper_name)
                    records = await result.data()
                else: records = []
            else:
                node_id = f"<hyperedge>{knowledge_text}"
                cypher_query = """
                MATCH (target)-[r:BELONG_TO]->(paper:Paper)
                WHERE target.name = $node_id
                RETURN paper.name AS src_id, paper.pmid AS pmid, paper.title AS title, paper.guidelines AS guidelines
                """
                result = await session.run(cypher_query, node_id=node_id)
                records = await result.data()
            
            for record in records:
                src_id = record.get("src_id", "Unknown").replace('"', '')
                raw_pmid = record.get("pmid")
                final_pmid = str(raw_pmid) if (raw_pmid and len(str(raw_pmid)) < 20) else src_id.replace("paper::", "") if "paper::" in src_id else "Unknown"
                raw_gl = record.get("guidelines")
                gl_str = ", ".join(raw_gl) if isinstance(raw_gl, list) else str(raw_gl or "General Evidence")
                
                sources_info.append({
                    "id": src_id, "pmid": final_pmid, 
                    "title": record.get("title") or "No Title", "guidelines": gl_str
                })
    except Exception as e:
        print(f"[Warning] Source lookup failed: {e}")
    return sources_info

# ================= PathoLLM：双语临床特征提取 =================
async def extract_bilingual_features(patient_case, client):
    sys_prompt = (
        "你是一个专门从事妇科肿瘤临床病理分析与文献检索特征提取的医学专家系统。\n"
        "你的任务是从用户输入的「病理报告」及「术后诊断/患者病史」中，精准提取出决定术后辅助治疗方案的【核心】临床特征。\n\n"
        "【🚨 极其重要：检索降噪规则】：\n"
        "为了防止检索库被噪音污染，请**绝对禁止**在检索词中包含 ER、PR、Ki-67、错配修复蛋白(MLH1/MSH2等)等常规免疫组化指标，除非它们是决定分型的核心高危突变（如 p53突变/野生）。\n\n"
        "你只能提取以下核心维度（若未提及当作“无”处理）：\n"
        "1. 疾病类型（如子宫内膜样癌）\n"
        "2. 组织学分级（如G1/G2/G3）\n"
        "3. FIGO分期（如IA期）\n"
        "4. 肌层浸润深度（如无肌层浸润、浅肌层浸润）\n"
        "5. 脉管内癌栓/LVSI状态\n"
        "6. 宫颈/子宫外受累情况及淋巴结状态\n"
        "7. 核心突变（p53等）\n"
        "8. 严重合并症（如肥胖、2型糖尿病等，这可能触发治疗禁忌）\n\n"
        "必须严格按照以下格式输出，使用 <keywords> 标签包裹最终结果：\n"
        "<think>\n逐步分析过程...\n</think>\n"
        "<keywords>\n"
        "[中文检索词]: 子宫内膜样癌, Ⅰ级, 无肌层浸润, 脉管内癌栓阴性, 淋巴结阴性, p53野生型, FIGO IA期, 2型糖尿病\n"
        "[英文检索词]: Endometrial endometrioid adenocarcinoma, Grade 1, No myometrial invasion, Negative LVSI, Negative lymph node, p53 wild-type, FIGO IA stage, Type 2 diabetes mellitus\n"
        "</keywords>"
    )
    try:
        response = await client.chat.completions.create(
            model=LLM_MODEL_NAME, 
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": patient_case}],
            temperature=0.0, max_tokens=2048  
        )
        raw_content = response.choices[0].message.content.strip()
        match = re.search(r'<keywords>(.*?)</keywords>', raw_content, re.DOTALL | re.IGNORECASE)
        bilingual_keywords = match.group(1).strip() if match else re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
            
        print(f"  -> 成功提取双语特征:\n{bilingual_keywords}")
        return f"术后辅助治疗方案与临床指南 (Adjuvant treatment guidelines, recommendations and management)\n{bilingual_keywords}"
    except Exception as e:
        print(f"[Warning] 提取双语检索词失败: {e}")
        return patient_case

# ================= Qwen-Reranker 打分逻辑 =================
async def compute_rerank_score(query, doc, client):
    instruction = "Given a clinical case, retrieve relevant clinical guidelines and evidence that help formulate a treatment plan."
    prompt = f"<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    try:
        response = await client.completions.create(model=RERANK_MODEL_NAME, prompt=prompt, max_tokens=1, temperature=0, logprobs=20)
        top_logprobs_dict = response.choices[0].logprobs.top_logprobs[0]
        true_logit, false_logit = -10.0, -10.0
        for token_str, logprob in top_logprobs_dict.items():
            clean_token = token_str.strip().lower()
            if clean_token == "yes": true_logit = max(true_logit, logprob)
            elif clean_token == "no": false_logit = max(false_logit, logprob)
        true_score, false_score = math.exp(true_logit), math.exp(false_logit)
        return 0.0 if true_score + false_score == 0 else true_score / (true_score + false_score)
    except Exception: return 0.0

print(f">>> [1/5] 正在配置 API 客户端...")
try:
    embed_client = AsyncOpenAI(base_url=EMBEDDING_API_URL, api_key=EMBEDDING_API_KEY)
    rerank_client = AsyncOpenAI(base_url=RERANK_API_URL, api_key=RERANK_API_KEY)
    
    @wrap_embedding_func_with_attrs(embedding_dim=EMBEDDING_DIM, max_token_size=8192)
    async def embedding_func(texts):
        if isinstance(texts, str): texts = [texts]
        response = await embed_client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=texts)
        return np.array([data.embedding for data in response.data])
    print("API 客户端配置成功。")
except Exception as e:
    print(f"API 客户端配置失败: {e}")
    exit(1)

async def vector_stream_reranker(query_str, docs_list):
    """图谱底层的粗排过滤流"""
    if not docs_list: return []
    scores = await asyncio.gather(*[compute_rerank_score(query_str, doc, rerank_client) for doc in docs_list])
    filtered_docs = [doc for doc, score in sorted(zip(docs_list, scores), key=lambda x: x[1], reverse=True) if score > 0.05]
    return filtered_docs

async def main():
    print(">>> [加载模型] 正在挂载 MoE 门控路由器...")
    moe_model = MoERouter(input_dim=EMBEDDING_DIM).to(DEVICE)
    try:
        moe_model.load_state_dict(torch.load(MOE_MODEL_PATH))
        moe_model.eval()
        print("✅ MoE 路由器挂载成功！")
    except Exception as e:
        print(f"❌ MoE 路由器挂载失败，请检查路径: {e}")
        exit(1)

    print(f">>> [2/5] 正在初始化 GraphR1 检索引擎...")
    try:
        graph_engine = GraphR1(
            working_dir=WORKING_DIR, embedding_func=embedding_func,
            kv_storage="JsonKVStorage", vector_storage="NanoVectorDBStorage", 
            graph_storage="Neo4JStorage", reranker_func=vector_stream_reranker
        )
    except Exception as e:
        print(f"GraphR1 初始化失败: {e}")
        exit(1)

    llm_client = AsyncOpenAI(base_url=VLLM_API_URL, api_key=VLLM_API_KEY)

    patient_case = (
        """现病史："""
        """既往月经规则，周期30天，经期7天，量中，无痛经。否认激素补品摄入，近3月月经不规则，经期时长时短，经量时多时少，2023-12月经经期延长，量多，淋漓不尽,伴有血块。2023-12-25外院超声示内膜增厚16mm，建议经净后复查阴超，2024-1-3复查超声示内膜增厚11mm，回声欠均，予黄体酮口服4粒 bid服用3天，2024-1-23月经来潮，月经第五天复查B超提示内膜19mm，内见18*8mm稍低回声区，边界欠清。2024-1-31外院(台州市中心医院)行宫腔镜子宫内膜病损切除+宫颈病损切除术，手术记录未见报告，术后病理：（宫腔、宫颈赘生物）子宫内膜样腺癌 FIGO I-II级。术后阴道少量流血至2-15，我院病理会诊：（宫腔、宫颈赘生物）子宫内膜样癌I级。2024-02-21 肿瘤指标：糖类抗原125：8.26U/ml 糖类抗原72-4：<1.50U/ml 人附睾蛋白4：63.8pmol/L。考虑“子宫内膜内膜样癌I级”建议手术治疗，门诊以“子宫内膜内膜样癌I级”收入院。患病以来，无恶心、呕吐、腹泻，无腹胀、腹痛、腰酸等症状，无尿频，尿急，尿痛，尿不尽感。神志清，精神可，饮食可，睡眠一般，大小便正常。"""
        """既往史："""
        """手术外伤史:否认手术外伤史
系统回顾：无殊
已婚，1-0-1-1，末次妊娠:2006年，2004年顺产，生产方式:自然流产 平时避孕方式:不避孕，不孕症:无，配偶健康状况:健康
"""
        """家族史："""
        """父母均健在,兄弟姐妹均健康，否认家族性遗传性病史，爷爷胃癌"""
        """术前辅助检查："""
        """肿瘤相关： 肿瘤相关2：2024-02-21 (糖类抗原125：8.26U/ml 糖类抗原72-4：<1.50U/ml 罗马指数(绝经前)：12.16% 罗马指数(绝经后)：9.78% 人附睾蛋白4：63.8pmol/L )糖类抗原15-3：6.40U/ml 糖类抗原19-9：4.73U/ml 癌胚抗原 CEA：<1.7ng/ml 铁蛋白：68.9ng/ml甲胎蛋白 AFP：2.43ng/ml 
B超：2024/2/20检查描述:【子宫】【经阴道】子宫位置：后位；子宫大小：长径 51mm，左右径 50mm，前后径 47mm； 子宫形态：规则；子宫回声：欠均匀； 肌层彩色血流星点状，内膜厚度8mm，内膜回声不均，宫内IUD: 无； 宫颈长度:32mm【附件】 右卵巢隐约见：大小19*10*20mm 左卵巢：大小18*16*11mm 【盆腔积液】：无。诊断结论:内膜不均。

下腹部和盆腔MR(平扫+增强)： 2024/2/22检查描述:子宫呈后位，宫体大小形态饱满，子宫肌层弥漫性增厚，信号混杂。宫腔中下段可见等T1稍长T2信号肿块影，下缘达宫颈内口水平，大小约2.2cm×1.0cm，DWI高信号，ADC值降低，内膜肌层交界区不清，前壁浅肌层可见浸润，增强后可见轻度强化。宫颈见多个小圆形囊性灶，较大者直径0.7cm，呈T2WI高信号，增强后未见强化。双侧附件区未见明显异常信号。膀胱充盈尚可，壁未见增厚，未见明显异常信号影。阴道、尿道、直肠内未见明显异常信号影。盆腔内未见明显肿大淋巴结影。MRU双侧尿路未见梗阻。诊断结论:宫腔中下段异常信号灶，考虑子宫内膜癌，累及前壁浅肌层可能。子宫弥漫性腺肌症。宫颈纳氏囊肿。
胸部CT：2024-1-31外院：右肺上叶散在小结节，请随访。
上中腹部平扫+增强(CT) ：上中腹部平扫+增强(CT) 2024/2/22检查描述:肝脏大小正常，形态规则，边缘光整，平扫肝脏密度略低于脾脏，约45Hu，增强后未见异常强化灶，肝内外胆管无扩张。门静脉显示清晰。胆囊壁光滑，平扫及增强未见明显异常。脾脏及双肾形态规则，无明显增大，平扫及增强后未见明显异常。 胰腺未见异常，胰腺周围间隙清楚，胰头周围血管走行自然。腹主动脉旁未见明显肿大淋巴结。诊断结论:轻度脂肪肝可能。
2024-1宫颈细胞学：NILM；HPV：（-）。
"""
        """手术："""
        """手术日期：2024-1-26
手术方式：1、腹腔镜下全子宫切除术(子宫＜10孕周)；2、腹腔镜下双侧输卵管切除术；3、腹腔镜下双侧前哨淋巴结活检术

探查：
子宫位置:[前位]位,大小:[5*5*4]cm,形态:[正常]。
左卵巢大小:[2*1*1]cm,形态:[正常]；左侧输卵管外观:[正常]。
右卵巢大小:[2*1*1]cm,形态:[正常]；右侧输卵管外观:[正常]。
腹水:[无] 。
余盆腹腔探查:[未见明显异常]。
可疑增大淋巴结:[无]

手术达到R0

术中冰冻病理：[NF2024-03816]1.子宫内膜样癌I级,。2.癌灶目前局限于内膜层。3.癌灶未累及宫颈。
"""
        """术后病理："""
        """N2024-07910]病理常规检查: 2024/3/1 
一、全子宫：
1.子宫内膜样癌，Ⅰ级，肿瘤大小2×2×0.5cm，浸润子宫浅表肌层，未见脉管内癌栓，未累及宫颈；周围子宫内膜呈复杂不典型增生。
2.慢性宫颈炎。
二、双侧输卵管未见病变。
三、（右侧前哨淋巴结（超分期））淋巴结1枚，未见癌转移。
    （左侧前哨淋巴结（超分期））淋巴结1枚，未见癌转移。
杨浦免疫结果：MLH1（+），MSH2（+），MSH6（+），PMS2（+），ER（+，90%，强），PR（+，80%，强），P53（野生表型），Ki-67（+，15%），PTEN（-），β-catenin（膜+），L1CAM（-）。
（右侧前哨淋巴结（超分期））：AE1/AE3（-）。
（左侧前哨淋巴结（超分期））：AE1/AE3（-）。

NC2024-00481细胞学: 2024/2/26 （腹腔冲洗液液基细胞学）未找到恶性细胞。

"""
        """术后诊断："""
        """1、子宫内膜恶性肿瘤:子宫内膜样癌G1 IA2期（FIGO2023）/IA期（FIGO2009）/T1aNsn0M0（AJCC2017），2、高血压可能，3、肺结节，4、肥胖，5、2型糖尿病可能"""
    )
    print(f"\n>>> [3/5] 输入原始病例:\n{patient_case}")
    print("\n>>> [3.5/5] 正在调用 PathoLLM 执行双语临床特征提取...")
    enhanced_query = await extract_bilingual_features(patient_case, llm_client)
    print("\n>>> [4/5] 正在执行图谱混合检索与 MoE 动态融合...")
    
    extended_knowledge_pool = {} 

    try:
        param = QueryParam(mode="hybrid", top_k=40, max_token_for_text_unit=4000)
        retrieved_results = await graph_engine.aquery(enhanced_query, param)
        
        if isinstance(retrieved_results, list):
            for item in retrieved_results:
                if isinstance(item, dict):
                    content = item.get('<knowledge>', str(item)).strip()
                    graph_score = float(item.get('<coherence>', 0.0))
                    
                    if content in ["RELATES_TO", "BELONG_TO", "EVIDENCE", ""] or len(content) < 5: 
                        continue
                        
                    if content.startswith("【权威循证溯源："):
                        if "⚠️ 临床绝对警报" in content:
                            tag_info = "🚨 禁忌症熔断警报"
                        else:
                            tag_info = "🧠 图谱高阶逻辑"
                        extended_knowledge_pool[content] = {"type": tag_info, "graph_score": graph_score}
                    else:
                        extended_knowledge_pool[content] = {"type": "🧩 纯向量语义召回", "graph_score": graph_score}

            # ==========================================
            # 🚀 核心阶段：MoE 门控网络自适应打分
            # ==========================================
            fragments_to_sort = []
            
            # 1. 指挥官只看 Anchor：瞬间计算出图谱信任权重 g
            anchor_emb_np = await embedding_func(enhanced_query)
            anchor_tensor = torch.tensor(anchor_emb_np[0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                g_weight = moe_model(anchor_tensor).item()
            print(f"\n🧠 [MoE 门控介入] 当前患者病情复杂度测算完毕，动态图谱信任权重: g = {g_weight:.3f}")
            
            candidate_contents = list(extended_knowledge_pool.keys())
            
            if candidate_contents:
                # 2. 并发计算候选方案的纯正语义分 (调用 Reranker)
                semantic_scores = await asyncio.gather(*[
                    compute_rerank_score(enhanced_query, content, rerank_client) 
                    for content in candidate_contents
                ])
                
                for idx, content in enumerate(candidate_contents):
                    info = extended_knowledge_pool[content]
                    sources = await get_source_details(graph_engine, content)
                    best_tier = min([get_guideline_tier(src.get('guidelines', '')) for src in sources] + [4])
                    
                    semantic_score = semantic_scores[idx]
                    graph_score = info["graph_score"]
                    
                    # 3. 严格执行 MoE 打分公式
                    final_score = (g_weight * graph_score) + ((1.0 - g_weight) * semantic_score)
                    
                    # 4. 绝对安全兜底：如果是禁忌症，强制置顶推给 LLM 去规避
                    if "🚨" in info["type"]:
                        final_score += 1000.0  
                        
                    # 5. 权威指南微调加成
                    tier_bonus = (4 - best_tier) * 0.1 
                    final_score += tier_bonus
                    
                    fragments_to_sort.append({
                        "content": content, "info": info, "sources": sources,
                        "best_tier": best_tier, "semantic_score": semantic_score, 
                        "graph_score": graph_score, "final_score": final_score
                    })
            
            # 按最终融合得分降序排列
            fragments_to_sort.sort(key=lambda x: x["final_score"], reverse=True)
            
            TOP_N_FOR_LLM = 10
            selected_fragments = fragments_to_sort[:TOP_N_FOR_LLM]

            print("\n" + "="*20 + f" MoE 动态融合重排证据 (Top {len(selected_fragments)}) " + "="*20)
            bibliography, ref_counter, llm_context_list = {}, 1, []
            
            for idx, frag in enumerate(selected_fragments):
                source_indices = []
                for src in frag["sources"]:
                    src_id = src['id']
                    if src_id not in bibliography:
                        src['ref_index'] = ref_counter
                        bibliography[src_id] = src
                        ref_counter += 1
                    source_indices.append(f"[{bibliography[src_id]['ref_index']}]")
                
                tier_name = TIER_NAMES[frag['best_tier']]
                ref_tag = f"【来源文献: {', '.join(source_indices)} | 证据级别: {tier_name}】" if source_indices else "【来源文献: 未知 | 证据级别: 缺乏指南支撑】"
                
                # 清晰展示 MoE 评分细节
                print(f"[{idx+1}] [{frag['info']['type']}] 复合总分: {frag['final_score']:.3f} | 图谱分: {frag['graph_score']:.3f} | 语义分: {frag['semantic_score']:.3f}")
                print(f"内容: {frag['content'][:100]}...") 
                print("-" * 30)
                llm_context_list.append(f"{ref_tag} {frag['content']}")
        
        context_str = "\n\n".join(llm_context_list)
        print("="*50)

    except Exception as e:
        print(f"检索出错: {e}")
        return

    print("\n>>> [5/5] 正在生成治疗方案...")
    system_prompt = (
        "你是一个极其严谨且保守的妇科肿瘤临床辅助决策系统。\n"
        "请严格且**仅根据**【参考证据】制定详细、可落地的术后辅助治疗方案。\n\n"
        "【🚨 医疗安全红线要求（绝对遵守）】：\n"
        "1. 严禁幻觉：在缺乏检索证据【针对该患者当前状态明确推荐】的情况下，严禁基于大模型自身记忆编造任何药物或方案。\n"
        "2. 宁缺毋滥：如果检索到的证据中【没有明确推荐】某项治疗，或者明确指出该患者属于低危/早期而【不推荐/禁忌】某项治疗，**请直接省略该治疗方式的输出（连标题都不要创造）**。\n"
        "3. 遇到【🚨 禁忌症熔断警报】的证据，该证据具有最高临床否决权，不仅要直接省略相关禁忌治疗，还需在报告首要位置简要高亮提示其原因。\n"
        "4. 【逻辑校验红线（防误判）】：必须核对患者的手术状态！如果病历显示患者**已行“全子宫切除术”**，则该患者的治疗目的为“术后辅助治疗”，**绝对禁止**向其推荐任何基于“保留生育功能”的内分泌治疗方案（如大剂量MPA/MA等）。请严格区分文献的适用人群！\n\n"
        "【📝 输出格式约束（必须严格遵循以下Markdown模板结构）】：\n"
        "1. 请先使用 <think>...</think> 进行内部循证推理分析（核对子宫是否切除、排除禁忌症、匹配推荐症）。\n"
        "2. 推理结束后，输出 <answer>...</answer> 结论。\n"
        "3. 在 <answer> 标签内部，请按照以下大纲输出（**注：1.x 的子标题请按需动态生成，没有相关证据的治疗方式直接不要写**）：\n\n"
        "根据患者提供的病历和参考证据，以下是详细的术后辅助治疗方案：\n\n"
        "### 1. **术后辅助治疗方案**\n"
        "（根据实际推荐情况输出对应的子标题并自动编号，例如 1.1, 1.2 等。如果全都不推荐，本节可直接写明“不推荐辅助放化疗及内分泌治疗，以随访为主”）\n"
        "#### 1.x **放射治疗**（仅在有证据推荐时输出）\n"
        "#### 1.x **化疗**（仅在有证据推荐时输出）\n"
        "#### 1.x **内分泌治疗**（仅在有证据推荐时输出）\n\n"
        "### 2. **随访计划**\n"
        "根据 [x] 的指南建议...（必须包含随访频率、妇科检查、影像学检查及肿瘤标志物等内容）\n\n"
        "### 3. **其他建议**\n"
        "（如参考证据中包含生活干预、减重、靶向治疗/免疫治疗、遗传咨询等其他建议，请在此列出；若无，本段可直接省略）\n\n"
        "### 4. **注意事项**\n"
        "（列出并发症监测及相互作用等）\n\n"
        "### 5. **总结**\n"
        "（对上述制定的核心方案进行精简的高度概括）\n\n"
        "【关键准则】：在建议中引用证据时，请直接使用原文序号标签，如 '根据 [1] 的指南建议...'。MoE引擎已对证据排序，排名越靠前的方案优先级越高。"
    )
    user_prompt = f"【患者信息】\n{patient_case}\n\n【参考证据】\n{context_str}\n\n请制定极其详细的术后辅助治疗方案。"

    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            stream=True, temperature=0.0, max_tokens=2048
        )
        print("\n>>> 模型回复:\n")
        async for chunk in response:
            if content := chunk.choices[0].delta.content: print(content, end="", flush=True)
        
        print("\n\n" + "="*20 + " 参考文献 (References) " + "="*20)
        if bibliography:
            for details in sorted(bibliography.values(), key=lambda x: x['ref_index']):
                idx, pmid_val, paper_id = details['ref_index'], details.get('pmid', 'Unknown'), details['id']
                print(f"[{idx}] PMID: {pmid_val}" if pmid_val != 'Unknown' else f"[{idx}] DocID: {paper_id[:8]}... (无标准PMID文献)")
                print(f"    Title: {details['title']}\n    Guidelines: {details['guidelines']}\n" + "-" * 10)
        else: print("（本次检索未关联到具体文献节点）")
        print("="*50)

    except Exception as e: print(f"LLM 调用失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
