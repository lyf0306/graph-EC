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

# Milvus 数据库配置
MILVUS_URI = "http://localhost:19530"
os.environ["MILVUS_URI"] = MILVUS_URI

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

# ================= PathoLLM：双语全息患者画像提取 =================
async def extract_patient_profile(patient_case, client):
    sys_prompt = (
        "你是一个极其严谨的妇科肿瘤病历特征提取专家系统。\n"
        "你的任务是从用户输入的 TB 报告中提取结构化信息。\n\n"
        "【🚨 核心提取法则】：\n"
        "1. 绝不能抄写模板，必须根据原始病历填入真实数据。\n"
        "2. 【分期绝对采信】：病历末尾的【术后诊断】给出了明确的 FIGO 分期，你必须直接提取，绝对禁止重新推演！\n\n"
        "请严格按照以下格式输出：\n\n"
        "### 【全息患者画像】\n"
        "- **基本体征**：[填入年龄、绝经状态、体能评分等]\n"
        "- **所有合并症与既往史**：[详尽填入所有疾病，如高血压、糖尿病、冠心病/支架、肺部炎症、胃炎等]\n"
        "- **既定分期**：[直接复制术后诊断分期，含FIGO与TNM]\n"
        "- **病理与转移特征**：\n"
        "  - 组织学类型：[填入如浆液性癌、内膜样癌等]\n"
        "  - 组织学分级：[填入 G1/G2/G3/低分化/高级别等]\n"
        "  - 浸润与周围受累：[填入肌层浸润深度、是否累及宫颈间质/输卵管/卵巢等]\n"
        "  - 脉管癌栓 (LVSI)：[填入 阳性/阴性/局灶/广泛等]\n"
        "  - 淋巴结状态：[填入转移情况及比例，如 0/2未转移]\n"
        "  - 分子分型关键指标：[重点提取 MMR(完整/缺失)、p53(野生/突变)、ER/PR、Ki-67 等]\n\n"
        "【极其重要】：在画像之后，你必须使用 <keywords> 标签提取出中英双语检索词。\n"
        "🚨【检索词防偏航红线】：检索词【只允许】包含核心肿瘤特征（分期、病理类型、LVSI、淋巴结、分子分型）。【绝对禁止】在检索词中加入高血压、糖尿病、冠心病等非肿瘤合并症，这会导致底层向量检索严重偏航，错失指南！\n"
        "格式必须为：\n"
        "<keywords>\n"
        "[中文检索词]: 浆液性癌, 低分化, 浸润深肌层, 累及宫颈间质, p53突变型, MMR正常, FIGO IIIA1期\n"
        "[英文检索词]: Serous carcinoma, Poorly differentiated, Deep myometrial invasion, Cervical stromal involvement, p53 mutated/abn, MMR proficient/pMMR, FIGO IIIA1 stage\n"
        "</keywords>"
    )
    
    user_prompt = f"【原始病历源数据】\n{patient_case}\n\n请提取患者画像并生成中英双语 <keywords>。"

    try:
        response = await client.chat.completions.create(
            model=LLM_MODEL_NAME, 
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0, max_tokens=2048  
        )
        raw_content = response.choices[0].message.content.strip()
        content_no_think = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
        
        profile_match = re.search(r'(### 【全息患者画像】.*?)(?=<keywords>|$)', content_no_think, re.DOTALL)
        profile_text = profile_match.group(1).strip() if profile_match else content_no_think
        
        keywords_match = re.search(r'<keywords>([\s\S]*?)</keywords>', content_no_think, re.DOTALL | re.IGNORECASE)
        if keywords_match:
            bilingual_keywords = keywords_match.group(1).strip()
        else:
            bilingual_keywords = "\n".join(content_no_think.split('\n')[-2:])
            
        print(f"  -> 成功提取全息患者画像与检索词:\n{profile_text}\n\n[检索词]:\n{bilingual_keywords}")
        
        # 🚨 增加 NCCN 和 ESGO 的“强力路标锚点”，确保底层图谱和向量库强制召回核心指南
        final_query = f"NCCN ESGO 指南、术后辅助治疗方案与预后生存率 (NCCN ESGO Adjuvant treatment guidelines, prognosis survival rate, recommendations and management)\n{bilingual_keywords}"
        return profile_text, final_query
    except Exception as e:
        print(f"[Warning] 提取特征失败: {e}")
        return patient_case, patient_case

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
            kv_storage="JsonKVStorage", 
            vector_storage="MilvusVectorDBStorge", 
            graph_storage="Neo4JStorage", reranker_func=vector_stream_reranker
        )
    except Exception as e:
        print(f"GraphR1 初始化失败: {e}")
        exit(1)

    llm_client = AsyncOpenAI(base_url=VLLM_API_URL, api_key=VLLM_API_KEY)

    # 此处为测试患者病历数据
    patient_case = (
        """## 现病史：
患者绝经10年， 近2年每月点滴样出血，小便后擦拭可见。每年未曾体检。2020-04-16因“绝经后阴道出血”当地医院就诊，B超检查， （未见报告）。遵医嘱行宫腔镜。4-29当地医院行宫腔镜。术后病理提示：（宫腔）宫内膜显示非典型增生伴输卵管上皮化生及靴钉样改变，灶区可疑癌变。建议手术。患者为进一步治疗，我院门诊就诊，病理会诊：NH2020-01874（宫腔）子宫内膜样癌，I级，周围内膜复杂不典型增生。5-8我院妇科常规彩色超声（经阴道）检查描述:【经阴道】 子宫位置：前位；子宫大小：长径 58mm，左右径 58mm，前后径 53mm； 子宫形态：不规则；子宫回声：不均匀； 肌层彩色血流星点状， 宫腔内中低回声区24*26*19mm   宫内IUD: 无。  宫颈长度:27mm子宫前壁突起中低回声区：27*24*22mm，右后壁向外突中低回声区：42*37*33mm，左侧壁下段肌层中高回声区：33*28*28mm，余肌层数枚低回声结节，最大直径18mm右卵巢：未暴露； 左卵巢：未暴露； 【盆腔积液】：无。诊断结论:宫腔内实质占位，符合病史。子宫多发肌瘤可能。门诊建议手术治疗，拟"子宫内膜癌"收住入院。

## 既往史：
糖尿病20年，二甲双胍0.5g qn，po，格列齐特 （达美康）1#，早餐前1#，午饭后1#。平素空腹血糖＜8mmol/l。高血压20年，口服厄贝沙坦氢氯噻嗪（依伦平）1#，qd，晨服，自测血压130-150/75-80mmHg。2012年“脑梗”口服药物治疗，现左食指、拇指麻木，活动受限。
生育史：1-0-0-1

## 家族史：
否认家族性肿瘤、遗传性病史

## 术前辅助检查：
1.       B超：妇科常规彩色超声（经阴道）检查描述:【经阴道】 子宫位置：前位；子宫大小：长径 58mm，左右径 58mm，前后径 53mm； 子宫形态：不规则；子宫回声：不均匀； 肌层彩色血流星点状， 宫腔内中低回声区24*26*19mm   宫内IUD: 无。  宫颈长度:27mm子宫前壁突起中低回声区：27*24*22mm，右后壁向外突中低回声区：42*37*33mm，左侧壁下段肌层中高回声区：33*28*28mm，余肌层数枚低回声结节，最大直径18mm右卵巢：未暴露； 左卵巢：未暴露； 【盆腔积液】：无。诊断结论:宫腔内实质占位，符合病史。子宫多发肌瘤可能。
2.      上腹部CT：1.肝脏右后叶近膈顶斑片状高密度影，介入术后改变？请结合临床病史。2.双肾小结石可能；双肾囊肿可能。
3.      盆腔MRI：子宫形态尚可，呈前倾前屈位，宫体大小约6.7cm×5.8cm×4.3cm。子宫肌壁间及浆膜下可见多发结节影，最大位于子宫后壁浆膜下，大小约4.1cm*4.2cm*3.5cm，边界清晰，病灶向宫体外突出，病灶呈T1WI等低信号，T2WI低等信号，增强后轻度均匀强化，强化程度同肌层相仿。宫腔内可见一异常信号肿物影，大小约4.4cm×3.4cm×3.2cm，呈T1W等信号T2W稍高信号，DWI呈高信号，内膜肌层交界区不清，局部可达深肌层，最深处距离浆膜面约1mm。增强后肿物可见明显强化。双侧附件区未见异常信号影及异常强化灶。膀胱充盈尚可，膀胱壁完整，未见增厚，阴道、直肠未见明显异常信号。前、后陷凹内未见明显异常信号灶。增强后亦未见异常强化灶。所扫范围盆腔内及双侧腹股沟区未见明显肿大淋巴结影。影像结论：宫腔内肿物，考虑为子宫内膜癌，累及深肌层，局部可达浆膜面； 子宫多发肌瘤。
4.      NH2020-01874（宫腔）子宫内膜样癌，I级，周围内膜复杂不典型增生。
5.      CA125抗原：125.80U/ml  人附睾蛋白4：270.6pmol/L肿瘤相关： CA199抗原：568.30U/ml

## 手术：
1.腹腔镜下全子宫切除术(子宫＜10孕周)；2、腹腔镜下双侧输卵管卵巢切除
3、腹腔镜下双侧前哨淋巴结清扫术（临床试验）

子宫前位，大小4*3*3cm，形态不规则，子宫前壁见直径3cm质硬突起，右后壁向外突直径4cm质硬结节，左侧壁下段外突直径3cm质硬结节。左输卵管外观未见异常，左卵巢大小2.5*2*1.5cm，外观未见异常。右输卵管外观未见异常，右卵巢大小2*1.5*1cm，外观未见异常。其他：肠管、肝、脾、横隔下及盆壁未见明显异常。

## 术后病理：
一、全子宫
1.子宫内膜样癌，Ⅱ级，病灶大小4×3.5×2.5cm，浸润子宫深肌层；脉管内见癌栓；癌灶未累及宫颈。周围子宫内膜呈单纯萎缩性改变。
2.子宫肌壁间多发平滑肌瘤。
3.子宫局限型腺肌病。
4.慢性宫颈炎。
二、左侧卵巢包涵囊肿伴周围炎。
    右侧卵巢包涵囊肿。
三、左侧输卵管慢性炎。
    右侧输卵管周围炎。
四、（右侧前哨（超分期））淋巴结1/4枚见癌转移。
    （左侧前哨（超分期））淋巴结1/3枚见癌转移。
免疫组化：MLH1（+），MSH2（+），MSH6（+），PMS2（+），ER（+，80%，强），PR（+，30%，强），P53（野生表型），Ki-67（+，30%），PTEN（+），AE1/AE3/CD31（脉管内见癌栓），AE1/AE3/D240（脉管内见癌栓）。
（右侧前哨（超分期））AE1/AE3（见单个肿瘤细胞转移）。
（左侧前哨（超分期））AE1/AE3（见单个肿瘤细胞转移）。
（腹腔冲洗液液基细胞学）未找到恶性细胞。

## 术后诊断：
子宫内膜样癌G2 III C1期（FIGO 2009）/T1bN1（sn）M0期"""
    )
    print(f"\n>>> [3/5] 输入原始病例:\n")
    print("\n>>> [3.5/5] 正在调用 PathoLLM 提取全息患者画像与双语特征...")
    
    # 提取画像和检索词
    patient_profile_md, enhanced_query = await extract_patient_profile(patient_case, llm_client)
    
    print("\n>>> [4/5] 正在执行图谱混合检索与 MoE 动态融合...")
    extended_knowledge_pool = {} 
    llm_context_list = []

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
            anchor_emb_np = await embedding_func(enhanced_query)
            anchor_tensor = torch.tensor(anchor_emb_np[0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                g_weight = moe_model(anchor_tensor).item()
            print(f"\n🧠 [MoE 门控介入] 当前患者病情复杂度测算完毕，动态图谱信任权重: g = {g_weight:.3f}")
            
            candidate_contents = list(extended_knowledge_pool.keys())
            
            if candidate_contents:
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
                    
                    final_score = (g_weight * graph_score) + ((1.0 - g_weight) * semantic_score)
                    if "🚨" in info["type"]:
                        final_score += 1000.0  
                        
                    tier_bonus = (4 - best_tier) * 0.1 
                    final_score += tier_bonus
                    
                    fragments_to_sort.append({
                        "content": content, "info": info, "sources": sources,
                        "best_tier": best_tier, "semantic_score": semantic_score, 
                        "graph_score": graph_score, "final_score": final_score
                    })
            
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
                
                print(f"[{idx+1}] [{frag['info']['type']}] 复合总分: {frag['final_score']:.3f} | 图谱分: {frag['graph_score']:.3f} | 语义分: {frag['semantic_score']:.3f}")
                print(f"内容: {frag['content'][:100]}...") 
                print("-" * 30)
                llm_context_list.append(f"{ref_tag} {frag['content']}")
        
        context_str = "\n\n".join(llm_context_list)
        
        # ==========================================
        # 🛡️ 安全防线：放宽拦截条件，兼容国内外主流指南
        # ==========================================
        guideline_keywords = ["ESGO", "NCCN", "FIGO", "中华医学会", "CSCO", "中国肿瘤", "ESMO"]
        if not any(kw in context_str.upper() for kw in guideline_keywords):
            print("\n⚠️ [安全警报]：未命中任何国内外主流指南，触发系统防幻觉兜底策略！")
            context_str = "【系统安全诊断】：本次 RAG 未能召回 ESGO/NCCN/FIGO/中华医学会/CSCO 等权威指南文本。请在最终报告开头明确提示证据缺失，仅基于现有信息分析，并强烈建议主治医师查阅原版指南。\n\n" + context_str
            
        print("="*50)

    except Exception as e:
        print(f"检索出错: {e}")
        return

    
    system_prompt = (
        "你是一个极其严谨的顶级肿瘤医院 MDT（多学科会诊）专家。你的任务是根据传入的【全息患者画像】和【参考证据】，撰写一份极其详尽、专业、总字数在 2000 字左右的临床 Tumor Board (TB) 会诊报告。\n\n"
        "【🚨 核心执业红线与引用规范（绝对遵守）】：\n"
        "1. **极其详尽**：这是真实的临床会诊记录，严禁一笔带过或压缩信息！每一项指南的解析、每一个临床试验的数据对比都必须单独成段，详细展开论述。\n"
        "2. **🚨绝对杜绝张冠李戴（最严厉警告）🚨**：你必须严格核对传入证据中每段开头的 `【来源文献: [x] | 证据级别: XXX】` 标签！\n"
        "   - 如果你在报告中写“根据 ESGO 指南...”，你打的标号 [x] 对应的证据必须真的是 ESGO！\n"
        "   - 如果你写“根据 NCCN 指南...”，你打的标号 [x] 必须真的是 NCCN！\n"
        "   - 绝对禁止把《中华医学会》或《中国肿瘤指南》硬冠上 NCCN/ESGO 的名号并瞎打标号！如果你在证据库里只看到了国内指南，请如实写“根据检索到的国内权威指南[x]推荐...”，绝不能强行伪造国际指南的出处！\n"
        "3. **内置逻辑免引用**：本系统已内置【2025 ESGO 风险推演逻辑】，这属于系统固有知识。在进行 ESGO 风险判定时，请直接写“基于系统内置的 2025 ESGO 风险评估原则”，【绝对不要】为了凑格式而给它瞎编一个 [x] 标号！\n"
        "4. **数据强制引用**：对于证据库中真实存在的临床试验数据（PFS/OS/HR等）、生存率、具体化疗方案，必须在句尾极其精准地打上对应的 [x]。\n\n"
        "【📝 强制文书结构（请严格按以下四大模块输出，每个模块极尽详细）】：\n\n"
        "病情分析：\n"
        "1. **病历摘要**：用一段话（约200-300字）高度凝练患者核心病史（年龄、手术方式、病理类型、浸润深度、脉管癌栓、淋巴结转移、核心分子分型及最终 FIGO 分期）。\n"
        "2. **ESGO 2025 风险分层与指南推荐**：\n"
        "   - **风险判定**：根据系统内置逻辑树（POLE型IA-IIC低危,III-IVA不确定；MMRd型IA/IC低危,IB或IIC无LVSI/宫浸中危,IIA/IIB或IIC伴LVSI/宫浸中高危,III/IVA高危；NSMP低级ER+型IA低危,IB/IIA中危,IIB中高危,III/IVA高危；NSMP高级/ER-与p53abn型IA1/IC不确定,IA2-IVA高危）。写出推断依据，并加粗输出该患者的最终风险等级。\n"
        "   - **ESGO 推荐主路径**：基于该风险等级，用代码块高亮显示 ESGO 指南的规范化推荐格式（必须严格形如：`Systemic Therapy ± EBRT ± vaginal brachytherapy` 或 `放化疗联合 / 单纯化疗` 等）。\n"
        "   - **详细解析**：结合参考证据，详细展开论述 ESGO 对该风险人群的具体干预指导意见（如引用了外部证据，准确打标号 [x]）。\n"
        "3. **NCCN 及其他权威指南推荐**：\n"
        "   - 仔细查阅【参考证据】，如实反映检索到的指南（NCCN或国内指南等）。\n"
        "   - **指南主路径**：用代码块高亮显示该指南的推荐公式（例如：`系统治疗 (Systemic therapy) ± 盆腔外照射 (EBRT)`）。\n"
        "   - **详细解析**：展开论述该指南针对此分期及高危因素的具体化放疗细节（精准打标号 [x]）。\n"
        "4. **核心临床试验深度解析**：【极其重要，必须长篇详写！】提取证据中与【本患者分子分型/分期】高度匹配的核心临床研究（如 PORTEC、ENGOT、GOG 等）。必须详细列出：研究背景与纳入人群、对照组与实验组的具体干预方案对比、**具体的 PFS/OS/HR 等统计学数值**，并深入剖析该试验结果应用到本患者身上的预期获益（精准打标号 [x]）。\n\n"
        "术后处理：\n"
        "1. **肿瘤专科主方案**：综合上述指南，下达果断的最终临床医嘱指令。明确写出具体的化疗药物组合（如紫杉醇+卡铂）、给药周期数、是否联合免疫/靶向药物，以及放疗的具体介入时机。\n"
        "2. **合并症与多学科管理（逐条详细列出）**：全面扫描患者画像中的【全部合并症】（高血压/糖尿病/脑梗/冠心病/胃病等）。详细写明抗肿瘤药物对该合并症的具体毒性警示（如紫杉醇加重糖尿病末梢神经病变、抗血管生成药物增加脑梗血栓复发风险等），并下达对应专科（内分泌科/心内科/神经内科等）的详细随诊指令。\n\n"
        "预后分析：\n"
        "提取检索证据中针对此 FIGO 分期的客观数据库生存率数据（如 NCDB 的 3年/5年 OS 率）。结合本例患者的高危因素（深肌层浸润、淋巴结阳性、LVSI等）展开一段不少于150字的个体化预后讨论（精准打标号 [x]）。若证据中无确切数据，请明确注明“当前证据库未包含针对该分期确切的大样本生存数据”。\n\n"
        "随访方案：\n"
        "1. **随访时间表**：明确前2年、3-5年、5年后的具体随访频率。\n"
        "2. **警示症状**：列举需立即就诊的异常体征（如盆腹腔疼痛、异常出血、血栓症状等）。\n"
        "3. **检查项目明细**：详述妇检、肿瘤标志物、盆腹腔增强MRI/CT的实施频次。\n"
        "4. **康复与心理支持**：涉及饮食指导、并发症管理、性健康及对复发恐惧的心理干预。"
    )
    
    user_prompt = f"以下是患者的【全息特征画像】：\n{patient_profile_md}\n\n以下是图谱召回的【参考证据】：\n{context_str}\n\n请你深呼吸，仔细核对每一条参考证据的【来源文献】标签。严格按照上述要求，撰写一份内容极其丰富、数据详实、具备顶级主治医师水准的 Tumor Board 会诊报告。🚨最后警告：宁可不标文献序号，也绝对不准把国内指南硬冠上国际指南的名字瞎标序号！"

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
