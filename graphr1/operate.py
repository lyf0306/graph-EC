# operate.py
import asyncio
import math
import json
import re
import torch
from tqdm.asyncio import tqdm as tqdm_async
from typing import Union
from collections import Counter, defaultdict
import warnings
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from .hyper_attention import GLOBAL_ENTITY_CACHE, compute_hyperedge_scores_sync


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    now_hyper_relation: str,
):
    # [关键修改] 检查长度：至少要有6个字段 (Head, Name, Type, Desc, Role, Score)
    if len(record_attributes) < 6 or record_attributes[0] != '"entity"' or now_hyper_relation == "":
        return None
        
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    
    # [关键修改] 提取 Role 字段 (索引 4)
    raw_role = clean_str(record_attributes[4].upper()).replace('"', '').replace("'", "").strip()
    #print(f"DEBUG ROLE: Raw='{raw_role}' | Original List={record_attributes}")
    
    # [关键修改] 角色白名单映射 (容错处理)
    ROLE_MAPPING = {
        # 标准输出
        "CONDITION": "CONDITION",
        "INPUT": "CONDITION",
        "PREREQUISITE": "CONDITION",
        
        "RECOMMENDATION": "RECOMMENDATION",
        "OUTCOME": "RECOMMENDATION",
        "ACTION": "RECOMMENDATION",
        "TREATMENT": "RECOMMENDATION",
        
        "CONTRAINDICATION": "CONTRAINDICATION",
        "EXCLUSION": "CONTRAINDICATION",
        
        "CONTEXT": "CONTEXT",
        "BACKGROUND": "CONTEXT",
        
        "EVIDENCE": "EVIDENCE",
        "SOURCE": "EVIDENCE"
    }
    
    # 如果不在白名单，默认回退到 CONDITION
    edge_role = ROLE_MAPPING.get(raw_role, "CONDITION")
    
    # 提取分数 (最后一个字段)
    weight_str = record_attributes[-1]
    weight = (
        float(weight_str) if is_float_regex(weight_str) else 0.0
    )
    
    # 过滤低分实体
    if weight < 50:
        return None

    hyper_relation = now_hyper_relation
    entity_source_id = chunk_key
    
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        edge_role=edge_role,  # [关键修改] 保存提取到的角色
        weight=weight,
        hyper_relation=hyper_relation,
        source_id=entity_source_id,
    )


async def _handle_single_hyperrelation_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 3 or record_attributes[0] != '"hyper-relation"':
        return None
    
    # 针对片段分数的过滤 (0-10分制)
    weight_str = record_attributes[-1]
    weight = (
        float(weight_str) if is_float_regex(weight_str) else 0.0
    )
    
    if weight < 7:
        return None

    # add this record as edge
    knowledge_fragment = clean_str(record_attributes[1])
    edge_source_id = chunk_key
    
    return dict(
        hyper_relation="<hyperedge>"+knowledge_fragment,
        weight=weight,
        source_id=edge_source_id,
    )
    

async def _merge_hyperedges_then_upsert(
    hyperedge_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    paper_name: str = None, # [新增参数] 传入当前正在处理的文献命名
):
    already_weights = []
    already_source_ids = []

    already_hyperedge = await knowledge_graph_inst.get_node(hyperedge_name)
    if already_hyperedge is not None:
        already_weights.append(already_hyperedge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_hyperedge["source_id"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in nodes_data] + already_weights)
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    node_data = dict(
        role = "hyperedge",
        weight=weight,
        source_id=source_id,
    )

    # A. 插入/更新超边节点
    await knowledge_graph_inst.upsert_node(
        hyperedge_name,
        node_data=node_data,
    )

    # 在超边 upsert 成功后，立即建立与 Paper 的关系。
    # B. 建立文献与超边的关系
    if paper_name:
        await knowledge_graph_inst.upsert_edge(
            paper_name,  # 源节点：文献PMID
            hyperedge_name,  # 目标节点：超边
            edge_data=dict(
                role = "EVIDENCE", # 关系角色为 EVIDENCE
                description="BELONG_TO",  # description 为 BELONG_TO
                weight=1.0,
                source_id=source_id  # 记录来源 chunk ID 以供证据追查
            ),
        )

    node_data["hyperedge_name"] = hyperedge_name
    return node_data


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        role="entity",
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    edge_data = []
    
    for node in nodes_data:
        source_id = node["source_id"]
        hyper_relation = node["hyper_relation"]
        weight = node["weight"]
        
        # [关键修改] 获取角色，默认为 CONDITION
        role = node.get("edge_role", "CONTEXT")
        
        already_weights = []
        already_source_ids = []
        
        if await knowledge_graph_inst.has_edge(hyper_relation, entity_name):
            already_edge = await knowledge_graph_inst.get_edge(hyper_relation, entity_name)
            already_weights.append(already_edge["weight"])
            already_source_ids.extend(
                split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
            )
        
        weight = sum([weight] + already_weights)
        source_id = GRAPH_FIELD_SEP.join(
            set([source_id] + already_source_ids)
        )

        # [关键修改] 将 role 写入边属性
        await knowledge_graph_inst.upsert_edge(
            hyper_relation,
            entity_name,
            edge_data=dict(
                weight=weight,
                source_id=source_id,
                role=role,  # <--- 存入 role
                description="RELATES_TO",
            ),
        )

        edge_data.append(dict(
            src_id=hyper_relation,
            tgt_id=entity_name,
            weight=weight,
            role=role # 保留以备后用
        ))

    return edge_data

# 修改点：新增 paper_name 参数，用于关联文献节点
# graphr1/operate.py 中的 extract_entities 函数

async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    hyperedge_vdb: BaseVectorStorage,
    global_config: dict,
    paper_name: str = None, # [新增参数]
) -> Union[BaseGraphStorage, None]:

    # 准备配置参数
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    # 按顺序处理 chunks
    ordered_chunks = list(chunks.items())
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    # 准备实体提取的 prompt 和示例
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    # 处理单个内容块
    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}"
        ).format(**context_base, input_text=content)

        final_result = await use_llm_func(hint_prompt)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        now_hyper_relation=""
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            
            # 安全的循环处理逻辑
            if len(record_attributes) > 0 and record_attributes[0] == '"hyper-relation"':
                if_relation = await _handle_single_hyperrelation_extraction(
                    record_attributes, chunk_key
                )
                if if_relation is not None:
                    maybe_edges[if_relation["hyper_relation"]].append(if_relation)
                    now_hyper_relation = if_relation["hyper_relation"]
                else:
                    now_hyper_relation = ""
                continue
            
            # 处理实体
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key, now_hyper_relation
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue
            
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # 并发处理所有内容块
    results = []
    for result in tqdm_async(
        asyncio.as_completed([_process_single_content(c) for c in ordered_chunks]),
        total=len(ordered_chunks),
        desc="Extracting entities from chunks",
        unit="chunk",
    ):
        results.append(await result)

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[k].extend(v)


    # ==========================
    # 关键修改：加入信号量并发控制
    # ==========================
    
    # 限制并发数为 30 (可根据数据库实际承受能力调整，20-50通常较安全)
    write_concurrency_limit = asyncio.Semaphore(30)

    # 定义带信号量的任务包装器
    async def _sem_task(coro):
        async with write_concurrency_limit:
            return await coro

    # 1. 插入超边 (Hyperedges)
    logger.info("Inserting hyperedges into storage...")
    all_hyperedges_data = []
    
    # 使用 _sem_task 包裹任务
    tasks = [
        _sem_task(_merge_hyperedges_then_upsert(
            k, v, knowledge_graph_inst, global_config, paper_name=paper_name
        ))
        for k, v in maybe_edges.items()
    ]
    
    for result in tqdm_async(
        asyncio.as_completed(tasks),
        total=len(maybe_edges),
        desc="Inserting hyperedges",
        unit="entity",
    ):
        all_hyperedges_data.append(await result)
            
    # 2. 插入实体 (Entities)
    logger.info("Inserting entities into storage...")
    all_entities_data = []

    # 使用 _sem_task 包裹任务
    tasks = [
        _sem_task(_merge_nodes_then_upsert(
            k, v, knowledge_graph_inst, global_config
        ))
        for k, v in maybe_nodes.items()
    ]
    
    for result in tqdm_async(
        asyncio.as_completed(tasks),
        total=len(maybe_nodes),
        desc="Inserting entities",
        unit="entity",
    ):
        all_entities_data.append(await result)

    # 3. 插入关系 (Relationships)
    logger.info("Inserting relationships into storage...")
    all_relationships_data = []

    # 使用 _sem_task 包裹任务
    tasks = [
        _sem_task(_merge_edges_then_upsert(
            k, v, knowledge_graph_inst, global_config
        ))
        for k, v in maybe_nodes.items()
    ]

    for result in tqdm_async(
        asyncio.as_completed(tasks),
        total=len(maybe_nodes),
        desc="Inserting relationships",
        unit="relationship",
    ):
        all_relationships_data.append(await result)

    # ==========================
    # 结束并发控制修改区域
    # ==========================

    if not len(all_hyperedges_data) and not len(all_entities_data) and not len(all_relationships_data):
        logger.warning(
            "Didn't extract any hyperedges and entities, maybe your LLM is not working"
        )
        return None

    if not len(all_hyperedges_data):
        logger.warning("Didn't extract any hyperedges")
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships")

    if hyperedge_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["hyperedge_name"], prefix="rel-"): {
                "content": dp["hyperedge_name"],
                "hyperedge_name": dp["hyperedge_name"],
            }
            for dp in all_hyperedges_data
        }
        await hyperedge_vdb.upsert(data_for_vdb)

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst

async def kg_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: list,
    hyperedges_vdb: list,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> str:
    hl_keywords, ll_keywords = query, query
    keywords = [ll_keywords, hl_keywords]
    context = await _build_query_context(
        keywords,
        knowledge_graph_inst,
        entities_vdb,
        hyperedges_vdb,
        text_chunks_db,
        query_param,
        global_config  # <--- [修改点] 把配置往下传
    )
    return context



async def _build_query_context(
    query: list,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict, 
):
    ll_kewwords, hl_keywrds = query[0], query[1]
    # 【流 A】图谱多跳流：返回 {内容: 原始图谱推演饱和度}
    graph_dict = await _get_node_data(
        ll_kewwords, knowledge_graph_inst, entities_vdb, text_chunks_db, query_param, global_config
    )
    # 【流 B】向量单跳流：返回 [内容列表]
    vector_list = await _get_edge_data(
        hl_keywrds, knowledge_graph_inst, hyperedges_vdb, text_chunks_db, query_param, global_config
    )
    
    # ========================================================
    # 🌟 采用平滑映射与自然量纲机制，彻底剔除魔法数字和硬截断
    # ========================================================
    
    # 从全局配置中获取图论/数学超参数，如果未配置则使用学术界经典默认值
    graph_prior_bias = global_config.get("graph_prior_bias", 1.0) # 图谱先验偏置，保证图谱路优先
    pagerank_damping = global_config.get("pagerank_damping", 0.85) # PageRank经典阻尼系数
    spatial_decay = global_config.get("spatial_decay", 0.95)       # 随机游走空间衰减率
    temperature = global_config.get("tanh_temperature", 3.0)       # tanh 平滑映射的温度系数，用于调节推演饱和度的缩放
    final_candidates = {}
    
    # 1. 处理图谱多跳方案
    if graph_dict:
        for content, raw_score in graph_dict.items():
            if raw_score == float('inf') or raw_score > 500:
                # 🚨 绝对禁忌症：保送入围，拥有绝对优先级，图谱逻辑特权分给 0.0（防守掩码）
                priority = raw_score 
                graph_logic_score = 0.0
            else:
                # 正常多跳方案：使用 tanh 进行平滑映射，将 [0, +∞) 的饱和度连续映射到 [0, 1.0)
                # 彻底解决 min(raw_score, 1.0) 导致高分方案全部被削平为 1.0 的问题
                graph_logic_score = math.tanh(float(raw_score) / temperature)
                
                # 赋予图谱先验霸体特权 (+1.0)，确保多跳推理出的方案优先级高于普通单跳向量
                priority = graph_logic_score + graph_prior_bias 
                
            final_candidates[content] = {"priority": priority, "coherence": graph_logic_score}
    # 2. 处理向量单跳方案
    for i, content in enumerate(vector_list):
        if content not in final_candidates:
            # 纯单跳方案：遵循经典的带阻尼随机游走 (Random Walk with Restart) 衰减
            decay_priority = pagerank_damping * (spatial_decay ** i)
            
            # 为向量召回赋予微弱的连续图谱边缘分（避免硬编码0.1），排名越靠后得分越低
            vector_graph_score = 0.1 * (spatial_decay ** i) 
            
            final_candidates[content] = {"priority": decay_priority, "coherence": vector_graph_score}
        else:
            # 双路共识方案：同时被图谱推演和向量命中，给予共识奖励分
            final_candidates[content]["priority"] += 0.5
            
    # 3. 严格按综合优先级截断 Top K
    sorted_candidates = sorted(final_candidates.items(), key=lambda x: x[1]["priority"], reverse=True)[:query_param.top_k]
    
    # 将平滑、连续的得分透传给底层的大模型/MoE层
    knowledge = [{"<knowledge>": k, "<coherence>": round(v["coherence"], 4)} for k, v in sorted_candidates]
    return knowledge


async def _get_node_data(
    query, # 患者病历原始文本
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict, # <--- [新增] 接住上层传来的配置
):  
    results = entities_vdb
    if not len(results):
        return {} # ⚠️ 修复：返回空字典，防止后续迭代报错

    node_keys = [r["entity_name"] if isinstance(r, dict) else r for r in results]

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(k) for k in node_keys]
    )
    if not all([n is not None for n in node_datas]):
        from .utils import logger
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(k) for k in node_keys]
    )
    
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(node_keys, node_datas, node_degrees)
        if n is not None
    ]  
    
    # [关键修改]：把 query 和 global_config 继续传给最底层的找边算分函数
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst, query, global_config
    )

    print("\n" + "="*20 + " 🚀 [超图多维特征同构召回效果演示] " + "="*20)
    for idx, edge in enumerate(use_relations[:5]): 
        matched_ents = edge.get('matched_entities', [])
        score = edge.get('coverage_score', 0)
        desc_snippet = edge.get('description', '')[:60].replace('\n', ' ') 
        print(f"Top {idx+1} | 覆盖实体数: {score:.4f} | 命中实体: {matched_ents}")
        print(f"  └─ 方案内容: {desc_snippet}...")
    print("="*75 + "\n")

    knowledge_dict = {}
    for s in use_relations:
        desc = s["description"].replace("<hyperedge>", "")
        knowledge_dict[desc] = s.get("coverage_score", 0.0)

    return knowledge_dict


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                    "relation_counts": 0,
                }

            if this_edges:
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        all_text_units_lookup[c_id]["relation_counts"] += 1

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units

# 💡 全局配置：定义临床语义角色截断优先级 (越小优先级越高，图爆炸时越不容易被截断)
ROLE_TRUNCATION_PRIORITY = {
    "CONTRAINDICATION": 0,  # 绝对禁忌，保命级，永不丢弃
    "RECOMMENDATION": 1,    # 核心治疗方案，优先保留
    "EVIDENCE": 2,          # 循证依据，中等保留
    "CONDITION": 3,         # 前提条件
    "CONTEXT": 4,           # 背景知识，图爆炸时最先淘汰
    "UNKNOWN": 5            # 未知角色兜底
}

import numpy as np
import torch
import asyncio
from collections import defaultdict
# 请确保顶部已经导入了 compute_hyperedge_scores_sync 和 GLOBAL_ENTITY_CACHE

async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam, # 这里的类型提示视你的实际代码而定
    knowledge_graph_inst,    # BaseGraphStorage
    query_str: str,       # <--- 患者病历原始文本
    global_config: dict   # <--- 全局配置
):
    # 设定安全扩展阈值，作为动态召回的软配额
    MAX_NEIGHBORS_PER_NODE = 50 

    # ==========================================
    # 🌟 预处理：前置计算 Query 向量，用于语义召回与后续推理
    # ==========================================
    embedding_func = knowledge_graph_inst.embedding_func
    query_emb_np = None
    if embedding_func:
        # 获取 Query 的 1536 维 (或其他维度) 稠密向量
        query_emb_np = (await embedding_func([query_str]))[0]

    def calc_cosine_sim(v1, v2):
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return float(np.dot(v1, v2) / norm) if norm > 0 else 0.0

    # ==========================================
    # 阶段一：获取带有 Role 的一阶边，并执行【Query感知的动态语义召回】
    # ==========================================
    all_related_edges_raw = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges_with_roles(dp["entity_name"]) for dp in node_datas]
    )

    hyperedge_to_entities = defaultdict(list)
    entity_to_hyperedges = defaultdict(list) 
    
    for dp, this_edges in zip(node_datas, all_related_edges_raw):
        if not this_edges:
            continue
        entity_name = dp["entity_name"]
        
        # 🛡️【核心防御与升级：动态语义召回 + 领域公理硬保留】
        if len(this_edges) > MAX_NEIGHBORS_PER_NODE:
            must_keep = []
            candidates = []
            
            # 1. 划定绝对安全红线 (Symbolic Constraints)
            for e in this_edges:
                role = e[2] if len(e) > 2 else "UNKNOWN"
                # 领域公理：绝对禁忌和核心推荐无条件保留，彻底无视 50 的上限
                if role in ["CONTRAINDICATION", "RECOMMENDATION"]:
                    must_keep.append(e)
                else:
                    candidates.append(e)
            
            # 2. 对剩余的候选边进行数据驱动的语义竞争 (Semantic Recall)
            quota = MAX_NEIGHBORS_PER_NODE - len(must_keep)
            
            if quota > 0 and len(candidates) > quota and query_emb_np is not None:
                # 提取候选超边的文本内容进行批量 Embedding
                cand_texts = [e[1] for e in candidates]
                cand_embs = await embedding_func(cand_texts)
                
                cand_with_sim = []
                for cand, emb in zip(candidates, cand_embs):
                    sim = calc_cosine_sim(query_emb_np, emb)
                    cand_with_sim.append((cand, sim))
                
                # 按与 Query 的相似度降序排列 (废弃静态字典序)
                cand_with_sim.sort(key=lambda x: (x[1], x[0][1]), reverse=True)
                candidates = [x[0] for x in cand_with_sim[:quota]]
                
            elif quota <= 0:
                # 如果禁忌症和推荐已经占满了配额，背景知识直接舍弃
                candidates = []
            elif query_emb_np is None and quota > 0:
                # 兜底：如果外部 Embedding 挂了，降级为原有的权重截断
                candidates = sorted(candidates, key=lambda x: -(x[-1] if isinstance(x[-1], float) else 1.0))[:quota]

            this_edges = must_keep + candidates

        for e in this_edges:
            he_name = e[1]
            role = e[2] if len(e) > 2 else "UNKNOWN"
            hyperedge_to_entities[he_name].append((entity_name, role))
            entity_to_hyperedges[entity_name].append(he_name)

    unique_hyperedges = sorted(list(hyperedge_to_entities.keys()))

    # ==========================================
    # 阶段二：获取二阶邻居，锁定文献归属并执行基础度截断
    # ==========================================
    hyperedge_neighbors_raw = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(he) for he in unique_hyperedges]
    )

    parent_aggregation = defaultdict(lambda: {"contained_hyperedges": set(), "matched_items": set()})
    hyperedge_neighbors_truncated = [] 
    
    for he, neighbors in zip(unique_hyperedges, hyperedge_neighbors_raw):
        if not neighbors:
            hyperedge_neighbors_truncated.append([])
            continue
            
        # 二阶邻居通常是结构化的 Paper ID，不含语义文本，保留权重截断即可
        if len(neighbors) > MAX_NEIGHBORS_PER_NODE:
            neighbors = sorted(
                neighbors, 
                key=lambda x: -(x[-1] if isinstance(x[-1], float) else 1.0)
            )[:MAX_NEIGHBORS_PER_NODE]
            
        hyperedge_neighbors_truncated.append(neighbors)

        for edge in neighbors:
            neighbor_name = edge[1]
            if neighbor_name.startswith("paper::"): 
                parent_aggregation[neighbor_name]["contained_hyperedges"].add(he)
                parent_aggregation[neighbor_name]["matched_items"].update(hyperedge_to_entities[he])


    # ==========================================
    # 🌟 阶段三：组装 HGNN 超边批处理张量 
    # ==========================================
    ROLE_VOCAB = {"EVIDENCE": 0, "CONTEXT": 1, "CONDITION": 2, "RECOMMENDATION": 3, "CONTRAINDICATION": 4, "PAD": 5}
    MAX_ENTITIES = 40 # 必须与训练时的数据结构对齐
    
    hyperedge_tensors = []
    hyperedge_roles = []
    hyperedge_masks = []
    valid_hyperedges = []
    
    contraindicated_hyperedges = set()
    hyperedge_role_profiles = defaultdict(set) 
    
    if len(GLOBAL_ENTITY_CACHE) == 0:
        print("🚨 致命警告: 实体缓存为0！请检查是否调用了 init_attention_system()！")

    for he in unique_hyperedges:
        hit_items = hyperedge_to_entities[he]
        embs, roles, masks = [], [], []
        
        for ent, role in hit_items:
            hyperedge_role_profiles[he].add(role)
            if role == "CONTRAINDICATION":
                contraindicated_hyperedges.add(he)
                
            ent_key = ent.upper()
            if ent_key in GLOBAL_ENTITY_CACHE:
                embs.append(GLOBAL_ENTITY_CACHE[ent_key])
                roles.append(ROLE_VOCAB.get(role.upper(), 1)) 
                masks.append(1.0)

        if len(embs) > 0:
            embs = embs[:MAX_ENTITIES]
            roles = roles[:MAX_ENTITIES]
            masks = masks[:MAX_ENTITIES]
            
            pad_len = MAX_ENTITIES - len(embs)
            if pad_len > 0:
                dim = embs[0].size(0)
                embs.extend([torch.zeros(dim)] * pad_len)
                roles.extend([ROLE_VOCAB["PAD"]] * pad_len)
                masks.extend([0.0] * pad_len)
                
            hyperedge_tensors.append(torch.stack(embs))
            hyperedge_roles.append(roles)
            hyperedge_masks.append(masks)
            valid_hyperedges.append(he)

    # ==========================================
    # 🌟 阶段四：调用 HGNN 进行端到端批量推断
    # ==========================================
    hyperedge_scores = {}
    hyperedge_baseline_scores = {}
    
    if valid_hyperedges and query_emb_np is not None:
        he_embs_batch = torch.stack(hyperedge_tensors)
        he_roles_batch = torch.tensor(hyperedge_roles, dtype=torch.long)
        he_masks_batch = torch.tensor(hyperedge_masks, dtype=torch.bool)
        
        # 💡 复用阶段一预计算的 Query 向量
        query_emb_tensor = torch.tensor(query_emb_np, dtype=torch.float32)
        
        hg_scores = await asyncio.to_thread(
            compute_hyperedge_scores_sync, 
            query_emb_tensor, 
            (he_embs_batch, he_roles_batch, he_masks_batch)
        )
        
        for he, score in zip(valid_hyperedges, hg_scores):
            # 💡 [安全防线]: 触发阶段五的强预警一票否决
            if he in contraindicated_hyperedges:
                hyperedge_scores[he] = float('-inf')
            else:
                hyperedge_scores[he] = float(score)
            hyperedge_baseline_scores[he] = float(score)
    else:
        print("⚠️ 警告: 缺少 embedding_func 或没有有效超边，打分降级为 0")

    for he in unique_hyperedges:
        if he not in hyperedge_scores:
            hyperedge_scores[he] = 0.0
            hyperedge_baseline_scores[he] = 0.0

    # ==========================================
    # 🌟 阶段五：字典序分层聚合排序 (Lexicographical Sorting)
    # ==========================================
    parent_scores = []
    for parent, data in parent_aggregation.items():
        contained_hes = sorted(list(data["contained_hyperedges"]))
        matched_items = sorted(list(data["matched_items"]), key=lambda x: x[0])
        
        valid_he_scores = [hyperedge_scores[he] for he in contained_hes if hyperedge_scores[he] > 0]
        
        contraindication_alerts = [] 
        for he in contained_hes:
            if hyperedge_scores[he] == float('-inf'):
                contraindication_alerts.extend([ent for ent, role in hyperedge_to_entities[he] if role == "CONTRAINDICATION"])

        if valid_he_scores or contraindication_alerts:
            coverage_score = sum(valid_he_scores)
            
            if contraindication_alerts:
                coverage_score += 1000.0 
                
            has_recommendation = any("RECOMMENDATION" in hyperedge_role_profiles.get(he, set()) for he in contained_hes)
            has_evidence = any("EVIDENCE" in hyperedge_role_profiles.get(he, set()) for he in contained_hes)
            
            if has_recommendation:
                tier = 3 
            elif has_evidence:
                tier = 2 
            else:
                tier = 1 

            parent_scores.append({
                "parent_name": parent,
                "coverage_score": round(coverage_score, 4),
                "tier": tier,
                "contained_hyperedges": list(contained_hes),
                "matched_items": matched_items,
                "contraindication_alerts": list(set(contraindication_alerts)) 
            })

    parent_scores = sorted(
        parent_scores,
        key=lambda x: (x["tier"], x["coverage_score"], len(x["contained_hyperedges"])),
        reverse=True
    )
    top_parents = parent_scores[:5] 

    # ==========================================
    # 阶段六：控制台打印与组装结构化上下文
    # ==========================================
    print("\n" + "="*15 + " 🧮 [神经符号学 (Neuro-Symbolic) 推演明细] " + "="*15)
    for i, p in enumerate(top_parents):
        display_score = p['coverage_score'] - 1000.0 if p["contraindication_alerts"] else p['coverage_score']
        print(f"[{i+1}] {p['parent_name']} | 逻辑层级: Tier {p['tier']} | ∑推荐度: {display_score:.4f}")
        
        for he in p["contained_hyperedges"]:
            score = hyperedge_scores[he]
            if score == float('-inf'):
                print(f"    ├─ ❌ [触发一票否决] 动作: {he[:25].replace(chr(10), '')}...")
                continue
                
            hit_items = hyperedge_to_entities[he]
            detail_strs = []
            for ent, role in hit_items:
                detail_strs.append(f"{ent}")
                
            he_snippet = he[:25].replace('\n', '') + "..." 
            
            trend_str = f"🚀 {score:.2%}" if score > 0.7 else f"🔻 {score:.2%}" if score < 0.3 else f"⚖️ {score:.2%}"
            print(f"    ├─ [{list(hyperedge_role_profiles[he])[0]}] 推荐度: {trend_str} | {he_snippet}")
            print(f"    │   └─ 命中特征: {' + '.join(detail_strs)}")
    print("="*66 + "\n")

    all_edges_data = []

    for p in top_parents:
        real_score = p['coverage_score'] - 1000.0 if p["contraindication_alerts"] else p['coverage_score']
        aggregated_desc = f"【权威循证溯源：{p['parent_name']}】(逻辑层级: Tier {p['tier']}, 推荐度: {real_score:.4f})\n"
        
        if p["contraindication_alerts"]:
            alerts_str = ", ".join(p["contraindication_alerts"])
            aggregated_desc += f"  [⚠️ 临床绝对警报]：患者基础病历与本指南发生【绝对禁忌症】冲突！特征：{alerts_str}\n"
            aggregated_desc += f"  🛑 请在最终的治疗方案输出中明确警示【不可使用以下方案及原因】：\n"
            
            for he in p["contained_hyperedges"]:
                if hyperedge_scores[he] == float('-inf'):
                    aggregated_desc += f"      ❌ 禁用方案: {he}\n"

        intra_group_edges = []
        for he in p["contained_hyperedges"]:
            if hyperedge_scores[he] > 0:
                hit_items = hyperedge_to_entities[he] 
                intra_group_edges.append((he, hyperedge_scores[he], hit_items))
            
        intra_group_edges.sort(key=lambda x: x[1], reverse=True)
        kept_edges = intra_group_edges[:4] 
        
        for idx, (he, sat_score, hit_items) in enumerate(kept_edges):
            trigger_reason = ", ".join([f"{ent}" for ent, role in hit_items])
            he_roles = "/".join(list(hyperedge_role_profiles[he]))
            aggregated_desc += (
                f"  > 候选路径 {idx+1} [{he_roles}] (神经推演推荐度: {sat_score:.2%})\n"
                f"    ├─ 方案内容: {he}\n"
                f"    ├─ 触发该方案的局部特征: 【{trigger_reason}】\n"
                f"    └─ ⚠️[系统强制检验指令]: 本方案因局部特征高匹配度被召回。作为主治医师，你在总结最终方案时，必须核对【患者完整病理与分期】是否满足此方案的【全部临床前提】。如患者当前阶段（如术后辅助期）不符合该方案（如晚期姑息）的标准触发条件，请主动将其废弃或降级为讨论项，绝不可与主路径机械叠加！\n"
            )

        all_edges_data.append({
            "description": aggregated_desc,
            "coverage_score": p["coverage_score"], 
            "rank": len(p["contained_hyperedges"]),
            "matched_entities": [item[0] for item in p["matched_items"]], 
            "weight": 1.0
        })

    return all_edges_data

async def _get_edge_data(
    query_str, # <--- 现在的入参是查询字符串
    knowledge_graph_inst: BaseGraphStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict, # <--- 接收配置
):  
    results = hyperedges_vdb
    if not len(results): return []

    edge_keys = [r["hyperedge_name"] if isinstance(r, dict) else r for r in results]
    edge_datas = await asyncio.gather(*[knowledge_graph_inst.get_node(k) for k in edge_keys])
    
    edge_datas = [{"hyperedge": k, "rank": v["weight"], **v} 
                  for k, v in zip(edge_keys, edge_datas) if v is not None]
    
    # 提取纯文本备用
    docs_to_rerank = [s["hyperedge"].replace("<hyperedge>","") for s in edge_datas]

    # ===== [核心] 调用注入的 Reranker 进行 Rank B 洗牌 =====
    reranker_func = global_config.get("reranker_func", None)
    if reranker_func and docs_to_rerank:
        # 由外部 test.py 提供打分逻辑，返回降序排列的 doc 列表
        knowledge_list = await reranker_func(query_str, docs_to_rerank) 
    else:
        # 降级方案
        edge_datas = sorted(edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True)
        knowledge_list = [s["hyperedge"].replace("<hyperedge>","") for s in edge_datas]
        
    return knowledge_list


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(edge["hyperedge"]) for edge in edge_datas]
    )
    
    entity_names = []
    seen = set()

    for node_data in node_datas:
        for e in node_data:
            if e[1] not in seen:
                entity_names.append(e[1])
                seen.add(e[1])

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]
    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                chunk_data = await text_chunks_db.get_by_id(c_id)
                # Only store valid data
                if chunk_data is not None and "content" in chunk_data:
                    all_text_units_lookup[c_id] = {
                        "data": chunk_data,
                        "order": index,
                    }

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    # Ensure all text chunks have content
    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    truncated_text_units = truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units: list[TextChunkSchema] = [t["data"] for t in truncated_text_units]

    return all_text_units


def combine_contexts(entities, relationships, sources):
    # Function to extract entities, relationships, and sources from context strings
    hl_entities, ll_entities = entities[0], entities[1]
    hl_relationships, ll_relationships = relationships[0], relationships[1]
    # Combine and deduplicate the entities
    combined_entities = process_combine_contexts(hl_entities, ll_entities)

    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )

    return combined_entities, combined_relationships, ""
