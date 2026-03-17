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
from .hyper_attention import GLOBAL_ENTITY_CACHE, compute_dynamic_weights_sync


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
    # [关键修改]：将 global_config 和 ll_kewwords 传入，用于唤醒 Attention 机制
    graph_dict = await _get_node_data(
        ll_kewwords, knowledge_graph_inst, entities_vdb, text_chunks_db, query_param, global_config
    )

    # 【流 B】向量单跳流：返回 [内容列表]
    vector_list = await _get_edge_data(
        hl_keywrds, knowledge_graph_inst, hyperedges_vdb, text_chunks_db, query_param, global_config
    )
    
    # ========================================================
    # 🌟 彻底剔除 W-RRF！使用纯净的优先级与自然量纲机制 (你的优秀重构)
    # ========================================================
    final_candidates = {}
    
    # 1. 处理图谱多跳方案
    if graph_dict:
        for content, raw_score in graph_dict.items():
            if raw_score > 500:
                # 🚨 绝对禁忌症：保送入围，但图谱逻辑特权分给 0.0
                priority = raw_score 
                graph_logic_score = 0.0
            else:
                # 正常多跳方案：不搞任何恶心的公式压缩，直接封顶 1.0！
                graph_logic_score = min(float(raw_score), 1.0)
                # 赋予海选霸体特权 (+1.0)，碾压普通向量
                priority = graph_logic_score + 1.0 
                
            final_candidates[content] = {"priority": priority, "coherence": graph_logic_score}

    # 2. 处理向量单跳方案
    for i, content in enumerate(vector_list):
        if content not in final_candidates:
            # 纯单跳方案：海选优先级平滑衰减
            decay_priority = 0.85 * (0.95 ** i)
            # 缺乏图谱背书，图谱底薪只给 0.1！
            final_candidates[content] = {"priority": decay_priority, "coherence": 0.1}
        else:
            # 双路共识方案：加分奖励
            final_candidates[content]["priority"] += 0.5
            
    # 3. 严格按海选优先级截断 Top 40
    sorted_candidates = sorted(final_candidates.items(), key=lambda x: x[1]["priority"], reverse=True)[:query_param.top_k]
    
    # 将极其纯净的自然得分透传给 MoE
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


# 💡 全局配置：定义临床语义角色权重乘数
ROLE_WEIGHT_MULTIPLIERS = {
    "CONDITION": 1.0,          # 适应症/前提条件 -> 基础权重
    "RECOMMENDATION": 1.5,     # 推荐动作 -> 提供分子放大增益，鼓励核心方案
    "CONTRAINDICATION": -999.0,# 禁忌症 -> 极负权重强行触发微观方案的一票否决
    "EVIDENCE": 0.8,           # 证据支撑 -> 略低于核心条件
    "CONTEXT": 0.2,            # 背景信息 -> 补充性权重
    "UNKNOWN": 1.0             # 未知角色 -> 默认保底
}

async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    query_str: str,       # <--- 患者病历原始文本
    global_config: dict   # <--- 全局配置
):
    # ==========================================
    # 阶段一：获取带有 Role (语义角色) 的一阶边
    # ==========================================
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges_with_roles(dp["entity_name"]) for dp in node_datas]
    )

    hyperedge_to_entities = defaultdict(list)
    entity_to_hyperedges = defaultdict(list) 
    
    for dp, this_edges in zip(node_datas, all_related_edges):
        if not this_edges:
            continue
        entity_name = dp["entity_name"]
        for e in this_edges:
            he_name = e[1]
            role = e[2] if len(e) > 2 else "UNKNOWN"
            hyperedge_to_entities[he_name].append((entity_name, role))
            entity_to_hyperedges[entity_name].append(he_name)

    unique_hyperedges = list(hyperedge_to_entities.keys())

    # ==========================================
    # 阶段二：获取二阶邻居，锁定文献归属并提取所有相关特征
    # ==========================================
    hyperedge_neighbors = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(he) for he in unique_hyperedges]
    )

    parent_aggregation = defaultdict(lambda: {"contained_hyperedges": set(), "matched_items": set()})
    all_neighbor_entities = set() 
    
    for he, neighbors in zip(unique_hyperedges, hyperedge_neighbors):
        if not neighbors:
            continue
        for edge in neighbors:
            neighbor_name = edge[1]
            if neighbor_name.startswith("paper::"): 
                parent_aggregation[neighbor_name]["contained_hyperedges"].add(he)
                parent_aggregation[neighbor_name]["matched_items"].update(hyperedge_to_entities[he])
            elif not neighbor_name.startswith("<hyperedge>"):
                all_neighbor_entities.add(neighbor_name)

    # ==========================================
    # 🌟 阶段三：基于 QA-HGAT 的动态超图信息熵计算
    # ==========================================
    hit_entity_idf = {dp["entity_name"]: float(dp.get("idf_weight", 1.0)) for dp in node_datas}

    unknown_entities = list(all_neighbor_entities - set(hit_entity_idf.keys()))
    unknown_entities_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(ent) for ent in unknown_entities]
    )
    
    all_entity_idf = hit_entity_idf.copy()
    for ent, data in zip(unknown_entities, unknown_entities_data):
        all_entity_idf[ent] = float(data.get("idf_weight", 1.0)) if data else 1.0

    # 🚀 1. 准备当前患者病历的张量
    # 💡 [修复 1]: 直接使用知识图谱实例自带的 embedding_func，防止在 global_config 中获取失败
    embedding_func = knowledge_graph_inst.embedding_func
    if embedding_func:
        # 取决于底层的异步封装，如果你在外部报错这里不需要 await，请去掉 await
        query_emb_np = await embedding_func([query_str]) 
        query_emb_tensor = torch.tensor(query_emb_np[0], dtype=torch.float32)
    else:
        query_emb_tensor = torch.zeros(1536) # 降级占位符，请确认你的模型维度，Qwen3-4B通常是3584或1536
        print("⚠️ 警告: knowledge_graph_inst 缺少 embedding_func，Attention 将降级！")

    # 🚀 2. 从本地缓存抽取所有相关特征实体的预计算张量
    all_involved_entities = list(all_entity_idf.keys())
    entity_tensors = []
    valid_entities = []
    
    if len(GLOBAL_ENTITY_CACHE) == 0:
        print("🚨 致命警告: 实体缓存为0！请检查 main/初始化代码 中是否调用了 init_attention_system()！")

    for ent in all_involved_entities:
        # 💡 [修复 2]: 强制转换为大写去匹配缓存，确保百分百命中！
        ent_key = ent.upper()
        if ent_key in GLOBAL_ENTITY_CACHE:
            entity_tensors.append(GLOBAL_ENTITY_CACHE[ent_key])
            valid_entities.append(ent) # 记录原始名字用于后续匹配
            
    # 🚀 3. 异步唤醒独立线程进行 Attention 打分
    if valid_entities and embedding_func:
        entity_embs_batch = torch.stack(entity_tensors)
        dynamic_weights = await asyncio.to_thread(
            compute_dynamic_weights_sync, 
            query_emb_tensor, 
            entity_embs_batch
        )
        all_entity_attention = {ent: float(weight) for ent, weight in zip(valid_entities, dynamic_weights)}
        
        # 💡 打印 AI 顿悟结果
        print(f"\n🧠 [AI 临床直觉] 成功推演 {len(valid_entities)} 个特征的上下文动态权重:")
        sorted_attn = sorted(all_entity_attention.items(), key=lambda x: x[1], reverse=True)
        if sorted_attn:
            print(f"   📈 极度关键 (最高提权): {sorted_attn[0][0]} -> {sorted_attn[0][1]:.2f}x 增幅")
            print(f"   📉 无关紧要 (最高降权): {sorted_attn[-1][0]} -> {sorted_attn[-1][1]:.2f}x 衰减")
    else:
        all_entity_attention = defaultdict(lambda: 1.0) 
        print(f"\n⚠️ [AI 临床直觉] 降级打分！找到有效实体数:{len(valid_entities)}, embedding存在:{bool(embedding_func)}")

    # 🚀 4. 执行严谨的临床数学核算 (双轨并行计分)
    hyperedge_scores = {}
    hyperedge_baseline_scores = {} # 记录纯静态基线分
    contraindicated_hyperedges = set()
    
    for he, neighbors in zip(unique_hyperedges, hyperedge_neighbors):
        hit_items = hyperedge_to_entities[he]
        
        total_weight = 0.0
        if neighbors:
            for edge in neighbors:
                n_name = edge[1]
                if n_name in all_entity_idf:
                    total_weight += all_entity_idf[n_name]
        total_weight = max(total_weight, 0.1)

        hit_weight = 0.0
        baseline_hit_weight = 0.0 
        has_contraindication = False

        for ent, role in hit_items:
            if role == "CONTRAINDICATION":
                has_contraindication = True
                contraindicated_hyperedges.add(he)
                break 
            
            role_m = ROLE_WEIGHT_MULTIPLIERS.get(role, 1.0)
            static_idf = all_entity_idf.get(ent, 1.0)
            attn_w = all_entity_attention.get(ent, 1.0) 
            
            # 双轨对比算分
            hit_weight += static_idf * attn_w * role_m
            baseline_hit_weight += static_idf * 1.0 * role_m 

        if has_contraindication:
            hyperedge_scores[he] = -999.0
            hyperedge_baseline_scores[he] = -999.0
        else:
            hyperedge_scores[he] = round(hit_weight / total_weight, 4)
            hyperedge_baseline_scores[he] = round(baseline_hit_weight / total_weight, 4)

    # ==========================================
    # 阶段四：文献级信息降维聚合与强制曝光 (宏观文献层)
    # ==========================================
    parent_scores = []
    for parent, data in parent_aggregation.items():
        contained_hes = data["contained_hyperedges"]
        matched_items = list(data["matched_items"]) 
        
        valid_he_scores = [hyperedge_scores[he] for he in contained_hes if hyperedge_scores[he] > 0]
        
        contraindication_alerts = [] 
        for he in contained_hes:
            if hyperedge_scores[he] < 0:
                contraindication_alerts.extend([ent for ent, role in hyperedge_to_entities[he] if role == "CONTRAINDICATION"])

        if valid_he_scores or contraindication_alerts:
            coverage_score = sum(valid_he_scores)
            
            if contraindication_alerts:
                coverage_score += 1000.0 

            parent_scores.append({
                "parent_name": parent,
                "coverage_score": round(coverage_score, 4),
                "contained_hyperedges": list(contained_hes),
                "matched_items": matched_items,
                "contraindication_alerts": list(set(contraindication_alerts)) 
            })

    parent_scores = sorted(
        parent_scores,
        key=lambda x: (x["coverage_score"], len(x["contained_hyperedges"])),
        reverse=True
    )
    top_parents = parent_scores[:5] 

    # ==========================================
    # 阶段五：在控制台打印可解释的“信息熵拓扑推演树”
    # ==========================================
    print("\n" + "="*15 + " 🧮 [高阶超图信息熵推演过程明细] " + "="*15)
    for i, p in enumerate(top_parents):
        display_score = p['coverage_score'] - 1000.0 if p["contraindication_alerts"] else p['coverage_score']
        print(f"[{i+1}] 归属文献: {p['parent_name']}")
        print(f"    ⭐ 文献总推演分 (∑饱和度): {display_score:.4f} {'(🚨 包含最高级禁忌警报)' if p['contraindication_alerts'] else ''}")
        
        for he in p["contained_hyperedges"]:
            score = hyperedge_scores[he]
            if score < 0:
                print(f"    ├─ ❌ [触发一票否决] 动作: {he[:25].replace(chr(10), '')}...")
                continue
                
            hit_items = hyperedge_to_entities[he]
            detail_strs = []
            for ent, role in hit_items:
                w = all_entity_idf.get(ent, 1.0)
                rm = ROLE_WEIGHT_MULTIPLIERS.get(role, 1.0)
                attn = all_entity_attention.get(ent, 1.0)
                detail_strs.append(f"{ent}(IDF:{w:.2f} | Attn:{attn:.2f})")
                
            he_snippet = he[:25].replace('\n', '') + "..." 
            
            # 💡 核心视觉冲击力：计算动态变化并打上箭头
            base_score = hyperedge_baseline_scores.get(he, score)
            diff = score - base_score
            if diff > 0.01:
                trend_str = f"🚀 由 {base_score:.2%} 跃升至 {score:.2%} (+{diff:.2%})"
            elif diff < -0.01:
                trend_str = f"🔻 由 {base_score:.2%} 降级至 {score:.2%} ({diff:.2%})"
            else:
                trend_str = f"⚖️ 平稳维持 {score:.2%}"

            print(f"    ├─ 动作饱和度: {trend_str} | {he_snippet}")
            print(f"    │   └─ 命中特征: {' + '.join(detail_strs)}")
    print("="*62 + "\n")

    # ==========================================
    # 阶段六：组装带警报防守的结构化上下文给 LLM 
    # ==========================================
    all_edges_data = []

    for p in top_parents:
        real_score = p['coverage_score'] - 1000.0 if p["contraindication_alerts"] else p['coverage_score']
        aggregated_desc = f"【权威循证溯源：{p['parent_name']}】(推演饱和度: {real_score:.4f})\n"
        
        if p["contraindication_alerts"]:
            alerts_str = ", ".join(p["contraindication_alerts"])
            aggregated_desc += f"  [⚠️ 临床绝对警报]：该患者具有合并症/特征 {alerts_str}，触发了本指南的【绝对禁忌症】！\n"
            aggregated_desc += f"  🛑 请在最终的治疗方案输出中，明确向医生指出【不可使用以下方案及原因】：\n"
            
            for he in p["contained_hyperedges"]:
                if hyperedge_scores[he] < 0:
                    aggregated_desc += f"      ❌ 禁用方案: {he}\n"

        intra_group_edges = []
        for he in p["contained_hyperedges"]:
            if hyperedge_scores[he] > 0:
                hit_items = hyperedge_to_entities[he] 
                intra_group_edges.append((he, hyperedge_scores[he], hit_items))
            
        intra_group_edges.sort(key=lambda x: x[1], reverse=True)
        kept_edges = intra_group_edges[:4] 
        
        for idx, (he, sat_score, hit_items) in enumerate(kept_edges):
            # 将推演出的权重随证据一起送给 LLM
            trigger_reason = ", ".join([f"{ent}(权重增益:{all_entity_attention.get(ent, 1.0):.2f})" for ent, role in hit_items])
            aggregated_desc += f"  > 核心正面推荐 {idx+1} [饱和度 {sat_score:.2%}, 关键证据: {trigger_reason}]: {he}\n"

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
