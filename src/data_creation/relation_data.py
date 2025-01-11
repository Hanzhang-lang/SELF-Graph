import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain.embeddings import (
    HuggingFaceEmbeddings,
)
from langchain_community.vectorstores import FAISS
# import sys
# sys.path.append('/media/disk1/chatgpt/zh/graph_data')
import datasets
import argparse
import networkx as nx
from collections import deque
import walker
import json
from typing import List
import random
import json
from typing import List
import multiprocessing as mp
from langchain.storage import LocalFileStore, RedisStore
from langchain.embeddings import CacheBackedEmbeddings
from multiprocess import set_start_method
from src.graph_utils import *
from src.sparql_utils import *
embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        # model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False},
    )


store = RedisStore(redis_url="redis://localhost:6379")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings, store, namespace="bge-large"
)
row_string = []
with open('data/clean_relations', 'r') as f:
    data = f.readlines()
db = FAISS.from_texts(data, cached_embedder)
retriever = db.as_retriever(search_kwargs={"k": 5})
def save_to_json(data: List, data_path='../output/chain_data.json'):
    if not os.path.isfile(data_path):
        # 文件不存在，创建新列表并写入文件
        with open(data_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        return
    try:
        # 尝试读取现有文件
        with open(data_path, 'r', encoding='utf-8') as file:
            # 加载现有的JSON数据
            existing_data = json.load(file)
            existing_data.extend(data)
    except json.JSONDecodeError:
        # 文件不是有效的JSON，打印错误信息并退出
        print(f"文件 {data_path} 不是有效的JSON格式。")
        return
    except ValueError as e:
        # 打印错误信息并退出
        print(e)
        return
    # 将更新后的数据写回文件
    with open(data_path, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)


def build_Digraph(graph: list) -> nx.Graph:
    G = nx.DiGraph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G


def bfs_with_rule(graph, start_node, target_rule, max_p=10):
    result_paths = []
    queue = deque([(start_node, [])])
    while queue:
        current_node, current_path = queue.popleft()
        if len(current_path) == len(target_rule):
            result_paths.append(current_path)
        if len(current_path) < len(target_rule):
            if current_node not in graph:
                continue
            for neighbor in graph.neighbors(current_node):
                rel = graph[current_node][neighbor]['relation']
                if rel != target_rule[len(current_path)] or len(current_path) > len(target_rule):
                    continue
                queue.append((neighbor, current_path +
                             [(current_node, rel, neighbor)]))
            for neighbor in graph.predecessors(current_node):
                rel = graph[neighbor][current_node]['relation']
                if rel != target_rule[len(current_path)] or len(current_path) > len(target_rule):
                    continue
                queue.append((neighbor, current_path +
                             [(current_node, rel, neighbor)]))

    return result_paths


def process_relation_data(line):
    # import multiprocess
    chain_data = []
    id = line['id']
    topic_entity = line['q_entity']
    answer = line['a_entity']
    di_graph = build_Digraph(line['graph'])
    paths = get_truth_paths(topic_entity, answer, di_graph)
    # every question sample at most 3 paths
    paths = random.sample(paths, min(3, len(paths)))
    sent_idx = 0
    for pid, p in enumerate(paths):
        for pidx, step in enumerate(p):
            real_relation = step[1]
            real_entity = step[2]
            candidate_entities = [tail for head, tail in di_graph.out_edges(
                step[0]) if di_graph[head][tail].get('relation') == step[1]]
            candidate_relation = [page.page_content.strip() for page in retriever.invoke(
                line['question'] + real_relation + real_entity)]
            if step[1] in candidate_relation:
                chain_data.append({
                    "sent_idx": sent_idx,
                    "chain_step": pidx + 1,
                    "candidate_relation": candidate_relation,
                    "candidate_entity": candidate_entities,
                    "real_relation": real_relation,
                    "real_entity": real_entity,
                    "paths": p,
                    "effective": True
                })
            else:
                chain_data.append({
                    "sent_idx": sent_idx,
                    "chain_step": pidx + 1,
                    "candidate_relation": candidate_relation,
                    "candidate_entity": candidate_entities,
                    "real_relation": real_relation,
                    "real_entity": real_entity,
                    "paths": p,
                    "effective": False
                })
            sent_idx += 1
    return {"qid": id, "query": line['question'], "topic_entity": topic_entity, "answer": answer, "chains": chain_data}

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="webqsp", help="choose the dataset.")
    args = parser.parse_args()
    relation_data_train = datasets.load_dataset('rmanluo/RoG-cwq', split='validation')
    processed_data = relation_data_train.map(process_relation_data, num_proc=8, remove_columns=relation_data_train.column_names)
    # processed_data.to_json('./output/chain_data/cwq_dev_chain_top_5.json')
    print('success')
