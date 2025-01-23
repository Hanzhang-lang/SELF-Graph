import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain.embeddings import (
    HuggingFaceEmbeddings,
)
import argparse
from vllm import LLM, SamplingParams
import numpy as np
import re
from src.sparql_utils import *
import random
from langchain.storage import LocalFileStore, RedisStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer
import json
from src.utils import save_to_json
# TODO: 提高树结构的效率，减少relation和entity的重复计算


def load_special_tokens(tokenizer, use_grounding=False, use_utility=False):
    rel_tokens = {}
    for token in ['[Unrelevant]', '[Partially Relevant]', '[Fully Relevant]']:
        rel_tokens[token] = tokenizer.convert_tokens_to_ids(token)
    return rel_tokens


def run_relation_generation_batch(model, prompt, new_retrieval, context, topic_entity, hypo=True, use_1hop=True, rel_tokens=None, embeddings=None, all_db=None, sampling_params=None, ):
    rel_score_dict = {}
    final_preds = []
    overall_scores = []
    final_contexts = []
    if new_retrieval:
        retrieval_token = "[New Retrieval]"
    else:
        retrieval_token = "[Continue to Retrieve Evidence]"
    if use_1hop:
        candidate_relations = []
        for entity in topic_entity:
            try:
                candidate_relations.extend(
                    get_1hop_relations_with_odbc(entity))
            except:
                continue
        if len(candidate_relations):
            vec_db = FAISS.from_texts(candidate_relations, embeddings)
        else:
            vec_db = all_db
        retriever = vec_db.as_retriever(search_kwargs={"k": 5})
    else:
        retriever = all_db.as_retriever(search_kwargs={"k": 5})
    paragraph = [page.page_content.strip() for page in retriever.invoke(
        prompt.split('\n\n')[1].split('<|eot_id|>')[0] + ' ' + context)]
    if hypo:
        hypo_rel = model.generate(
            prompt + retrieval_token, sampling_params)[0].outputs[0].text
        pattern = r'(\w+\.\w+\.\w+)\[(.*?)\]'
        if '[Retrieve Entity]' in hypo_rel:
            hypo_rel = hypo_rel.split('[Retrieve Entity]')[0]
        matches = dict(re.findall(pattern, hypo_rel))
        string = ''
        # for k,v in matches.items():
        #     if v in ['Fully Relevant', 'Partially Relevant']:
        #         for extra_rel in retriever.invoke(k):
        #             if extra_rel.page_content.strip() not in paragraph:
        #                 paragraph.append(extra_rel.page_content.strip())
    # aug_prompts = ["<paragraph>{}</paragraph>".format(';'.join(paragraph))]
        for k, v in matches.items():
            if v in ['Fully Relevant', 'Partially Relevant']:
                string += k + ' '
        for extra_rel in retriever.invoke(string):
            if extra_rel.page_content.strip() not in paragraph:
                paragraph.append(extra_rel.page_content.strip())
    aug_prompts = ["<paragraph>{}</paragraph>".format(
        ';'.join(p)) for p in [paragraph[i: i+5] for i in range(0, len(paragraph), 5)]]

    preds = model.generate(
        [prompt + retrieval_token + aug for aug in aug_prompts], sampling_params)
    for p_id, pred in enumerate(preds):
        pred_token_ids = pred.outputs[0].token_ids
        pred_text_1 = pred.outputs[0].text
        pred_log_probs = pred.outputs[0].logprobs
        seq_score = pred.outputs[0].cumulative_logprob / \
            max(len(pred.outputs[0].token_ids), 1)
        relevance_indices = []
        for tok_idx, tok in enumerate(pred_token_ids):
            if tok in rel_tokens.values():
                relevance_indices.append(tok_idx)
        if len(relevance_indices) > 0:
            for idx in relevance_indices:
                for token, token_id in rel_tokens.items():
                    prob = pred_log_probs[idx][token_id].logprob if token_id in pred_log_probs[idx] else -100
                    rel_score_dict[token] = np.exp(prob)
        relevance_score = rel_score_dict['[Fully Relevant]'] + \
            rel_score_dict['[Partially Relevant]'] / \
            np.sum(list(rel_score_dict.values()))
        if '[Retrieve Entity]' in pred_text_1:
            processed_pred = pred_text_1.split('[Retrieve Entity]')[
                0] + '[Retrieve Entity]'
        else:
            processed_pred = pred_text_1
        final_preds.append(retrieval_token +
                           aug_prompts[p_id] + processed_pred)
        overall_scores.append(0)
        final_contexts.append(pred_text_1.split('[Retrieve Entity]')[0])

    return final_preds, overall_scores, final_contexts


def run_entity_generation_batch(model, prompt, topic_entities, context, sampling_params=None, score_type='hard'):
    # 初始化变量
    final_predictions = []
    overall_scores = {}
    final_entities = []
    final_contexts = []
    entity_prompts = []
    effective_count = 0
    name_to_id = {}

    # 提取上下文中的实体和相关性匹配
    context_pattern = r'(.*?)\[(.*?)\]'
    context_matches = dict(re.findall(context_pattern, context))

    # 构建实体提示
    for entity in topic_entities:
        for key, relevance in context_matches.items():
            if relevance in ['Fully Relevant', 'Partially Relevant']:
                try:
                    related_entities = get_another_entity(
                        entity, key, return_label=True)
                except:
                    related_entities = {}

                if related_entities:
                    # 初始化分数和实体映射
                    name_to_id[effective_count] = related_entities
                    overall_scores[effective_count] = {
                        'r_relevance': 1.0 if relevance == 'Fully Relevant' else 0.5
                    }
                    # 构造实体提示
                    entities = [
                        f'({get_label(entity)}, {key}, {rel_entity})'
                        for rel_entity in related_entities.keys()
                    ]
                    entity_prompts.append(
                        f"<paragraph>{';'.join(entities[:5])}</paragraph>"
                    )
                    effective_count += 1

    # 生成模型预测
    augmented_prompts = [
        f"{prompt}[Retrieve Entity]{entity_prompt}"
        for entity_prompt in entity_prompts
    ]
    predictions = model.generate(augmented_prompts, sampling_params)

    # 解析预测结果
    for idx, prediction in enumerate(predictions):
        pred_output = prediction.outputs[0]
        pred_text = pred_output.text
        pred_log_probs = pred_output.logprobs
        pred_token_ids = pred_output.token_ids
        seq_score = pred_output.cumulative_logprob / \
            max(len(pred_token_ids), 1)

        processed_pred, rel_score, reason_score, return_entities = process_prediction(
            idx, pred_text, context_pattern, name_to_id, overall_scores
        )

        # 更新分数和最终结果
        overall_scores[idx].update({
            "e_relevance": rel_score,
            "reason": reason_score,
            "final_score": np.exp(rel_score) * np.exp(reason_score) * np.exp(overall_scores[idx]['r_relevance'])
        })

        final_predictions.append(
            f"[Retrieve Entity]{entity_prompts[idx]}{processed_pred}")
        final_entities.append(list(return_entities.values()))
        final_contexts.append(' '.join(return_entities.keys()))

    # 返回最终结果
    return final_predictions, [
        overall_scores[idx]['final_score'] for idx in overall_scores
    ], final_entities, final_contexts, overall_scores


def process_prediction(idx, pred_text, pattern, name_to_id, overall_scores):
    """处理单条预测结果，计算相关性分数和返回实体。"""
    rel_score = 0
    reason_score = 0
    return_entities = {}
    count = 0

    if '[Continue to Retrieve Evidence]' in pred_text:
        processed_pred = pred_text.split('[Continue to Retrieve Evidence]')[0]
        matches = dict(re.findall(pattern, processed_pred))
        for key, relevance in matches.items():
            if relevance in ['Fully Relevant', 'Partially Relevant']:
                if key in name_to_id[idx]:
                    return_entities[key] = name_to_id[idx][key]
                elif key == 'Unknown Entity':
                    random_key = random.choice(list(name_to_id[idx].keys()))
                    return_entities[random_key] = name_to_id[idx][random_key]
                # 更新相关性分数
                rel_score = ((1 if relevance == 'Fully Relevant' else 0.5) +
                             count * rel_score) / (count + 1)
                count += 1
        processed_pred += '[Continue to Retrieve Evidence]'

    elif '[No Retrieval]' in pred_text:
        processed_pred = pred_text
        matches = dict(re.findall(
            pattern, pred_text.split('[No Retrieval]')[0]))
        for key, relevance in matches.items():
            if relevance in ['Fully Relevant', 'Partially Relevant']:
                rel_score = ((1 if relevance == 'Fully Relevant' else 0.5) +
                             count * rel_score) / (count + 1)
                count += 1
    else:
        processed_pred = '[No Retrieval]Answer: None'
        rel_score = 0

    # 处理理由分数
    if '[Fully Reasonable]' in processed_pred:
        reason_score += 1
    elif '[Partially Reasonable]' in processed_pred:
        reason_score += 0.5

    return processed_pred, rel_score, reason_score, return_entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='/media/disk2/llama_factory/generation_0110_no_mask')
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument('--cached_embedding', action='store_false', default=True,
                        help='Whether to use cached embeddings (default: True)')
    parser.add_argument('--input_file', type=str,
                        default='./data/merged/WebQSP_test.json')
    parser.add_argument('--output_file', type=str,
                        default='./output/inference/webqsp_code_0111_res.json')
    parser.add_argument('--max_new_tokens', type=int, default=100)

    args = parser.parse_args()
    os.environ["AZURE_OPENAI_API_KEY"] = "2b219db0d2984f9dae28b651ab8ab3d9"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://smsh.openai.azure.com/"
    os.environ["AZURE_OPENAI_API_VERSION"] = "2024-03-01-preview"
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        # model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False},
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    rel_tokens = load_special_tokens(tokenizer)
    if args.world_size is not None:
        model = LLM(model=args.model_name, trust_remote_code=True,
                    tensor_parallel_size=args.world_size)
    if args.cached_embedding:
        store = RedisStore(redis_url="redis://localhost:6379")
        embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, store, namespace="bge-large"
        )
        with open('./data/clean_relations', 'r') as f:
            r_data = [line.strip() for line in f]
        all_db = FAISS.from_texts(r_data, embeddings)
    else:
        with open('./data/clean_relations', 'r') as f:
            r_data = [line.strip() for line in f]
        all_db = FAISS.from_texts(r_data, embeddings)
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens, logprobs=5, skip_special_tokens=False, include_stop_str_in_output=True)
    PROMPT_DICT = {
        "llama3": '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'}

    with open(args.input_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print("Input File Length:", len(test_data))
    prediction_trees = []
    count = 0
    correct_ids = []
    for index in range(0, len(test_data)):
        hit = 0
        print(f'Process {index}')
        data_input = test_data[index]['question']
        prompt = PROMPT_DICT['llama3'].format(input=data_input)
        max_depth = 5
        # topic_entity = list(test_data[index]['topic_entity'].keys())
        topic_entity = list(test_data[index]['gold_entity_map'].keys())
        curr_depth = 1
        # set early stop
        terminated = False
        node_id = 0
        prediction_tree = {}
        levels = {}
        prediction_tree[node_id] = {"prompt": prompt, "pred": "[New Retrieval]",
                                    "processed_pred": "", "score": None, "topic_entity": topic_entity, "parent": None, "context": ''}
        levels[0] = [0]
        while curr_depth < max_depth:
            levels[curr_depth] = []
            if curr_depth-1 in levels:
                for node in levels[curr_depth-1]:
                    curr_pred = prediction_tree[node]["pred"]
                    if "<|eot_id|>" in curr_pred or 'No Retrieval' in curr_pred:
                        continue
                    prompt = prediction_tree[node]["prompt"]
                    prev_generation = prediction_tree[node]["processed_pred"]
                    score = prediction_tree[node]["score"]
                    topic_entity = prediction_tree[node]["topic_entity"]
                    context = prediction_tree[node]['context']
                    cur_prompt = prompt + prev_generation
                    if "Retrieve Entity" in curr_pred.split('[')[-1]:
                        retrieval_results = {}
                        preds, scores, next_entities, contexts, _ = run_entity_generation_batch(
                            model, cur_prompt, topic_entity, context, sampling_params=sampling_params)
                        for i, (pred, p_score, next_topic, context) in enumerate(zip(preds, scores, next_entities, contexts)):
                            retrieval_results[i] = {
                                "pred": pred, "score": p_score, "next_topic": next_topic, "context": context}

                        for i, result in retrieval_results.items():
                            node_id += 1
                            node_score = (
                                result["score"] + score) / (curr_depth // 2)
                            pred = result["pred"]
                            next_entity = result['next_topic']
                            if len(next_entity) == 0:
                                next_entity = topic_entity
                            prediction_tree[node_id] = {"prompt": cur_prompt, "pred": pred, "context": result['context'],
                                                        "score": node_score, "parent": node,
                                                        "topic_entity": next_entity}
                            if "[Continue to Retrieve Evidence]" in pred:
                                gen_result_index = pred.index(
                                    "[Continue to Retrieve Evidence]")
                                prev_generation = pred[:gen_result_index]
                            else:
                                prev_generation = pred
                            prediction_tree[node_id]["processed_pred"] = prev_generation
                            levels[curr_depth].append(node_id)
                    # 存在前后逻辑粘连
                    if "New Retrieval" in curr_pred.split('[')[-1] or "Continue to Retrieve Evidence" in curr_pred.split('[')[-1]:
                        retrieval_results = {}
                        preds, scores, contexts = run_relation_generation_batch(
                            model, cur_prompt, new_retrieval=True if ("[New Retrieval]" in curr_pred) else False, context=context, topic_entity=topic_entity, hypo=True,
                            rel_tokens=rel_tokens, embeddings=embeddings, all_db=all_db, sampling_params=sampling_params)
                        for i, (pred, p_score, context) in enumerate(zip(preds, scores, contexts)):
                            retrieval_results[i] = {
                                "pred": pred, "score": p_score, "context": context}

                        for i, result in retrieval_results.items():
                            node_id += 1
                            node_score = result["score"] + \
                                score if score is not None else result["score"]
                            pred = result["pred"]
                            context = result["context"]
                            prediction_tree[node_id] = {"prompt": cur_prompt, "pred": pred,
                                                        "score": node_score, "parent": node,
                                                        "topic_entity": topic_entity, "context": context}
                            if "[Retrieve Entity]" in pred:
                                gen_result_index = pred.index(
                                    "[Retrieve Entity]")
                                prev_generation = pred[:gen_result_index]
                            else:
                                prev_generation = pred
                            prediction_tree[node_id]["processed_pred"] = prev_generation
                            levels[curr_depth].append(node_id)
                current_rank = levels[curr_depth]
                node2score = {
                    node_id: prediction_tree[node_id]["score"] for node_id in current_rank}
                top_nodes = sorted(node2score.items(), key=lambda x: x[1], reverse=True)[
                    :3]
                levels[curr_depth] = [node[0] for node in top_nodes]
                curr_depth += 1
            else:
                break
        prediction_trees.append({"index": index, "tree": prediction_tree})
        labels = [get_label(ans) if ans.startswith(
            'm.') else ans for ans in test_data[index]['answer']]
        for tree_node in prediction_tree.values():
            if 'Answer' in tree_node['processed_pred']:
                answer = tree_node['processed_pred'].split('Answer:')[-1]
        # max_score = 0
        # for ind, tree_node in enumerate(prediction_tree.values()):
        #     if 'Answer' in tree_node['processed_pred']:
        #         if tree_node['score'] > max_score:
        #             max_score = tree_node['score']
        #             answer = tree_node['processed_pred'].split('Answer:')[-1]
                for label in labels:
                    if label and label in answer:
                        hit = 1
        if hit == 1:
            print('Correct')
            count += 1
            correct_ids.append(index)
        if len(prediction_trees) == 20:
            print('saving logging res')
            save_to_json(prediction_trees, args.output_file)
            prediction_trees = []
    if len(prediction_trees):
        save_to_json(prediction_trees, args.output_file)
    print('============================ Output Result Analysis =============================')
    print('Test data length: ', len(test_data))
    print('All correct count: ', count)
    print('Accuracy: ', count / len(test_data))


if __name__ == "__main__":
    main()
