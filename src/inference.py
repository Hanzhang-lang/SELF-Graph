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
def load_special_tokens(tokenizer, use_grounding=False, use_utility=False):
    rel_tokens = {}
    for token in ['[Unrelevant]','[Partially Relevant]','[Fully Relevant]']:
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
                candidate_relations.extend(get_1hop_relations_with_odbc(entity))
            except:
                continue
        if len(candidate_relations):
            vec_db = FAISS.from_texts(candidate_relations, embeddings)
        else:
            vec_db = all_db
        retriever = vec_db.as_retriever(search_kwargs={"k": 5})
    else:
        retriever = all_db.as_retriever(search_kwargs={"k": 5})
    paragraph = [page.page_content.strip() for page in retriever.invoke(prompt.split('\n\n')[1].split('<|eot_id|>')[0] + ' '+ context)]
    if hypo:
        hypo_rel = model.generate(prompt + retrieval_token, sampling_params)[0].outputs[0].text
        pattern = r'(\w+\.\w+\.\w+)\[(.*?)\]'
        if '[Retrieve Entity]' in hypo_rel:
            hypo_rel = hypo_rel.split('[Retrieve Entity]')[0]
        matches =  dict(re.findall(pattern, hypo_rel))
        string = ''
        for k,v in matches.items():
            if v in ['  ', 'Partially Relevant']:
                string += k + ' '
        for extra_rel in retriever.invoke(string):
            if extra_rel.page_content.strip() not in paragraph:
                paragraph.append(extra_rel.page_content.strip())
    
    aug_prompts =  ["<paragraph>{}</paragraph>".format(';'.join(p))  for p in [paragraph[i: i+5] for i in range(0, len(paragraph), 5)]]
    
    preds = model.generate([prompt + retrieval_token + aug for aug in aug_prompts], sampling_params)
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
        relevance_score = rel_score_dict['[Fully Relevant]']+ rel_score_dict['[Partially Relevant]'] / np.sum(list(rel_score_dict.values()))
        if '[Retrieve Entity]' in pred_text_1:
            processed_pred = pred_text_1.split('[Retrieve Entity]')[0] + '[Retrieve Entity]'
        else:
            processed_pred = pred_text_1
        final_preds.append(retrieval_token + aug_prompts[p_id] + processed_pred)
        overall_scores.append(relevance_score)
        final_contexts.append(pred_text_1.split('[Retrieve Entity]')[0])

    return final_preds, overall_scores, final_contexts

def run_entity_generation_batch(model, prompt, topic_entity, context, score_type='hard', sampling_params=None):
    final_preds = []
    overall_scores = {}
    final_entities = []
    final_context = []
    pattern = r'(.*?)\[(.*?)\]'
    matches =  dict(re.findall(pattern,context))
    name2id = {}
    effective_count = 0
    entity_prompts = []
    for _, entity in enumerate(topic_entity):
        for k, v in matches.items():
            if v in ['Fully Relevant', 'Partially Relevant']:
                entities = []
                try:
                    another_entities = get_another_entity(entity, k, return_label=True)
                except:
                    another_entities = []
                # handle Unkown mid entities
                # whether mix another entities
                if len(another_entities):
                    name2id.setdefault(effective_count, {})
                    name2id[effective_count].update(another_entities)
                    effective_count += 1
                    entities.extend([f'({get_label(entity)}, {k}, {e})' for e in another_entities.keys()])
                    entity_prompts.append("<paragraph>{}</paragraph>".format(';'.join(entities[:5])))
    # print(aug_prompts)
    preds = model.generate([prompt+  '[Retrieve Entity]' + entity_prompts[i] for i in range(len(entity_prompts))], sampling_params)
    
    for p_idx, pred in enumerate(preds):
        return_entities = dict()
        pred_token_ids = pred.outputs[0].token_ids
        pred_text_2 = pred.outputs[0].text
        pred_log_probs = pred.outputs[0].logprobs
        # hard decode 分数
        rel_score = 0
        # rel_score_dict = {}
        # relevance_indices = []
        # for tok_idx, tok in enumerate(pred_token_ids):
        #     if tok in rel_tokens.values():
        #         relevance_indices.append(tok_idx)
        # if len(relevance_indices) > 0:
        #     # print(relevance_indices)
        #     for idx in relevance_indices:
        #         for token, token_id in rel_tokens.items():
        #             prob = pred_log_probs[idx][token_id].logprob if token_id in pred_log_probs[idx] else -100
        #             rel_score_dict[token] = np.exp(prob)
        # if len(rel_score_dict) == 3:
        #     overall_scores[p_idx] = rel_score_dict['[Fully Relevant]'] + rel_score_dict['[Partially Relevant]']/ np.sum(list(rel_score_dict.values()))
        if '[Continue to Retrieve Evidence]' in pred_text_2:
            processed_pred = pred_text_2.split('[Continue to Retrieve Evidence]')[0]
            matches =  dict(re.findall(pattern, processed_pred))
            for k, v in matches.items():
                if v in ['Fully Relevant', 'Partially Relevant']:
                    if k in name2id[p_idx]:
                        return_entities[k] = name2id[p_idx][k]
                    elif k == 'Unknown Entity':
                        random_key = random.choice(list(name2id[p_idx].keys()))
                        return_entities[random_key] = name2id[p_idx][random_key]
                    rel_score += 1 if v =='Fully Relevant' else 0.5
            processed_pred += '[Continue to Retrieve Evidence]'

        elif '[No Retrieval]' in pred_text_2:
            processed_pred = pred_text_2
            rel_score += 100
        else:
            processed_pred = '[No Retrieval]'
        overall_scores[p_idx] = rel_score
        final_preds.append('[Retrieve Entity]' + entity_prompts[p_idx]+processed_pred)
        final_entities.append(list(return_entities.values()))
        final_context.append(' '.join(return_entities.keys()))
    return final_preds, [overall_scores[p_idx] for p_idx in overall_scores], final_entities, final_context
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='/media/disk2/llama_factory/generation_0110_no_mask')
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument('--cached_embedding', action='store_false', default=True,
                    help='Whether to use cached embeddings (default: True)')
    parser.add_argument('--input_file', type=str, default='./data/merged/WebQSP_test.json')
    parser.add_argument('--output_file', type=str, default='data/test_output.json')
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
        model = LLM(model=args.model_name, trust_remote_code=True, tensor_parallel_size=args.world_size)
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
            temperature=0.01, top_p=1.0,max_tokens=args.max_new_tokens, logprobs=5, skip_special_tokens=False, include_stop_str_in_output=True)
    PROMPT_DICT = {"llama3": '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'}
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print("Input File Length:", len(test_data))

    count = 0
    correct_ids = []
    for index in range(0, len(test_data)):
        hit = 0
        print(f'Process {index}')
        data_input = test_data[index]['question']
        prompt = PROMPT_DICT['llama3'].format(input= data_input)
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
                    if "<|eot_id|>" in curr_pred:
                        continue
                    prompt = prediction_tree[node]["prompt"]
                    prev_generation = prediction_tree[node]["processed_pred"]
                    score = prediction_tree[node]["score"]
                    topic_entity = prediction_tree[node]["topic_entity"]
                    context = prediction_tree[node]['context']
                    cur_prompt = prompt + prev_generation
                    if "Retrieve Entity" in curr_pred.split('[')[-1]:
                        retrieval_results = {}
                        preds, scores, next_entities, contexts = run_entity_generation_batch(
                            model, cur_prompt, topic_entity, context, sampling_params=sampling_params)
                        for i, (pred, p_score,next_topic, context) in enumerate(zip(preds, scores, next_entities, contexts)):
                            retrieval_results[i] = {
                                "pred": pred, "score": p_score, "next_topic": next_topic, "context": context}

                        for i, result in retrieval_results.items():
                            node_id += 1
                            node_score = result["score"] * \
                                score if score is not None else result["score"]
                            pred = result["pred"]
                            next_entity = result['next_topic']
                            if len(next_entity) == 0:
                                next_entity = topic_entity
                            prediction_tree[node_id] = {"prompt": cur_prompt, "pred": pred, "context": result['context'],
                                                        "score": node_score, "parent": node,
                                                        "topic_entity": next_entity}
                            if "[Continue to Retrieve Evidence]" in pred:
                                gen_result_index = pred.index("[Continue to Retrieve Evidence]")
                                prev_generation = pred[:gen_result_index]
                            else:
                                prev_generation = pred
                            prediction_tree[node_id]["processed_pred"] = prev_generation
                            levels[curr_depth].append(node_id)
                    #存在前后逻辑粘连   
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
                            node_score = result["score"] * \
                                score if score is not None else result["score"]
                            pred = result["pred"]
                            context = result["context"]
                            prediction_tree[node_id] = {"prompt": cur_prompt, "pred": pred,
                                                        "score": node_score, "parent": node,
                                                        "topic_entity": topic_entity, "context": context}
                            if "[Retrieve Entity]" in pred:
                                gen_result_index = pred.index("[Retrieve Entity]")
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
        labels = [get_label(ans) if ans.startswith('m.') else ans for ans in test_data[index]['answer']]
        # labels = [ans['entity_name'] for ans in test_data[index]['answer']]
        # print(labels)
        for tree_node in prediction_tree.values():
            if 'Answer' in tree_node['processed_pred']:
                answer = tree_node['processed_pred'].split('Answer:')[-1]
                for label in labels:
                    if label and label in answer:
                        hit = 1
        if hit == 1:
            print('Correct')
            count += 1
            correct_ids.append(index)
            break

if __name__ == "__main__":
    main()