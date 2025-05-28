import os
import json
import random
import re
import argparse
import numpy as np
from typing import List, Dict, Set, Tuple
from langchain_openai import AzureOpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore, RedisStore
from langchain_community.vectorstores import FAISS
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from src.sparql_utils import get_1hop_relations_with_odbc, get_another_entity, get_label
from src.utils import save_to_json

def load_special_tokens(tokenizer: AutoTokenizer) -> Tuple[Dict, Dict, Dict]:
    """Load special tokens and convert to token IDs."""
    token_sets = [
        ['[Unrelevant]', '[Partially Relevant]', '[Fully Relevant]'],
        ['[Fully Reasonable]', '[Partially Reasonable]', '[Unreasonable]'],
        ['[Utility:5]', '[Utility:4]', '[Utility:3]', '[Utility:2]', '[Utility:1]']
    ]
    return tuple({token: tokenizer.convert_tokens_to_ids(token) for token in tokens} 
                 for tokens in token_sets)

def random_sample(items: List, k: int = 3) -> List:
    """Sample k items randomly from a list."""
    return random.sample(items, min(k, len(items)))

def calculate_metrics(predictions: Set, answers: List) -> Tuple[float, float, float, float]:
    """Calculate precision, recall, F1 score, and hits."""
    correct = sum(1 for entity in predictions if entity in answers)
    total = len(predictions)
    ans_len = len(answers)
    
    if ans_len == 0:
        return (1.0, 1.0, 1.0, 1.0) if total == 0 else (0.0, 1.0, 0.0, 1.0)
    
    hits = 1 if correct > 0 else 0
    if total == 0:
        return 1.0, 0.0, 0.0, hits
    
    precision = correct / total
    recall = correct / ans_len
    f1 = 2.0 / (1.0 / precision + 1.0 / recall) if precision and recall else 0.0
    return precision, recall, f1, hits

def initialize_embeddings(cached: bool, model_name: str, redis_url: str = None) -> CacheBackedEmbeddings:
    """Initialize embeddings with optional caching."""
    embeddings = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={'normalize_embeddings': False})
    if cached and redis_url:
        store = RedisStore(redis_url=redis_url)
        embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, store, namespace="bge-large")
    return embeddings

def get_retriever(topic_entities: List[str], use_1hop: bool, cached_embedder: CacheBackedEmbeddings, 
                 all_db: FAISS) -> FAISS:
    """Initialize retriever based on 1-hop relations or full database."""
    if use_1hop:
        relations = set()
        for entity in topic_entities:
            try:
                relations.update(get_1hop_relations_with_odbc(entity))
            except:
                continue
        return FAISS.from_texts(list(relations), cached_embedder).as_retriever(search_kwargs={"k": 5})
    return all_db.as_retriever(search_kwargs={"k": 5})

def process_retrieved_paragraphs(prompt: str, retriever: FAISS, context: str, 
                               llama: str) -> Set[str]:
    """Retrieve and process paragraphs from the prompt."""
    paragraphs = set()
    try:
        query = (prompt.split('\n\n')[1].split('<|eot_id|>')[0] if llama == 'llama3' 
                else prompt.split('[INST]')[1].split('[/INST]')[0].strip()) + ' ' + context
        paragraphs.update(page.page_content.strip() for page in retriever.invoke(query))
    except:
        pass
    return paragraphs

def run_relation_generation_batch(model: LLM, prompt: str, llama: str, context: str, 
                                topic_entities: List[str], hypo: bool, income: str, 
                                cached_embedder: CacheBackedEmbeddings, all_db: FAISS, 
                                sampling_params: SamplingParams, rel_tokens: Dict) -> Tuple[List, List, List]:
    """Generate relation predictions in batch."""
    retriever = get_retriever(topic_entities, use_1hop=True, cached_embedder=cached_embedder, all_db=all_db)
    paragraphs = process_retrieved_paragraphs(prompt, retriever, context, llama)
    
    if hypo:
        hypo_rel = model.generate(prompt + "[Retrieve Relation]", sampling_params)[0].outputs[0].text
        matches = dict(re.findall(r'(\w+\.\w+\.\w+)\[(.*?)\]', hypo_rel.split('[Retrieve Entity]')[0]))
        paragraphs.update(page.page_content.strip() for page in retriever.invoke(
            ' '.join(k for k, v in matches.items() if v in ['Fully Relevant', 'Partially Relevant'])))
    
    paragraphs.discard(income)
    aug_prompts = ["<paragraph>{}</paragraph>".format(';'.join(p)) 
                   for p in [list(paragraphs)[i:i+5] for i in range(0, len(paragraphs), 5)]]
    
    final_preds, scores, contexts = [], [], []
    for pred in model.generate([prompt + "[Retrieve Relation]" + aug for aug in aug_prompts], sampling_params):
        pred_text = pred.outputs[0].text
        pred_text = pred_text.split('[Retrieve Entity]')[0] + '[Retrieve Entity]' if '[Retrieve Entity]' in pred_text else pred_text
        score_dict = process_relation_prediction(pred, rel_tokens)
        final_preds.append("[Retrieve Relation]" + aug_prompts[len(final_preds)] + pred_text)
        scores.append(pred.outputs[0].cumulative_logprob / max(len(pred.outputs[0].token_ids), 1))
        contexts.append(score_dict)
    
    return final_preds, scores, contexts

def process_relation_prediction(pred, rel_tokens: Dict) -> Dict:
    """Process a single relation prediction."""
    score_dict = {}
    pred_text = pred.outputs[0].text
    pred_log_probs = pred.outputs[0].logprobs
    pred_token_ids = pred.outputs[0].token_ids
    
    matches = re.findall(r'(\w+\.\w+\.\w+)\[(.*?)\]', pred_text.split('[Retrieve Entity]')[0])
    relevance_indices = [i for i, tok in enumerate(pred_token_ids) if tok in rel_tokens.values()]
    
    for i, idx in enumerate(relevance_indices[:len(matches)]):
        rel_scores = {token: np.exp(pred_log_probs[idx][token_id].logprob 
                      if token_id in pred_log_probs[idx] else -100) 
                      for token, token_id in rel_tokens.items()}
        score_sum = sum(rel_scores.values())
        score_dict[matches[i][0]] = {
            "relevance": matches[i][1],
            "score": (rel_scores['[Fully Relevant]'] + 0.5 * rel_scores['[Partially Relevant]']) / score_sum
        }
    return score_dict

def run_entity_generation_batch(model: LLM, prompt: str, topic_entities: List[str], context: Dict, 
                              sampling_params: SamplingParams, utility_tokens: Dict, 
                              rel_tokens: Dict, reason_tokens: Dict, tokenizer: AutoTokenizer) -> Tuple[List, List, List, List, Dict, List]:
    """Generate entity predictions in batch."""
    entity_prompts, income_rel, overall_scores, name_to_id = [], [], {}, {}
    effective_count = 0
    
    for entity in topic_entities:
        for key, relevance in context.items():
            if relevance['relevance'] not in ['Fully Relevant', 'Partially Relevant']:
                continue
            try:
                related_entities = get_another_entity(entity, key, return_label=True)
            except:
                continue
                
            if related_entities:
                income_rel.append(key)
                name_to_id[effective_count] = related_entities
                overall_scores[effective_count] = {'r_relevance': relevance['score']}
                entities = [f'({get_label(entity)}, {key}, {rel_entity})' 
                           for rel_entity in related_entities.keys()][:5]
                overall_scores[effective_count]['path'] = entities
                entity_prompts.append(f"<paragraph>{';'.join(entities)}</paragraph>")
                effective_count += 1
    
    predictions = model.generate([f"{prompt}[Retrieve Entity]{ep}" for ep in entity_prompts], sampling_params)
    final_preds, final_entities, final_contexts, scores = [], [], [], []
    
    for idx, pred in enumerate(predictions):
        pred_output = pred.outputs[0]
        processed_pred, rel_score, reason_score, utility_score, return_entities, matches, rationality, utility = process_prediction(
            idx, pred_output.text, pred_output.logprobs, pred_output.token_ids, 
            r'(.+?)\[(.*?)\]', name_to_id, utility_tokens, rel_tokens, reason_tokens, tokenizer
        )
        
        overall_scores[idx].update({
            "r_match": {income_rel[idx]: context[income_rel[idx]]},
            "e_match": matches,
            "reason_score": reason_score,
            "rationality": rationality,
            "utility": utility,
            "utility_score": utility_score,
            "seq_score": pred_output.cumulative_logprob / max(len(pred_output.token_ids), 1),
            "final_score": rel_score * overall_scores[idx]['r_relevance'] * reason_score
        })
        
        final_preds.append(f"[Retrieve Entity]{entity_prompts[idx]}{processed_pred}")
        final_entities.append(list(return_entities.values()))
        final_contexts.append(' '.join(return_entities.keys()))
        scores.append(overall_scores[idx]['final_score'])
    
    return final_preds, scores, final_entities, final_contexts, overall_scores, income_rel

def process_prediction(idx: int, pred_text: str, pred_log_probs: List, pred_token_ids: List, 
                      pattern: str, name_to_id: Dict, utility_tokens: Dict, 
                      rel_tokens: Dict, reason_tokens: Dict, tokenizer: AutoTokenizer) -> Tuple:
    """Process a single entity prediction."""
    rel_score, reason_score, utility_score = 0, 0, 0
    rationality, utility = '', ''
    return_entities = {}
    
    relevance_indices = [i for i, tok in enumerate(pred_token_ids) if tok in rel_tokens.values()]
    reason_indices = [i for i, tok in enumerate(pred_token_ids) if tok in reason_tokens.values()]
    utility_indices = [i for i, tok in enumerate(pred_token_ids) if tok in utility_tokens.values()]
    
    if '[Retrieve Relation]' in pred_text:
        processed_pred = pred_text.split('[Retrieve Relation]')[0].strip() + '[Retrieve Relation]'
        matches = dict(re.findall(pattern, processed_pred))
        for key, relevance in matches.items():
            if relevance.strip() in ['Fully Relevant', 'Partially Relevant']:
                if key in name_to_id[idx]:
                    return_entities[key] = name_to_id[idx][key]
                elif key.strip() == 'Unknown Entity':
                    for random_key in random_sample(list(name_to_id[idx].keys())):
                        return_entities[random_key] = name_to_id[idx][random_key]
    elif '[No Retrieval]' in pred_text:
        processed_pred = pred_text
        matches = dict(re.findall(pattern, pred_text.split('[No Retrieval]')[0]))
        if utility_indices:
            uti_scores = {token: np.exp(pred_log_probs[utility_indices[0]][token_id].logprob 
                         if token_id in pred_log_probs[utility_indices[0]] else -100) 
                         for token, token_id in utility_tokens.items()}
            ut_sum = sum(uti_scores.values())
            if len(uti_scores) == 5:
                utility_score = sum(i * (uti_scores[f"[Utility:{i}]"] / ut_sum) 
                                  for i in range(1, 6)) / 5
            utility = tokenizer.convert_ids_to_tokens(pred_token_ids[utility_indices[0]])
    else:
        matches = {}
        processed_pred = '[No Retrieval]Answer: None'
    
    if relevance_indices:
        rel_scores = {token: np.exp(pred_log_probs[idx][token_id].logprob 
                     if token_id in pred_log_probs[idx] else -100) 
                     for token, token_id in rel_tokens.items()}
        rl_sum = sum(rel_scores.values())
        rel_score = max((rel_scores['[Fully Relevant]'] + 0.5 * rel_scores['[Partially Relevant]']) / rl_sum, rel_score)
    
    if reason_indices:
        reason_scores = {token: np.exp(pred_log_probs[reason_indices[0]][token_id].logprob 
                        if token_id in pred_log_probs[reason_indices[0]] else -100) 
                        for token, token_id in reason_tokens.items()}
        rs_sum = sum(reason_scores.values())
        reason_score = (reason_scores['[Fully Reasonable]'] + 0.5 * reason_scores['[Partially Reasonable]']) / rs_sum
        rationality = tokenizer.convert_ids_to_tokens(pred_token_ids[reason_indices[0]])
    
    return processed_pred, rel_score, reason_score, utility_score, return_entities, matches, rationality, utility

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='/media/disk2/llama_factory/generation_0202_uti_no_mask/')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--cached_embedding', action='store_false', default=True)
    parser.add_argument('--input_file', type=str, default='./data/merged/WebQSP_test.json')
    parser.add_argument('--output_file', type=str, default='./output/inference/webqsp_code_0111_res.json')
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--save_output', action='store_true')
    args = parser.parse_args()

    # Initialize model and resources
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    rel_tokens, reason_tokens, utility_tokens = load_special_tokens(tokenizer)
    model = LLM(model=args.model_name, trust_remote_code=True, tensor_parallel_size=args.world_size)
    embeddings = initialize_embeddings(args.cached_embedding, "BAAI/bge-large-en-v1.5", "redis://localhost:6379")
    
    with open('./data/clean_relations', 'r') as f:
        all_db = FAISS.from_texts([line.strip() for line in f], embeddings)
    
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens, 
                                  logprobs=5, skip_special_tokens=False, include_stop_str_in_output=True)
    PROMPT_DICT = {
        "llama3": '<|begin_of_text|><|start_header_id|>user<|end_header_id>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id>',
        "llama2": "<s>[INST] {input} [/INST]"
    }

    with open(args.input_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    precisions, recalls, f1s, hits, logging_res = [], [], [], [], []
    
    for index in range(min(10, len(test_data))):
        print(f'Processing {index}')
        data_input = test_data[index]['question']
        prompt = PROMPT_DICT['llama3'].format(input=data_input)
        topic_entities = list(test_data[index]['gold_entity_map'].keys())
        prediction_tree = {
            0: {
                "prompt": prompt, "pred": "[Retrieve Relation]", "processed_pred": "",
                "score": None, "topic_entity": topic_entities, "parent": None,
                "context": "", "income": "", "depth": 0
            }
        }
        levels = {0: [0]}
        curr_depth = 1
        node_id = 0
        max_depth = 5

        while curr_depth < max_depth:
            levels[curr_depth] = []
            for node in levels.get(curr_depth-1, []):
                node_data = prediction_tree[node]
                if "<|eot_id|>" in node_data["pred"] or 'No Retrieval' in node_data["pred"]:
                    continue
                
                cur_prompt = node_data["prompt"] + node_data["processed_pred"]
                score = node_data["score"] or 0
                
                if "Retrieve Entity" in node_data["pred"].split('[')[-1]:
                    preds, scores, next_entities, contexts, overall_scores, income_rel = run_entity_generation_batch(
                        model, cur_prompt, node_data["topic_entity"], node_data["context"],
                        sampling_params, utility_tokens, rel_tokens, reason_tokens, tokenizer
                    )
                    for i, (pred, p_score, next_topic, context, rel) in enumerate(zip(preds, scores, next_entities, contexts, income_rel)):
                        node_id += 1
                        node_score = (p_score + score) / (curr_depth // 2 + 1)
                        next_entity = next_topic if next_topic else node_data["topic_entity"]
                        prediction_tree[node_id] = {
                            "prompt": cur_prompt, "pred": pred, "context": context,
                            "score": node_score, "parent": node, "topic_entity": next_entity,
                            "income": rel, "verbose": {
                                "path": overall_scores[i]['path'], "e_match": overall_scores[i]['e_match'],
                                "r_match": overall_scores[i]['r_match'], "reason_score": overall_scores[i]['reason_score'],
                                "rationality": overall_scores[i]['rationality'], "utility": overall_scores[i]['utility'],
                                "utility_score": overall_scores[i]['utility_score'], "seq_score": overall_scores[i]['seq_score']
                            }, "depth": curr_depth
                        }
                        prediction_tree[node_id]["processed_pred"] = pred.split("[Retrieve Relation]")[0] if "[Retrieve Relation]" in pred else pred
                        levels[curr_depth].append(node_id)
                
                if "Retrieve Relation" in node_data["pred"].split('[')[-1]:
                    preds, scores, contexts = run_relation_generation_batch(
                        model, cur_prompt, 'llama3', node_data["context"], node_data["topic_entity"],
                        hypo=False, income=node_data["income"], cached_embedder=embeddings, 
                        all_db=all_db, sampling_params=sampling_params, rel_tokens=rel_tokens
                    )
                    for i, (pred, p_score, context) in enumerate(zip(preds, scores, contexts)):
                        node_id += 1
                        node_score = p_score + score if score is not None else p_score
                        prediction_tree[node_id] = {
                            "prompt": cur_prompt, "pred": pred, "score": node_score,
                            "parent": node, "topic_entity": node_data["topic_entity"],
                            "context": context, "income": node_data["income"], "depth": curr_depth
                        }
                        prediction_tree[node_id]["processed_pred"] = pred.split("[Retrieve Entity]")[0] if "[Retrieve Entity]" in pred else pred
                        levels[curr_depth].append(node_id)
            
            if not levels[curr_depth]:
                break
            levels[curr_depth] = [node for node, _ in sorted(
                [(n, prediction_tree[n]["score"]) for n in levels[curr_depth]],
                key=lambda x: x[1], reverse=True)][:3]
            curr_depth += 1

        labels = [get_label(ans) if ans.startswith('m.') else ans for ans in test_data[index]['answer']]
        end_nodes = [n for n, node in prediction_tree.items() if 'Answer' in node['processed_pred']]
        
        queues = []
        for n_ind in end_nodes:
            queue = []
            node = prediction_tree[n_ind]
            answer = node['processed_pred'].split('Answer:')[-1]
            score = node['score'] + (0.5 * (node['verbose']['utility_score'] + 1) + node['verbose']['seq_score']
                                   if 'verbose' in node else 0)
            while node:
                if 'verbose' in node:
                    queue.append(node['verbose'])
                node = prediction_tree[node['parent']] if node['parent'] is not None else None
            queues.append({'verbose': queue, "answer": answer, "score": score})
        
        answers = set()
        for q in sorted(queues, key=lambda x: x['score'], reverse=True)[:3]:
            candidates = q['answer'].strip().split(';')
            answers.update(re.findall(r'(.+?)\[(.*?)\]', c)[0][0] for c in candidates 
                          if re.findall(r'(.+?)\[(.*?)\]', c))
        
        precision, recall, f1, hit = calculate_metrics(answers, labels)
        if hit == 1:
            print(f'Correct {index}')
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        hits.append(hit)
        logging_res.append({"index": index, "tree": prediction_tree})
        
        if len(logging_res) == 20:
            save_to_json(logging_res, './output/inference/cwq_test_top3_0214_beam3_k5_detph9_wohyper2.json')
            logging_res = []

    print('============================ Output Result Analysis =============================')
    print(f'Test data length: {len(test_data)}')
    print(f'Average F1: {np.mean(f1s):.4f}')
    print(f'Average Hits: {np.mean(hits):.4f}')

if __name__ == "__main__":
    main()