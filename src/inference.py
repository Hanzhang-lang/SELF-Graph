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



def load_special_tokens(tokenizer):
    """加载特殊标记并转换为对应的token ID。"""
    rel_tokens = {token: tokenizer.convert_tokens_to_ids(token) for token in ['[Unrelevant]', '[Partially Relevant]', '[Fully Relevant]']}
    reason_tokens = {token: tokenizer.convert_tokens_to_ids(token) for token in ['[Fully Reasonable]', '[Partially Reasonable]', '[Unreasonable]']}
    ut_tokens = {token: tokenizer.convert_tokens_to_ids(token) for token in ['[Utility:5]', '[Utility:4]', '[Utility:3]', '[Utility:2]', '[Utility:1]']}
    return rel_tokens, reason_tokens, ut_tokens


def random_sample(lst, k=3):
    return random.sample(lst, min(k, len(lst)))

def cal_eval_metric(preds, answers):
    correct, total = 0.0, 0.0
    for entity in preds:
        if entity in answers:
            correct += 1
        total += 1
    if len(answers) == 0:
        if total == 0:
            return 1.0, 1.0, 1.0, 1.0 # precision, recall, f1, hits
        else:
            return 0.0, 1.0, 0.0, 1.0 # precision, recall, f1, hits
    if correct != 0:
        hits = 1
    else: 
        hits = 0
    if total == 0:
        return 1.0, 0.0, 0.0, hits # precision, recall, f1, hits
    else:
        precision, recall = correct / total, correct / len(answers)
        f1 = 2.0 / (1.0 / precision + 1.0 / recall) if precision != 0 and recall != 0 else 0.0
        return precision, recall, f1, hits



def run_relation_generation_batch(model, prompt, context, topic_entity, hypo=True, use_1hop=True, income=None, cached_embedder=None, all_db=None, sampling_params=None, rel_tokens=None ):
    # 初始化关系得分字典、最终预测列表、总体得分列表、最终上下文列表、段落集合
    final_preds = []
    overall_scores = []
    final_contexts = []
    paragraph = set()
    # 如果是新检索，则设置检索标记为"[Retrieve Relation]"，否则设置为"[Retrieve Relation]"
    
    retrieval_token = "[Retrieve Relation]"
    # 如果使用1跳关系，则获取1跳关系，否则使用所有关系
    if use_1hop:
        candidate_relations = set()
        for entity in topic_entity:
            try:
                candidate_relations.update(get_1hop_relations_with_odbc(entity))
            except:
                continue
        if len(list(candidate_relations)):
            vec_db = FAISS.from_texts(list(candidate_relations), cached_embedder)
        else:
            vec_db = all_db
        retriever = vec_db.as_retriever(search_kwargs={"k": 5})
    else:
        retriever = all_db.as_retriever(search_kwargs={"k": 5})
    # 获取段落内容
    try:
        paragraph.update([page.page_content.strip() for page in retriever.invoke(prompt.split('\n\n')[1].split('<|eot_id|>')[0] + ' '+ context)])
    except:
        pass
    # 如果使用假设，则生成假设关系
    if hypo:
        hypo_rel = model.generate(prompt + retrieval_token, sampling_params)[0].outputs[0].text
        pattern = r'(\w+\.\w+\.\w+)\[(.*?)\]'
        if '[Retrieve Entity]' in hypo_rel:
            hypo_rel = hypo_rel.split('[Retrieve Entity]')[0]
        matches =  dict(re.findall(pattern, hypo_rel))
        string = ''
        for k,v in matches.items():
            if v in ['Fully Relevant', 'Partially Relevant']:
                string += k + ' '
        paragraph.update([page.page_content.strip() for page in retriever.invoke(string)])
    paragraph.discard(income)
    aug_prompts =  ["<paragraph>{}</paragraph>".format(';'.join(p))  for p in [list(paragraph)[i: i+5] for i in range(0, len(paragraph), 5)]]
    
    preds = model.generate([prompt + retrieval_token + aug for aug in aug_prompts], sampling_params)
    for p_id, pred in enumerate(preds):
        score_dict = dict()
        pred_token_ids = pred.outputs[0].token_ids
        pred_text_1 = pred.outputs[0].text
        pred_log_probs = pred.outputs[0].logprobs
        seq_score = pred.outputs[0].cumulative_logprob / \
            max(len(pred.outputs[0].token_ids), 1)
        if '[Retrieve Entity]' in pred_text_1:
            processed_pred = pred_text_1.split('[Retrieve Entity]')[0] + '[Retrieve Entity]'
        else:
            processed_pred = pred_text_1

        context_pattern = r'(\w+\.\w+\.\w+)\[(.*?)\]'
        context_matches = re.findall(context_pattern, pred_text_1.split('[Retrieve Entity]')[0])
        relevance_indices = []
        for tok_idx, tok in enumerate(pred_token_ids):
            if tok in rel_tokens.values():
                relevance_indices.append(tok_idx)
        if len(relevance_indices) > 0:
            for i, idx in enumerate(relevance_indices[:len(context_matches)]):
                rel_score_dict = {}
                for token, token_id in rel_tokens.items():
                    prob = pred_log_probs[idx][token_id].logprob if token_id in pred_log_probs[idx] else -100
                    rel_score_dict[token] = np.exp(prob)
                score_dict[context_matches[i][0]] = {"relevance":context_matches[i][1],  "score":(rel_score_dict['[Fully Relevant]']+ 0.5 * rel_score_dict['[Partially Relevant]']) / np.sum(list(rel_score_dict.values()))}
        
        final_preds.append(retrieval_token + aug_prompts[p_id] + processed_pred)
        overall_scores.append(0)
        final_contexts.append(score_dict)

    return final_preds, overall_scores, final_contexts


def run_entity_generation_batch(model, prompt, topic_entities, context, sampling_params, utility_tokens, rel_tokens, reason_tokens, ):
    # 初始化变量
    final_predictions, final_entities, final_contexts = [], [], []
    entity_prompts, income_rel = [], []
    overall_scores, name_to_id = {}, {}
    effective_count = 0
    context_pattern = r'(.+?)\[(.*?)\]'
    # 构建实体提示
    for entity in topic_entities:
        for key, relevance in context.items():
            if relevance['relevance'] in ['Fully Relevant', 'Partially Relevant']:
                try:
                    related_entities = get_another_entity(entity, key, return_label=True)
                except Exception:
                    related_entities = {}

                if related_entities:
                    # 初始化分数和实体映射
                    income_rel.append(key)
                    name_to_id[effective_count] = related_entities
                    overall_scores[effective_count] = {
                        'r_relevance': relevance['score']
                    }
                    # 构造实体提示
                    entities = [
                        f'({get_label(entity)}, {key}, {rel_entity})'
                        for rel_entity in related_entities.keys()
                    ]
                    overall_scores[effective_count]['path'] = entities[:5]
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
        seq_score = pred_output.cumulative_logprob / max(len(pred_token_ids), 1)

        processed_pred, rel_score, reason_score, utility_score, return_entities, matches = process_prediction(
            idx, pred_text, pred_log_probs, pred_token_ids, context_pattern, name_to_id, overall_scores, utility_tokens, rel_tokens, reason_tokens,
        )

        # 更新分数和最终结果
        overall_scores[idx].update({
            "r_match": {income_rel[idx]: context[income_rel[idx]]},
            "e_match": matches,
            "reason_score": reason_score,
            "utility_score": utility_score,
            "seq_score": seq_score,
            "final_score":  rel_score * overall_scores[idx]['r_relevance'] * reason_score
        })

        final_predictions.append(f"[Retrieve Entity]{entity_prompts[idx]}{processed_pred}")
        final_entities.append(list(return_entities.values()))
        final_contexts.append(' '.join(return_entities.keys()))

    # 返回最终结果
    return final_predictions, [
        overall_scores[idx]['final_score'] for idx in overall_scores
    ], final_entities, final_contexts, overall_scores, income_rel


def process_prediction(idx, pred_text, pred_log_probs, pred_token_ids,pattern, name_to_id, overall_scores, utility_tokens, rel_tokens, reason_tokens,):
    """处理单条预测结果，计算相关性分数和返回实体。"""
    rel_score = 0
    reason_score = 0
    utility_score = 0
    return_entities = {}
    count = 0
    relevance_indices = []
    reason_indices = []
    utility_indices = []
    
    if '[Retrieve Relation]' in pred_text:
        processed_pred = pred_text.split('[Retrieve Relation]')[0]
        matches = dict(re.findall(pattern, processed_pred))
        for key, relevance in matches.items():
            if relevance in ['Fully Relevant', 'Partially Relevant']:
                if key in name_to_id[idx]:
                    return_entities[key] = name_to_id[idx][key]
                elif key == 'Unknown Entity':
                    # random_key = random.choice(list(name_to_id[idx].keys()))
                    random_keys = random_sample(list(name_to_id[idx].keys()))
                    for random_key in random_keys:
                        return_entities[random_key] = name_to_id[idx][random_key]
        processed_pred += '[Retrieve Relation]'

    elif '[No Retrieval]' in pred_text:
        processed_pred = pred_text
        matches = dict(re.findall(pattern, pred_text.split('[No Retrieval]')[0]))
        for tok_idx, tok in enumerate(pred_token_ids):
            if tok in utility_tokens.values():
                utility_indices.append(tok_idx)
        if len(utility_indices):
            uti_score_dict = {}
            for token, token_id in utility_tokens.items():
                prob = pred_log_probs[utility_indices[0]][token_id].logprob if token_id in pred_log_probs[utility_indices[0]] else -100
                uti_score_dict[token] = np.exp(prob)
            if len(uti_score_dict) == 5:
                ut_sum = np.sum(list(uti_score_dict.values()))
                ut_scores = [-1, -0.5, 0, 0.5, 1]
                utility_score = np.sum([ut_scores[i] * (uti_score_dict["[Utility:{}]".format(i+1)] / ut_sum)
                                    if "[Utility:{}]".format(i+1) in uti_score_dict else 0.0 for i in range(0, 5)])
    else:
        matches = dict()
        processed_pred = '[No Retrieval]Answer: None'
        rel_score = 0
    for tok_idx, tok in enumerate(pred_token_ids):
        if tok in rel_tokens.values():
            relevance_indices.append(tok_idx)
        if tok in reason_tokens.values():
            reason_indices.append(tok_idx)
    if len(relevance_indices) > 0:
        for i, idx in enumerate(relevance_indices[:len(matches)]):
            rel_score_dict = {}
            for token, token_id in rel_tokens.items():
                prob = pred_log_probs[idx][token_id].logprob if token_id in pred_log_probs[idx] else -100
                rel_score_dict[token] = np.exp(prob)
            if len(rel_score_dict) == 3:
                rl_sum =  np.sum(list(rel_score_dict.values()))
            rel_score = max((rel_score_dict['[Fully Relevant]'] / rl_sum)+ 0.5* (rel_score_dict['[Partially Relevant]'] / rl_sum), rel_score)
    if len(reason_indices): 
        reason_score_dict = {}
        for token, token_id in reason_tokens.items():
            prob = pred_log_probs[reason_indices[0]][token_id].logprob if token_id in pred_log_probs[reason_indices[0]] else -100
            reason_score_dict[token] = np.exp(prob)
        if len(reason_score_dict) == 3:
            rs_sum = np.sum(list(reason_score_dict.values()))
            reason_score = (reason_score_dict['[Fully Reasonable]'] / rs_sum) + 0.5 * (reason_score_dict['[Partially Reasonable]'] / rs_sum)

    return processed_pred, rel_score, reason_score, utility_score, return_entities, matches

def initialize_embeddings(cached, model_name, redis_url=None):
    """初始化嵌入模型和数据库。"""
    embeddings = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={'normalize_embeddings': False})
    if cached:
        store = RedisStore(redis_url=redis_url) if redis_url else None
        embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, store, namespace="bge-large")
    return embeddings



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='/media/disk2/llama_factory/generation_0202_uti_no_mask/')
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument('--cached_embedding', action='store_false', default=True,
                        help='Whether to use cached embeddings (default: True)')
    parser.add_argument('--input_file', type=str,
                        default='./data/merged/WebQSP_test.json')
    parser.add_argument('--output_file', type=str,
                        default='./output/inference/webqsp_code_0111_res.json')
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--save_output', action='store_true')

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
    rel_tokens, reason_tokens, utility_tokens = load_special_tokens(tokenizer)
    if args.world_size is not None:
        model = LLM(model=args.model_name, trust_remote_code=True,
                    tensor_parallel_size=args.world_size)
    with open('./data/clean_relations', 'r') as f:
        r_data = [line.strip() for line in f]
    embeddings = initialize_embeddings(args.cached_embedding, "BAAI/bge-large-en-v1.5", redis_url="redis://localhost:6379")
    all_db = FAISS.from_texts(r_data, embeddings)
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens, logprobs=5, skip_special_tokens=False, include_stop_str_in_output=True)
    PROMPT_DICT = {
        "llama3": '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'}

    with open(args.input_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print("Input File Length:", len(test_data))
    count = 0
    precisions = []
    recalls = []
    f1s = []
    hitss = []
    logging_res = []
    logging_res = []
    context_pattern = r'(.+?)\[(.*?)\]'
    for index in range(0, 10):
        print(f'Process {index}')
        data_input = test_data[index]['question']
        prompt = PROMPT_DICT['llama3'].format(input= data_input)
        max_depth = 9
        # topic_entity = list(test_data[index]['topic_entity'].keys())
        topic_entity = list(test_data[index]['gold_entity_map'].keys())
        curr_depth = 1
        node_id = 0
        prediction_tree = {
            node_id: {
                "prompt": prompt,
                "pred": "[Retrieve Relation]",
                "processed_pred": "",
                "score": None,
                "topic_entity": topic_entity,
                "parent": None,
                "context": "",
                "income": ""
            }
        }
        levels = {0: [node_id]}
        while curr_depth < max_depth:
            levels[curr_depth] = []
            if curr_depth-1 in levels:
                for node in levels[curr_depth-1]:
                    curr_pred = prediction_tree[node]["pred"]
                    if "<|eot_id|>" in curr_pred or 'No Retrieval' in curr_pred:
                        continue
                    cur_prompt = prediction_tree[node]["prompt"] + prediction_tree[node]["processed_pred"]
                    score = prediction_tree[node]["score"] or 0
                    topic_entity = prediction_tree[node]["topic_entity"]
                    context = prediction_tree[node]["context"]
                    income_rel = prediction_tree[node]["income"]
                    if "Retrieve Entity" in curr_pred.split('[')[-1]:
                        retrieval_results = {}
                        preds, scores, next_entities, contexts, overall_scores, income_rel = run_entity_generation_batch(
                            model, cur_prompt, topic_entity, context, sampling_params, utility_tokens, rel_tokens, reason_tokens,)
                        for i, (pred, p_score,next_topic, context, rel) in enumerate(zip(preds, scores, next_entities, contexts,  income_rel)):
                            retrieval_results[i] = {
                                "pred": pred, "score": p_score, "next_topic": next_topic, "context": context, "income": rel, "verbose": {"path": overall_scores[i]['path'], "e_match": overall_scores[i]['e_match'], "r_match": overall_scores[i]['r_match'], 'reason_score': overall_scores[i]['reason_score'], 'utility_score': overall_scores[i]['utility_score'], 'seq_score': overall_scores[i]['seq_score']}}
                        for i, result in retrieval_results.items():
                            node_id += 1
                            node_score = (result["score"] + score) / (curr_depth // 2)
                            pred = result["pred"]
                            next_entity = result['next_topic']
                            if len(next_entity) == 0:
                                next_entity = topic_entity  
                            prediction_tree[node_id] = {"prompt": cur_prompt, "pred": pred, "context": result['context'],
                                                        "score": node_score, "parent": node,
                                                        "topic_entity": next_entity, "income": result['income'], "verbose": result['verbose'],'depth': curr_depth}
                            if "[Retrieve Relation]" in pred:
                                gen_result_index = pred.index("[Retrieve Relation]")
                                prev_generation = pred[:gen_result_index]
                            else:
                                prev_generation = pred
                            prediction_tree[node_id]["processed_pred"] = prev_generation
                            levels[curr_depth].append(node_id)
                    #存在前后逻辑粘连   
                    if "Retrieve Relation" in curr_pred.split('[')[-1]:
                        retrieval_results = {}
                        preds, scores, contexts = run_relation_generation_batch(
                            model, cur_prompt, context=context, topic_entity=topic_entity, hypo=False, income=income_rel, cached_embedder = embeddings, all_db=all_db, sampling_params= sampling_params, rel_tokens= rel_tokens)
                        for i, (pred, p_score, context) in enumerate(zip(preds, scores, contexts)):
                            retrieval_results[i] = {
                                "pred": pred, "score": p_score, "context": context}

                        for i, result in retrieval_results.items():
                            node_id += 1
                            #计算score
                            node_score = result["score"] + score if score is not None else result["score"]
                            # node_score = result['score']
                            pred = result["pred"]
                            context = result["context"]
                            prediction_tree[node_id] = {"prompt": cur_prompt, "pred": pred,
                                                        "score": node_score, "parent": node,
                                                    "topic_entity": topic_entity, "context": context, 'income': income_rel, 'depth': curr_depth}
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

        end_nodes = []
        for n_ind, node in prediction_tree.items():
            if 'Answer' in node['processed_pred']:
                end_nodes.append(n_ind)

        queues = []
        for n_ind in end_nodes:

            queue = []
            node = prediction_tree[n_ind]
            answer = node['processed_pred'].split('Answer:')[-1]
            if 'verbose' in node:
                score =  node['score'] + 0.5 * (node['verbose']['utility_score']+ 1) + node['verbose']['seq_score']
            else:
                score = node['score']
            while node:
                parent = node['parent']
                if 'verbose' in node:
                    queue.append(node['verbose'])
                if parent == None:
                    queues.append({'verbose': queue, "answer": answer, "score": score})

                    break
                node = prediction_tree[parent]
        queues.sort(key=lambda x: x['score'], reverse=True)
        answers = set()
        for q in queues[:3]:
            answer = q['answer']
            candidate =  answer.strip().split(';')
            answers.update([re.findall(context_pattern, candidate[i])[0][0]  for i in range(len(candidate)) if len(re.findall(context_pattern, candidate[i]))])
        precision, recall, f1, hits =cal_eval_metric(list(answers), labels)
        if hits == 1:
            print(f'correct {index}')
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        hitss.append(hits)
        logging_res.append({"index": index, "tree": prediction_tree})
        if len(logging_res) == 20:
            save_to_json(logging_res, './output/inference/cwq_test_top3_0214_beam3_k5_detph9_wohyper2.json')
            logging_res = []
    print('============================ Output Result Analysis =============================')
    print('Test data length: ', len(test_data))
    print('All correct count: ', count)
    print('Accuracy: ', count / len(test_data))


if __name__ == "__main__":
    main()
