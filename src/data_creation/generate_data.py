from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import os
from langchain.chains import LLMChain
from src.data_creation.prompt_list import *
import argparse
import random
import json
import re
from src.utils import save_to_json

def extract_relationship_and_score(s):
    match_dict = dict()
    pattern = r'{(.+?) \(Score: (.+?)\)}'
    for match in re.findall(pattern, s):
        match_dict[match[0]] = match[1]
    return match_dict


def invoke_chain(chain, fallback_chain, params):
    """通用的链调用，带异常处理"""
    try:
        return chain.invoke(params)['text']
    except ValueError:
        return fallback_chain.invoke(params)['text']

def process_utility(query, answer):
    utility_text = utility_chain.invoke({"query": query, "output":answer})['text']
    try:
        content = eval(utility_text)
        individual = content['individual_scores']
        overall = content['overall_scores']
    except:
        individual = {}
        overall = "[Utility:3]"
    return individual, overall

def process_relationship(query, processed_relationship, topic_entity):
    """处理关系相关性"""
    rel_relevance = invoke_chain(
        relevance_chain,
        relevance_chain_oa,
        {
            'query': query,
            'evidence': processed_relationship,
            'preceding_sentences': preceding_sentences,
            'topic': topic_entity
        }
    )
    return extract_relationship_and_score(rel_relevance)


def process_entities(query, processed_triplet):
    """处理实体相关性"""
    entity_relevance = invoke_chain(
        triplet_chain,
        triplet_chain_oa,
        {
            'query': query,
            'evidence': ';'.join(processed_triplet),
            'preceding_sentences': preceding_sentences
        }
    )
    return extract_relationship_and_score(entity_relevance)


def process_reasoning_score(query, answer, reasoning_path):
    """处理推理分数"""
    reasoning_output = reason_chain.invoke({
        "query": query,
        "output": ';'.join(answer),
        "preceding_sentences": ','.join(reasoning_path)
    })['text'].split('\n')[0].strip()
    return reasoning_output.split('Score:')[-1].strip() if 'Score:' in reasoning_output else reasoning_output


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--chain_data", type=str,
                        default="./output/chain_data/cwq_train_chain_top_5_0116.json", help="choose the dataset.")
    parser.add_argument("--model_name", type=str,
                        default="gpt-4o-mini", help="choose the model.")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--task", type=str, default='r_relevance')
    parser.add_argument("--output_file", type=str,
                        default='output/generate/test.json')
    parser.add_argument("--openai_api_key")
    args = parser.parse_args()
    dev_chain_data = []
    with open(args.chain_data, 'r') as f:
        for line in f.readlines():
            dev_chain_data.append(json.loads(line))
    # dev_chain_data = random.sample(dev_chain_data, 1000)
    print('Input data length:', len(dev_chain_data))
    openai_model = ChatOpenAI(model='gpt-3.5-turbo', base_url="https://api.chatanywhere.tech/v1",
                              api_key="sk-bLZSHx4pKfPRZkYyIyyvUHSEjrlqj5sh2QIsxOM23yJnyoGD")
    if args.model_name == "gpt-4o-mini":
        os.environ["AZURE_OPENAI_API_KEY"] = "aa183bb914bb4858b15bed161fb47ba5"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://bxcl-prod.openai.azure.com/"
        os.environ["AZURE_OPENAI_API_VERSION"] = "2024-08-01-preview"
        os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-4o-mini"

        model = AzureChatOpenAI(
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            temperature=1,
            n=3,
            max_retries=5, request_timeout=600
        )
    relevance_chain = LLMChain(
        llm=model, prompt=few_shot_all_relation_prompt, verbose=args.verbose)
    utility_chain = LLMChain(llm=model, prompt=few_shot_is_useful, verbose=args.verbose)
    triplet_chain = LLMChain(
        llm=model, prompt=few_shot_entity_prompt, verbose=args.verbose)
    relevance_chain_oa = LLMChain(
        llm=openai_model, prompt=few_shot_relation_prompt, verbose=args.verbose)
    triplet_chain_oa = LLMChain(
        llm=openai_model, prompt=few_shot_entity_prompt, verbose=args.verbose)
    reason_chain = LLMChain(
        llm=model, prompt=few_shot_path_prompt_meta, verbose=args.verbose)
    if args.task == 'all':
        current_task = ['r_relevance', 'e_relevance', 'reasoness', 'utility']
    else:
        current_task = [args.task]
    with open('./output/generate/metaqa_sample_1500_0207.json', 'r') as f:
        already_data = json.load(f)
    already_ids = set([d['qid'] for d in already_data])
    print(len(already_ids))
    preceding_sentences = ""
    starter = '[New Retrieval]'
    reasoning_path = []
    output = []
    output_utility = []
    flag = True
    # already_ids = set()
    # random.seed(42)
    for count, dev_data in enumerate(dev_chain_data):
        query = dev_data['query']
        tmp_scores = []
        if len(output) == 10:
            print(f'Saving {count}')
            save_to_json(output, args.output_file)
            output = []
        if len(output_utility) == 10:
            print(f'Saving {count}')
            save_to_json(output_utility, './output/generate/metaqa_sample_1500_0210_uti.json')
            output_utility = []
        if 'utility' in current_task:
            if dev_data['qid'] in already_ids:
                continue
            individual, overall = process_utility(query, ', '.join(dev_data['answer'][:10]))
            output_utility.append({"qid": dev_data['qid'], "query": query, "answer": dev_data['answer'][:10], "scores": [{"utility_score": overall, "individual_score": individual}],"score_type": 'utility'})
        for chain_line in dev_data['chains']:
            if dev_data['qid'] in already_ids:
                continue
            score_dict = {}
            topic_entity = chain_line['paths'][chain_line['chain_step'] - 1][0]
            

            # 处理关系相关性任务
            if 'r_relevance' in current_task:
                processed_relationship = ';'.join(
                    chain_line['candidate_relation'][:4] +
                    [chain_line['real_relation']]
                    if not chain_line['effective'] else chain_line['candidate_relation']
                )
                score_dict['r_relevance'] = process_relationship(
                    query, processed_relationship, topic_entity)
                score_dict['r_context'] = processed_relationship

            # 处理实体相关性任务
            if 'e_relevance' in current_task:
                candidate_entities = chain_line['candidate_entity'][:5] if chain_line['real_entity'] in chain_line['candidate_entity'][:5] \
                    else chain_line['candidate_entity'][:4] + [chain_line['real_entity']]
                processed_triplet = [
                    f"({chain_line['paths'][chain_line['chain_step'] - 1][0]}, {chain_line['paths'][chain_line['chain_step'] - 1][1]}, {entity})"
                    for entity in candidate_entities
                ]
                if chain_line['real_entity'].startswith('m.'):
                    score_dict['e_relevance'] = {
                        'Unknown Entity': '[Partially Relevant]'}
                else:
                    score_dict['e_relevance'] = process_entities(
                        query, processed_triplet)
                score_dict['e_context'] = processed_triplet

            # 处理推理任务
            if 'reasoness' in current_task:
                reasoning_path.append(
                    f"({topic_entity}, {chain_line['real_relation']}, {chain_line['real_entity']})"
                )
                score_dict['reasoness'] = process_reasoning_score(
                    query, dev_data['answer'][:5], reasoning_path
                )


            # 更新前置上下文句子
            preceding_sentences += f"({topic_entity}, {chain_line['real_relation']}, {chain_line['real_entity']})"

            # 添加分数记录
            tmp_scores.append({
                "sent_idx": chain_line['sent_idx'],
                "chain_step": chain_line['chain_step'],
                "score": score_dict,
            })
            if 'reasoness' in current_task:
                if score_dict['reasoness'] == '[Unreasonable]':
                    preceding_sentences = ""
                    reasoning_path = []
                    output.append({"qid": dev_data['qid'], "query": dev_data['query'], "answer": dev_data['answer'], "scores": tmp_scores,"score_type": args.task})
                    tmp_scores = []
                    flag = True
                    already_ids.add(dev_data['qid'])
                    break
            # 判断是否到达路径末尾，重置上下文
            if chain_line['chain_step'] == len(chain_line['paths']):
                preceding_sentences = ""
                reasoning_path = []
                output.append({"qid": dev_data['qid'], "query": dev_data['query'], "answer": dev_data['answer'], "scores": tmp_scores,"score_type": args.task})
                tmp_scores = []
                flag = True
                already_ids.add(dev_data['qid'])
                break
    save_to_json(output, args.output_file)
    # save_to_json(output_utility, args.output_file)
    save_to_json(output_utility, './output/generate/metaqa_sample_1500_0210_uti.json')
