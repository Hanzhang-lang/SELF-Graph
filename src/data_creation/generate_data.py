import os
import argparse
import random
import json
import re
import logging
from typing import Dict, List, Tuple, Any
from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from src.data_creation.prompt_list import (
    few_shot_all_relation_prompt,
    few_shot_is_useful,
    few_shot_entity_prompt,
    few_shot_path_prompt_meta,
)
from src.utils import save_to_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def get_model_configs(args: argparse.Namespace) -> Dict[str, Any]:
    """Returns model configuration based on use_azure flag."""
    common_config = {
        'temperature': args.temperature or 1.0,
        'n': args.n or 3,
        'max_retries': args.max_retries or 5,
        'request_timeout': args.request_timeout or 600,
    }
    
    if args.use_azure:
        if not args.azure_api_key or not args.azure_endpoint:
            raise ValueError("Azure API key and endpoint are required when use_azure is True")
        return {
            'type': 'azure',
            'config': {
                **common_config,
                'openai_api_version': args.azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15"),
                'azure_deployment': args.azure_deployment or os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-35-turbo"),
                'api_key': args.azure_api_key,
                'azure_endpoint': args.azure_endpoint,
            }
        }
    return {
        'type': 'openai',
        'config': {
            **common_config,
            'model': args.model_name,
            'base_url': args.openai_base_url or "https://api.chatanywhere.tech/v1",
            'api_key': args.openai_api_key or "sk-",
        }
    }

def initialize_models(model_config: Dict[str, Any], verbose: bool = False) -> Dict[str, LLMChain]:
    """Initialize language model and chains based on config."""
    model_type = model_config['type']
    config = model_config['config']
    
    try:
        model = (
            AzureChatOpenAI(**config) if model_type == 'azure'
            else ChatOpenAI(**config)
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise
    
    chains = {
        'relevance': LLMChain(llm=model, prompt=few_shot_all_relation_prompt, verbose=verbose),
        'utility': LLMChain(llm=model, prompt=few_shot_is_useful, verbose=verbose),
        'triplet': LLMChain(llm=model, prompt=few_shot_entity_prompt, verbose=verbose),
        'reason': LLMChain(llm=model, prompt=few_shot_path_prompt_meta, verbose=verbose)
    }
    return chains

def extract_relationship_and_score(text: str) -> Dict[str, str]:
    """Extract relationships and their scores from text."""
    matches = re.findall(r'{(.+?) \(Score: (.+?)\)}', text)
    return dict(matches)

def invoke_chain(chain: LLMChain, params: Dict[str, str]) -> str:
    """Invoke a chain with parameters."""
    try:
        return chain.invoke(params)['text']
    except Exception as e:
        logger.error(f"Chain invocation failed: {e}")
        raise

def process_utility(query: str, answer: str, utility_chain: LLMChain) -> Tuple[Dict, str]:
    """Process utility scores for a query and answer."""
    try:
        utility_text = invoke_chain(utility_chain, {"query": query, "output": answer})
        content = json.loads(utility_text)
        return content.get('individual_scores', {}), content.get('overall_scores', '[Utility:3]')
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse utility output: {e}")
        return {}, "[Utility:3]"

def process_relationship(query: str, processed_relationship: str, topic_entity: str, preceding_sentences: str, relevance_chain: LLMChain) -> Dict[str, str]:
    """Process relationship relevance."""
    rel_relevance = invoke_chain(
        relevance_chain,
        {
            'query': query,
            'evidence': processed_relationship,
            'preceding_sentences': preceding_sentences,
            'topic': topic_entity
        }
    )
    return extract_relationship_and_score(rel_relevance)

def process_entities(query: str, processed_triplet: List[str], preceding_sentences: str,  triplet_chain: LLMChain) -> Dict[str, str]:
    """Process entity relevance."""
    entity_relevance = invoke_chain(
        triplet_chain,
        {
            'query': query,
            'evidence': ';'.join(processed_triplet),
            'preceding_sentences': preceding_sentences
        }
    )
    return extract_relationship_and_score(entity_relevance)

def process_reasoning_score(query: str, answer: List[str], reasoning_path: List[str], reason_chain: LLMChain) -> str:
    """Process reasoning score."""
    reasoning_output = invoke_chain(
        reason_chain,
        {
            "query": query,
            "output": ';'.join(answer),
            "preceding_sentences": ','.join(reasoning_path)
        }
    ).split('\n')[0].strip()
    return reasoning_output.split('Score:')[-1].strip() if 'Score:' in reasoning_output else reasoning_output

def load_chain_data(file_path: str) -> List[Dict]:
    """Load chain data from JSONL file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    except Exception as e:
        logger.error(f"Failed to load chain data from {file_path}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Process chain data with language model chains.")
    parser.add_argument("--chain_data", type=str, default="./output/chain_data/cwq_train_chain_top_5_0116.json",
                        help="Path to the input dataset (JSONL format)")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo",
                        help="OpenAI model name")
    parser.add_argument("--verbose", action='store_true',
                        help="Enable verbose logging for chains")
    parser.add_argument("--task", type=str, default='all',
                        choices=['all', 'r_relevance', 'e_relevance', 'reasoness', 'utility'],
                        help="Task to perform")
    parser.add_argument("--output_file", type=str, default="./output/generate/webqsp_0524_gpt35.json",
                        help="Path to save output results")
    parser.add_argument("--openai_api_key", type=str, help="OpenAI API key")
    parser.add_argument("--azure_api_key", type=str, help="Azure OpenAI API key")
    parser.add_argument("--azure_endpoint", type=str, help="Azure OpenAI endpoint")
    parser.add_argument("--azure_api_version", type=str, default="2023-05-15",
                        help="Azure OpenAI API version")
    parser.add_argument("--azure_deployment", type=str, default="gpt-35-turbo",
                        help="Azure OpenAI deployment name")
    parser.add_argument("--openai_base_url", type=str, default="https://api.chatanywhere.tech/v1",
                        help="OpenAI base URL")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Model temperature (0.0 to 2.0)")
    parser.add_argument("--n", type=int, default=3,
                        help="Number of completions to generate")
    parser.add_argument("--max_retries", type=int, default=5,
                        help="Maximum number of retries for API calls")
    parser.add_argument("--request_timeout", type=int, default=600,
                        help="Request timeout in seconds")
    parser.add_argument("--use_azure", action='store_true',
                        help="Use Azure OpenAI model instead of OpenAI")
    args = parser.parse_args()

    # Validate arguments
    if args.use_azure and (not args.azure_api_key or not args.azure_endpoint):
        parser.error("Both --azure_api_key and --azure_endpoint are required when --use_azure is set")
    if args.temperature < 0.0 or args.temperature > 2.0:
        parser.error("Temperature must be between 0.0 and 2.0")
    if args.n < 1:
        parser.error("Number of completions (n) must be at least 1")
    if args.max_retries < 0:
        parser.error("Max retries must be non-negative")
    if args.request_timeout < 1:
        parser.error("Request timeout must be at least 1 second")

    # Load model configurations
    model_config = get_model_configs(args)
    chains = initialize_models(model_config, args.verbose)
    tasks = ['r_relevance', 'e_relevance', 'reasoness', 'utility'] if args.task == 'all' else [args.task]

    # Load data
    dev_chain_data = load_chain_data(args.chain_data)
    logger.info(f"Loaded {len(dev_chain_data)} data points from {args.chain_data}")

    preceding_sentences = ""
    reasoning_path: List[str] = []
    output: List[Dict] = []
    already_ids: set = set()

    for count, dev_data in enumerate(dev_chain_data):
        query = dev_data['query']
        qid = dev_data['qid']
        tmp_scores = []

        if qid in already_ids:
            continue

        # Process utility task
        utility_result = None
        if 'utility' in tasks:
            answer = ', '.join(dev_data['answer'][:10])
            individual, overall = process_utility(query, answer, chains['utility'])
            utility_result = {"utility_score": overall, "individual_score": individual}

        # Process chain data
        for chain_line in dev_data['chains']:
            score_dict = {}
            topic_entity = chain_line['paths'][chain_line['chain_step'] - 1][0]

            # Relationship relevance
            if 'r_relevance' in tasks:
                processed_relationship = ';'.join(
                    chain_line['candidate_relation'][:4] + [chain_line['real_relation']]
                    if not chain_line['effective'] else chain_line['candidate_relation']
                )
                score_dict['r_relevance'] = process_relationship(
                    query, processed_relationship, topic_entity, preceding_sentences, chains['relevance']
                )
                score_dict['r_context'] = processed_relationship

            # Entity relevance
            if 'e_relevance' in tasks:
                candidate_entities = (
                    chain_line['candidate_entity'][:5]
                    if chain_line['real_entity'] in chain_line['candidate_entity'][:5]
                    else chain_line['candidate_entity'][:4] + [chain_line['real_entity']]
                )
                processed_triplet = [
                    f"({chain_line['paths'][chain_line['chain_step'] - 1][0]}, "
                    f"{chain_line['paths'][chain_line['chain_step'] - 1][1]}, {entity})"
                    for entity in candidate_entities
                ]
                score_dict['e_relevance'] = (
                    {'Unknown Entity': '[Partially Relevant]'}
                    if chain_line['real_entity'].startswith('m.')
                    else process_entities(query, processed_triplet, preceding_sentences, chains['triplet'])
                )
                score_dict['e_context'] = processed_triplet

            # Reasoning score
            if 'reasoness' in tasks:
                reasoning_path.append(
                    f"({topic_entity}, {chain_line['real_relation']}, {chain_line['real_entity']})"
                )
                score_dict['reasoness'] = process_reasoning_score(
                    query, dev_data['answer'][:5], reasoning_path, chains['reason']
                )

            # Update context
            preceding_sentences += f"({topic_entity}, {chain_line['real_relation']}, {chain_line['real_entity']})"
            tmp_scores.append({
                "sent_idx": chain_line['sent_idx'],
                "chain_step": chain_line['chain_step'],
                "score": score_dict,
            })

            # Handle unreasonable reasoning
            if 'reasoness' in tasks and score_dict.get('reasoness') == '[Unreasonable]':
                output.append({
                    "qid": qid,
                    "query": query,
                    "answer": dev_data['answer'],
                    "scores": tmp_scores,
                    "uti_scores": [utility_result] if utility_result else [],
                    "score_type": args.task
                })
                preceding_sentences = ""
                reasoning_path = []
                tmp_scores = []
                already_ids.add(qid)
                break

            # Reset context at path end
            if chain_line['chain_step'] == len(chain_line['paths']):
                output.append({
                    "qid": qid,
                    "query": query,
                    "answer": dev_data['answer'],
                    "scores": tmp_scores,
                    "uti_scores": [utility_result] if utility_result else [],
                    "score_type": args.task
                })
                preceding_sentences = ""
                reasoning_path = []
                tmp_scores = []
                already_ids.add(qid)
                break

        # Save output in batches
        if len(output) >= 2:
            logger.info(f"Saving batch at index {count}")
            save_to_json(output, args.output_file)
            output = []

    # Save remaining output
    if output:
        logger.info("Saving final batch")
        save_to_json(output, args.output_file)

if __name__ == "__main__":
    main()


# from langchain.chains import LLMChain
# from langchain_openai import AzureChatOpenAI, ChatOpenAI
# import os
# from langchain.chains import LLMChain
# from src.data_creation.prompt_list import *
# import argparse
# import random
# import json
# import re
# from src.utils import save_to_json

# def extract_relationship_and_score(s):
#     match_dict = dict()
#     pattern = r'{(.+?) \(Score: (.+?)\)}'
#     for match in re.findall(pattern, s):
#         match_dict[match[0]] = match[1]
#     return match_dict


# def invoke_chain(chain, fallback_chain, params):
#     """通用的链调用，带异常处理"""
#     try:
#         return chain.invoke(params)['text']
#     except ValueError:
#         return fallback_chain.invoke(params)['text']

# def process_utility(query, answer):
#     utility_text = utility_chain.invoke({"query": query, "output":answer})['text']
#     try:
#         content = eval(utility_text)
#         individual = content['individual_scores']
#         overall = content['overall_scores']
#     except:
#         individual = {}
#         overall = "[Utility:3]"
#     return individual, overall

# def process_relationship(query, processed_relationship, topic_entity):
#     """处理关系相关性"""
#     rel_relevance = invoke_chain(
#         relevance_chain,
#         relevance_chain_oa,
#         {
#             'query': query,
#             'evidence': processed_relationship,
#             'preceding_sentences': preceding_sentences,
#             'topic': topic_entity
#         }
#     )
#     return extract_relationship_and_score(rel_relevance)


# def process_entities(query, processed_triplet):
#     """处理实体相关性"""
#     entity_relevance = invoke_chain(
#         triplet_chain,
#         triplet_chain_oa,
#         {
#             'query': query,
#             'evidence': ';'.join(processed_triplet),
#             'preceding_sentences': preceding_sentences
#         }
#     )
#     return extract_relationship_and_score(entity_relevance)


# def process_reasoning_score(query, answer, reasoning_path):
#     """处理推理分数"""
#     reasoning_output = reason_chain.invoke({
#         "query": query,
#         "output": ';'.join(answer),
#         "preceding_sentences": ','.join(reasoning_path)
#     })['text'].split('\n')[0].strip()
#     return reasoning_output.split('Score:')[-1].strip() if 'Score:' in reasoning_output else reasoning_output


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--chain_data", type=str,
#                         default="./output/chain_data/cwq_train_chain_top_5_0116.json", help="choose the dataset.")
#     parser.add_argument("--model_name", type=str,
#                         default="gpt-3.5-turbo", help="choose the model.")
#     parser.add_argument("--verbose", action='store_true')
#     parser.add_argument("--task", type=str, default='r_relevance')
#     parser.add_argument("--output_file", type=str,
#                         default='./output/generate/webqsp_0331_gpt35.json')
#     parser.add_argument("--openai_api_key")
#     args = parser.parse_args()
#     dev_chain_data = []
#     with open(args.chain_data, 'r') as f:
#         for line in f.readlines():
#             dev_chain_data.append(json.loads(line))
#     # dev_chain_data = random.sample(dev_chain_data, 1000)
#     print('Input data length:', len(dev_chain_data))
#     openai_model = ChatOpenAI(model='gpt-3.5-turbo', base_url="https://api.chatanywhere.tech/v1",
#                               api_key="sk-bLZSHx4pKfPRZkYyIyyvUHSEjrlqj5sh2QIsxOM23yJnyoGD")
#     if args.model_name == "gpt-4o-mini":
#         os.environ["AZURE_OPENAI_API_KEY"] = "aa183bb914bb4858b15bed161fb47ba5"
#         os.environ["AZURE_OPENAI_ENDPOINT"] = "https://bxcl-prod.openai.azure.com/"
#         os.environ["AZURE_OPENAI_API_VERSION"] = "2024-08-01-preview"
#         os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-4o-mini"
#     if args.model_name == "gpt-3.5-turbo":
#         os.environ["AZURE_OPENAI_API_KEY"] = "2b219db0d2984f9dae28b651ab8ab3d9"
#         os.environ["AZURE_OPENAI_ENDPOINT"] = "https://smsh.openai.azure.com/"
#         os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
#         os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-35-turbo"
#     model = AzureChatOpenAI(
#         openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
#         azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
#         temperature=1,
#         n=3,
#         max_retries=5, request_timeout=600
#     )
#     relevance_chain = LLMChain(
#         llm=model, prompt=few_shot_all_relation_prompt, verbose=args.verbose)
#     utility_chain = LLMChain(llm=model, prompt=few_shot_is_useful, verbose=args.verbose)
#     triplet_chain = LLMChain(
#         llm=model, prompt=few_shot_entity_prompt, verbose=args.verbose)
#     relevance_chain_oa = LLMChain(
#         llm=openai_model, prompt=few_shot_all_relation_prompt, verbose=args.verbose)
#     triplet_chain_oa = LLMChain(
#         llm=openai_model, prompt=few_shot_entity_prompt, verbose=args.verbose)
#     reason_chain = LLMChain(
#         llm=model, prompt=few_shot_path_prompt_meta, verbose=args.verbose)
#     if args.task == 'all':
#         current_task = ['r_relevance', 'e_relevance', 'reasoness', 'utility']
#     else:
#         current_task = [args.task]
#     # with open('./output/generate/metaqa_sample_1500_0207.json', 'r') as f:
#     #     already_data = json.load(f)
#     # already_ids = set([d['qid'] for d in already_data])
#     preceding_sentences = ""
#     starter = '[New Retrieval]'
#     reasoning_path = []
#     output = []
#     output_utility = []
#     flag = True
#     already_ids = set()
#     # random.seed(42)
#     for count, dev_data in enumerate(dev_chain_data):
#         query = dev_data['query']
#         tmp_scores = []
#         if len(output) == 10:
#             print(f'Saving {count}')
#             save_to_json(output, args.output_file)
#             output = []
#         if len(output_utility) == 10:
#             print(f'Saving {count}')
#             save_to_json(output_utility, './output/generate/webqsp_0331_gpt35_uti.json')
#             output_utility = []
#         if 'utility' in current_task:
#             if dev_data['qid'] in already_ids:
#                 continue
#             individual, overall = process_utility(query, ', '.join(dev_data['answer'][:10]))
#             output_utility.append({"qid": dev_data['qid'], "query": query, "answer": dev_data['answer'][:10], "scores": [{"utility_score": overall, "individual_score": individual}],"score_type": 'utility'})
#         for chain_line in dev_data['chains']:
#             if dev_data['qid'] in already_ids:
#                 continue
#             score_dict = {}
#             topic_entity = chain_line['paths'][chain_line['chain_step'] - 1][0]
            

#             # 处理关系相关性任务
#             if 'r_relevance' in current_task:
#                 processed_relationship = ';'.join(
#                     chain_line['candidate_relation'][:4] +
#                     [chain_line['real_relation']]
#                     if not chain_line['effective'] else chain_line['candidate_relation']
#                 )
#                 score_dict['r_relevance'] = process_relationship(
#                     query, processed_relationship, topic_entity)
#                 score_dict['r_context'] = processed_relationship

#             # 处理实体相关性任务
#             if 'e_relevance' in current_task:
#                 candidate_entities = chain_line['candidate_entity'][:5] if chain_line['real_entity'] in chain_line['candidate_entity'][:5] \
#                     else chain_line['candidate_entity'][:4] + [chain_line['real_entity']]
#                 processed_triplet = [
#                     f"({chain_line['paths'][chain_line['chain_step'] - 1][0]}, {chain_line['paths'][chain_line['chain_step'] - 1][1]}, {entity})"
#                     for entity in candidate_entities
#                 ]
#                 if chain_line['real_entity'].startswith('m.'):
#                     score_dict['e_relevance'] = {
#                         'Unknown Entity': '[Partially Relevant]'}
#                 else:
#                     score_dict['e_relevance'] = process_entities(
#                         query, processed_triplet)
#                 score_dict['e_context'] = processed_triplet

#             # 处理推理任务
#             if 'reasoness' in current_task:
#                 reasoning_path.append(
#                     f"({topic_entity}, {chain_line['real_relation']}, {chain_line['real_entity']})"
#                 )
#                 score_dict['reasoness'] = process_reasoning_score(
#                     query, dev_data['answer'][:5], reasoning_path
#                 )


#             # 更新前置上下文句子
#             preceding_sentences += f"({topic_entity}, {chain_line['real_relation']}, {chain_line['real_entity']})"

#             # 添加分数记录
#             tmp_scores.append({
#                 "sent_idx": chain_line['sent_idx'],
#                 "chain_step": chain_line['chain_step'],
#                 "score": score_dict,
#             })
#             if 'reasoness' in current_task:
#                 if score_dict['reasoness'] == '[Unreasonable]':
#                     preceding_sentences = ""
#                     reasoning_path = []
#                     output.append({"qid": dev_data['qid'], "query": dev_data['query'], "answer": dev_data['answer'], "scores": tmp_scores,"score_type": args.task})
#                     tmp_scores = []
#                     flag = True
#                     already_ids.add(dev_data['qid'])
#                     break
#             # 判断是否到达路径末尾，重置上下文
#             if chain_line['chain_step'] == len(chain_line['paths']):
#                 preceding_sentences = ""
#                 reasoning_path = []
#                 output.append({"qid": dev_data['qid'], "query": dev_data['query'], "answer": dev_data['answer'], "scores": tmp_scores,"score_type": args.task})
#                 tmp_scores = []
#                 flag = True
#                 already_ids.add(dev_data['qid'])
#                 break
#     save_to_json(output, args.output_file)
#     # save_to_json(output_utility, args.output_file)
#     save_to_json(output_utility, './output/generate/webqsp_0331_gpt35_uti.json')
