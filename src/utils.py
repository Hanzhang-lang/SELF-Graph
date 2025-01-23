import argparse
import glob
import json
import os
import re
import string
from sklearn.metrics import precision_score
from typing import List
import random



def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1

def eval_acc(prediction, answer):
    matched = 0.
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)

def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0

def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = ' '.join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall

def extract_topk_prediction(prediction, k=-1):
    results = {}
    for p in prediction:
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]

def save_to_json(data: List, data_path=''):
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

def random_sample(lst, k=3):
    try:
        # 尝试从列表中随机抽取k个不重复的元素
        return random.sample(lst, k)
    except ValueError:
        # 如果列表长度小于k，返回整个列表
        return lst