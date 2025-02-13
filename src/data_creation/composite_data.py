import argparse
import os
import json
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        default="./output/generate/test.json", help="choose the dataset.")
    parser.add_argument("--task", type=str, default='r_relevance')
    parser.add_argument("--output_file", type=str,
                        default='output/generate/test_output.json')
    outputs = []
    starter = ''
    args = parser.parse_args()
    with open(args.input_file, 'r') as f:
        generate_data = json.load(f)
    with open('./output/generate/metaqa_sample_1500_0210_uti_copy.json', 'r') as f:
        uti_data = json.load(f)
    id2uti = {}
    for uti in uti_data:
        id2uti[uti['qid']] = uti['scores'][0]
    for data in generate_data:
        if data['qid'] not in id2uti:
            continue
        cur_sent = 0
        for step in data['scores']:
            content = step['score']
            cur_sent = step['sent_idx']
            starter += '[Retrieve Relation]'
            starter += "<paragraph>{}</paragraph>".format(content['r_context']) + ''.join([f"{r}{s}" for r,s in content['r_relevance'].items()]) + '[Retrieve Entity]'
            starter += "<paragraph>{}</paragraph>".format(';'.join(content['e_context'])) + ''.join([f"{r}{s}" for r,s in content['e_relevance'].items()])
            reason_score = content['reasoness']
            starter += reason_score
            if reason_score == '[Unreasonable]':
                break
        if len(starter):
            starter += '[No Retrieval]'
            starter += 'Answer: '+ ';'.join([f"{r}{s}" for r, s in id2uti[data['qid']]['individual_score'].items()])
            starter += id2uti[data['qid']]["utility_score"]
            outputs.append({"instruction": data['query'], "input": "", "output": starter})
            starter = ''

    with open(args.output_file, 'w') as f:
        json.dump(outputs, f)
            
                
            
            
