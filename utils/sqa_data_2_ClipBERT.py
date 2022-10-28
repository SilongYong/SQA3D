import argparse
import json
import os
import numpy as np
import jsonlines

def parse(response):
    with jsonlines.open('output.jsonl',mode='a') as writer:
        writer.write(response)

def sqa2clipbert(split, anno_dir, output_dir, data_version="balanced"):
    answer_dict = json.load(open(os.path.join(anno_dir, "answer_dict.json"), 'r'))
    anno_dir = os.path.join(anno_dir, "balanced")
    questions = json.load(open(os.path.join(anno_dir, f'v1_{data_version}_questions_{split}_scannetv2.json'), 'r'))['questions']
    annotations = json.load(open(os.path.join(anno_dir, f'v1_{data_version}_sqa_annotations_{split}_scannetv2.json'), 'r'))['annotations']
    qid2annoid = {}
    for i in range(len(annotations)):
        qid2annoid[annotations[i]["question_id"]] = i
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(len(questions)):
        entry = {}
        s = questions[i]["situation"]
        q = questions[i]["question"]
        q_id = questions[i]["question_id"]
        scene_id = questions[i]["scene_id"]
        answer = annotations[qid2annoid[q_id]]["answers"][0]['answer']
        entry['answer'] = answer if answer in answer_dict[0].keys() else "unknown"
        entry['question'] = s + ' ' + q
        answer_type = q.split(' ')
        entry['video_id'] = scene_id
        entry['answer_type'] = answer_type[0].lower() if answer_type[0].lower() in ["what", "is", "how", "can", "which"] else "others"
        with jsonlines.open(os.path.join(output_dir, f'{split}.jsonl'),mode='a') as writer:
            writer.write(entry)
    with open(os.path.join(output_dir, "train_ans2label.json"), 'w') as f:
        json.dump(answer_dict[0], f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_dir', default="../assets/data/sqa_task")
    parser.add_argument('--output_dir', default="../ClipBERT/data/txt_db/sqa")
    parser.add_argument('--data_version', default="balanced", help="USE full or balanced")
    args = parser.parse_args()
    sqa2clipbert('val', args.anno_dir, data_version=args.data_version)
    sqa2clipbert('train', args.anno_dir, data_version=args.data_version)
