import argparse
import json
import os
import numpy as np

def sqa2scanqa(split, anno_dir, data_version="balanced"):
    random_pair_ratio = 0.5
    answer_dict = json.load(open(os.path.join(anno_dir, "answer_dict.json"), 'r'))
    answer2class = answer_dict[0]
    class2answer = answer_dict[1]
    anno_dir = os.path.join(anno_dir, data_version)
    questions = json.load(open(os.path.join(anno_dir, f'v1_{data_version}_questions_{split}_scannetv2.json'), 'r'))['questions']
    annotations = json.load(open(os.path.join(anno_dir, f'v1_{data_version}_sqa_annotations_{split}_scannetv2.json'), 'r'))['annotations']
    qid2annoid = {}
    for i in range(len(annotations)):
        qid2annoid[annotations[i]["question_id"]] = i
    
    entries = []
    for i in range(len(questions)):
        entry = {}
        s = questions[i]["situation"]
        q = questions[i]["question"]
        q_id = questions[i]["question_id"]
        scene_id = questions[i]["scene_id"]
        pos = [v for (k, v) in annotations[qid2annoid[q_id]]["position"].items()]
        rot = [v for [k, v] in annotations[qid2annoid[q_id]]["rotation"].items()]
        situation = s
        question = q
        answer = annotations[qid2annoid[q_id]]["answers"][0]['answer']
        entry['answers'] = [answer] if answer in answer_dict[0].keys() else ["unknown"]
        entry['object_ids'] = []
        entry['object_names'] = []
        entry['question'] = question
        entry['situation'] = situation
        entry['question_id'] = q_id
        entry['scene_id'] = scene_id
        entry['position'] = pos + rot
        entries.append(entry)
    with open(f"../ScanQA/data/qa/SQA_balanced_{split}.json", "w") as f:
        json.dump(entries, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_dir', default="../assets/data/sqa_task")
    parser.add_argument('--data_version', default="balanced", help="USE full or balanced")
    args = parser.parse_args()
    sqa2scanqa('val', args.anno_dir, data_version=args.data_version)
    sqa2scanqa('train', args.anno_dir, data_version=args.data_version)
    sqa2scanqa('test', args.anno_dir, data_version=args.data_version)
