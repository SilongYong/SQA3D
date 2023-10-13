import argparse
import hashlib
import json
import pickle
import random
import re
import sys
import time

import openai
import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

openai.api_key = '' # <-- your key goes here

data_file = 'v1_balanced_questions_test_scannetv2.json'
annotation_file = 'v1_balanced_sqa_annotations_test_scannetv2.json'
tokenizer = None
uqa_model = None
captions = None
num_captions = 30
prompt_template_gpt = '''
    Context: there is a book on the desk. A laptop with a green cover is to the left of the book.
    Q: I'm working by the desk. What is on the desk beside the book?
    A: laptop
    Context: {}
    Q: {}
    A:
'''
prompt_template_uqa = '''
    {}
    Q: {}
    A:
'''

def clean_answer(data):
    key = 'answer'
    for index in range(len(data)):
        data[index][key] = data[index][key].lower()
        data[index][key] = re.sub('[ ]+$' ,'', data[index][key])
        data[index][key] = re.sub('^[ ]+' ,'', data[index][key])
        data[index][key] = re.sub(' {2,}', ' ', data[index][key])

        data[index][key] = re.sub('\.[ ]{2,}', '. ', data[index][key])
        data[index][key] = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data[index][key])
        data[index][key] = re.sub('ç' ,'c', data[index][key])
        data[index][key] = re.sub('’' ,'\'', data[index][key])
        data[index][key] = re.sub(r'\bletf\b' ,'left', data[index][key])
        data[index][key] = re.sub(r'\blet\b' ,'left', data[index][key])
        data[index][key] = re.sub(r'\btehre\b' ,'there', data[index][key])
        data[index][key] = re.sub(r'\brigth\b' ,'right', data[index][key])
        data[index][key] = re.sub(r'\brght\b' ,'right', data[index][key])
        data[index][key] = re.sub(r'\bbehine\b', 'behind', data[index][key])
        data[index][key] = re.sub(r'\btv\b' ,'TV', data[index][key])
        data[index][key] = re.sub(r'\bchai\b' ,'chair', data[index][key])
        data[index][key] = re.sub(r'\bwasing\b' ,'washing', data[index][key])
        data[index][key] = re.sub(r'\bwaslked\b' ,'walked', data[index][key])
        data[index][key] = re.sub(r'\boclock\b' ,'o\'clock', data[index][key])
        data[index][key] = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data[index][key])

        # digit to word, only for answer
        data[index][key] = re.sub(r'\b0\b', 'zero', data[index][key])
        data[index][key] = re.sub(r'\bnone\b', 'zero', data[index][key])
        data[index][key] = re.sub(r'\b1\b', 'one', data[index][key])
        data[index][key] = re.sub(r'\b2\b', 'two', data[index][key])
        data[index][key] = re.sub(r'\b3\b', 'three', data[index][key])
        data[index][key] = re.sub(r'\b4\b', 'four', data[index][key])
        data[index][key] = re.sub(r'\b5\b', 'five', data[index][key])
        data[index][key] = re.sub(r'\b6\b', 'six', data[index][key])
        data[index][key] = re.sub(r'\b7\b', 'seven', data[index][key])
        data[index][key] = re.sub(r'\b8\b', 'eight', data[index][key])
        data[index][key] = re.sub(r'\b9\b', 'nine', data[index][key])
        data[index][key] = re.sub(r'\b10\b', 'ten', data[index][key])
        data[index][key] = re.sub(r'\b11\b', 'eleven', data[index][key])
        data[index][key] = re.sub(r'\b12\b', 'twelve', data[index][key])
        data[index][key] = re.sub(r'\b13\b', 'thirteen', data[index][key])
        data[index][key] = re.sub(r'\b14\b', 'fourteen', data[index][key])
        data[index][key] = re.sub(r'\b15\b', 'fifteen', data[index][key])
        data[index][key] = re.sub(r'\b16\b', 'sixteen', data[index][key])
        data[index][key] = re.sub(r'\b17\b', 'seventeen', data[index][key])
        data[index][key] = re.sub(r'\b18\b', 'eighteen', data[index][key])
        data[index][key] = re.sub(r'\b19\b', 'nineteen', data[index][key])
        data[index][key] = re.sub(r'\b20\b', 'twenty', data[index][key])
        data[index][key] = re.sub(r'\b23\b', 'twenty-three', data[index][key])

        # misc
        # no1, mat2, etc
        data[index][key] = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data[index][key])
        data[index][key] = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data[index][key])
        data[index][key] = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data[index][key])
        data[index][key] = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data[index][key])

        data[index][key] = re.sub(r'\bbackwards\b', 'backward', data[index][key])
    return data

def merge_data(data, annotation):
    ret_data = []
    for i in data['questions']:
        qid = i['question_id']
        for ind, j in enumerate(annotation['annotations']):
            if j['question_id'] == qid:
                ret_data.append((i, j))
                break
    return ret_data

def chat_llm(history, temperature=0, max_tokens=100, model='gpt-3.5-turbo'):
    if model == 'uqa_large':
        global uqa_model, tokenizer
        if uqa_model is None:
            model_name = "allenai/unifiedqa-v2-t5-{}-1251000".format('large')
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            uqa_model = T5ForConditionalGeneration.from_pretrained(model_name).cuda()

        assert type(history) == str
        input_ids = tokenizer.encode(history, return_tensors="pt").cuda()
        res = uqa_model.generate(input_ids)
        return tokenizer.batch_decode(res, skip_special_tokens=True)[0]
    else:
        if type(history) == str:
            history = [('user', history)]

        chat_history = []
        for i in history:
            if i[0] == 'user':
                chat_history.append({
                    'role': 'user',
                    'content': i[1]
                })
            elif i[0] == 'assistant':
                chat_history.append({
                    'role': 'assistant',
                    'content': i[1]
                })
            else:
                raise NotImplementedError

        total_trials = 0
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model = model,
                    messages=chat_history,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                time.sleep(1)
                break
            except openai.OpenAIError as e:
                total_trials += 1
                print(e)
                time.sleep(1)
            except KeyboardInterrupt:
                print("Interrupted by user. Exiting...")
                sys.exit(1)

        return response.choices[0].message.content

def inference_llm(model):
    global captions, num_captions
    data = merge_data(
        json.load(open(data_file, 'r')),
        json.load(open(annotation_file, 'r')),
    )
    all_preds = []
    for i in tqdm.tqdm(range(len(data))):
        sid = data[i][0]['scene_id']
        sit = data[i][0]['situation']
        q = data[i][0]['question']
        a = data[i][1]['answers'][0]['answer']
        rot = data[i][1]['rotation']
        pos = data[i][1]['position']
        try:
            desc = captions[sid]
        except KeyError:
            desc = []
        if len(desc) > num_captions:
            desc = random.sample(desc, k=num_captions)
        desc = '. '.join([i[1] for i in desc])
        if 'gpt' in model:
            input_s = prompt_template_gpt.format(desc, ' '.join([sit, q]))
        else:
            input_s = prompt_template_uqa.format(desc, ' '.join([sit, q]))
        pred = chat_llm(input_s, model=model)
        if 'gpt' in model:
            pred = re.sub('^[ ]+' ,'', pred.split('\n')[-1])
        all_preds.append({
            'scene_id': sid,
            'situation': sit,
            'question': q,
            'answer': pred,
            'agent_rot': rot,
            'agent_pos': pos,
        })
    return all_preds


def gpt_llm_eval(q, pred, a):
    # Given the question "{q}", does the answer "{pred}" imply the answer "{a}"? Answer with Yes or No.
    eval_prompt = f"""
    Given the question "{q}", given the true answer "{a}", does the prediction "{pred}" imply the true answer? Answer with Yes or No.
    """
    output = chat_llm(eval_prompt)
    return output.lower().strip() == 'yes'

def compute_accuracy(data, annotation, prediction, use_gpt_metric=False):
    # situation+question+agent_pos+agent_rot as index:
    merged = merge_data(data, annotation)
    prediction = clean_answer(prediction)
    index = {}
    for i in range(len(merged)):
        hash = hashlib.md5((str({
                'scene_id': merged[i][0]['scene_id'],
                'agent_rot': merged[i][1]['rotation'],
                'agent_pos': merged[i][1]['position'],
                'question': merged[i][0]['question'],
                'situation': merged[i][0]['situation']
        })).encode()).hexdigest()
        index[hash] = i
    corr = {}
    for ind, i in enumerate(prediction):
        print(f'{ind}/{len(prediction)}')
        hash = hashlib.md5((str({
                'scene_id': i['scene_id'],
                'agent_rot': i['agent_rot'],
                'agent_pos': i['agent_pos'],
                'question': i['question'],
                'situation': i['situation']
        })).encode()).hexdigest()
        if hash not in corr:
            corr[hash] = 0
        try:
            answer = merged[index[hash]][1]['answers'][0]['answer']

            if use_gpt_metric:
                if i['answer'] == answer:
                    corr[hash] += 1
                elif gpt_llm_eval(i['question'], i['answer'], answer):
                    print(i['answer'], '==', answer)
                    corr[hash] += 1
                else:
                    print(i['answer'], '!=', answer)
            else:
                if i['answer'] == answer:
                    corr[hash] += 1
                # FIXME: better documented behaviour
                elif i['answer'] in answer:
                    corr[hash] += 1
                elif ''.join(i['answer'].split()) in ''.join(answer.split()):
                    print(i['question'])
                    print(i['answer'], '=====', answer)
                    corr[hash] += 1
                # TODO: we also need to get rid of "a", "the", etc
                elif len(set(i['answer'].split()).intersection(answer.split())) > 0:
                    print(i['question'])
                    print(i['answer'], '=====', answer)
                    corr[hash] += 1
                else:
                    continue
        except KeyError:
            # print('Answer not found. Aborted.')
            pass

    total = len(corr)
    cnt = sum([1 if v>=1 else 0 for v in corr.values()])

    print('Acc: {}/{} = {:.4f}'.format(cnt, total, cnt/total))

def main(args):
    if args.model == 'gpt':
        print('Eval with GPT model')
        data = inference_llm('gpt-3.5-turbo')
    elif args.model == 'uqa':
        print('Eval with Unified QA model')
        data = inference_llm('uqa_large')
    else:
        raise NotImplementedError(f'{args.model} is not supported.')

    compute_accuracy(
        json.load(open(data_file, 'r')),
        json.load(open(annotation_file, 'r')),
        data,
        use_gpt_metric=args.use_gpt_metric,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="check accuracy")

    # Add the arguments
    parser.add_argument("--model", choices=['uqa', 'gpt'])
    parser.add_argument('--scene_captions', type=str)
    parser.add_argument('--use_gpt_metric', action='store_true')

    # Parse the arguments
    args = parser.parse_args()

    captions = pickle.load(open(args.scene_captions, 'rb'))
    main(args)