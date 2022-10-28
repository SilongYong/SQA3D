from data_utils import ques_load, tokenize
from data_utils import proc_ques
import collections
import numpy as np
import glob, json, torch, time
import torch.utils.data as Data
from PIL import Image
import os
from torchvision import transforms

TYPE_2_IDX = {"what" : 0, "is" : 1, "how" : 2, "can" : 3, "which" : 4}

class DataSet(Data.Dataset):
    def __init__(self, flags, split, answer_counter=None):
        self.split = split
        self.q_type = TYPE_2_IDX
        self.answer_counter = answer_counter
        self.flags = flags
        anno_dir = os.path.join(flags.anno_dir, "balanced")
        self.questions = json.load(open(os.path.join(anno_dir, f'v1_balanced_questions_{self.split}_scannetv2.json'), 'r'))['questions']
        self.annotations = json.load(open(os.path.join(anno_dir, f'v1_balanced_sqa_annotations_{self.split}_scannetv2.json'), 'r'))['annotations']
        self.qid2annoid = {}
        for i in range(len(self.annotations)):
            self.qid2annoid[self.annotations[i]["question_id"]] = i
        answer_dict = json.load(open(os.path.join(flags.anno_dir, "answer_dict.json"), 'r'))
        self.answer2class = answer_dict[0]
        self.class2answer = answer_dict[1]
        self.pix_mean = (0.485, 0.456, 0.406)
        self.pix_std = (0.229, 0.224, 0.225)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224, 224]),
            transforms.Normalize(mean=self.pix_mean, std=self.pix_std)
            ])
        self.all_questions = \
            json.load(open(os.path.join(anno_dir, 'v1_balanced_questions_train_scannetv2.json'), 'r'))['questions'] + \
            json.load(open(os.path.join(anno_dir, 'v1_balanced_questions_val_scannetv2.json'), 'r'))['questions'] + \
            json.load(open(os.path.join(anno_dir, 'v1_balanced_questions_test_scannetv2.json'), 'r'))['questions']
        self.token_to_ix, self.pretrained_emb = tokenize(self.all_questions, flags.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        self.qid_to_ques = ques_load(self.all_questions)
        self.data_size = len(self.questions)
        
    def __getitem__(self, index):
        q = self.questions[index]["question"].lower().replace(',', '').replace('.', '').replace('?', '').replace('\'s', ' \'s')
        q_type_id = self.q_type[q.split()[0]] if q.split()[0] in self.q_type.keys() else 5 
        s = self.questions[index]["situation"].lower().replace(',', '').replace('.', '').replace('?', '').replace('\'s', ' \'s')
        q_id = self.questions[index]["question_id"]
        scene_id = self.questions[index]["scene_id"]   # scene0352_00 : str
        question = s + ' ' + q
        # question = q
        a = self.annotations[self.qid2annoid[q_id]]["answers"][0]['answer']
        if a not in self.answer2class.keys():
            a = len(self.answer2class)
        else:
            a = self.answer2class[a]
    
        ques_ix_iter = proc_ques(question, self.token_to_ix, self.flags.MAX_TOKEN)
        img = Image.open(os.path.join(self.flags.img_dir, f"{scene_id}_BEV.png")).convert('RGB')
        img = self.transform(img)
        return img, torch.from_numpy(ques_ix_iter), torch.from_numpy(np.array(a)).unsqueeze(0), q_type_id
            
    def __len__(self):
        return self.data_size
