""" 
Modified from: https://github.com/daveredrum/ScanRefer/blob/master/lib/dataset.py
"""

import re
import os
import sys
import time
import pickle
import numpy as np
import multiprocessing as mp
from scipy.spatial.transform import Rotation as R
#from sklearn import preprocessing
from torch.utils.data import Dataset
from data.scannet.model_util_scannet import ScannetDatasetConfig
sys.path.append(os.path.join(os.getcwd(), 'lib')) # HACK add the lib folder
from lib.config import CONF
from utils.pc_utils import random_sampling, rotx, roty, rotz
from data.scannet.model_util_scannet import ScannetDatasetConfig, rotate_aligned_boxes_along_axis

# data setting
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# data path
SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, 'scannetv2-labels.combined.tsv')
MULTIVIEW_DATA = CONF.MULTIVIEW
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, 'glove.p')

def get_index(lst=None, item=''):
    return [index for (index,value) in enumerate(lst) if value == item]

def get_answer_score(freq):
    if freq == 0:
        return .0
    elif freq == 1:
        return .3
    elif freq == 2:
        return .6
    elif freq == 3:
        return .9
    else:
        return 1.

class ScannetQADatasetConfig(ScannetDatasetConfig):
    def __init__(self):
        super().__init__()
        self.num_answers = -1

class Answer(object):
    def __init__(self, answers=None, unk_token='<unk>', ignore_idx=-100):
        if answers is None:
            answers = []
        self.unk_token = unk_token
        self.ignore_idx = ignore_idx
        self.vocab = {x: i for i, x in enumerate(answers)}
        self.rev_vocab = dict((v, k) for k, v in self.vocab.items())
        
    def itos(self, i):
        if i == self.ignore_idx:
            return self.unk_token
        return self.rev_vocab[i]

    def stoi(self, v):
        if v not in self.vocab:
            #return self.vocab[self.unk_token]
            return self.ignore_idx
        return self.vocab[v]

    def __len__(self):
        return len(self.vocab)    


class ScannetQADataset(Dataset):
    def __init__(self, sqa, sqa_all_scene, 
            use_unanswerable=False,
            answer_cands=None,
            answer_counter=None,
            answer_cls_loss='ce',
            split='train', 
            num_points=40000,
            use_height=False, 
            use_color=False, 
            use_normal=False, 
            use_multiview=False, 
            tokenizer=None,
            augment=False,
            debug=False,
            wos=False,
            test=False,
        ):

        self.debug = debug
        self.all_data_size = -1
        self.answerable_data_size = -1
        self.wos = wos
        self.answer_features = None
        self.use_unanswerable = use_unanswerable

        if split == 'train':
            # remove unanswerble qa samples for training
            self.all_data_size = len(sqa)
            if use_unanswerable: 
                self.sqa = sqa
            else:
                self.sqa = [data for data in sqa if len(set(data['answers']) & set(answer_cands)) > 0]
            self.answerable_data_size = len(self.sqa)
            print('all train:', self.all_data_size)
            print('answerable train', self.answerable_data_size)
        elif split == 'val':
            self.all_data_size = len(sqa)
            if use_unanswerable:
                self.sqa = sqa
            else:
                self.sqa = [data for data in sqa if len(set(data['answers']) & set(answer_cands)) > 0]
                
            self.answerable_data_size = len(self.sqa)
            print('all val:', self.all_data_size)
            print('answerable val', self.answerable_data_size)

        self.sqa_all_scene = sqa_all_scene # all scene_ids in sqa
        self.answer_cls_loss = answer_cls_loss
        self.answer_cands = answer_cands
        self.answer_counter = answer_counter
        self.answer_vocab = Answer(answer_cands)
        self.num_answers = 0 if answer_cands is None else len(answer_cands) 

        self.split = split
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.use_normal = use_normal        
        self.use_multiview = use_multiview
        self.augment = augment
        self.test = test
        # tokenize a question to tokens
        scene_ids = sorted(set(record['scene_id'] for record in self.sqa))
        self.scene_id_to_number = {scene_id:int(''.join(re.sub('scene', '', scene_id).split('_'))) for scene_id in scene_ids}
        self.scene_number_to_id = {v: k for k, v in self.scene_id_to_number.items()}

        if tokenizer is None:
            from spacy.tokenizer import Tokenizer
            from spacy.lang.en import English
            nlp = English()
            # Create a blank Tokenizer with just the English vocab
            spacy_tokenizer = Tokenizer(nlp.vocab)
            
            def tokenize(sent):
                sent = sent.replace('?', ' ?').replace('.', ' .')
                return [token.text for token in spacy_tokenizer(sent)]

            for record in self.sqa:
                record.update(question=tokenize(record['question'])) 
                record.update(situation=tokenize(record['situation']))
        else:
            raise NotImplementedError("BERT-based unimplemented")
            
        # load data
        self._load_data()
        self.multiview_data = {}


    def __len__(self):
        return len(self.sqa)

    def __getitem__(self, idx):
        start = time.time()
        scene_id = self.sqa[idx]['scene_id']
        position = self.sqa[idx]['position']

        object_ids = None
        object_names = None            

        question_id = self.sqa[idx]['question_id']
        answers = self.sqa[idx].get('answers', [])
        answer_cats = np.zeros(self.num_answers) 
        answer_inds = [self.answer_vocab.stoi(answer) for answer in answers]

        if self.answer_counter is not None:        
            answer_cat_scores = np.zeros(self.num_answers)
            for answer, answer_ind in zip(answers, answer_inds):
                if answer_ind < 0:
                    continue                    
                answer_cats[answer_ind] = 1
                answer_cat_score = get_answer_score(self.answer_counter.get(answer, 0))
                answer_cat_scores[answer_ind] = answer_cat_score

            if not self.use_unanswerable:
                assert answer_cats.sum() > 0
                assert answer_cat_scores.sum() > 0
        else:
            raise NotImplementedError

        answer_cat = answer_cats.argmax()

        # get language features
        s_len = self.lang[scene_id][question_id]['s_len']
        q_len = self.lang[scene_id][question_id]['q_len']
        
        s_len = s_len if s_len <= CONF.TRAIN.MAX_TEXT_LEN else CONF.TRAIN.MAX_TEXT_LEN
        q_len = q_len if q_len <= CONF.TRAIN.MAX_TEXT_LEN else CONF.TRAIN.MAX_TEXT_LEN
        s_feat = self.lang[scene_id][question_id]['s_feat']
        q_feat = self.lang[scene_id][question_id]['q_feat']
        #
        # get point cloud features
        #
        mesh_vertices = self.scene_data[scene_id]['mesh_vertices']
        instance_labels = self.scene_data[scene_id]['instance_labels']
        semantic_labels = self.scene_data[scene_id]['semantic_labels']
        instance_bboxes = self.scene_data[scene_id]['instance_bboxes']
        bs_center = self.scene_data[scene_id]['bs_center']
        axis_align_matrix = self.scene_data[scene_id]['axis_align_matrix']
        coord_situation = np.array(position[ : 3])
        coord_situation += bs_center
        quat_situation = position[3 : ]
        # quat_situation.insert(0, quat_situation.pop())
        quat_situation = np.array(quat_situation)
        augment_vector = np.ones((1, 4))
        augment_vector[:, 0 : 3] = coord_situation
        augment_vector = np.dot(augment_vector, axis_align_matrix.transpose())
        coord_situation = augment_vector[:, 0 : 3]
        coord_situation = coord_situation.reshape(-1)
        rot_situation = R.from_quat(quat_situation)
        rot_mat_situation = np.array(rot_situation.as_matrix())
        rot_mat_situation = np.dot(axis_align_matrix[0 : 3, 0 : 3], rot_mat_situation)
        rot_situation = R.from_matrix(rot_mat_situation)
        quat_situation = np.array(rot_situation.as_quat())
        num_bboxes = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3]
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
            pcl_color = point_cloud[:,3:6]
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1) # p (50000, 7)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1)

        '''
        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA + '.hdf5', 'r', libver='latest')
            multiview = self.multiview_data[pid][scene_id]
            point_cloud = np.concatenate([point_cloud, multiview],1)
        '''

        #'''
        if self.use_multiview:
            # load multiview database
            enet_feats_file = os.path.join(MULTIVIEW_DATA, scene_id) + '.pkl'
            multiview = pickle.load(open(enet_feats_file, 'rb'))
            point_cloud = np.concatenate([point_cloud, multiview],1) # p (50000, 135)
        #'''
        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)        
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]
        
        # ------------------------------- LABELS ------------------------------    
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))

        if self.split != 'test':
            num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
            target_bboxes_mask[0:num_bbox] = 1
            target_bboxes[0:num_bbox,:] = instance_bboxes[:MAX_NUM_OBJ,0:6]

            point_votes = np.zeros([self.num_points, 3])
            point_votes_mask = np.zeros(self.num_points)

            # ------------------------------- DATA AUGMENTATION ------------------------------        
            if self.augment and not self.debug:
                
                if np.random.random() > 0.5:
                    # Flipping along the YZ plane
                    point_cloud[:,0] = -1 * point_cloud[:,0]
                    target_bboxes[:,0] = -1 * target_bboxes[:,0]
                    coord_situation[0] = -1 * coord_situation[0]
                    rot_situation = R.from_quat(quat_situation).as_matrix()
                    rot_situation[0, 0] *= -1
                    rot_situation[1, 1] *= -1
                    quat_situation = list(R.from_matrix(rot_situation).as_quat())
                    
                if np.random.random() > 0.5:
                    # Flipping along the XZ plane
                    point_cloud[:,1] = -1 * point_cloud[:,1]
                    target_bboxes[:,1] = -1 * target_bboxes[:,1]                                
                    coord_situation[1] = -1 * coord_situation[1]
                    rot_situation = R.from_quat(quat_situation).as_matrix()
                    rot_situation = rot_situation[[1, 0, 2], :]
                    rot_situation = rot_situation[:, [1, 0, 2]]
                    quat_situation = list(R.from_matrix(rot_situation).as_quat())

                # Rotation along X-axis
                
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = rotx(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, 'x')
                coord_situation = np.dot(coord_situation.reshape(1, -1), np.transpose(rot_mat)).reshape(-1)
                rot_situation = R.from_quat(quat_situation)
                rot_mat_situation = rot_situation.as_matrix()
                rot_mat_situation = np.dot(rot_mat, rot_mat_situation)
                rot_situation = R.from_matrix(rot_mat_situation)
                quat_situation = np.array(rot_situation.as_quat())

                # Rotation along Y-axis
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = roty(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, 'y')
                coord_situation = np.dot(coord_situation.reshape(1, -1), np.transpose(rot_mat)).reshape(-1)
                rot_situation = R.from_quat(quat_situation)
                rot_mat_situation = rot_situation.as_matrix()
                rot_mat_situation = np.dot(rot_mat, rot_mat_situation)
                rot_situation = R.from_matrix(rot_mat_situation)
                quat_situation = np.array(rot_situation.as_quat())

                # Rotation along up-axis/Z-axis
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = rotz(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, 'z')
                coord_situation = np.dot(coord_situation.reshape(1, -1), np.transpose(rot_mat)).reshape(-1)
                rot_situation = R.from_quat(quat_situation)
                rot_mat_situation = rot_situation.as_matrix()
                rot_mat_situation = np.dot(rot_mat, rot_mat_situation)
                rot_situation = R.from_matrix(rot_mat_situation)
                quat_situation = np.array(rot_situation.as_quat())

                # Translation
                point_cloud, target_bboxes, factor = self._translate(point_cloud, target_bboxes)
                coord_situation += factor
            # compute votes *AFTER* augmentation
            # generate votes
            # Note: since there's no map between bbox instance labels and
            # pc instance_labels (it had been filtered 
            # in the data preparation step) we'll compute the instance bbox
            # from the points sharing the same instance label. 
            for i_instance in np.unique(instance_labels):            
                # find all points belong to that instance
                ind = np.where(instance_labels == i_instance)[0]
                # find the semantic label            
                if semantic_labels[ind[0]] in DC.nyu40ids:
                    x = point_cloud[ind,:3]
                    center = 0.5*(x.min(0) + x.max(0))
                    point_votes[ind, :] = center - x
                    point_votes_mask[ind] = 1.0
            point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 
            
            class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox,-2]]
            # NOTE: set size class as semantic class. Consider use size2class.
            size_classes[0:num_bbox] = class_ind
            size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind,:]

        else:
            num_bbox = 1
            point_votes = np.zeros([self.num_points, 9]) # make 3 votes identical 
            point_votes_mask = np.zeros(self.num_points)

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        try:
            target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:,-2][0:num_bbox]]
        except KeyError:
            pass

        object_name = None if object_names is None else object_names[0]
        auxiliary_task = list(coord_situation) + list(quat_situation)
        data_dict = {}
        # data_dict['lang_feat'] = lang_feat.astype(np.float32) # language feature vectors
        data_dict['s_feat'] = s_feat.astype(np.float32)
        data_dict['q_feat'] = q_feat.astype(np.float32)
        data_dict['point_clouds'] = point_cloud.astype(np.float32) # point cloud data including features
        data_dict['s_len'] = np.array(s_len).astype(np.int64)
        data_dict['q_len'] = np.array(q_len).astype(np.int64)
        data_dict['center_label'] = target_bboxes.astype(np.float32)[:,0:3] # (MAX_NUM_OBJ, 3) for GT box center XYZ
        data_dict['heading_class_label'] = angle_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict['heading_residual_label'] = angle_residuals.astype(np.float32) # (MAX_NUM_OBJ,)
        data_dict['size_class_label'] = size_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict['size_residual_label'] = size_residuals.astype(np.float32) # (MAX_NUM_OBJ, 3)
        data_dict['num_bbox'] = np.array(num_bbox).astype(np.int64)
        data_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64) # (MAX_NUM_OBJ,) semantic class index
        data_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32) # (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        data_dict['vote_label'] = point_votes.astype(np.float32) # 
        data_dict['vote_label_mask'] = point_votes_mask.astype(np.int64) # point_obj_mask (gf3d)
        data_dict['scan_idx'] = np.array(idx).astype(np.int64)
        data_dict['pcl_color'] = pcl_color
        data_dict['auxiliary_task'] = np.array(auxiliary_task).astype(np.float32)
        data_dict['scene_id'] = np.array(int(self.scene_id_to_number[scene_id])).astype(np.int64)
        if type(question_id) == str:
            data_dict['question_id'] = np.array(int(question_id.split('-')[-1])).astype(np.int64)
        else:
            data_dict['question_id'] = np.array(int(question_id)).astype(np.int64)
        data_dict['pcl_color'] = pcl_color
        data_dict['load_time'] = time.time() - start
        data_dict['answer_cat'] = np.array(int(answer_cat)).astype(np.int64) # 1
        data_dict['answer_cats'] = answer_cats.astype(np.int64) # num_answers
        if self.test:
            data_dict["qid"] = question_id 
        if self.answer_cls_loss == 'bce' and self.answer_counter is not None:
            data_dict['answer_cat_scores'] = answer_cat_scores.astype(np.float32) # num_answers
        return data_dict

    
    def _get_raw2label(self):
        # mapping
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label

    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.sqa:
            scene_id = data['scene_id']

            for object_id, object_name in zip(data['object_ids'], data['object_names']):
                object_id = data['object_ids'][0]
                object_name = ' '.join(object_name.split('_'))

                if scene_id not in all_sem_labels:
                    all_sem_labels[scene_id] = []

                if scene_id not in cache:
                    cache[scene_id] = {}

                if object_id not in cache[scene_id]:
                    cache[scene_id][object_id] = {}
                    try:
                        all_sem_labels[scene_id].append(self.raw2label[object_name])
                    except KeyError:
                        all_sem_labels[scene_id].append(17)

        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.sqa:
            scene_id = data['scene_id']
            question_id = data['question_id']

            unique_multiples = []
            for object_id, object_name in zip(data['object_ids'], data['object_names']):
                object_id = data['object_ids'][0]
                object_name = ' '.join(object_name.split('_'))
                try:
                    sem_label = self.raw2label[object_name]
                except KeyError:
                    sem_label = 17

                unique_multiple_ = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1
                unique_multiples.append(unique_multiple_)

            unique_multiple = max(unique_multiples)

            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            unique_multiple_lookup[scene_id][question_id] = unique_multiple

        return unique_multiple_lookup

    def _tranform_text_glove(self, token_type='token'):
        with open(GLOVE_PICKLE, 'rb') as f:
            glove = pickle.load(f)

        lang = {}
        for data in self.sqa:
            scene_id = data['scene_id']
            question_id = data['question_id']

            if scene_id not in lang:
                lang[scene_id] = {}

            if question_id in lang[scene_id]:
                continue
            lang[scene_id][question_id] = {}
            # tokenize the description
            s_tokens = data["situation"]
            q_tokens = data["question"]
            s_embeddings = np.zeros((CONF.TRAIN.MAX_TEXT_LEN, 300))
            q_embeddings = np.zeros((CONF.TRAIN.MAX_TEXT_LEN, 300))
            for token_id in range(CONF.TRAIN.MAX_TEXT_LEN):
                if token_id < len(s_tokens):
                    token = s_tokens[token_id]
                    if not self.wos:     
                        if token in glove:
                            s_embeddings[token_id] = glove[token]
                        else:
                            s_embeddings[token_id] = glove['unk']
                    else:
                        s_embeddings[token_id] = glove['unk']
            for token_id in range(CONF.TRAIN.MAX_TEXT_LEN):
                if token_id < len(q_tokens):
                    token = q_tokens[token_id]
                    if token in glove:
                        q_embeddings[token_id] = glove[token]
                    else:
                        q_embeddings[token_id] = glove['unk']
            # store
            lang[scene_id][question_id]['s_feat'] = s_embeddings
            lang[scene_id][question_id]['s_len'] = len(s_tokens)
            lang[scene_id][question_id]['q_feat'] = q_embeddings
            lang[scene_id][question_id]['q_len'] = len(q_tokens)
            lang[scene_id][question_id]['s_token'] = s_tokens
            lang[scene_id][question_id]['q_token'] = q_tokens
        temp = list(DC.type2class.keys())
        class_embedding = np.zeros((len(temp), 300))
        for token_id in range(len(temp)):
            token = temp[token_id]
            if token in glove:
                class_embedding[token_id] = glove[token]
            else:
                class_embedding[token_id] = glove['unk']
        self.class_embedding = class_embedding
        return lang

    def _load_data(self):
        print('loading data...')
        # load language features
        self.lang = self._tranform_text_glove('token')

        # add scannet data
        self.scene_list = sorted(list(set([data['scene_id'] for data in self.sqa])))

        # load scene data
        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            self.scene_data[scene_id]['mesh_vertices'] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+'_aligned_vert.npy') # axis-aligned
            temp = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+'_vert.npy')
            self.scene_data[scene_id]['bs_center'] = (np.max(temp[:, 0 : 3], axis=0) + np.min(temp[:, 0 : 3], axis=0)) / 2
            meta_file = open(os.path.join(CONF.PATH.SCANNET_SCANS, scene_id, scene_id+".txt")).readlines()
            axis_align_matrix = None
            for line in meta_file:
                if 'axisAlignment' in line:
                    axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            if axis_align_matrix != None:
                axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
            self.scene_data[scene_id]['instance_labels'] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+'_ins_label.npy')
            self.scene_data[scene_id]['semantic_labels'] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+'_sem_label.npy')
            self.scene_data[scene_id]['axis_align_matrix'] = axis_align_matrix if axis_align_matrix is not  None else np.eye(4)
            self.scene_data[scene_id]['instance_bboxes'] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+'_aligned_bbox.npy')

        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()
        self.label2raw = {v: k for k, v in self.raw2label.items()}
        # if self.split != 'test':
        #     self.unique_multiple_lookup = self._get_unique_multiple_lookup()

    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]
        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]
        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox, factor
