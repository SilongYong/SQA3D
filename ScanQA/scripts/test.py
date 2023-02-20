import os
import sys
import json
import argparse
import collections
import torch
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
from torch.utils.data import DataLoader
from datetime import datetime
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.sepdataset import ScannetQADataset, ScannetQADatasetConfig
from lib.config import CONF 
from models.sqa_module import ScanQA
from collections import OrderedDict

# constants
DC = ScannetQADatasetConfig()

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debugging mode")
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. XYZ_COLOR", default="")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    # Training
    parser.add_argument("--cur_criterion", type=str, default="answer_acc_at1", help="data augmentation type")
    parser.add_argument("--batch_size", type=int, help="batch size", default=16)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=1000) # 5000
    parser.add_argument("--train_num_scenes", type=int, default=-1, help="Number of train scenes [default: -1]")
    parser.add_argument("--val_num_scenes", type=int, default=-1, help="Number of val scenes [default: -1]")
    parser.add_argument("--test_num_scenes", type=int, default=-1, help="Number of test scenes [default -1]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # Optimizer   
    parser.add_argument("--optim_name", type=str, help="optimizer name", default="adam")
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--lr", type=float, help="initial learning rate", default=5e-4)
    parser.add_argument("--adam_beta1", type=float, help="beta1 hyperparameter for the Adam optimizer", default=0.9)
    parser.add_argument("--adam_beta2", type=float, help="beta2 hyperparameter for the Adam optimizer", default=0.999) # 0.98
    parser.add_argument("--adam_epsilon", type=float, help="epsilon hyperparameter for the Adam optimizer", default=1e-8) # 1e-9
    parser.add_argument("--amsgrad", action="store_true", help="Use amsgrad for Adam")
    parser.add_argument('--lr_decay_step', nargs='+', type=int, default=[15, 35]) # 15
    parser.add_argument("--lr_decay_rate", type=float, help="decay rate of learning rate", default=0.2) # 01, 0.2
    parser.add_argument('--bn_decay_step', type=int, default=20)
    parser.add_argument("--bn_decay_rate", type=float, help="bn rate", default=0.5)
    parser.add_argument("--max_grad_norm", type=float, help="Maximum gradient norm ", default=1.0)
    # Data
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use data augmentations.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    # Model
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden layer size[default: 256]")
    ## pointnet & votenet & proposal
    parser.add_argument("--vote_radius", type=float, help="", default=0.3) # 5
    parser.add_argument("--vote_nsample", type=int, help="", default=16) # 512
    parser.add_argument("--pointnet_width", type=int, help="", default=1)
    parser.add_argument("--pointnet_depth", type=int, help="", default=2)
    parser.add_argument("--seed_feat_dim", type=int, help="", default=256) # or 288
    parser.add_argument("--proposal_size", type=int, help="", default=128)    
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--use_seed_lang", action="store_true", help="Fuse seed feature and language feature.")    
    ## module option
    parser.add_argument("--no_object_mask", action="store_true", help="objectness_mask for qa")
    parser.add_argument("--no_aux_reg", action="store_true", help="Do NOT use auxiliary task regressor.")
    parser.add_argument("--no_answer", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    # Pretrain
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    # Loss
    parser.add_argument("--vote_loss_weight", type=float, help="vote_net loss weight", default=1.0)
    parser.add_argument("--objectness_loss_weight", type=float, help="objectness loss weight", default=0.5)
    parser.add_argument("--box_loss_weight", type=float, help="box loss weight", default=1.0)
    parser.add_argument("--sem_cls_loss_weight", type=float, help="sem_cls loss weight", default=0.1)
    parser.add_argument("--ref_loss_weight", type=float, help="reference loss weight", default=0.1)
    parser.add_argument("--aux_loss_weight", type=float, help="auxiliary task loss weight", default=0.1)
    parser.add_argument("--answer_loss_weight", type=float, help="answer loss weight", default=0.1)  
    # Answer
    parser.add_argument("--answer_cls_loss", type=str, help="answer classifier loss", default="bce") # ce, bce
    parser.add_argument("--answer_max_size", type=int, help="maximum size of answer candicates", default=-1) # default use all
    parser.add_argument("--answer_min_freq", type=int, help="minimum frequence of answers", default=1)
    parser.add_argument("--answer_pdrop", type=float, help="dropout_rate of answer_cls", default=0.3)
    # Question
    parser.add_argument("--tokenizer_name", type=str, help="Pretrained tokenizer name", default="spacy_blank")
    parser.add_argument("--lang_num_layers", type=int, default=1, help="Number of GRU layers")
    parser.add_argument("--lang_use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--lang_pdrop", type=float, help="dropout_rate of lang_cls", default=0.3)
    ## MCAN
    parser.add_argument("--mcan_pdrop", type=float, help="", default=0.1)
    parser.add_argument("--mcan_flat_mlp_size", type=int, help="", default=256) # mcan: 512
    parser.add_argument("--mcan_flat_glimpses", type=int, help="", default=1)
    parser.add_argument("--mcan_flat_out_size", type=int, help="", default=512) # mcan: 1024
    parser.add_argument("--mcan_num_heads", type=int, help="", default=8)
    parser.add_argument("--mcan_num_layers", type=int, help="", default=2) # mcan: 6
    ## Ablation
    parser.add_argument("--wo3d", action="store_true", help="DO NOT use 3D branch")
    parser.add_argument("--wos", action="store_true", help="DO NOT use situation")
    parser.add_argument("--Hpos", type=float, default=1.0, help="position loss weight")
    parser.add_argument("--Hrot", type=float, default=1.0, help="rotation loss weight")

    ## which split to evaluate
    parser.add_argument("--split", type=str, choices=['train', 'val', 'test'], default='train')
    
    ## checkpoint
    parser.add_argument("--ckpt", type=str, help="checkpoint to evaluate")
    args = parser.parse_args()
    return args
    

def get_answer_cands(args, answer_counter_list):
    answer_counter = answer_counter_list
    answer_counter = collections.Counter(sorted(answer_counter))
    num_all_answers = len(answer_counter)
    answer_max_size = args.answer_max_size
    if answer_max_size < 0:
        answer_max_size = len(answer_counter)
    answer_counter = dict([x for x in answer_counter.most_common()[:answer_max_size] if x[1] >= args.answer_min_freq])
    print("using {} answers out of {} ones".format(len(answer_counter), num_all_answers))    
    answer_cands = sorted(answer_counter.keys())
    return answer_cands, answer_counter


def get_dataloader(args, sqa, all_scene_list, split, config, augment, answer_counter_list, test=False):
    answer_cands, answer_counter = get_answer_cands(args, answer_counter_list)
    config.num_answers = len(answer_cands)

    tokenizer = None

    dataset = ScannetQADataset(
        sqa=sqa[split], 
        sqa_all_scene=all_scene_list, 
        answer_cands=answer_cands,
        answer_counter=answer_counter,
        answer_cls_loss=args.answer_cls_loss,
        split=split, 
        num_points=args.num_points, 
        use_height=(not args.no_height),
        use_color=args.use_color, 
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        tokenizer=tokenizer,
        augment=augment,
        debug=args.debug,
        wos=args.wos,
        test=test
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    return dataset, dataloader


def get_model(args, config):
    lang_emb_size = 300 # glove emb_size

    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)

    model = ScanQA(
        num_answers=config.num_answers,
        # proposal
        input_feature_dim=input_channels,            
        num_object_class=config.num_class, 
        num_heading_bin=config.num_heading_bin,
        num_size_cluster=config.num_size_cluster,
        mean_size_arr=config.mean_size_arr,
        num_proposal=args.num_proposals, 
        seed_feat_dim=args.seed_feat_dim,
        proposal_size=args.proposal_size,
        pointnet_width=args.pointnet_width,
        pointnet_depth=args.pointnet_depth,        
        vote_radius=args.vote_radius, 
        vote_nsample=args.vote_nsample,            
        # qa
        #answer_cls_loss="ce",
        answer_pdrop=args.answer_pdrop,
        mcan_num_layers=args.mcan_num_layers,
        mcan_num_heads=args.mcan_num_heads,
        mcan_pdrop=args.mcan_pdrop,
        mcan_flat_mlp_size=args.mcan_flat_mlp_size, 
        mcan_flat_glimpses=args.mcan_flat_glimpses,
        mcan_flat_out_size=args.mcan_flat_out_size,
        # lang
        lang_use_bidir=args.lang_use_bidir,
        lang_num_layers=args.lang_num_layers,
        lang_emb_size=lang_emb_size,
        lang_pdrop=args.lang_pdrop,
        # common
        hidden_size=args.hidden_size,
        # option
        use_object_mask=(not args.no_object_mask),
        use_aux_reg=(not args.no_aux_reg),
        use_answer=(not args.no_answer),
        wo3d = args.wo3d,
    )

    # to CUDA
    model = model.cuda()
    print(next(model.parameters()).device)
    return model


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataloader):
    model = get_model(args, DC)
    #wandb.watch(model, log_freq=100)
    if not args.no_aux_reg:
        from lib.solver_aux import Solver
    else:
        from lib.solver import Solver
    if args.optim_name == 'adam':
        model_params = [{"params": model.parameters()}]
        optimizer = optim.Adam(
            model_params,
            lr=args.lr, 
            betas=[args.adam_beta1, args.adam_beta2],
            eps=args.adam_epsilon,
            weight_decay=args.wd, 
            amsgrad=args.amsgrad)
    elif args.optim_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, 
                                betas=[args.adam_beta1, args.adam_beta2],
                                eps=args.adam_epsilon,
                                weight_decay=args.wd, 
                                amsgrad=args.amsgrad)
    elif args.optim_name == 'adamw_cb':
        from transformers import AdamW
        optimizer = AdamW(model.parameters(), lr=args.lr, 
                                betas=[args.adam_beta1, args.adam_beta2],
                                eps=args.adam_epsilon,
                                weight_decay=args.wd)
    else:
        raise NotImplementedError

    print('set optimizer...')
    print(optimizer)
    print()

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_"+args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    loss_weights = {}
    loss_weights['vote_loss']       = args.vote_loss_weight
    loss_weights['objectness_loss'] = args.objectness_loss_weight 
    loss_weights['box_loss']        = args.box_loss_weight
    loss_weights['sem_cls_loss']    = args.sem_cls_loss_weight
    loss_weights['aux_loss']       = args.aux_loss_weight
    loss_weights['answer_loss']     = args.answer_loss_weight

    solver = Solver(
        model=model, 
        config=DC, 
        dataloader=dataloader, 
        optimizer=optimizer, 
        stamp=stamp, 
        val_step=args.val_step,
        cur_criterion=args.cur_criterion,
        detection=not args.no_detection,
        use_reference=not args.no_reference, 
        use_answer=not args.no_answer,
        use_aux_regressor=not args.no_aux_reg,
        max_grad_norm=args.max_grad_norm,
        lr_decay_step=args.lr_decay_step,
        lr_decay_rate=args.lr_decay_rate,
        bn_decay_step=args.bn_decay_step,
        bn_decay_rate=args.bn_decay_rate,
        loss_weights=loss_weights,
        loss_pos = args.Hpos,
        loss_rot = args.Hrot,
    )
    num_params = get_num_params(model)

    return solver, num_params, root, stamp

def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value
    
    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    answer_vocab = train_dataset.answer_counter
    with open(os.path.join(root, "answer_vocab.json"), "w") as f:
        json.dump(answer_vocab, f, indent=4)        



def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])
    return scene_list

def get_sqa(sqa_train, sqa_val, sqa_test, train_num_scenes, val_num_scenes, test_num_scenes):
    # get initial scene list
    train_scene_list = sorted(list(set([data["scene_id"] for data in sqa_train])))
    val_scene_list = sorted(list(set([data["scene_id"] for data in sqa_val])))
    test_scene_list = sorted(list(set([data["scene_id"] for data in sqa_test])))
    # set train_num_scenes
    if train_num_scenes <= -1: 
        train_num_scenes = len(train_scene_list)
    else:
        assert len(train_scene_list) >= train_num_scenes

    # slice train_scene_list
    train_scene_list = train_scene_list[:train_num_scenes]

    # filter data in chosen scenes
    new_sqa_train = []
    for data in sqa_train:
        if data["scene_id"] in train_scene_list:
            new_sqa_train.append(data)

    # set val_num_scenes
    if val_num_scenes <= -1: 
        val_num_scenes = len(val_scene_list)
    else:
        assert len(val_scene_list) >= val_num_scenes

    # slice val_scene_list
    val_scene_list = val_scene_list[:val_num_scenes]        
    
    new_sqa_val = []
    for data in sqa_val:
        if data["scene_id"] in val_scene_list:
            new_sqa_val.append(data)
    
    # set val_num_scenes
    if test_num_scenes <= -1: 
        test_num_scenes = len(test_scene_list)
    else:
        assert len(test_scene_list) >= test_num_scenes

    # slice val_scene_list
    test_scene_list = test_scene_list[:test_num_scenes]        
    
    new_sqa_test = []
    for data in sqa_test:
        if data["scene_id"] in test_scene_list:
            new_sqa_test.append(data)

    # all sqa scene
    all_scene_list = train_scene_list + val_scene_list + test_scene_list
    return new_sqa_train, new_sqa_val, new_sqa_test, all_scene_list

def test(args, SQA_TRAIN, SQA_VAL, SQA_TEST, path, answer_counter_list):

    sqa_train, sqa_val, sqa_test, all_scene_list = get_sqa(SQA_TRAIN, SQA_VAL, SQA_TEST, args.train_num_scenes, args.val_num_scenes, args.test_num_scenes)
    sqa = {
        "train" : sqa_train,
        "val" : sqa_val,
        "test" : sqa_test
    }
    val_dataset, val_dataloader = get_dataloader(args, sqa, all_scene_list, args.split, DC, False, answer_counter_list, test=True)

    ckpt = torch.load(path, map_location="cuda:0")
    new_ckpt = OrderedDict()
    # for key in ckpt.keys():                  # for pretrained models
    #     if "lang_cls" in key:
    #         newkey = key.replace("lang_cls", "aux_reg")
    #         new_ckpt[newkey] = ckpt[key]
    #     else:
    #         new_ckpt[key] = ckpt[key]
    
    model = get_model(args, DC)

    sd_before_load = deepcopy(model.state_dict())
    model.load_state_dict(ckpt, strict=False)
    sd_after_load = deepcopy(model.state_dict())
    same_keys = [k for k in sd_before_load if torch.equal(sd_before_load[k], sd_after_load[k])]
    new_keys = []
    for key in same_keys:
        new_keys.append(key)
    print('-------------------- Loaded weights --------------------')
    print(f'Weights unloaded:{new_keys}')
    print('----------------------------')

    
    model.eval()
    with torch.no_grad():
        count = 0
        right_count = 0
        for data_dict in val_dataloader:
            # assert False
            for key in data_dict:
                if type(data_dict[key]) is dict:
                    data_dict[key] = {k:v.cuda() for k, v in  data_dict[key].items()}
                else:
                    data_dict[key] = data_dict[key].cuda()
            data_dict = model(data_dict)
            pred_answer = torch.argmax(data_dict["answer_scores"], 1).cpu().detach().item()
            gt_answer = torch.argmax(data_dict["answer_cats"].squeeze()).cpu().detach().item()
            if pred_answer == gt_answer:
                right_count += 1
            count += 1
        print("overall acc:", right_count / count)

    return right_count / count

if __name__ == "__main__":
    args = parse_option()
    # setting
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    project_name = "SQA"
    SQA_TRAIN = json.load(open(os.path.join(CONF.PATH.SQA, project_name + "_train.json"))) 
    SQA_VAL = json.load(open(os.path.join(CONF.PATH.SQA, project_name + "_val.json")))
    SQA_TEST = json.load(open(os.path.join(CONF.PATH.SQA, project_name + "_test.json")))
    answer_counter_list = json.load(open(os.path.join(CONF.PATH.SQA, "answer_counter.json")))
    torch.cuda.set_device('cuda:{}'.format(args.gpu))
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed) 
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    path = args.ckpt
    save_list = test(args, SQA_TRAIN, SQA_VAL, SQA_TEST, path, answer_counter_list)
    
