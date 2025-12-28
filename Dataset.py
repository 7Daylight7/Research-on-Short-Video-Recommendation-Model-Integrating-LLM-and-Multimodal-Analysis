import time
import random
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from config.config import USE_CUDA


def data_load(dataset, has_v=True, has_a=True, has_t=True):
    dir_str = './Data/' + dataset
    train_edge = np.load(dir_str+'/train_sample.npy', allow_pickle=True)
    user_item_dict = np.load(dir_str+'/user_item_dict_sample.npy', allow_pickle=True).item()

    if dataset == 'movielens':
        num_user = 55485
        num_item = 5986
        v_feat = np.load(dir_str+'/FeatureVideo_normal.npy', allow_pickle=True) if has_v else None
        a_feat = np.load(dir_str+'/FeatureAudio_avg_normal.npy', allow_pickle=True) if has_a else None
        t_feat = np.load(dir_str+'/FeatureText_stl_normal.npy', allow_pickle=True) if has_t else None
        if USE_CUDA:
            v_feat = torch.tensor(v_feat, dtype=torch.float).cuda() if has_v else None
            a_feat = torch.tensor(a_feat, dtype=torch.float).cuda() if has_a else None
            t_feat = torch.tensor(t_feat, dtype=torch.float).cuda() if has_t else None
        else:
            v_feat = torch.tensor(v_feat, dtype=torch.float) if has_v else None
            a_feat = torch.tensor(a_feat, dtype=torch.float) if has_a else None
            t_feat = torch.tensor(t_feat, dtype=torch.float) if has_t else None
    elif dataset == 'tiktok':
        num_user = 100
        num_item = 1651
        if has_v:
            v_feat = torch.load(dir_str+'/v_feat_sample.pt')
            if USE_CUDA:
                v_feat = torch.tensor(v_feat, dtype=torch.float).cuda()
            else:
                v_feat = torch.tensor(v_feat, dtype=torch.float)
        else:
            v_feat = None

        if has_a:
            a_feat = torch.load(dir_str+'/a_feat_sample.pt')
            if USE_CUDA:
                a_feat = torch.tensor(a_feat, dtype=torch.float).cuda()
            else:
                a_feat = torch.tensor(a_feat, dtype=torch.float)
        else:
            a_feat = None
        
        t_feat = torch.load(dir_str+'/t_feat_sample.pt') if has_t else None
    elif dataset == 'Kwai':
        num_user = 7010
        num_item = 86483
        v_feat = torch.load(dir_str+'/feat_v.pt')
        if USE_CUDA:
            v_feat = torch.tensor(v_feat, dtype=torch.float).cuda()
        else:
            v_feat = torch.tensor(v_feat, dtype=torch.float)
        a_feat = t_feat = None

    return num_user, num_item, train_edge, user_item_dict, v_feat, a_feat, t_feat

class TrainingDataset(Dataset):
    def __init__(self, num_user, num_item, user_item_dict, edge_index):
        self.edge_index = edge_index
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.all_list = list(range(num_user, num_user+num_item))

    def __len__(self):
        return len(self.edge_index)

    def __getitem__(self, index):
        user, pos_item = self.edge_index[index]
        while True:
            neg_item = random.sample(self.all_list, 1)[0]
            if neg_item not in self.user_item_dict[user]:
                break
        return torch.LongTensor([user,user]), torch.LongTensor([pos_item, neg_item])
