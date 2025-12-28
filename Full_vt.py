"""
MMGCN模型评估模块
该文件实现了MMGCN模型的评估功能，主要包括full_vt函数，用于计算模型的precision、recall和NDCG指标，
并通过TensorBoard记录评估结果和可视化数据。
"""
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import no_grad
import numpy as np


def full_vt(epoch, model, data, prefix, writer=None):
    """
    模型评估函数

    参数：
    - epoch: 当前训练轮数
    - model: 训练好的MMGCN模型
    - data: 评估数据集（验证集或测试集）
    - prefix: 评估类型前缀（'Val'或'Test'）
    - writer: TensorBoard写入器（可选）

    返回：
    - precision: 模型在评估数据集上的Precision指标
    - recall: 模型在评估数据集上的Recall指标
    - ndcg_score: 模型在评估数据集上的NDCG指标
    """
    print(prefix+' start...')
    model.eval()

    with no_grad():
        precision, recall, ndcg_score = model.full_accuracy(data)
        print('---------------------------------{0}-th Precition:{1:.4f} Recall:{2:.4f} NDCG:{3:.4f}---------------------------------'.format(
            epoch, precision, recall, ndcg_score))
        if writer is not None:
            writer.add_scalar(prefix+'_Precition', precision, epoch)
            writer.add_scalar(prefix+'_Recall', recall, epoch)
            writer.add_scalar(prefix+'_NDCG', ndcg_score, epoch)

            writer.add_histogram(prefix+'_visual_distribution', model.v_rep, epoch)
            writer.add_histogram(prefix+'_acoustic_distribution', model.a_rep, epoch)
            writer.add_histogram(prefix+'_textual_distribution', model.t_rep, epoch)
            
            writer.add_histogram(prefix+'_user_visual_distribution', model.user_preferences[:,:44], epoch)
            writer.add_histogram(prefix+'_user_acoustic_distribution', model.user_preferences[:, 44:-44], epoch)
            writer.add_histogram(prefix+'_user_textual_distribution', model.user_preferences[:, -44:], epoch)

            writer.add_embedding(model.v_rep)
            writer.add_embedding(model.a_rep)
            writer.add_embedding(model.t_rep)
            
        return precision, recall, ndcg_score



