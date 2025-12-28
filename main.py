"""
MMGCN项目入口文件
该文件是MMGCN（Multi-Modal Graph Convolutional Network）项目的主入口，负责：
1. 参数解析与配置
2. 数据加载与预处理
3. 模型初始化
4. 训练循环控制
5. 模型评估与保存
"""
import argparse
import os
import time
import numpy as np
import torch
from Dataset import TrainingDataset, data_load  # 数据加载模块
from Model_MMGCN import Net  # 核心模型
from torch.utils.data import DataLoader  # 数据加载器
from Train import train  # 训练函数
from Full_vt import full_vt  # 评估函数
# from torch.utils.tensorboard import SummaryWriter  # TensorBoard可视化


if __name__ == '__main__':
    # 参数解析器，用于配置模型训练的各种超参数
    parser = argparse.ArgumentParser(description='MMGCN: Multi-Modal Graph Convolutional Network for Recommender Systems')

    # 随机种子设置，保证实验可重复性
    parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    # 是否使用CUDA进行训练
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    # 选用的数据集
    parser.add_argument('--data_path', default='movielens', help='Dataset path')
    # 保存文件名
    parser.add_argument('--save_file', default='', help='Filename')

    # 模型权重加载和保存路径
    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default=None, help='Writing weight filename.')

    # 优化器参数
    parser.add_argument('--l_r', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=1, help='Workers number.')

    # 模型参数
    parser.add_argument('--dim_E', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--prefix', default='', help='Prefix of save_file.')
    parser.add_argument('--aggr_mode', default='add', help='Aggregation Mode.')
    parser.add_argument('--topK', type=int, default=10, help='Workers number.')

    # 模态选择参数，控制是否使用特定模态
    parser.add_argument('--has_entropy_loss', default='False', help='Has Cross Entropy loss.')
    parser.add_argument('--has_weight_loss', default='False', help='Has Weight Loss.')
    parser.add_argument('--has_v', default='True', help='Has Visual Features.')
    parser.add_argument('--has_a', default='True', help='Has Acoustic Features.')
    parser.add_argument('--has_t', default='True', help='Has Textual Features.')

    # 解析命令行参数
    args = parser.parse_args()

    # 设置随机种子，确保实验可重复性
    seed = args.seed
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置CUDA随机种子
    # 设置设备（GPU或CPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # 提取参数值到变量
    data_path = args.data_path
    save_file = args.save_file
    learning_rate = args.l_r
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epoch = args.num_epoch
    topK = args.topK
    prefix = args.prefix
    aggr_mode = args.aggr_mode

    has_v = True if args.has_v == 'True' else False
    has_a = True if args.has_a == 'True' else False
    has_t = True if args.has_t == 'True' else False
    has_entropy_loss = True if args.has_entropy_loss == 'True' else False
    has_weight_loss = True if args.has_weight_loss == 'True' else False
    dim_E = args.dim_E

    # TensorBoard可视化器
    writer = None  # SummaryWriter()
    # with open(data_path+'/result/result{0}_{1}.txt'.format(l_r, weight_decay), 'w') as save_file:
    #  save_file.write('---------------------------------lr: {0} \t Weight_decay:{1} ---------------------------------\r\n'.format(l_r, weight_decay))
    print('Data loading ...')

    # 加载数据集：用户数、物品数、训练边、用户-物品映射字典、多模态特征
    num_user, num_item, train_edge, user_item_dict, v_feat, a_feat, t_feat = data_load(data_path)

    v_feat = torch.tensor(v_feat, dtype=torch.float).to(device) if has_v else None
    a_feat = torch.tensor(a_feat, dtype=torch.float).to(device) if has_a else None
    t_feat = torch.tensor(t_feat, dtype=torch.float).to(device) if has_t else None

    # 创建训练数据集和数据加载器
    train_dataset = TrainingDataset(num_user, num_item, user_item_dict, train_edge)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)

    # 加载验证和测试数据
    val_data = np.load('./Data/'+data_path+'/val_full.npy', allow_pickle=True)
    test_data = np.load('./Data/'+data_path+'/test_full.npy', allow_pickle=True)
    print('Data has been loaded.')

    # 初始化模型
    # 参数说明：
    # v_feat, a_feat, t_feat: 多模态特征
    # words_tensor: 文本词张量(None)
    # train_edge: 训练边索引对
    # batch_size: 批次大小
    # num_user, num_item: 用户数和物品数
    # aggr_mode: 聚合方式
    # concate: 是否拼接（这里为False表示使用加和）
    # num_layer: 图卷积层数
    # has_id: 是否使用ID嵌入
    # user_item_dict: 用户-物品映射字典
    # weight_decay: 权重衰减
    # dim_E: 嵌入维度
    model = Net(v_feat, a_feat, t_feat, None, train_edge, batch_size, num_user, num_item, 'mean', False, 2, True, user_item_dict, weight_decay, dim_E).to(device)
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}])

    # 用于记录最佳性能指标
    max_precision = 0.0
    max_recall = 0.0
    max_NDCG = 0.0
    val_max_recall = 0.0
    num_decreases = 0  # 用于早停机制

    # 训练循环
    for epoch in range(num_epoch):
        print(f"\n===== 第 {epoch+1}/{num_epoch} 轮训练 =====")
        loss = train(epoch, len(train_dataset), train_dataloader, model, optimizer, batch_size, writer)

        # 检查是否出现NaN损失
        if torch.isnan(loss):
            # 记录NaN错误信息
            with open('./Data/'+data_path+'/result_{0}.txt'.format(save_file), 'a') as save_file:
                    save_file.write('lr: {0} \t Weight_decay:{1} is Nan\r\n'.format(learning_rate, weight_decay))
            break
        # 释放GPU缓存
        torch.cuda.empty_cache()

        val_precision, val_recall, val_ndcg = full_vt(epoch, model, val_data, 'Val', writer)
        test_precision, test_recall, test_ndcg = full_vt(epoch, model, test_data, 'Test', writer)

        if val_recall > val_max_recall:
            val_max_recall = val_recall
            max_precision = test_precision
            max_recall = test_recall
            max_NDCG = test_ndcg
            num_decreases = 0
        else:
            if num_decreases > 20:
                with open('./Data/'+data_path+'/result_{0}.txt'.format(save_file), 'a') as save_file:
                    save_file.write('lr: {0} \t Weight_decay:{1} =====> Precision:{2} \t Recall:{3} \t NDCG:{4}\r\n'.
                                    format(learning_rate, weight_decay, max_precision, max_recall, max_NDCG))
                break
            else:
                num_decreases += 1
