"""
MMGCN核心模型实现

该文件定义了MMGCN（Multi-Modal Graph Convolutional Network）的核心模型结构，包括：
1. GCN类：实现了多层图卷积网络，用于处理单模态信息
2. Net类：整合多模态GCN，实现用户和物品的多模态表示学习
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from BaseModel import BaseModel  # 基础消息传递模型
from torch_geometric.utils import scatter  # 图数据聚合工具


class GCN(torch.nn.Module):
    """
    图卷积网络（Graph Convolutional Network）类

    该类实现了一个多层的图卷积网络，用于在用户-物品交互图上进行信息传播和特征学习。
    每个GCN模块处理一种模态的特征信息。
    """
    def __init__(self, edge_index, batch_size, num_user, num_item, dim_feat, dim_id, aggr_mode, concate, num_layer, has_id, dim_latent=None):
        """
        初始化GCN模块

        参数：
        - edge_index: 边索引，定义用户-物品交互图的结构
        - batch_size: 批次大小
        - num_user: 用户数量
        - num_item: 物品数量
        - dim_feat: 特征维度
        - dim_id: ID嵌入维度
        - aggr_mode: 聚合模式，如'add'、'mean'等
        - concate: 是否拼接原始特征和聚合特征
        - num_layer: 图卷积层数（虽然参数存在但当前实现固定为3层）
        - has_id: 是否使用ID嵌入
        - dim_latent: 潜在空间维度，用于特征降维（可选）
        """
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer  # 目前未使用，默认为3层
        self.has_id = has_id

        # 根据是否有潜在空间维度选择不同的初始化方式(原始高维特征投影至设定D维空间)
        if self.dim_latent:
            # 初始化用户特征向量
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).cuda()
            # MLP用于特征降维
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            # 第一层图卷积嵌入层
            self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            # 线性变换层
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_latent, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight) 

        else:
            # 直接使用特征维度的用户特征向量初始化
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).cuda()
            # 第一层图卷积嵌入层
            self.conv_embed_1 = BaseModel(self.dim_feat, self.dim_feat, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            # 线性变换层
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        # 第二层图卷积嵌入层
        self.conv_embed_2 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id, self.dim_id)

        # 第三层图卷积嵌入层
        self.conv_embed_3 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_3.weight)
        self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer3.weight)
        self.g_layer3 = nn.Linear(self.dim_id+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id, self.dim_id)  

    def forward(self, features, id_embedding):
        """
        GCN前向传播

        参数：
        - features: 物品特征（视觉/声学/文本）
        - id_embedding: 用户和物品的ID嵌入

        返回：
        - x: 经过多层图卷积后的节点表示
        """
        # 如果定义了潜在维度，则对特征进行降维
        temp_features = self.MLP(features) if self.dim_latent else features

        # 拼接用户偏好和物品特征
        x = torch.cat((self.preference, temp_features),dim=0)
        # 特征归一化
        x = F.normalize(x).cuda()

        # 第一层图卷积
        # 聚合邻居信息
        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index))  # equation 1
        # 原始特征的线性变换，并可选地添加ID嵌入
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer1(x))  # equation 5
        # 融合聚合特征和变换后的原始特征
        x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer1(h)+x_hat)

        # 第二层图卷积（与第一层类似）
        h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))  # equation 1
        x_hat = F.leaky_relu(self.linear_layer2(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer2(x))  # equation 5
        x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer2(h)+x_hat)

        # 第三层图卷积（与前两层类似）
        h = F.leaky_relu(self.conv_embed_3(x, self.edge_index))  # equation 1
        x_hat = F.leaky_relu(self.linear_layer3(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer3(x))#equation 5
        x = F.leaky_relu(self.g_layer3(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer3(h)+x_hat)

        return x


class Net(torch.nn.Module):
    """
    MMGCN主网络类
    """
    def __init__(self, v_feat, a_feat, t_feat, words_tensor, edge_index, batch_size, num_user, num_item, aggr_mode, concate, num_layer, has_id, user_item_dict, reg_weight, dim_x):
        """
        初始化MMGCN主网络

        参数：
        - v_feat: 视觉特征
        - a_feat: 声学特征
        - t_feat: 文本特征
        - words_tensor: 文本词张量
        - edge_index: 边索引
        - batch_size: 批次大小
        - num_user: 用户数量
        - num_item: 物品数量
        - aggr_mode: 聚合模式
        - concate: 是否拼接
        - num_layer: 图卷积层数
        - has_id: 是否使用ID嵌入
        - user_item_dict: 用户-物品交互字典
        - reg_weight: 正则化权重
        - dim_x: 嵌入维度
        """
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.user_item_dict = user_item_dict
        # 用于BPR损失计算的权重
        self.weight = torch.tensor([[1.0],[-1.0]]).cuda()
        self.reg_weight = reg_weight

        # 构建无向图（用户-物品和物品-用户）
        self.edge_index = torch.tensor(edge_index).t().contiguous().cuda()
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1,0]]), dim=1)
        self.num_modal = 0

        # 初始化视觉GCN模块
        self.v_feat = torch.tensor(v_feat,dtype=torch.float).cuda()
        self.v_gcn = GCN(self.edge_index, batch_size, num_user, num_item, self.v_feat.size(1), dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id, dim_latent=256)

        # 初始化声学GCN模块
        self.a_feat = torch.tensor(a_feat,dtype=torch.float).cuda()
        self.a_gcn = GCN(self.edge_index, batch_size, num_user, num_item, self.a_feat.size(1), dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id)

        # 初始化文本GCN模块
        self.t_feat = torch.tensor(t_feat,dtype=torch.float).cuda()
        self.t_gcn = GCN(self.edge_index, batch_size, num_user, num_item, self.t_feat.size(1), dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id)

        #  文本特征的另一种实现方式（使用词嵌入，当前未使用）
        # self.words_tensor = torch.tensor(words_tensor, dtype=torch.long).cuda()
        # self.word_embedding = nn.Embedding(torch.max(self.words_tensor[1])+1, 128)
        # nn.init.xavier_normal_(self.word_embedding.weight) 
        # self.t_gcn = GCN(self.edge_index, batch_size, num_user, num_item, 128, dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id)

        # 初始化用户和物品的ID嵌入
        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x), requires_grad=True)).cuda()
        # 存储最终表示的结果
        self.result = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x))).cuda()


    def forward(self):
        """
        主网络前向传播

        返回：
        - representation: 融合后的用户和物品表示
        """
        # 通过三个模态的GCN获取各自的表示
        v_rep = self.v_gcn(self.v_feat, self.id_embedding)
        a_rep = self.a_gcn(self.a_feat, self.id_embedding)

        #  文本特征的另一种处理方式（当前未使用）
        #  self.t_feat = torch.tensor(scatter_('mean', self.word_embedding(self.words_tensor[1]), self.words_tensor[0])).cuda()
        t_rep = self.t_gcn(self.t_feat, self.id_embedding)

        # 简单平均融合三个模态的表示
        representation = (v_rep+a_rep+t_rep)/3
        # 保存结果用于后续评估
        self.result = representation
        return representation

    def loss(self, user_tensor, item_tensor):
        """
        计算模型损失

        参数：
        - user_tensor: 用户索引张量
        - item_tensor: 物品索引张量（包含正样本和负样本）

        返回：
        - loss: 总损失（BPR损失 + 正则化损失）
        - model_loss: BPR损失
        - reg_loss: 总正则化损失
        - reg_embedding_loss: 嵌入正则化损失
        - reg_embedding_loss: 重复返回嵌入正则化损失（可能是为了API一致性）
        """
        # 重塑张量维度
        user_tensor = user_tensor.view(-1)
        item_tensor = item_tensor.view(-1)
        # 获取模型前向传播的结果
        out = self.forward()
        # 获取用户和物品的表示
        user_score = out[user_tensor]
        item_score = out[item_tensor]
        # 计算用户-物品交互得分（点积）
        score = torch.sum(user_score * item_score, dim=1).view(-1, 2)
        # 计算BPR损失
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))
        # 计算嵌入正则化损失(L2正则化)
        reg_embedding_loss = (self.id_embedding[user_tensor]**2 + self.id_embedding[item_tensor]**2).mean() + (self.v_gcn.preference**2).mean()
        # 计算总正则化损失
        reg_loss = self.reg_weight * (reg_embedding_loss)
        # 总损失
        total_loss = bpr_loss + reg_loss

        return total_loss, bpr_loss, reg_loss, reg_embedding_loss, reg_embedding_loss

    def accuracy(self, step=2000, topk=10):
        """
        计算训练集上的推荐性能指标

        参数：
        - step: 处理用户的批次大小
        - topk: 推荐列表长度

        返回：
        - precision: 精确率
        - recall: 召回率
        - ndcg: 归一化折损累积增益
        """
        # 分离用户和物品表示
        user_tensor = self.result[:self.num_user]
        item_tensor = self.result[self.num_user:]

        # 分批次处理用户
        start_index = 0
        end_index = self.num_user if step==None else step

        all_index_of_rank_list = torch.LongTensor([])
        while end_index <= self.num_user and start_index < end_index:
            # 获取当前批次的用户表示
            temp_user_tensor = user_tensor[start_index:end_index]
            # 计算用户-物品得分矩阵
            score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())

            # 获取得分最高的topk个物品
            _, index_of_rank_list = torch.topk(score_matrix, topk)
            all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu()+self.num_user), dim=0)

            # 更新批次索引
            start_index = end_index
            if end_index + step < self.num_user:
                end_index += step
            else:
                end_index = self.num_user

        # 计算评估指标
        length = self.num_user      
        precision = recall = ndcg = 0.0

        # 对每个用户计算指标
        for row, col in self.user_item_dict.items():
            user = row
            pos_items = set(col)  # 用户实际交互的物品集合
            num_pos = len(pos_items)
            items_list = all_index_of_rank_list[user].tolist()  # 推荐的物品列表

            items = set(items_list)

            # 计算命中数量
            num_hit = len(pos_items.intersection(items))

            # 计算精确率和召回率
            precision += float(num_hit / topk)
            recall += float(num_hit / num_pos)

            # 计算NDCG
            ndcg_score = 0.0
            max_ndcg_score = 0.0

            # 计算最大可能的NDCG分数
            for i in range(min(num_hit, topk)):
                max_ndcg_score += 1 / math.log2(i+2)
            if max_ndcg_score == 0:
                continue

            # 计算实际的NDCG分数
            for i, temp_item in enumerate(items_list):
                if temp_item in pos_items:
                    ndcg_score += 1 / math.log2(i+2)

            ndcg += ndcg_score/max_ndcg_score

        # 返回平均指标
        return precision/length, recall/length, ndcg/length


    def full_accuracy(self, val_data, step=2000, topk=10):
        """
        计算验证集或测试集上的推荐性能指标

        参数：
        - val_data: 验证或测试数据
        - step: 处理用户的批次大小
        - topk: 推荐列表长度

        返回：
        - precision: 精确率
        - recall: 召回率
        - ndcg: 归一化折损累积增益
        """
        # 分离用户和物品表示
        user_tensor = self.result[:self.num_user]
        item_tensor = self.result[self.num_user:]

        # 分批次处理用户
        start_index = 0
        end_index = self.num_user if step==None else step

        all_index_of_rank_list = torch.LongTensor([])
        while end_index <= self.num_user and start_index < end_index:
            # 获取当前批次的用户表示
            temp_user_tensor = user_tensor[start_index:end_index]
            # 计算用户-物品得分矩阵
            score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())

            # 排除训练集中已有的交互（将训练集中的物品得分设为很低的值）
            for row, col in self.user_item_dict.items():
                if row >= start_index and row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col))-self.num_user
                    score_matrix[row][col] = 1e-5

            # 获取得分最高的topk个物品
            _, index_of_rank_list = torch.topk(score_matrix, topk)
            all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu()+self.num_user), dim=0)
            start_index = end_index

            # 更新批次索引
            if end_index+step < self.num_user:
                end_index += step
            else:
                end_index = self.num_user
        # 计算评估指标
        length = 0        
        precision = recall = ndcg = 0.0

        # 对验证/测试数据中的每个用户计算指标
        for data in val_data:
            user = data[0]
            pos_items = set(data[1:])  # 验证集中的正样本物品
            num_pos = len(pos_items)
            if num_pos == 0:
                continue
            length += 1
            items_list = all_index_of_rank_list[user].tolist()  # 推荐的物品列表

            items = set(items_list)

            # 计算命中数量
            num_hit = len(pos_items.intersection(items))

            # 计算精确率和召回率
            precision += float(num_hit / topk)
            recall += float(num_hit / num_pos)

            # 计算NDCG
            ndcg_score = 0.0
            max_ndcg_score = 0.0

            # 计算最大可能的NDCG分数
            for i in range(min(num_pos, topk)):
                max_ndcg_score += 1 / math.log2(i+2)
            if max_ndcg_score == 0:
                continue

            # 计算实际的NDCG分数
            for i, temp_item in enumerate(items_list):
                if temp_item in pos_items:
                    ndcg_score += 1 / math.log2(i+2)

            ndcg += ndcg_score/max_ndcg_score

        # 返回平均指标
        return precision/length, recall/length, ndcg/length