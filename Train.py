import torch
import torch.nn as nn
from tqdm import tqdm


def train(epoch, length, dataloader, model, optimizer, batch_size, writer=None):
    """
    模型训练函数

    参数：
    - args: 命令行参数，包含训练配置
    - edge_index: 边索引，定义用户-物品交互图
    - user_item_dict: 用户-物品交互字典
    - v_feat: 视觉特征
    - a_feat: 声学特征
    - t_feat: 文本特征
    - words_tensor: 词张量
    - train_data: 训练数据
    - val_data: 验证数据
    - test_data: 测试数据
    - num_user: 用户数量
    - num_item: 物品数量
    - save_file_path: 模型保存路径

    返回：
    - 无，但会保存训练好的模型并输出评估结果
    """
    model.train()
    print('Now, training start ...')
    sum_loss = 0.0
    sum_model_loss = 0.0
    sum_reg_loss = 0.0
    step = 0.0
    pbar = tqdm(total=length)
    num_pbar = 0
    for user_tensor, item_tensor in dataloader:
        optimizer.zero_grad()
        #  loss: 总损失值（包含正则化项）
        #  model_loss: 模型主损失（如BPR损失）
        #  reg_loss: 正则化损失
        #  weight_loss: 权重相关损失
        #  entropy_loss: 熵损失
        loss, model_loss, reg_loss = model.loss(user_tensor, item_tensor)
        loss.backward()
        optimizer.step()
        pbar.update(batch_size)
        num_pbar += batch_size
        sum_loss += loss.cpu().item()
        sum_model_loss += model_loss.cpu().item()
        sum_reg_loss += reg_loss.cpu().item()
        step += 1.0
    pbar.close()
    print('----------------- loss value:{}  model_loss value:{}  reg_loss value:{}--------------'
        .format(sum_loss/step, sum_model_loss/step, sum_reg_loss/step))
    if writer is not None:
        writer.add_scalar('loss', sum_loss/step, epoch)
        writer.add_scalar('model_loss', sum_model_loss/step, epoch)
        writer.add_scalar('reg_loss', sum_reg_loss/step, epoch)

    return loss
