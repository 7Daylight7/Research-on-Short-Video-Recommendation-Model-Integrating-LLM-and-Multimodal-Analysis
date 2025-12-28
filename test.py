import torch

tensor = torch.randn(1651, 128)
torch.save(tensor, 'Data/tiktok/t_feat_sample.pt')  # 保存张量
print(tensor.shape)  # 打印形状