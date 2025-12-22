"""
MMGCN基础图卷积模型模块
定义 MMGCN 模型的基础图卷积操作类 BaseModel, 继承自 PyTorch Geometric的 MessagePassing类,
实现了图神经网络中的消息传递机制, 是整个 MMGCN 模型的基础组件。
"""
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch_geometric.nn.inits import uniform


class BaseModel(MessagePassing):
	"""
	基础图卷积模型类

	该类继承自PyTorch Geometric的MessagePassing, 实现了图神经网络中的消息传递机制, 是MMGCN中所有GCN层的基类。
	"""
	def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
		"""
		初始化基础图卷积层

		参数：
		- in_channels: 输入特征维度
		- out_channels: 输出特征维度
		- normalize: 是否对输出进行归一化处理
		- bias: 是否使用偏置项
		- aggr: 聚合函数类型，默认为'add'（求和）
		- **kwargs: 其他传递给父类的参数
		"""
		# 调用父类MessagePassing的初始化方法，设置聚合方式
		super(BaseModel, self).__init__(aggr=aggr, **kwargs)
		self.aggr = aggr
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.normalize = normalize
		# 定义权重参数，形状为[in_channels, out_channels]
		self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
		# 初始化权重参数
		self.reset_parameters()

	def reset_parameters(self):
		"""
		重置模型参数
		使用均匀分布初始化权重参数，范围为[-in_channels^(-0.5), in_channels^(-0.5)]
		"""
		# 使用PyTorch Geometric提供的uniform函数初始化权重
		uniform(self.in_channels, self.weight)

	def forward(self, x, edge_index, size=None):
		"""
		前向传播函数

		参数：
		- x: 节点特征矩阵，形状为[num_nodes, in_channels]
		- edge_index: 边索引，形状为[2, num_edges]，表示图中的连接关系
		- size: 图的大小，默认为None

		返回：
		- 更新后的节点特征矩阵，形状为[num_nodes, out_channels]
		"""
		# 对节点特征进行线性变换
		x = torch.matmul(x, self.weight)
		# 调用propagate方法进行消息传递(启动GCN消息传递过程)
		return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

	def message(self, x_j, edge_index, size):
		"""
		消息生成函数
		在消息传递过程中，为每条边生成消息，这里直接返回邻居节点的特征作为消息

		参数：
		- x_j: 源节点的特征（邻居节点的特征）
		- edge_index: 边索引
		- size: 图的大小

		返回：
		- 生成的消息
		"""
		# 直接返回源节点（邻居节点）的特征作为消息
		return x_j

	def update(self, aggr_out):
		"""
		节点更新函数
		聚合消息后，更新节点特征
		参数：
		- aggr_out: 聚合后的消息，形状为[num_nodes, out_channels]

		返回：
		- 更新后的节点特征
		"""
		# 直接返回聚合后的消息作为更新后的节点特征
		return aggr_out

	def __repr__(self):
		"""
		模型的字符串表示
		返回：
		- 模型的字符串表示，包含类名和通道维度
		"""
		return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
