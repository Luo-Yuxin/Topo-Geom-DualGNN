import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import (
    SAGEConv, 
    GINEConv, 
    GATConv,
    GATv2Conv, 
    GENConv, 
    DeepGCNLayer,
    PNAConv
)
from models.layers.basic import MLP # 假设您有一个基础 MLP 模块，如果没有，可以使用 nn.Sequential

# GatedGCN Layer
class GatedGCNLayer(MessagePassing):
    """
    完整的 GatedGCN 层实现 (aligned with Dwivedi et al. / DGL implementation)
    
    Ref: "Benchmarking Graph Neural Networks", Dwivedi et al.
    
    参数对应关系 (Reference to DGL snippet):
    - A: 变换自身节点特征 (for node update)
    - B: 变换邻居节点特征 (for node update)
    - C: 变换边特征 (for edge update)
    - D: 变换目标节点特征 (for edge update)
    - E: 变换源节点特征 (for edge update)
    """
    def __init__(self, in_dim, out_dim, dropout=0.0, batch_norm=True, residual=True):
        # aggr='add': 聚合邻居消息时使用求和 (Standard GatedGCN uses Sum aggregation)
        super().__init__(aggr='add', flow='source_to_target')
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.residual = residual
        self.batch_norm = batch_norm

        # --- 1. 定义5个线性变换矩阵 (对应原版公式) ---
        
        # 用于节点特征更新 (Node Feature Update)
        # h_i_new = A*h_i + sum(sigma * B*h_j)
        self.A = nn.Linear(in_dim, out_dim, bias=True) # self (dst)
        self.B = nn.Linear(in_dim, out_dim, bias=True) # neighbor (src)
        
        # 用于边特征更新 (Edge Feature Update)
        # e_ij_new = C*e_ij + D*h_i + E*h_j
        self.C = nn.Linear(in_dim, out_dim, bias=True) # edge
        self.D = nn.Linear(in_dim, out_dim, bias=True) # dst node (i)
        self.E = nn.Linear(in_dim, out_dim, bias=True) # src node (j)
        
        # --- 2. 归一化与激活 ---
        if self.batch_norm:
            self.bn_node = nn.BatchNorm1d(out_dim)
            self.bn_edge = nn.BatchNorm1d(out_dim)
        
        # --- 3. 残差连接投影 ---
        if self.residual and self.in_dim != self.out_dim:
            self.node_residual_proj = nn.Linear(in_dim, out_dim)
            self.edge_residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.node_residual_proj = None
            self.edge_residual_proj = None

    def forward(self, h, edge_index, edge_attr):
        """
        :param h: [N, in_dim]
        :param edge_index: [2, E]
        :param edge_attr: [E, in_dim]
        """
        # 保存残差连接的原始输入
        h_in = h
        e_in = edge_attr
        
        # --- 步骤 A: 预计算所有线性变换 ---
        # 相比于在 message 中计算，提前计算可以利用矩阵乘法加速
        
        # Node transformations
        Ah = self.A(h) # For self-loop in node update
        Bh = self.B(h) # For neighbor aggregation
        Dh = self.D(h) # For edge update (dst)
        Eh = self.E(h) # For edge update (src)
        
        # Edge transformation
        Ce = self.C(edge_attr) 
        
        # --- 步骤 B: 边特征更新 (Edge Update) ---
        # DGL: eij = Ce + Dh_dst + Eh_src
        # PyG: edge_index[0] = src, edge_index[1] = dst (flow='source_to_target')
        
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        
        # e_hat = C*e + D*h_i + E*h_j
        # 注意：D对应目标节点(dst/i), E对应源节点(src/j)
        e_hat = Ce + Dh[dst_idx] + Eh[src_idx]
        
        # 门控系数 sigma
        sigma = torch.sigmoid(e_hat)
        
        # 边特征残差更新: e_new = e_in + ReLU(BN(e_hat))
        e_out = e_hat
        if self.batch_norm:
            e_out = self.bn_edge(e_out)
        e_out = F.relu(e_out)
        e_out = F.dropout(e_out, self.dropout, training=self.training)
        
        if self.residual:
            if self.edge_residual_proj is not None:
                e_in = self.edge_residual_proj(e_in)
            edge_attr_new = e_in + e_out
        else:
            edge_attr_new = e_out

        # --- 步骤 C: 节点特征聚合 (Node Update) ---
        # DGL: h = Ahi + sum(sigma * Bhj) 
        # (注：原DGL代码片段中有除法归一化，但标准 GatedGCN 论文使用 Sum。这里采用标准 Sum)
        
        # 启动消息传递: 传入 Bh 和 sigma
        # message(Bh_j, sigma) -> return sigma * Bh_j
        # aggregate -> sum
        sum_sigma_Bh = self.propagate(edge_index, Bh=Bh, sigma=sigma)
        
        # h_hat = Ah + sum_sigma_Bh
        h_hat = Ah + sum_sigma_Bh
        
        # 节点特征残差更新: h_new = h_in + ReLU(BN(h_hat))
        h_out = h_hat
        if self.batch_norm:
            h_out = self.bn_node(h_out)
        h_out = F.relu(h_out)
        h_out = F.dropout(h_out, self.dropout, training=self.training)
        
        if self.residual:
            if self.node_residual_proj is not None:
                h_in = self.node_residual_proj(h_in)
            h_new = h_in + h_out
        else:
            h_new = h_out
            
        return h_new, edge_attr_new

    def message(self, Bh_j, sigma):
        """
        :param Bh_j: 源节点(neighbor)经B变换后的特征 [E, out_dim]
        :param sigma: 边门控系数 [E, out_dim]
        """
        # Element-wise product of Gate and Neighbor Feature
        return sigma * Bh_j

class GatedGCNLayer_light(MessagePassing):

    """
    GatedGCN 层 (Based on Bresson et al. "Benchmarking Graph Neural Networks")
    
    核心特性:
    1. 显式利用边特征 (Edge Features) 进行消息传递
    2. 门控机制 (Gating): 边特征控制信息流的强度 (Soft Attention)
    3. 残差连接 (Residual): 自动包含输入特征
    4. 关闭了边更新(D, E 矩阵及 edge_bn 被关闭)
    """
    def __init__(self, in_dim, out_dim, dropout=0.0, batch_norm=True, residual=True):
        # aggr='add': 聚合邻居消息时使用求和
        super(GatedGCNLayer_light, self).__init__(aggr='add')
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.residual = residual
        
        # --- 门控系数计算层 ---
        # A, B, C 矩阵分别变换 源节点、目标节点、边特征
        self.A = nn.Linear(in_dim, out_dim, bias=True)
        self.B = nn.Linear(in_dim, out_dim, bias=True)
        self.C = nn.Linear(in_dim, out_dim, bias=True)
        
        # --- 消息转换层 ---
        # W 矩阵变换节点特征
        self.W = nn.Linear(in_dim, out_dim, bias=True)
        
        # --- 边特征更新层 (可选，如果网络层数很深，更新边特征有助于长程依赖) ---
        # 本light方法并不更新边特征
        
        # --- 后处理 ---
        self.bn_node = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()
        # 移除了 self.bn_edge (因为不更新边)
        # self.bn_edge = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()
        
    def forward(self, h, edge_index, edge_attr):
        """
        :param h: 节点特征 [N, in_dim]
        :param edge_index: 边索引 [2, E]
        :param edge_attr: 边特征 [E, in_dim] (注意: 边特征维度需先映射到与节点一致，或在内部映射)
        """
        # 1. 线性变换 (预计算，减少消息传递中的计算量)
        Ah = self.A(h)
        Bh = self.B(h)
        Ce = self.C(edge_attr) 
        Wh = self.W(h)

        src, dst = edge_index
        
        # 2. 计算门控 (Attention)
        # sigma_ij = sigmoid(Ah_i + Bh_j + Ce_ij)
        # 注意: 这里只计算系数，不作为新的边特征输出
        gate = torch.sigmoid(Ah[dst] + Bh[src] + Ce)

        # 3. 消息聚合
        # msg = gate * Wh_j
        out_node = self.propagate(edge_index, gate=gate, Wh=Wh)
        
        # 4. 节点特征更新
        h_out = out_node
        h_out = self.bn_node(h_out)
        h_out = F.relu(h_out)
        h_out = F.dropout(h_out, self.dropout, training=self.training)

        # 残差项
        if self.residual:
            h_new = h + h_out
        else:
            h_new = h_out
        
        # 4. 边特征更新 (这是 GatedGCN 的特色，边特征也会随层数演化)
        # 我们需要重新计算一遍 eta (门控值)，这在 propagate 内部是隐式的
        # 为了高效，我们可以在这里近似更新，或者在 message 中返回 edge_update
        # 这里为了接口简洁，我们暂时返回 原始边特征 或 简单处理后的边特征
        # 如果需要更强的边推理，需要修改 propagate 的返回值机制
        
        # 简单策略：利用上一层的门控信息来更新边
        # 这里的实现略有简化，标准 GatedGCN 会输出更新后的边特征供下一层使用
        # 为了保持 PyG 风格，我们暂不返回更新后的 edge_attr，除非显式需要
        
        return h_new, edge_attr
    
    def message(self, gate, Wh_j):
        """
        gate: [E, D]
        Wh_j: [E, D] (源节点特征)
        """
        return gate * Wh_j

# SAGE Layer
class SageLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0, batch_norm=True, residual=True):
        super().__init__()
        self.conv = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.bn = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()
        self.dropout = dropout
        self.residual = residual
        
    def forward(self, h, edge_index):
        h_in = h
        h = self.conv(h, edge_index)
        h = self.bn(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        if self.residual:
            h = h + h_in
        return h

# GINE Layer
class GineLayer(nn.Module):
    """
    Graph Isomorphism Network with Edge Features (GINE)
    """
    def __init__(self, hidden_dim, edge_dim=None, dropout=0.0, batch_norm=True, residual=True):
        super().__init__()
        # 确定边特征维度
        self.edge_dim = edge_dim if edge_dim is not None else hidden_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.residual = residual
        # 对边特征进行投影
        # PyG的GINEConv会自动的对齐边特征
        if self.edge_dim != hidden_dim:
            self.edge_linear = nn.Linear(self.edge_dim, hidden_dim)
        else:
            self.edge_linear = nn.Identity()

        self.mlp = MLP(input_dim=hidden_dim,
                       hidden_dim=hidden_dim * 2,
                       output_dim=hidden_dim,
                       num_layers=2,
                       norm_layer=nn.BatchNorm1d,
                       bias=False,
                       dropout=dropout,
                       act_layer=nn.ReLU,
                       # Act params
                       )
        # 初始化GINE
        # train_eps=True 允许模型学习中心节点自环的权重，这是一个重要技巧
        self.conv = GINEConv(self.mlp, train_eps=True)
        # 层归一化 (一般在卷积和激活之后)
        self.bn = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()
        
        
    def forward(self, h, edge_index, edge_attr):
        """
        :param h: 节点特征 [N, hidden_dim]
        :param edge_index: 边索引 [2, E]
        :param edge_attr: 边特征 [E, edge_dim]

        :return: 更新后的节点特征 [N, hidden_dim]
        """
        # 保存输入用于残差连接
        h_in = h
        # 对边特征进行对齐
        if self.edge_linear is not None:
            # 将边特征投影到 hidden_dim: [E, edge_dim] -> [E, hidden_dim]
            edge_attr = self.edge_linear(edge_attr)

        # PyG 的 GINEConv 会执行: MLP( (1+eps)*h_i + sum(ReLU(h_j + edge_attr)) )
        h = self.conv(x=h, edge_index=edge_index, edge_attr=edge_attr)
        # 后处理: 归一化 -> 激活 -> Dropout
        h = self.bn(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)

        if self.residual:
            h = h + h_in
        return h

# GAT Layer
class GATLayer(nn.Module):
    def __init__(self, hidden_dim, heads=4, dropout=0.0, batch_norm=True, residual=True):
        super().__init__()
        # 为了保证输入输出维度一致 (hidden_dim -> hidden_dim)
        # 我们需要设置每个 head 的维度为 hidden_dim // heads
        # 这样拼接 (concat) 后的总维度将回到 hidden_dim
        assert hidden_dim % heads == 0, f"Hidden dim {hidden_dim} must be divisible by heads {heads}"
        out_channels = hidden_dim // heads
        
        self.conv = GATConv(hidden_dim, out_channels, heads=heads, 
                            dropout=dropout, concat=True, add_self_loops=True)
        
        self.bn = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()
        self.dropout = dropout
        self.residual = residual
        # 使用 ELU 激活
        self.activation = nn.ELU()
        
    def forward(self, h, edge_index):
        h_in = h
        h = self.conv(h, edge_index)
        h = self.bn(h)
        h = self.activation(h) # 使用 ELU
        h = F.dropout(h, self.dropout, training=self.training)
        
        if self.residual:
            h = h + h_in
        return h

# GATv2 Layer
class GATv2Layer(nn.Module):
    def __init__(self, hidden_dim, heads=4, dropout=0.0, batch_norm=True, residual=True):
        super().__init__()
        # 确保 hidden_dim 能被 heads 整除
        assert hidden_dim % heads == 0
        self.conv = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, 
                              dropout=dropout, concat=True, add_self_loops=True)
        self.bn = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()
        self.dropout = dropout
        self.residual = residual
        
        # GAT 拼接后维度会自动变回 hidden_dim，无需额外投影
        
    def forward(self, h, edge_index):
        h_in = h
        h = self.conv(h, edge_index)
        h = self.bn(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        if self.residual:
            h = h + h_in
        return h

# DeeperGCN Layer
class DeeperGCNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0, batch_norm=True, residual=True):
        super().__init__()
        # GENConv 是 DeeperGCN 的核心
        conv = GENConv(hidden_dim, hidden_dim, aggr='softmax', 
                            t=1.0, learn_t=True, num_layers=2, norm='layer')
        # 定义 Normalization 和 Activation
        norm = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()
        act = nn.ReLU(inplace=True)
        # 使用 DeepGCNLayer
        self.layer = DeepGCNLayer(conv=conv, norm=norm, act=act, 
                                  block='res+', dropout=dropout)
        
    def forward(self, h, edge_index, edge_attr):
        
        h = self.layer(h, edge_index, edge_attr=edge_attr)
        return h

# PNA Layer
class PNALayer(nn.Module):
    """
    PNA Layer: 使用 PyG 原生实现。
    支持多种聚合器 (mean, min, max, std) 和 缩放器 (identity, amplification, attenuation)。
    需要传入 degree 分布 (deg)。
    """
    def __init__(self, hidden_dim, deg, dropout=0.0, batch_norm=True, residual=True, edge_dim=None):
        super().__init__()
        
        # PNA 标准配置
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        # PyG PNAConv
        # edge_dim: 如果为 None，则不使用边特征；如果不为 None，则要求 forward 中传入 edge_attr
        self.conv = PNAConv(in_channels=hidden_dim, out_channels=hidden_dim,
                            aggregators=aggregators, scalers=scalers, deg=deg,
                            edge_dim=edge_dim, towers=1, pre_layers=1, post_layers=1,
                            divide_input=False)
                            
        self.bn = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()
        self.dropout = dropout
        self.residual = residual
        
    def forward(self, h, edge_index, edge_attr=None):
        h_in = h
        # PNAConv 会自动处理 edge_attr (如果 init 时指定了 edge_dim)
        h = self.conv(h, edge_index, edge_attr=edge_attr)
        
        h = self.bn(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        
        if self.residual:
            h = h + h_in
        return h


# Identity Layer
class IdentityGNNLayer(nn.Module):
    """
    Identity GNN Layer
    不进行任何图卷积操作，直接返回输入的节点特征和边特征。
    用于消融实验：保留数据流但不使用 GNN 处理。
    """
    def __init__(self, hidden_dim, dropout=0.0, **kwargs):
        super().__init__()
        # 即使不使用，为了兼容接口，我们也接收这些参数
        self.identity = nn.Identity()
        
    def forward(self, h, edge_index, edge_attr=None):
        """
        :param h: [N, D]
        :param edge_index: [2, E]
        :param edge_attr: [E, D_e]
        :return: (h, edge_attr)
        """
        # 直接透传
        return h, edge_attr