import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from models.layers.basic import MLP

class GatedGCNLayer_light(MessagePassing):

    """
    GatedGCN 层 (Based on Bresson et al. "Benchmarking Graph Neural Networks")
    
    核心特性:
    1. 显式利用边特征 (Edge Features) 进行消息传递
    2. 门控机制 (Gating): 边特征控制信息流的强度 (Soft Attention)
    3. 残差连接 (Residual): 自动包含输入特征
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
        # e_new = e_old + sigma(...)
        self.edge_update_mlp = nn.Linear(out_dim, out_dim) # 用于将门控值映射回边特征空间
        
        # --- 后处理 ---
        self.bn_node = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()
        self.bn_edge = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()
        
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
        
        # 2. 开始消息传递
        # propagate 会自动调用 message -> aggregate -> update
        # 传入 Ah, Bh, Ce, Wh 以便在 message 函数中使用
        # out_node 是聚合后的节点特征
        out_node = self.propagate(edge_index, Ah=Ah, Bh=Bh, Ce=Ce, Wh=Wh, edge_attr=edge_attr)
        
        # 3. 节点特征更新 (Residual + BN + ReLU)
        if self.residual:
            h_new = h + out_node
        else:
            h_new = out_node
            
        h_new = self.bn_node(h_new)
        h_new = F.relu(h_new)
        h_new = F.dropout(h_new, self.dropout, training=self.training)
        
        # 4. 边特征更新 (这是 GatedGCN 的特色，边特征也会随层数演化)
        # 我们需要重新计算一遍 eta (门控值)，这在 propagate 内部是隐式的
        # 为了高效，我们可以在这里近似更新，或者在 message 中返回 edge_update
        # 这里为了接口简洁，我们暂时返回 原始边特征 或 简单处理后的边特征
        # 如果需要更强的边推理，需要修改 propagate 的返回值机制
        
        # 简单策略：利用上一层的门控信息来更新边
        # 这里的实现略有简化，标准 GatedGCN 会输出更新后的边特征供下一层使用
        # 为了保持 PyG 风格，我们暂不返回更新后的 edge_attr，除非显式需要
        
        return h_new

    def message(self, Ah_i, Bh_j, Ce, Wh_j):
        """
        定义单条边的消息计算逻辑
        _i: 目标节点 (Receiver)
        _j: 源节点 (Source)
        """
        # 计算门控系数 (Hadamard product attention)
        # sigma_ij = sigmoid( A*h_i + B*h_j + C*e_ij )
        gate = torch.sigmoid(Ah_i + Bh_j + Ce)
        
        # 消息 = 门控 * 变换后的邻居节点特征
        msg = gate * Wh_j
        
        return msg
    

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


class GatedGCN(nn.Module):
    """
    堆叠多层 GatedGCNLayer 的完整网络
    """
    def __init__(self, in_node_dim, in_edge_dim, hidden_dim, out_dim, n_layers, 
                 dropout=0.0, share_weights=False):
        super().__init__()
        self.n_layers = n_layers
        self.share_weights = share_weights

        self.node_embedding = nn.Linear(in_node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(in_edge_dim, hidden_dim)
        
        if self.share_weights:
            # 模式 A: 参数共享 (只实例化一层)
            shared_layer = GatedGCNLayer(hidden_dim, hidden_dim, dropout=dropout, batch_norm=True, residual=True)
            # 2. 将同一个实例添加多次到 ModuleList
            # 注意：PyTorch 能够正确处理这种情况，参数实际上只会被注册一次
            for _ in range(n_layers):
                self.layers.append(shared_layer)
        else:
            # 模式 B: 独立参数 (实例化 n_layers 层)
            self.layers = nn.ModuleList()
            for _ in range(n_layers):
                self.layers.append(
                    GatedGCNLayer(hidden_dim, hidden_dim, dropout=dropout, batch_norm=True, residual=True)
                )
            
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, h, edge_index, edge_attr):
        h = self.node_embedding(h)
        edge_attr = self.edge_embedding(edge_attr)
        
        for layer in self.layers:
            h, edge_attr = layer(h, edge_index, edge_attr)
            
        # Global Pooling (Mean) - 简单的 Readout 示例
        # 实际使用中根据任务替换为 global_mean_pool(h, batch) 等
        out = h.mean(dim=0, keepdim=True) 
        out = self.classifier(out)
        
        return out