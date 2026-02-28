import torch
import torch.nn as nn
from models.layers.gnn_layer import (
    GatedGCNLayer,
    GatedGCNLayer_light,
    SageLayer,
    GineLayer,
    GATLayer,
    GATv2Layer,
    DeeperGCNLayer,
    PNALayer,
    IdentityGNNLayer
)

class GNNPostProcess(nn.Module):
    """
    [GNN 增强模块] Pre-Activation Residual MLP Block
    逻辑: Output = Input + MLP(BatchNorm(Input))
    作用: 
    1. BatchNorm 稳定分布，防止 GNN 层数加深后的梯度消失/爆炸。
    2. MLP 增加非线性特征变换能力 (Channel-mixing)。
    3. 残差连接防止退化。
    """
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        
        self.norm = nn.BatchNorm1d(hidden_dim)
        
        # 使用双层 MLP 以获得更强的拟合能力
        # 结构: Linear -> Act -> Dropout -> Linear
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        # x: [N, D]
        residual = x
        
        # Pre-Norm
        out = self.norm(x)
        # MLP
        out = self.mlp(out)
        
        # Residual Connection
        return residual + out


class GNNAdapter(nn.Module):
    """
    [接口适配器]
    将不同 GNN Layer 的 I/O 统一为 DualStreamNet 所需的格式。
    统一接口: forward(h, edge_index, edge_attr) -> (h_new, e_new)
    """
    def __init__(self, gnn_layer, layer_type, post_process_module=None):
        super().__init__()
        self.gnn = gnn_layer
        self.layer_type = layer_type
        self.post_process = post_process_module
        
    def forward(self, h, edge_index, edge_attr):
        """
        统一接口: forward(h, edge_index, edge_attr) -> (h_new, e_new)
        """
        
        h_new = None
        e_new = edge_attr # 默认边特征不变，除非 GNN 更新了它
        # --- Case 1: 需要节点与边特征的GNN (返回 h, e) ---
        if self.layer_type in ['gated_gcn', 'gated_gcn_light', 'identity']:
            h_new, e_new = self.gnn(h, edge_index, edge_attr)
            
        # --- Case 2: 需要边特征的 GNN (GINE, DeeperGCN, PNA) ---
        # 增加 'pna' 到支持边特征的列表
        elif self.layer_type in ['gine', 'deepergcn', 'pna']:
            h_new = self.gnn(h, edge_index, edge_attr=edge_attr)
            # e_new 保持原状
            
        # --- Case 3: 仅节点的 GNN (SAGE, GAT, GATv2) ---
        else:
            h_new = self.gnn(h, edge_index)
            # e_new 保持原状
        
        # 应用节点特征后处理 (BN + MLP + Res)
        # 注意: 只对节点特征 h 进行后处理，边特征 e 通常由 GNN 内部处理或保持原样
        if self.post_process is not None:
            h_new = self.post_process(h_new)

        return h_new, e_new

def build_gnn_layer(gnn_type, hidden_dim, dropout, use_post_process=True, **kwargs):
    """
    [工厂函数]
    kwargs: 
       - deg: (Tensor) PNA 需要的度分布直方图
    """
    if gnn_type is None:
        return None
        
    gnn_type = gnn_type.lower()
    layer = None
    
    # 1. 实例化 GNN 核心层
    if gnn_type == 'gated_gcn':
        layer = GatedGCNLayer(hidden_dim, hidden_dim, dropout=dropout)

    elif gnn_type == 'gated_gcn_light':
        layer = GatedGCNLayer_light(hidden_dim, hidden_dim, dropout=dropout)
        
    elif gnn_type == 'sage':
        layer = SageLayer(hidden_dim, dropout=dropout)
        
    elif gnn_type == 'gine':
        layer = GineLayer(hidden_dim, dropout=dropout)
        
    elif gnn_type == 'gat':
        layer = GATLayer(hidden_dim, heads=4, dropout=dropout)

    elif gnn_type == 'gatv2':
        layer = GATv2Layer(hidden_dim, heads=4, dropout=dropout)
        
    elif gnn_type == 'deepergcn':
        layer = DeeperGCNLayer(hidden_dim, dropout=dropout)
        
    elif gnn_type == 'pna':
        # PNA 特有参数 deg
        deg = kwargs.get('deg', None)
        if deg is None:
            raise ValueError("Using PNA requires passing 'deg' (degree histogram) in kwargs.")
        # PNA 默认使用 edge_dim=hidden_dim (因为 Embedder 输出是 hidden_dim)
        layer = PNALayer(hidden_dim, deg=deg, dropout=dropout, edge_dim=hidden_dim)

    elif gnn_type == 'identity':
        layer = IdentityGNNLayer(hidden_dim, dropout=dropout)
        
    else:
        raise ValueError(f"不支持的 GNN 类型: {gnn_type}")
        
    # 2. 实例化后处理模块 (Post Process)
    post_proc = None
    if use_post_process:
        post_proc = GNNPostProcess(hidden_dim, dropout=dropout)

    # 3. 包装并返回
    return GNNAdapter(layer, gnn_type, post_process_module=post_proc)