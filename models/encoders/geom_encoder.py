import torch
import torch.nn as nn
from models.layers.basic import MLP

class GeomEncoder(nn.Module):
    """
    [统一接口] 几何流主编码器
    处理全局几何属性和几何关系边
    """
    def __init__(self, 
                 node_in_dim, 
                 edge_in_dim, 
                 embed_dim=128,
                 dropout=0.0,
                 act_layer=nn.LeakyReLU,
                 **act_args):
        super(GeomEncoder, self).__init__()
        
        # 节点编码器: 将 (通用属性 + 类型 OneHot) 映射到潜在空间
        self.node_mlp = MLP(
            input_dim=node_in_dim,
            hidden_dim=embed_dim * 2,
            output_dim=embed_dim,
            num_layers=2,
            # 激活函数参数
            act_layer=act_layer,
            dropout=dropout,
            **act_args
        )
        
        # 边编码器: 将 (关系 OneHot) 映射到潜在空间
        self.edge_mlp = MLP(
            input_dim=edge_in_dim,
            hidden_dim=embed_dim * 2,
            output_dim=embed_dim,
            num_layers=2,
            # 激活函数参数
            act_layer=act_layer,
            dropout=dropout,
            **act_args
        )
        
    def forward(self, x_geom, edge_attr_geom):
        """
        :param x_geom: 几何节点特征 [N, D_node]
        :param edge_attr_geom: 几何边特征 [E, D_edge]
        """
        h = self.node_mlp(x_geom)
        e = self.edge_mlp(edge_attr_geom)
        
        return h, e