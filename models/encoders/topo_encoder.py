import torch
import torch.nn as nn
from models.layers.basic import conv1d_block, conv2d_block, fc_block

class UVNetCurveEncoder(nn.Module):
    """
    [移植自 EAGNet] 处理边的采样点序列特征 (1D CNN)
    输入: (Batch_Edges, Channels=6, Num_Points)
    输出: (Batch_Edges, Embed_Dim)
    """
    def __init__(self, 
                 in_channels=6, 
                 output_dim=64,
                 act_layer=nn.LeakyReLU,
                 **act_args):
        super(UVNetCurveEncoder, self).__init__()
        # 编码器核心
        self.encoder = nn.Sequential(
            # 1D 卷积提取序列特征
            conv1d_block(in_channels, 32, kernel_size=3, act_layer=act_layer, **act_args),
            conv1d_block(32, 64, kernel_size=3, act_layer=act_layer, **act_args),
            conv1d_block(64, 128, kernel_size=3, act_layer=act_layer, **act_args),
            # 全局池化，消除点数 N 的影响
            nn.AdaptiveAvgPool1d(1),
            # 展平
            nn.Flatten(1),
            # 映射到目标维度
            fc_block(128, output_dim, act_layer=act_layer, **act_args))

    def forward(self, x):
        # x shape: [E, 6, M]
        # 注意: 如果输入是 [E, M, 6], 需要在外部 transpose
        return self.encoder(x)

class UVNetSurfaceEncoder(nn.Module):
    """
    [移植自 EAGNet] 处理面的 UV 网格特征 (2D CNN)
    输入: (Batch_Nodes, Channels=7, H, W)
    输出: (Batch_Nodes, Embed_Dim)
    """
    def __init__(self, in_channels=7, 
                 output_dim=64,
                 act_layer=nn.LeakyReLU,
                 **act_args):
        super(UVNetSurfaceEncoder, self).__init__()
        
        # 2D 卷积提取空间特征
        # 假设输入 grid 较小 (5x5 或 10x10)，不需要太深的网络
        self.encoder = nn.Sequential(
            # 2D 卷积提取序列特征
            conv2d_block(in_channels, 32, kernel_size=3, act_layer=act_layer, **act_args),
            conv2d_block(32, 64, kernel_size=3, act_layer=act_layer, **act_args),
            conv2d_block(64, 128, kernel_size=3, act_layer=act_layer, **act_args),
            # 全局池化，消除点数 N 的影响
            nn.AdaptiveAvgPool2d(1),
            # 展平
            nn.Flatten(1),
            fc_block(128, output_dim, act_layer=act_layer, **act_args))

    def forward(self, x):
        # x shape: [N, 7, H, W]
        return self.encoder(x)

class TopoEncoder(nn.Module):
    """
    [统一接口] 拓扑流主编码器
    负责调用子编码器处理节点和边特征
    """
    def __init__(self, 
                 node_in_channels=7, 
                 edge_in_channels=6, 
                 embed_dim=128,
                 act_layer=nn.LeakyReLU,
                 **act_args
                 ):
        super(TopoEncoder, self).__init__()
        # 节点特征编码器
        self.node_encoder = UVNetSurfaceEncoder(in_channels=node_in_channels, 
                                                output_dim=embed_dim,
                                                act_layer=act_layer,
                                                **act_args)

        # 边特征编码器
        self.edge_encoder = UVNetCurveEncoder(in_channels=edge_in_channels, 
                                              output_dim=embed_dim,
                                              act_layer=act_layer,
                                              **act_args)
        
    def forward(self, x_topo, edge_attr_topo):
        """
        :param x_topo: 节点特征 [N, 7, H, W]
        :param edge_attr_topo: 边特征 [E, M, 6] -> 需要转置为 [E, 6, M] 供 Conv1d 使用
        """
        # 1. 编码节点 (面)
        h = self.node_encoder(x_topo)
        
        # 2. 编码边 (连接)
        # Converter 输出的是 [E, M, 6]，Conv1d 需要 [E, 6, M]
        # 检查维度并转置
        if edge_attr_topo.shape[-1] == 6:
            edge_attr_topo = edge_attr_topo.transpose(1, 2)
            
        e = self.edge_encoder(edge_attr_topo)
        
        return h, e