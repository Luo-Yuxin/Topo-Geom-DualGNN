import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data

class DualStreamGraphConverter:
    """
    将 NetworkX 格式的拓扑图和几何图转换为 PyTorch Geometric 的 Data 对象
    用于双流图神经网络的输入。
    """
    
    def __init__(self, to_tensor_fn=None):
        """
        :param to_tensor_fn: 可选的自定义转换函数
        """
        self.to_tensor = to_tensor_fn if to_tensor_fn else torch.tensor

    def convert(self, topo_graph, geom_graph, extra_attributes=None):
        """
        用于将拓扑关系图与几何关系图转换为PyG的数据类型
        两类图将被保存在一个数据类型中
        
        :param topo_graph: build_graph.build_topo_graph 返回的 NetworkX 图
        :param geom_graph: build_graph.build_geom_graph 返回的 NetworkX 图
        :param extra_attributes: 字典, 格式 {'attr_name': (dtype, value_data), ...}
                                 用于通用的标签或额外属性注入
        :return: torch_geometric.data.Data 对象，包含以 _topo 和 _geom 后缀区分的属性
        """
        
        # 基础一致性检查
        # 我们需要最后对两图中节点数量进行比较, 以防止出现两图的数据冲突
        num_nodes = topo_graph.number_of_nodes()
        # 断言两者节点数量应当相同
        assert num_nodes == geom_graph.number_of_nodes(), \
            f"拓扑图节点数 ({num_nodes}) 与几何图节点数 ({geom_graph.number_of_nodes()}) 不一致！"

        # ---------------------------------------------------------------------
        # Part A: 处理拓扑流 (Topology Stream)
        # ---------------------------------------------------------------------
        
        # A1. 拓扑节点特征 (Face UV Grids)
        # 原始格式: list of (N, N, 7) numpy arrays
        # 目标格式: (Num_Nodes, 7, N, N) -> PyTorch Conv2d 偏好 (C, H, W)
        topo_node_feats = []
        # 确保按节点 ID 排序 (0, 1, 2...)
        for i in range(num_nodes):
            feat = topo_graph.nodes[i]['sample'] # shape: (N, N, 7)
            
            # 检查特征是否为空或形状不对
            if feat is None:
                # 假设默认 5x5
                feat = np.zeros((5, 5, 7), dtype=np.float32)
            
            # 转置: (H, W, C) -> (C, H, W)
            feat = np.transpose(feat, (2, 0, 1)) 
            topo_node_feats.append(feat)
            
        x_topo = self.to_tensor(np.array(topo_node_feats), dtype=torch.float)

        # A2. 拓扑边特征 (Edge Curves)
        # 目标格式: edge_index (2, E), edge_attr (E, M, 6)
        topo_edges = []
        topo_edge_attrs = []
        
        for u, v, data in topo_graph.edges(data=True):
            # 无向图在 PyG 中通常需要存储双向边，或者使用 ToUndirected 转换
            # NetworkX 的边是无向的，我们这里先存单向，依靠 PyG Loader 的处理或手动添加双向
            # 建议：手动添加双向以确保信息完整
            
            # 正向 u -> v
            topo_edges.append([u, v])
            sample = data['sample'] # shape: (M, 6)
            topo_edge_attrs.append(sample)
            
            # 反向 v -> u
            topo_edges.append([v, u])
            # 反向边中所存储的值不变
            topo_edge_attrs.append(sample) 
        # 若几何体中存在拓扑边
        if len(topo_edges) > 0:
            # 索引使用长整型, 特征使用浮点型
            edge_index_topo = self.to_tensor(topo_edges, dtype=torch.long).t().contiguous()
            edge_attr_topo = self.to_tensor(np.array(topo_edge_attrs), dtype=torch.float)
        else:
            # 处理没有边的情况
            edge_index_topo = torch.empty((2, 0), dtype=torch.long)
            edge_attr_topo = torch.empty((0, 5, 6), dtype=torch.float) # 假设 M=5

        # ---------------------------------------------------------------------
        # Part B: 处理几何流 (Geometry Stream)
        # ---------------------------------------------------------------------

        # B1. 几何节点特征 (Global Geom Properties + Type One-Hot)
        # 原始格式: 'feat_concat' in node data
        geom_node_feats = []
        for i in range(num_nodes):
            # 直接获取 build_graph 中拼接好的向量
            feat = geom_graph.nodes[i]['feat_concat'] 
            geom_node_feats.append(feat)
        # 将节点信息转化为张量
        x_geom = self.to_tensor(np.array(geom_node_feats), dtype=torch.float)

        # B2. 几何边特征 (Relations)
        # 目标格式: edge_index (2, E), edge_attr (E, 5) -> 5 dim one-hot
        geom_edges = []
        geom_edge_attrs = []
        # 从图中读取几何图边特征
        for u, v, data in geom_graph.edges(data=True):
            relation = data.get('one_hot_relation')
            
            if relation is None:
                continue

            # 正向
            geom_edges.append([u, v])
            geom_edge_attrs.append(relation)

            # 反向
            geom_edges.append([v, u])
            geom_edge_attrs.append(relation)
        
        if len(geom_edges) > 0:
            edge_index_geom = self.to_tensor(geom_edges, dtype=torch.long).t().contiguous()
            edge_attr_geom = self.to_tensor(np.array(geom_edge_attrs), dtype=torch.float)
        else:
            edge_index_geom = torch.empty((2, 0), dtype=torch.long)
            edge_attr_geom = torch.empty((0, 5), dtype=torch.float)

        # ---------------------------------------------------------------------
        # Part C: 构建 Data 对象
        # ---------------------------------------------------------------------
        
        data_dict = {
            # 拓扑流数据
            'x_topo': x_topo,               # (N, 7, H, W)
            'edge_index_topo': edge_index_topo, 
            'edge_attr_topo': edge_attr_topo, # (E_topo, M, 6)
            
            # 几何流数据
            'x_geom': x_geom,                 # (N, D_geom)
            'edge_index_geom': edge_index_geom,
            'edge_attr_geom': edge_attr_geom,   # (E_geom, 5)
            
            # 元数据
            'num_nodes': num_nodes
        }

        # 添加标签 (如果存在)
        # 遍历传入的字典，动态添加到 Data 对象中
        if extra_attributes is not None:
            for key, (dtype, value) in extra_attributes.items():
                if value is not None:
                    # 使用显式 dtype 转换
                    data_dict[key] = self.to_tensor(value, dtype=dtype)

        return Data(**data_dict)