import torch
import torch.nn as nn
import os
import sys
from models.encoders.topo_encoder import TopoEncoder
from models.encoders.geom_encoder import GeomEncoder
from models.heads.decoder import TaskHeads

from models.layers.gnn_loader import build_gnn_layer
from models.fusion.fusion_layers import build_fusion_layer
from models.heads.final_fusion_head import build_final_fusion_layer
# 引入 dropout_adj 用于实现 DropEdge
from torch_geometric.utils import dropout_edge

class DualStreamNet(nn.Module):
    """
    [主模型] DS-IGAT
    采用 "资源定义 (Definitions)" + "执行流 (Flows)" 的方式构建网络。
    """
    def __init__(self, config):
        super(DualStreamNet, self).__init__()
        self.config = config

        # =====================================================================
        # Part 0: 判断哪些流被开启与规范性检查
        # =====================================================================
        self.topo_enable = config.get('topo_enable', False)
        self.geom_enable = config.get('geom_enable', False)
        # 规范性检查
        # self._check_config(config)

        # =====================================================================
        # Part 1: 输入特征维度与切片处理 (保持之前逻辑不变)
        # =====================================================================
        self.geom_node_indices = config.get('geom_node_feat_indices', None)
        self.geom_edge_indices = config.get('geom_edge_feat_indices', None)
        self.topo_node_indices = config.get('topo_node_feat_indices', None)
        self.topo_edge_indices = config.get('topo_edge_feat_indices', None)

        # 维度计算逻辑
        topo_node_in = self._calculate_dim(config.get('topo_node_in', 7), self.topo_node_indices)
        topo_edge_in = self._calculate_dim(config.get('topo_edge_in', 6), self.topo_edge_indices)
        geom_node_in = self._calculate_dim(config.get('geom_node_in', 20), self.geom_node_indices)
        geom_edge_in = self._calculate_dim(config.get('geom_edge_in', 5), self.geom_edge_indices)
        
        # 安全检查
        if (self.geom_node_indices is not None and len(self.geom_node_indices) == 0) and \
           (self.topo_node_indices is not None and len(self.topo_node_indices) == 0):
             raise ValueError("[DualStreamNet] 禁止同时清空 Topo 和 Geom 的节点特征!")

        # 模型超参数
        hidden_dim = config.get('hidden_dim', 128)
        dropout = config.get('dropout', 0.1)

        # =====================================================================
        # Part 2: 特征编码器 (Encoders)
        # =====================================================================
        if self.topo_enable:
            self.topo_encoder = TopoEncoder(node_in_channels=topo_node_in, 
                                            edge_in_channels=topo_edge_in, 
                                            embed_dim=hidden_dim)
        else:
            self.topo_encoder = None

        if self.geom_enable:                                
            self.geom_encoder = GeomEncoder(node_in_dim=geom_node_in, 
                                            edge_in_dim=geom_edge_in, 
                                            embed_dim=hidden_dim)
        else:
            self.geom_encoder = None

        # =====================================================================
        # Part 3: 资源池构建 (Resource Pools)
        # =====================================================================
        
        # 3.1 获取定义字典 (Definitions)
        # 默认值仅作示例，实际应由 Config 传入
        self.topo_defs = config.get('topo_defs', {'0': {'gnn': 'gated_gcn'}})
        self.geom_defs = config.get('geom_defs', {'0': {'gnn': 'gated_gcn'}})
        self.fusion_defs = config.get('fusion_defs', {'0': {'method': 'cross_attn'}})

        # 3.2 构建 ModuleDict (物理层容器)
        self.topo_pool = nn.ModuleDict()
        self.geom_pool = nn.ModuleDict()
        self.fusion_pool = nn.ModuleDict()

        # -> 构建 Topo GNN 池
        # GNN后处理判断
        self.use_post_process = config.get('use_GNN_post')
        
        for key, params in self.topo_defs.items():
            gnn_type = params.get('gnn')
            # 若topo流中不存在GNN
            if gnn_type == None:
                break
            # 提取 kwargs (除了 'gnn' 之外的所有参数)
            gnn_kwargs = {k: v for k, v in params.items() if k != 'gnn'}
            # 针对PNA进行补充
            if gnn_type == 'pna': gnn_kwargs['deg'] = config.get('topo_deg', None)
            self.topo_pool[key] = build_gnn_layer(gnn_type, hidden_dim, dropout, use_post_process=self.use_post_process, **gnn_kwargs)

        # -> 构建 Geom GNN 池
        for key, params in self.geom_defs.items():
            gnn_type = params.get('gnn')
            # 若geom流中不存在GNN
            if gnn_type == None:
                break
            gnn_kwargs = {k: v for k, v in params.items() if k != 'gnn'}
            # 针对PNA进行补充
            if gnn_type == 'pna': gnn_kwargs['deg'] = config.get('geom_deg', None)
            self.geom_pool[key] = build_gnn_layer(gnn_type, hidden_dim, dropout, use_post_process=self.use_post_process, **gnn_kwargs)

        # -> 构建 Fusion 池
        for key, params in self.fusion_defs.items():
            method = params.get('method')
            # 若fusion流中不存在method
            if method == None:
                break
            fusion_kwargs = {k: v for k, v in params.items() if k != 'method'}
            # nn.ModuleDict 必须存储 nn.Module，不能存 None
            layer = build_fusion_layer(method, hidden_dim, dropout, **fusion_kwargs)
            if layer is None:
                raise ValueError(f"[DualStreamNet] Fusion method '{method}' returned None. \
                                 Ensure factory returns a Module (e.g. Identity or SumFusion).")
            self.fusion_pool[str(key)] = layer

        # =====================================================================
        # Part 4: 执行流指令 (Execution Flows)
        # =====================================================================
        
        # 获取指令列表
        # 示例: ['0', '1', '1']
        # 一个冗余配置以保证没有在里面写数字而不是字符
        self.topo_flow = tuple(str(x) if x is not None else None for x in config.get('topo_flow', ('0')))
        self.geom_flow = tuple(str(x) if x is not None else None for x in config.get('geom_flow', ('0')))
        # 示例: ('0<', '1>', '1=')
        self.fusion_flow = tuple(str(x) if x is not None else None for x in config.get('fusion_flow', ('0=')))

        # 校验: 三个流的长度必须一致 (代表网络的深度/步数)
        # 经之前的config_check处理后应当是一致的
        flow_check_ERROE = False
        topo_flow_len = len(self.topo_flow)
        geom_flow_len = len(self.geom_flow)
        fusion_flow_len = len(self.fusion_flow)

        if topo_flow_len == 0:
            if geom_flow_len == 0 or fusion_flow_len != 0:
                flow_check_ERROE = True
        elif geom_flow_len == 0:
            if topo_flow_len == 0 or fusion_flow_len != 0:
                flow_check_ERROE = True
        elif topo_flow_len != fusion_flow_len or geom_flow_len != fusion_flow_len:
            flow_check_ERROE = True
        if flow_check_ERROE:
            raise ValueError(f"[DualStreamNet] Flow lengths mismatch! Topo:{topo_flow_len}, \
                             Geom:{geom_flow_len}, Fusion:{fusion_flow_len}")
        # 以最大值作为后续更新的根据
        self.num_steps = max((topo_flow_len, geom_flow_len))

        # =====================================================================
        # Part 5: 解码器 (Heads)
        # =====================================================================
        # 定义最终融合
        # 获取 Final Fusion 配置
        final_fusion_config = config.get('final_fusion', {})
        if final_fusion_config is None: final_fusion_config = {}
        # 最终融合方法
        final_fusion_method = final_fusion_config.get('method', None)
        # 最终融合方法参数
        final_fusion_kwargs = {k: v for k, v in final_fusion_config.items() if k != 'method'}

        # 构建 Final Fusion 层并获取输出维度
        self.final_fusion_layer, fused_dim = build_final_fusion_layer(final_fusion_method, 
                                                                      hidden_dim, dropout, 
                                                                      **final_fusion_kwargs)
        
        # 处理无融合层时的默认行为 (Fallback to Concat)
        # 如果 fused_dim 为 0 (表示没有融合层), 我们需要手动计算默认拼接的维度
        if self.final_fusion_layer is None:
            # 默认行为: 简单的 torch.cat
            # 如果双流开启 -> 2 * hidden_dim
            # 如果单流开启 -> 1 * hidden_dim
            if self.topo_enable and self.geom_enable:
                fused_dim = hidden_dim * 2
            elif self.topo_enable or self.geom_enable:
                fused_dim = hidden_dim
            else:
                fused_dim = 0 # Should not happen based on Part 0 check
            
            print(f"[DualStreamNet] Warning - No Final Fusion Layer. Defaulting to CONCAT/IDENTITY. Out Dim: {fused_dim}")
        else:
            pass
            # print(f"[DualStreamNet] Final Fusion: {final_fusion_method} ({final_fusion_kwargs}). Out Dim: {fused_dim}")

        num_classes = config.get('num_classes', 24)
        decoder_type = config.get('decoder_type', {'sem':'mlp_multi_class','inst':'inner_product_head','bot': 'mlp_bin_class'})
        self.heads = TaskHeads(input_dim=fused_dim, 
                               num_semantic_classes=num_classes,
                               hidden_dim=fused_dim, # 通常保持与输入相同
                               dropout=dropout,
                               decoder_type=decoder_type)

    def _check_config(self, config):
        """
        规范化 Config: 检查必要参数, 并自动补全关闭的流为 None
        """
        topo_enable = config.get('topo_enable', False)
        geom_enable = config.get('geom_enable', False)

        topo_defs = config.get('topo_defs', None)
        topo_flow = config.get('topo_flow', None)

        geom_defs = config.get('geom_defs', None)
        geom_flow = config.get('geom_flow', None)
        
        fusion_defs = config.get('fusion_defs', None)
        fusion_flow = config.get('fusion_flow', None)

        final_fusion = config.get('final_fusion', None)
        # 规范性检查
        if topo_enable:
            # 检查topo定义状态
            if topo_defs == None or len(topo_defs) == 0:
                raise ValueError("'topo_defs' in config.model is missing or incorrect. \
                                 example: {'0':{'gnn':'identity'}}")
            if topo_flow == None or len(topo_flow) == 0:
                raise ValueError("'topo_flow' in config.model is missing or incorrect. \
                                 example: (0, 1, 2, 3)")
        if geom_enable:
            # 检查geom定义状态
            if geom_defs == None or len(geom_defs) == 0:
                raise ValueError("'geom_defs' in config.model is missing or incorrect. \
                                 example: {'0':{'gnn':'identity'}}")
            if geom_flow == None or len(geom_flow) == 0:
                raise ValueError("'geom_flow' in config.model is missing or incorrect. \
                                 example: (0, 1, 2, 3)")
        if topo_enable and geom_enable:
            # 两流长度必须相等
            if len(topo_flow) != len(geom_flow):
                raise ValueError("'topo_flow' and 'geom_flow' in config.model \
                                 are not of equal length")
            # 检查fusion定义状态
            if fusion_defs == None or len(geom_defs) == 0:
                raise ValueError("'fusion_defs' in config.model is missing or incorrect. \
                                 example: {'0': {'method': 'concat'}}")
            if fusion_flow == None or len(fusion_flow) != len(topo_flow):
                raise ValueError("'fusion_flow' in config.model is missing or not of equal length. \
                                 example: ('0<', '0=', '0=', '0=')")
        
        # 关闭部分通道时初始化用不到的通道
        if not self.topo_enable and not self.geom_enable:
            raise ValueError(f"All flows had been shut down")
        if not topo_enable:
            config['topo_defs'] = {'0':{'gnn':None}}
            config['fusion_defs'] = {'0':{'method':None}}
            config['topo_flow'] = tuple([None] * len(geom_flow))
            config['fusion_flow'] = tuple([None] * len(geom_flow))
            # 将final_fusion更换输出
            final_fusion['stream_usage'] = 'geom'
            config['final_fusion'] = final_fusion
        if not geom_enable:
            config['geom_defs'] = {'0':{'gnn':None}}
            config['fusion_defs'] = {'0':{'method':None}}
            config['geom_flow'] = tuple([None] * len(topo_flow))
            config['fusion_flow'] = tuple([None] * len(topo_flow))
            # 将final_fusion更换输出
            final_fusion['stream_usage'] = 'topo'
            config['final_fusion'] = final_fusion

    def _calculate_dim(self, default_dim, indices):
        """辅助函数: 计算切片后的维度"""
        if indices is None: return default_dim
        if len(indices) == 0: return 1
        return len(indices)

    def _process_features(self, feat, indices, data_type, shape_hint=None):
        """
        统一的特征切片/清空处理逻辑，显式指定数据类型。
        
        :param feat: 输入特征张量
        :param indices: 切片索引列表 或 [] 或 None
        :param data_type: 数据类型, 用于指定切片维度和构造逻辑
                          选项: 'topo_node', 'topo_edge', 'geom_node', 'geom_edge'
        :param shape_hint: 用于构造零向量的 dim_0 (通常是 N 或 E)
        :return: 处理后的特征张量
        """
        # Case 1: 全选 (None)
        if indices is None:
            return feat
            
        # Case 2: 全空 (Empty List) -> 构造零向量
        if len(indices) == 0:
            # 确定 batch 维度 (N 或 E)
            if shape_hint is not None:
                dim_0 = shape_hint
            else:
                dim_0 = feat.size(0)
            
            # 根据显式的数据类型构造占位符
            if data_type == 'topo_node':
                # [N, 7, H, W] -> [N, 1, H, W]
                # 这里假设 H, W 在 feat 中存在，即 feat 至少是一个 4D 形状 (即便内容可能无效，但形状信息在)
                # 如果 feat 本身不可用，则需要更强的 shape_hint，但通常原始数据是存在的
                _, C, h, w = feat.shape
                return torch.zeros((dim_0, C, h, w), device=feat.device)
            
            elif data_type == 'topo_edge':
                # [E, M, 6] -> [E, M, 1]
                _, m, C = feat.shape
                return torch.zeros((dim_0, m, C), device=feat.device)
            
            elif data_type in ['geom_node', 'geom_edge']:
                # [N, D] -> [N, 1] 或 [E, D] -> [E, 1]
                return torch.zeros((dim_0, 1), device=feat.device)
            
            else:
                raise ValueError(f"Unknown data_type: {data_type}")
                
        # Case 3: 部分切片 (Slicing)
        
        if data_type == 'topo_node':
            # [N, C, H, W] -> 切 Dim 1 (Channel)
            return feat[:, indices, :, :]
            
        elif data_type == 'topo_edge':
            # [E, M, C] -> 切 Dim 2 (Channel, Last Dim)
            return feat[:, :, indices]
            
        elif data_type in ['geom_node', 'geom_edge']:
            # [N, D] 或 [E, D] -> 切 Dim 1
            return feat[:, indices]
            
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
        
    def _get_module_from_pool(self, pool, key, pool_name="Pool"):
        """辅助函数: 安全地从 ModuleDict 获取模块"""
        # 处理 None 或 'None' 字符串 (表示跳过该层)
        if key is None or key == 'None':
            return None
        
        if key not in pool:
            raise KeyError(f"[DualStreamNet] Key '{key}' not found in {pool_name}. Available keys: {list(pool.keys())}")
        
        return pool[key]

    def forward(self, data):
        # -----------------------------------------------------------
        # A. 编码阶段 (Encoding) - 保持不变
        # -----------------------------------------------------------
        h_topo, e_topo = None, None
        h_geom, e_geom = None, None
        # 特征切片逻辑
        if self.topo_enable and self.topo_encoder is not None:
            x_geom_raw = data.x_geom
            x_topo_raw = data.x_topo
            # 切片逻辑
            x_topo = self._process_features(x_topo_raw, self.topo_node_indices, 'topo_node',
                                            shape_hint=x_geom_raw.size(0) \
                                                if self.topo_node_indices is not None and \
                                                    len(self.topo_node_indices)==0 else None)
            num_edges_topo = data.edge_index_topo.size(1)
            edge_topo = self._process_features(data.edge_attr_topo, 
                                               self.topo_edge_indices, 
                                               'topo_edge', shape_hint=num_edges_topo)
            # 进入编码
            h_topo, e_topo = self.topo_encoder(x_topo, edge_topo)
        
        if self.geom_enable and self.geom_encoder is not None:
            x_geom_raw = data.x_geom
            x_topo_raw = data.x_topo
            # 切片逻辑
            x_geom = self._process_features(x_geom_raw, self.geom_node_indices, 'geom_node',
                                            shape_hint=x_topo_raw.size(0) \
                                                if self.geom_node_indices is not None and \
                                                    len(self.geom_node_indices)==0 else None)
            num_edges_geom = data.edge_index_geom.size(1)
            # 进入编码
            edge_geom = self._process_features(data.edge_attr_geom, self.geom_edge_indices, 'geom_edge', shape_hint=num_edges_geom)
            # 几何图边特征赋予几何关系权重
            priority_relation = self.config.get('priority_relation', [4, 3, 2, 1, 1])
            if self.geom_node_indices is not None:
                priority_relation = [priority_relation[i] for i in self.geom_node_indices]
            # 为几何图边特征赋予权重

            # 为了利用原数据集信息, 我们将边特征还原回 0-1 向量逻辑 (Binarization)
            # 目的: 将已经加权归一化的特征还原为 0/1 存在性向量，以便重新加权
            # 只要数值大于 1e-6 (float精度容差)，就认为该关系存在
            if self.config.get('revert_norm', True):
                edge_geom = (edge_geom > 1e-6).float()
            # 转化为 Tensor 并移动到对应设备
            # edge_geom 形状: [E, D]
            # priority_tensor 形状: [1, D]
            priority_tensor = torch.tensor(priority_relation, device=edge_geom.device, dtype=torch.float32).view(1, -1)
            
            # 4. 计算加权特征 (Vectorized)
            # edge_geom 中的元素通常为 0 或 1 (代表是否存在该关系)
            # weighted_edges[i, j] = 存在(0/1) * 权重(P_j)
            weighted_edges = edge_geom * priority_tensor
            
            # 5. 计算分母 (Sum of active weights)
            # 对每一条边(行)的所有关系权重求和
            # 添加 eps 防止除以 0 (如果某条边没有任何关系)
            sum_weights = weighted_edges.sum(dim=1, keepdim=True) + 1e-6
            
            # 6. 归一化: P_k / Sum(P_active)
            edge_geom = weighted_edges / sum_weights
            # =========================================================

            h_geom, e_geom = self.geom_encoder(x_geom, edge_geom)

        # -----------------------------------------------------------
        # [NEW] DropEdge (Edge Dropout) 策略
        # -----------------------------------------------------------
        # 从 config 读取丢弃概率 (默认 0.0 表示不开启)
        drop_edge_topo = self.config.get('drop_edge_topo', 0.0)
        drop_edge_geom = self.config.get('drop_edge_geom', 0.0)

        # 初始化当前使用的边索引 (默认使用原始数据)
        curr_edge_index_topo = data.edge_index_topo
        curr_edge_index_geom = data.edge_index_geom

        # 仅在训练阶段执行
        if self.training:
            # 1. Topo Stream DropEdge
            if drop_edge_topo > 0:
                # [FIX] 使用 dropout_edge 并正确解包
                # dropout_edge(edge_index, p=0.5, force_undirected=False, training=True) -> (edge_index, edge_mask)
                curr_edge_index_topo, topo_mask = dropout_edge(curr_edge_index_topo, p=drop_edge_topo, training=self.training)
                
                # [关键] 必须同步裁剪对应的边特征 e_topo
                if e_topo is not None:
                    # 确保 e_topo 的长度与 mask 长度一致（原始边数）
                    if e_topo.size(0) == topo_mask.size(0):
                        e_topo = e_topo[topo_mask]
                    else:
                        print(f"Warning: e_topo size {e_topo.size(0)} != topo_mask size {topo_mask.size(0)}. Skip edge attr drop.")
            
            # 2. Geom Stream DropEdge (仅当 Geom 流启用时)
            if self.geom_enable and drop_edge_geom > 0:
                curr_edge_index_geom, geom_mask = dropout_edge(curr_edge_index_geom, p=drop_edge_geom, training=self.training)
                if e_geom is not None:
                    if e_geom.size(0) == geom_mask.size(0):
                        e_geom = e_geom[geom_mask]

        # -----------------------------------------------------------
        # B. 动态指令执行阶段 (Dynamic Execution Loop)
        # -----------------------------------------------------------
        
        # 遍历每一步 (Step)
        for i in range(self.num_steps):
            # 1. 获取指令 Key
            t_key = self.topo_flow[i]
            g_key = self.geom_flow[i]
            f_instr = self.fusion_flow[i] # 例如 "0<" 或 "1x"
            
            # 解析 Fusion 指令
            f_key = None
            f_mode = 'none'
            # 兼容: 如果 f_instr 是 None 或 'None'，则无融合
            if f_instr and f_instr != 'None' and len(f_instr) >= 2:
                f_key = f_instr[:-1]
                f_mode = f_instr[-1] # <, >, x, =
            
            # 从资源池检索实例 (从 Pool 中)
            # 允许 flow 中填 None 跳过该流的 GNN (非对称结构支持)
            topo_layer = self._get_module_from_pool(self.topo_pool, t_key, "Topo Pool")
            geom_layer = self._get_module_from_pool(self.geom_pool, g_key, "Geom Pool")
            fusion_layer = self._get_module_from_pool(self.fusion_pool, f_key, "Fusion Pool")

            # ---------------------------------------
            # Phase 1: Pre-Fusion (前置融合)
            # 模式: '>' 或 'x'
            # ---------------------------------------
            if f_mode in ['>', 'x'] and fusion_layer is not None:
                # 只有当两个流的数据都存在时，才进行融合
                if h_topo is not None and h_geom is not None:
                    h_topo, h_geom = fusion_layer(h_topo, h_geom)

            # ---------------------------------------
            # Phase 2: MPNN (独立消息传递)
            # ---------------------------------------
            if topo_layer is not None and h_topo is not None:
                # h_topo, e_topo = topo_layer(h_topo, data.edge_index_topo, e_topo)
                h_topo, e_topo = topo_layer(h_topo, curr_edge_index_topo, e_topo)
            
            if geom_layer is not None and h_geom is not None:
                # h_geom, e_geom = geom_layer(h_geom, data.edge_index_geom, e_geom)
                h_geom, e_geom = geom_layer(h_geom, curr_edge_index_geom, e_geom)

            # ---------------------------------------
            # Phase 3: Post-Fusion (后置融合)
            # 模式: '<' 或 'x'
            # ---------------------------------------
            if f_mode in ['<', 'x'] and fusion_layer is not None:
                if h_topo is not None and h_geom is not None:
                    h_topo, h_geom = fusion_layer(h_topo, h_geom)

        # -----------------------------------------------------------
        # C. 解码阶段 (Decoding)
        # -----------------------------------------------------------
        if self.final_fusion_layer is not None:
            # 使用配置好的融合层 (自动处理流选择和维度)
            # 传入 data.batch 以支持 Pooling
            h_final = self.final_fusion_layer(h_topo, h_geom, batch=data.batch)
        else:
            # Fallback: 默认拼接或单流透传
            if h_topo is not None and h_geom is not None:
                h_final = torch.cat([h_topo, h_geom], dim=-1)
            elif h_topo is not None:
                h_final = h_topo
            elif h_geom is not None:
                h_final = h_geom
            else:
                raise RuntimeError("[DualStreamNet] No features available for decoding!")
            
        batch_ptr = getattr(data, 'ptr', None)
        sem_logits, inst_matrix, inst_mask, bottom_logits = self.heads(h_final, batch_ptr)
        
        return sem_logits, inst_matrix, inst_mask, bottom_logits