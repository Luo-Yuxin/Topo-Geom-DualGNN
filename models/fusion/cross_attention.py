import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.basic import MLP

class CrossGatedFusionLightLayer(nn.Module):
    """
    交叉图注意力模块 (Cross-Graph Attention, CGA)
    用于融合两个同构图(共享节点)的特征。
    
    机制:
    Stream A (Source/Query) <--- Attention --- Stream B (Target/Key/Value)
    让 Stream A 的节点根据自身需求，从 Stream B 中抓取相关信息。
    """
    def __init__(self, feature_dim, dropout=0.0, residual=True):
        super(CrossGatedFusionLightLayer, self).__init__()
        
        self.feature_dim = feature_dim
        self.residual = residual
        
        # 融合后的特征变换层
        self.linear_fuse = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_query, h_key_value):
        """
        :param h_query: 当前流的节点特征 (作为 Query) [Num_Nodes, D]
        :param h_key_value: 另一流的节点特征 (作为 Key/Value) [Num_Nodes, D]
        """
        # 调整形状以适应 MultiheadAttention: (L, N, E) -> (Num_Nodes, 1, D)
        # 这种全局 Attention 假设所有节点之间都可见（全连接），计算量是 O(N^2)
        # 对于包含多个 STEP 文件的大 Batch，这会让不同 STEP 的节点互见，这通常是可以的（作为 Batch Norm 的一种形式）
        # 或者我们需要根据 batch 索引进行 mask。
        
        # 优化: 简单的 Global Cross Attention 可能计算量过大且引入噪声。
        # 考虑到我们的节点是一一对应的 (Face i in Topo == Face i in Geom)
        # 最简单的 "交叉" 其实是 Element-wise 交互，或者局部邻域交互。
        # 但研究报告提到的 CGA 是为了捕获"非局部"关系。
        
        # 这里为了遵循报告精神，我们实现一个简化版的 "对应节点增强" 或者 "全局上下文交互"
        # 方案 A: 严格的 Self-Attention (O(N^2)) - 显存消耗大
        # 方案 B: 仅对应节点融合 (Element-wise) - 丢失长程
        # 方案 C: 线性注意力 (Linear Attention)
        
        # 鉴于 PyG 的 batching 机制，直接用 MultiheadAttention 会混合不同图的节点。
        # 这里我们先实现一个 "对应节点融合 + 门控" 的轻量级方案，
        # 如果确实需要 O(N^2) 的全局 Attention，需要引入 batch mask。
        
        # --- 修正方案：基于对应节点的门控融合 (Gated Fusion) ---
        # 这比全图 Attention 更稳健，且符合 "双流" 互补的直觉
        # 即：Geom流的信息 补充给 Topo流对应的节点
        
        # 计算 Attention Scores (基于对应节点的相似度)
        # Q * K
        scores = torch.sum(h_query * h_key_value, dim=-1, keepdim=True) / (self.feature_dim ** 0.5)
        attn_weights = torch.sigmoid(scores) # 门控机制
        
        # 融合
        h_fused = h_query + attn_weights * self.linear_fuse(h_key_value)
        
        h_fused = self.norm(h_fused)
        h_fused = self.dropout(h_fused)
        
        return h_fused


class CrossGatedFusionLayer(nn.Module):
    """
    交叉门控融合机制
    """
    def __init__(self, feature_dim, dropout=0.0, residual=True):
        super(CrossGatedFusionLayer, self).__init__()
        
        self.feature_dim = feature_dim
        self.residual = residual
        
        # 可学习的门控参数生成器
        self.gate_net = MLP(input_dim=feature_dim*2,
                            hidden_dim=feature_dim*4,
                            output_dim=feature_dim,
                            num_layers=2,
                            norm_layer=None,
                            bias=False,
                            dropout=dropout,
                            norm_layer_2=None,
                            act_layer=nn.LeakyReLU
                            )
        
        # 值变换
        self.value_transform = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
    
    def _init_weights(self):
        """初始化权重，使门控初始接近中性"""
        # 门控网络的最后一层初始化为接近0
        nn.init.normal_(self.gate_net[-1].weight, std=0.01)
        nn.init.zeros_(self.gate_net[-1].bias)  # 这样初始门控值接近0.5
        
        # 值变换层正常初始化
        nn.init.xavier_uniform_(self.value_transform.weight)
        nn.init.zeros_(self.value_transform.bias)

    def forward(self, h_query, h_key_value):
        """
        :param h_query: 当前流的节点特征 (作为 Query) [Num_Nodes, D]
        :param h_key_value: 另一流的节点特征 (作为 Key/Value) [Num_Nodes, D]
        """
        
        # 拼接查询和键值特征
        combined = torch.cat([h_query, h_key_value], dim=-1)
        # 生成逐元素的门控信号
        gate = torch.sigmoid(self.gate_net(combined))

        # 变换键值
        transformed_value = self.value_transform(h_key_value)

        # 应用门控
        gated_value = gate * transformed_value

        
        # 残差融合
        if self.residual:
            output = h_query + gated_value
            output = self.norm(output)
        else:
            output = self.norm(gated_value)
        output = self.dropout(output)
        
        return output


class CrossGraphAttentionLayer(nn.Module):
    """
    标准交叉图注意力模块 (Standard Cross-Graph Attention)
    增加了 batch_mask 支持，防止跨图错误的注意力交互。
    """
    def __init__(self, feature_dim, num_heads=4, dropout=0.0, residual=True):
        super(CrossGraphAttentionLayer, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.residual = residual
        
        # 1. Multi-Head Attention
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, 
                                          num_heads=num_heads, 
                                          dropout=dropout, 
                                          batch_first=True)
        
        self.norm1 = nn.LayerNorm(feature_dim)
        
        # 2. Feed Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_query, h_key_value, batch=None):
        """
        :param h_query: [N, D]
        :param h_key_value: [N, D]
        :param batch: [N] PyG 的 batch 索引向量，用于构建 mask
        """
        # Q: [1, N, D], K: [1, N, D], V: [1, N, D]
        Q = h_query.unsqueeze(0)
        K = h_key_value.unsqueeze(0)
        V = h_key_value.unsqueeze(0)
        
        # --- [新增] 构建 Batch Mask ---
        attn_mask = None
        if batch is not None:
            # 这里的逻辑是：
            # 如果 node_i 和 node_j 属于同一个图 (batch[i] == batch[j]) -> False (不遮挡/允许Attention)
            # 如果 node_i 和 node_j 属于不同图 (batch[i] != batch[j]) -> True  (遮挡/禁止Attention)
            
            # batch.unsqueeze(1): [N, 1]
            # batch.unsqueeze(0): [1, N]
            # Broadcasting -> [N, N]
            attn_mask = batch.unsqueeze(1) != batch.unsqueeze(0)
            
            # 注意：生成的 attn_mask 可能会非常大 (N^2)，请确保显存足够
        
        # 1. Attention
        # 传入 attn_mask
        attn_output, _ = self.attn(query=Q, key=K, value=V, attn_mask=attn_mask)
        
        # 挤压回 [N, D]
        attn_output = attn_output.squeeze(0)
        
        # 2. Residual + Norm 1
        if self.residual:
            h_out = self.norm1(h_query + self.dropout(attn_output))
        else:
            h_out = self.norm1(attn_output)
            
        # 3. FFN + Residual + Norm 2
        h_ffn = self.ffn(h_out)
        if self.residual:
            h_out = self.norm2(h_out + h_ffn)
        else:
            h_out = self.norm2(h_ffn)
            
        return h_out


class CrossGraphAttention_old(nn.Module):
    """
    --- 已过时 ---
    当前方法没有加上batch mask, 使得crossattention将在整个batch上的所有图中进行
    标准的交叉图注意力模块 (Standard Cross-Graph Attention)
    基于 Transformer Decoder 的 Cross Attention 块设计。
    
    Flow:
    1. Query 来自 Stream A
    2. Key, Value 来自 Stream B
    3. Attn = Softmax(Q @ K.T / sqrt(d)) @ V
    4. Residual Connection + LayerNorm
    5. Feed Forward Network (FFN) + Residual + LayerNorm
    """
    def __init__(self, feature_dim, num_heads=4, dropout=0.0, residual=True):
        super(CrossGraphAttention_old, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.residual = residual
        
        # 1. Multi-Head Attention
        # batch_first=True: 输入格式为 (Batch, Seq_Len, Dim)
        # 在 PyG 中，我们将所有节点视为一个序列: (1, Num_Nodes, Dim)
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, 
                                          num_heads=num_heads, 
                                          dropout=dropout, 
                                          batch_first=True)
        
        self.norm1 = nn.LayerNorm(feature_dim)
        
        # 2. Feed Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_query, h_key_value):
        """
        :param h_query: [N, D] (Stream A)
        :param h_key_value: [N, D] (Stream B)
        """
        # PyTorch 的 MultiheadAttention 需要 3D 输入 (Batch, Seq, Dim)
        # 我们将整个图的节点列表视为 Sequence，Batch Size = 1
        # Q: [1, N, D], K: [1, N, D], V: [1, N, D]
        Q = h_query.unsqueeze(0)
        K = h_key_value.unsqueeze(0)
        V = h_key_value.unsqueeze(0)
        
        # 1. Attention
        # attn_output: [1, N, D]
        attn_output, _ = self.attn(query=Q, key=K, value=V)
        
        # 挤压回 [N, D]
        attn_output = attn_output.squeeze(0)
        
        # 2. Residual + Norm 1
        if self.residual:
            h_out = self.norm1(h_query + self.dropout(attn_output))
        else:
            h_out = self.norm1(attn_output)
            
        # 3. FFN + Residual + Norm 2
        h_ffn = self.ffn(h_out)
        if self.residual:
            h_out = self.norm2(h_out + h_ffn)
        else:
            h_out = self.norm2(h_ffn)
            
        return h_out


class DualStreamFusion_CrossAttn(nn.Module):
    """
    --- 调用统一整合至fusion_layers 这里不再被使用 ---
    双向交叉注意力融合
    Topo <- Attn(Q=Topo, KV=Geom)
    Geom <- Attn(Q=Geom, KV=Topo)
    """
    def __init__(self, feature_dim, num_heads=4, dropout=0.0):
        super(DualStreamFusion_CrossAttn, self).__init__()
        self.fusion_t2g = CrossGraphAttentionLayer(feature_dim, num_heads, dropout) # Geom 增强 Topo (Q=Topo)
        self.fusion_g2t = CrossGraphAttentionLayer(feature_dim, num_heads, dropout) # Topo 增强 Geom (Q=Geom)
        
    def forward(self, h_topo, h_geom):
        # 对于存在None
        if h_topo == None or h_geom == None:
            return h_topo, h_geom
        # Topo 查询 Geom 的信息来增强自己
        h_topo_new = self.fusion_t2g(h_query=h_topo, h_key_value=h_geom)
        
        # Geom 查询 Topo 的信息来增强自己
        h_geom_new = self.fusion_g2t(h_query=h_geom, h_key_value=h_topo)
        
        return h_topo_new, h_geom_new

class DualStreamFusion_CrossGated(nn.Module):
    """
    --- 调用统一整合至fusion_layers 这里不再被使用 ---
    双向融合层
    同时执行 Topo -> Geom 和 Geom -> Topo 的融合
    """
    def __init__(self, feature_dim, num_heads=4, dropout=0.0):
        super(DualStreamFusion_CrossGated, self).__init__()
        # 两个方向的融合模块
        self.fusion_t2g = CrossGatedFusionLayer(feature_dim, num_heads, dropout) # Topo -> Geom
        self.fusion_g2t = CrossGatedFusionLayer(feature_dim, num_heads, dropout) # Geom -> Topo
        
    def forward(self, h_topo, h_geom):
        # 对于存在None
        if h_topo == None or h_geom == None:
            return h_topo, h_geom
        # 几何流增强拓扑流
        h_topo_new = self.fusion_g2t(h_query=h_topo, h_key_value=h_geom)
        
        # 拓扑流增强几何流
        h_geom_new = self.fusion_t2g(h_query=h_geom, h_key_value=h_topo)
        
        return h_topo_new, h_geom_new