import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from models.layers.basic import MLP

class InnerProductHead(nn.Module):
    """
    [基于 AAGNet] 内积解码器
    通过计算 Query 和 Key 的内积来预测节点间的相关性矩阵 (Adjacency Matrix)。
    用于实例分割任务。
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=256, num_layers=2, dropout=0.0):
        super(InnerProductHead, self).__init__()
        
        # Query 映射网络
        self.Wq = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            norm_layer=nn.LayerNorm,
            act_layer=nn.Mish,
            dropout=dropout
        )
        
        # Key 映射网络
        self.Wk = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            norm_layer=nn.LayerNorm,
            act_layer=nn.Mish,
            dropout=dropout
        )

    def forward(self, h_fused, batch_ptr=None):
        """
        :param h_fused: [Total_Nodes, D] 融合后的节点特征
        :param batch_ptr: [Batch_Size + 1] PyG 的 ptr 属性，指示每个图的节点范围
        :return: inst_matrix [Batch_Size, Max_Nodes, Max_Nodes] 预测的相关性矩阵
                 mask [Batch_Size, Max_Nodes, Max_Nodes] 有效区域掩码
        """
        if batch_ptr is None:
            # 如果没有 batch 信息，假设只有一个图
            q = self.Wq(h_fused) # [N, D]
            k = self.Wk(h_fused) # [N, D]
            inst_matrix = torch.matmul(q, k.t()) # [N, N]
            return inst_matrix.unsqueeze(0), None

        # 1. 将大图拆分为 Batch 列表
        # ptr: [0, n1, n1+n2, ...]
        batch_num_nodes = (batch_ptr[1:] - batch_ptr[:-1]).tolist()
        hidden_list = torch.split(h_fused, batch_num_nodes, dim=0)
        
        # 2. Pad 到最大节点数以便批量计算
        # padded_h: [Batch, Max_Nodes, D]
        padded_h = pad_sequence(hidden_list, batch_first=True)
        
        # 3. 生成有效掩码 (用于 Loss 计算时忽略 Padding 部分)
        B, Max_N, _ = padded_h.shape
        device = h_fused.device
        
        # 创建 mask: [B, Max_N]
        # 比如节点数为 3，Max为 5，则 mask 为 [1, 1, 1, 0, 0]
        # range_tensor: [0, 1, 2, ..., Max_N-1]
        range_tensor = torch.arange(Max_N, device=device).unsqueeze(0).expand(B, Max_N)
        # lengths: [B, 1]
        lengths = torch.tensor(batch_num_nodes, device=device).unsqueeze(1)
        node_mask = range_tensor < lengths # [B, Max_N]
        
        # 扩展为矩阵掩码: [B, Max_N, Max_N]
        # 只有 row 和 col 都在有效范围内才是有效的
        matrix_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)

        # 4. 计算 Q 和 K
        q = self.Wq(padded_h) # [B, Max_N, Out_Dim]
        k = self.Wk(padded_h) # [B, Max_N, Out_Dim]
        
        # 5. 批量矩阵乘法 (BMM)
        # Result: [B, Max_N, Max_N]
        inst_matrix = torch.bmm(q, k.transpose(1, 2))
        
        return inst_matrix, matrix_mask


class BilinearHead(nn.Module):
    """
    [新增] 双线性解码器 (Bilinear Decoder)
    原理: Score = h_i^T * W * h_j
    优势: 引入可学习的交互矩阵 W, 可以捕捉特征之间的互补关系 (Complementarity)
          适合非对称连接或异质特征匹配 (如凸台匹配凹槽)
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=256, num_layers=2, dropout=0.0, num_outputs=2, symmetric_model='none'):
        super(BilinearHead, self).__init__()
        # 对称处理标识符
        # 'none': 完全不对称 (独立 Q/K, 自由 W)
        # 'hard': 硬对称 (Q=K, W=W.T) -> 物理上强制对称
        # 'soft': 软对称 (独立 Q/K, 自由 W, 输出取平均) -> 结果强制对称
        self.symmetric_model = symmetric_model.lower()
        
        # 特征投影层Q (先降维/特征变换，减少计算量)
        # 这里依然保留两路投影，允许非对称特征提取
        self.project_q = MLP(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=output_dim,
            num_layers=num_layers,
            norm_layer=nn.LayerNorm,
            act_layer=nn.Mish,
            dropout=dropout
        )
        
        if self.symmetric_model == 'hard':
            # 硬对称模式：共享投影层
            # 只有当 Q(x) == K(x) 且 W == W.T 时，x^T W y 才是对称的
            self.project_k = self.project_q
        else:
            # 不对称处理
            self.project_k = MLP(
                    input_dim=input_dim, 
                    hidden_dim=hidden_dim, 
                    output_dim=output_dim,
                    num_layers=num_layers,
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.Mish,
                    dropout=dropout)
        
        # 核心双线性权重矩阵 W: [D, D]
        # 使用 nn.Bilinear 的底层逻辑，但手动实现以适配矩阵运算
        self.bilinear_weight = nn.Parameter(torch.Tensor(output_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # 使用 Xavier 初始化
        nn.init.xavier_uniform_(self.bilinear_weight)

    def forward(self, h_fused, batch_ptr=None):
        # 1. 拆分 Batch & Padding (逻辑同 InnerProduct)
        if batch_ptr is not None:
            batch_num_nodes = (batch_ptr[1:] - batch_ptr[:-1]).tolist()
            hidden_list = torch.split(h_fused, batch_num_nodes, dim=0)
            h_padded = pad_sequence(hidden_list, batch_first=True)

            # Mask generation
            B, Max_N, _ = h_padded.shape
            range_tensor = torch.arange(Max_N, device=h_fused.device).unsqueeze(0).expand(B, Max_N)
            lengths = torch.tensor(batch_num_nodes, device=h_fused.device).unsqueeze(1)
            node_mask = range_tensor < lengths 
            matrix_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        else:
            h_padded = h_fused.unsqueeze(0)
            matrix_mask = None

        # 2. 投影特征
        Q = self.project_q(h_padded) # [B, N, D_out]
        if self.symmetric_model == 'hard':
            # 硬对称模式：共享投影层
            # 只有当 Q(x) == K(x) 且 W == W.T 时，x^T W y 才是对称的
            K = Q
        else:
            K = self.project_k(h_padded) # [B, N, D_out]

        # 准备权重矩阵 W
        W = self.bilinear_weight
        
        # [核心逻辑] 硬对称：强制 W 对称
        # 即使 Q=K，如果 W 不对称，结果也不对称 (x^T W y != y^T W x)
        # 所以 Hard 模式必须同时保证 Q=K 和 W=W.T
        if self.symmetric_model == 'hard':
            W = (W + W.transpose(0, 1)) / 2
        
        # 3. 双线性计算: Q @ W @ K^T
        # Step A: Q_w = Q @ W -> [B, N, D] @ [D, D] -> [B, N, D]
        # Step B: Logits = Q_w @ K^T -> [B, N, D] @ [B, D, N] -> [B, N, N]
        # logits = torch.matmul(Q_w, K.transpose(1, 2))
        # 计算：bij = sum_{d1,d2} Q_{b,i,d1} * W_{d1,d2} * K_{b,j,d2}
        # 使用einsum以实现更高效的内存访问
        # Einsum 计算双线性积分: Batch, i-node, dim_1; dim_1, dim_2; Batch, j-node, dim_2 -> Batch, i, j
        # logits[b, i, j] = sum(Q[b, i, d] * W[d, e] * K[b, j, e])
        logits = torch.einsum('bid,de,bje->bij', Q, W, K)

        # 若为'soft', 则取上三角和下三角的平均
        if self.symmetric_model == 'soft':
            logits = (logits + logits.transpose(1, 2)) / 2
        
        return logits, matrix_mask


class MLPConcatHead_origin(nn.Module):
    """
    [新增] MLP 拼接解码器 (Concat Decoder)
    原理: Score = MLP(Concat(h_i, h_j))
    优势: 具有最强的非线性拟合能力，可以学习任意复杂的连接规则。
    劣势: 显存占用较高 (O(N^2))，计算量大。
    """
    def __init__(self, input_dim, hidden_dim=256, projected_dim=128, num_layers=2, dropout=0.0):
        super(MLPConcatHead, self).__init__()
        
        # 1. 预处理层: 先将高维特征压缩，减少后续 N^2 拼接时的显存压力
        self.pre_proj = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=projected_dim, # 压缩到较小维度 (如 128)
            num_layers=num_layers,
            norm_layer=nn.LayerNorm,
            act_layer=nn.Mish,
            dropout=dropout
        )
        
        # 2. 评分网络: 输入是 2 * feature_dim，输出是 1
        # 这是一个针对 Pair 的二分类器
        self.classifier = nn.Sequential(
            nn.Linear(projected_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1) # 输出 Logits
        )

    def forward(self, h_fused, batch_ptr=None):
        # 1. 拆分 Batch & Padding
        if batch_ptr is not None:
            batch_index = batch_ptr
            _, counts = torch.unique(batch_index, return_counts=True)
            h_list = torch.split(h_fused, counts.tolist())
            h_padded = pad_sequence(h_list, batch_first=True) # [B, Max_N, D_in]
            lengths = counts
        else:
            h_padded = h_fused.unsqueeze(0)
            lengths = torch.tensor([h_fused.size(0)], device=h_fused.device)

        # 2. 预处理特征
        H = self.pre_proj(h_padded) # [B, N, D_feat]
        B, N, D = H.shape
        
        # 3. 构建 Pairwise 特征 (Broadcasting)
        # h_i: [B, N, 1, D] (行扩展)
        h_i = H.unsqueeze(2).expand(-1, -1, N, -1)
        # h_j: [B, 1, N, D] (列扩展)
        h_j = H.unsqueeze(1).expand(-1, N, -1, -1)
        
        # 拼接: [B, N, N, 2D]
        # 注意: 这里会分配大量显存，混合精度(AMP)对此处帮助很大
        h_pairs = torch.cat([h_i, h_j], dim=-1)
        
        # 4. MLP 评分
        # [B, N, N, 2D] -> [B, N, N, 1]
        logits = self.classifier(h_pairs)
        
        # [B, N, N]
        logits = logits.squeeze(-1)
        
        # 5. 生成掩码
        range_tensor = torch.arange(N, device=H.device).unsqueeze(0).expand(B, N)
        lengths = lengths.unsqueeze(1)
        node_mask = range_tensor < lengths 
        matrix_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        
        return logits, matrix_mask


class MLPConcatHead(nn.Module):
    """
    [高性能优化版] MLP 拼接解码器
    
    优化策略: 
    不显式构造 [B, N, N, 2D] 的拼接张量。
    利用 Linear(Cat(A, B)) = Linear(A) + Linear(B) + bias 的性质。
    先将特征投影到 hidden_dim, 再广播相加, 最后过激活函数。
    这能显著降低峰值显存占用。
    """
    def __init__(self, input_dim, hidden_dim=256, projected_dim=32, num_layers=2, dropout=0.2):
        super(MLPConcatHead, self).__init__()
        
        # 1. 特征压缩/预处理
        self.pre_proj = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=projected_dim,
            num_layers=num_layers,
            norm_layer=nn.LayerNorm,
            act_layer=nn.ReLU,
            dropout=dropout
        )
        
        # 2. 分解后的第一层线性变换 (对应输入层)
        # 输入是 projected_dim，输出是 hidden_dim
        # 我们用两个独立的 Linear 层分别处理 i 和 j，代替一个处理 2*projected_dim 的层
        self.proj_i = nn.Linear(projected_dim, hidden_dim)
        self.proj_j = nn.Linear(projected_dim, hidden_dim)
        
        # 3. MLP 的后续部分 (Norm -> Act -> Dropout -> Linear_Out)
        self.mlp_tail = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Mish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1) # 输出 Logits
        )

    def forward(self, h_fused, batch_ptr=None):
        # 1. 拆分 Batch
        if batch_ptr is not None:
            batch_num_nodes = (batch_ptr[1:] - batch_ptr[:-1]).tolist()
            hidden_list = torch.split(h_fused, batch_num_nodes, dim=0)
            h_padded = pad_sequence(hidden_list, batch_first=True)
            
            B, Max_N, _ = h_padded.shape
            range_tensor = torch.arange(Max_N, device=h_fused.device).unsqueeze(0).expand(B, Max_N)
            lengths = torch.tensor(batch_num_nodes, device=h_fused.device).unsqueeze(1)
            node_mask = range_tensor < lengths 
            matrix_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        else:
            h_padded = h_fused.unsqueeze(0)
            matrix_mask = None

        # 2. 特征预处理
        H = self.pre_proj(h_padded) # [B, N, D_proj]
        
        # 3. 分解计算 (Decomposed Computation)
        # 替代 H_pairs = cat(H_i, H_j) -> Linear
        
        # 计算 H_i 的贡献 [B, N, Hidden]
        feat_i = self.proj_i(H)
        # 计算 H_j 的贡献 [B, N, Hidden]
        feat_j = self.proj_j(H)
        
        # 广播相加: [B, N, 1, H] + [B, 1, N, H] -> [B, N, N, H]
        # 这里发生了广播，得到了 N*N 的 Pairwise 特征，但维度是 Hidden 而不是 2*D_proj
        # 通常 Hidden (256) ≈ 2*D_proj (2*128)，显存占用相似，但省去了 Cat 的内存拷贝开销
        h_pair_fused = feat_i.unsqueeze(2) + feat_j.unsqueeze(1)
        
        # 4. 后续 MLP 计算
        logits = self.mlp_tail(h_pair_fused) # [B, N, N, 1]
        
        return logits.squeeze(-1), matrix_mask


class MLP_Multi_Class(nn.Module):
    """
    多分类头
    1. 隐层有激活函数 (Mish)
    2. 最后一层是纯线性 (无激活)，输出 Logits
    """
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: Input -> Hidden
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(inplace=True),
            nn.Dropout(dropout),
            
            # Layer 2: Hidden -> Hidden (增加深度)
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(inplace=True),
            nn.Dropout(dropout),
            
            # Layer 3: Output (Logits)
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)
    

class MLP_Bin_Class(nn.Module):
    """
    二分类头 (底面预测)
    确保输出 logits (未经过 Sigmoid/Mish 截断)
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(input_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            # nn.Mish(inplace=True),
            # nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Mish(inplace=True),
            nn.Dropout(dropout),
            
            # Output Logits (1 dim)
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x)


class NoneHead(nn.Module):
    """
    显式占位任务头：
    根据设定的 num_outputs 返回固定长度的 None 元组
    """
    def __init__(self, num_outputs=1, *args, **kwargs):
        super().__init__()
        self.num_outputs = num_outputs

    def forward(self, *args, **kwargs):
        # 情况 1: 如果只需要 1 个返回量，直接返回 None
        if self.num_outputs == 1:
            return None
        
        # 情况 2: 如果需要多个返回量，返回对应长度的元组
        return (None,) * self.num_outputs

    def __repr__(self):
        return f"NoneHead(num_outputs={self.num_outputs})"


class TaskHeads(nn.Module):
    """
    多任务解码头集合
    包含:
    1. 语义分割头 (Semantic Segmentation) -> 分类
    2. 实例分割头 (Instance Segmentation) -> 嵌入向量
    3. 底面预测头 (Bottom Prediction) -> 二分类
    """
    def __init__(self, 
                 input_dim, 
                 num_semantic_classes, 
                 instance_embed_dim=16, 
                 hidden_dim=256,
                 dropout=0.2,
                 decoder_type={'sem' :'mlp_multi_class',
                               'inst':'inner_product_head',
                               'bot': 'mlp_bin_class',
                               'symmetric_model': 'none'}):
        super(TaskHeads, self).__init__()
        
        # 1. 语义分割头
        self.semantic_head = build_decoder_layer(decoder_type['sem'],
                                                 feature_dim=input_dim,
                                                 hidden_dim=hidden_dim,
                                                 output_dim=num_semantic_classes,
                                                 dropout=dropout,
                                                 num_outputs=1)
        
        # 2. 实例分割头 (Metric Learning Embedding)
        self.instance_head = build_decoder_layer(decoder_type['inst'],
                                                 feature_dim=input_dim,
                                                 hidden_dim=hidden_dim,
                                                 output_dim=instance_embed_dim,
                                                 dropout=dropout,
                                                 symmetric_model=decoder_type['symmetric_model'],
                                                 num_outputs=2)
        
        # 3. 底面预测头 (二值分类: 是底面/不是底面)
        self.bottom_head = build_decoder_layer(decoder_type['bot'],
                                                 feature_dim=input_dim,
                                                 hidden_dim=hidden_dim,
                                                 output_dim=1,
                                                 dropout=dropout,
                                                 num_outputs=1)
        
    def forward(self, h_fused, batch_ptr=None):
        """
        :param h_fused: 融合后的节点特征 [N, D]
        :param batch_ptr: PyG batch pointer
        :return: (semantic_logits, instance_embeds, bottom_logits)
        """
        # 1. 语义分割
        sem_logits = self.semantic_head(h_fused)
        # 2. 实例分割 (返回矩阵和掩码)
        inst_matrix, inst_mask = self.instance_head(h_fused, batch_ptr)
        # 3. 底面预测
        bottom_logits = self.bottom_head(h_fused)
        
        return sem_logits, inst_matrix, inst_mask, bottom_logits


def build_decoder_layer(decoder_method, feature_dim, hidden_dim, output_dim, dropout=0.0, **kwargs):
    """
    [融合层工厂]
    method: 'inner_product_head', 'bilinear_head', 'mlp_concat_head', 'mlp_multi_class', 'mlp_bin_class'

    :param fusion_method:
    :param feature_dim:
    :param dropout:
    """
        
    dtype = decoder_method.lower()

    if dtype == 'none':
        return NoneHead(input_dim=feature_dim, 
                        hidden_dim=hidden_dim, 
                        output_dim=output_dim, dropout=dropout, **kwargs)
    
    if dtype == 'inner_product_head':
        return InnerProductHead(input_dim=feature_dim, 
                                hidden_dim=hidden_dim, 
                                output_dim=output_dim, dropout=dropout)
    
    elif dtype == 'bilinear_head':
        return BilinearHead(input_dim=feature_dim, 
                            hidden_dim=hidden_dim, 
                            output_dim=output_dim, dropout=dropout, **kwargs)
    
    elif dtype == 'mlp_concat_head':
        return MLPConcatHead(input_dim=feature_dim, 
                            hidden_dim=hidden_dim, dropout=dropout if dropout >= 0.4 else 0.4)
    
    elif dtype == 'mlp_multi_class':
        return MLP_Multi_Class(input_dim=feature_dim, hidden_dim=hidden_dim, 
                               num_classes=output_dim, dropout=dropout)
        
    elif dtype == 'mlp_bin_class':
        return MLP_Bin_Class(input_dim=feature_dim, hidden_dim=hidden_dim,
                             dropout=dropout)
        
    else:
        raise ValueError(f"Unknown fusion type: {decoder_method}")