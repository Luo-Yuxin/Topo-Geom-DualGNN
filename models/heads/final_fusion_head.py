import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from models.layers.basic import fc_block, MLP

class BaseFusion(nn.Module):
    """
    融合层基类
    负责处理流的清洗、对齐和补全
    """
    def __init__(self, stream_usage='both'):
        super(BaseFusion, self).__init__()
        self.stream_usage = stream_usage.lower()
        
        if self.stream_usage not in ['both', 'topo', 'geom']:
            raise ValueError(f"Invalid stream_usage: {self.stream_usage}")

    def _clean_inputs(self, h_topo, h_geom):
        """
        根据 stream_usage 规范化输入数据
        """
        # 1. 如果需要 Topo，但输入为 None -> 报错或补零
        # 2. 如果需要 Both，缺一 -> 补零
        # 3. 如果只需要 Topo -> 忽略 Geom
        
        # 策略 A: 仅使用 Topo
        if self.stream_usage == 'topo':
            if h_topo is None:
                # 极端情况：配置要 Topo 但没给数据，通常应报错，但在消融中可能希望返回全0
                # 这里假设 h_geom 可能存在，借用其形状补0，如果都不存在则返回 None
                if h_geom is not None:
                    h_topo = torch.zeros_like(h_geom)
                else:
                    return None, None # 无法补救
            return h_topo, None # 强制丢弃 h_geom

        # 策略 B: 仅使用 Geom
        elif self.stream_usage == 'geom':
            if h_geom is None:
                if h_topo is not None:
                    h_geom = torch.zeros_like(h_topo)
                else:
                    return None, None
            return None, h_geom # 强制丢弃 h_topo

        # 策略 C: 双流融合 (Both)
        else:
            if h_topo is None and h_geom is None:
                return None, None
            
            # 自动补全缺失的一方
            if h_topo is None:
                h_topo = torch.zeros_like(h_geom)
            if h_geom is None:
                h_geom = torch.zeros_like(h_topo)
                
            return h_topo, h_geom
        
class SumFusion(BaseFusion):
    """
    相加融合
    Usage 'both': h_topo + h_geom
    Usage 'topo': h_topo (等价于 Identity)
    """
    def __init__(self, stream_usage='both', **kwargs):
        super(SumFusion, self).__init__(stream_usage)
    
    def forward(self, h_topo, h_geom, **kwargs):
        h_topo, h_geom = self._clean_inputs(h_topo, h_geom)
        # 流控制
        if self.stream_usage == 'topo':
            return h_topo
        if self.stream_usage == 'geom':
            return h_geom
        # 简单的双向相加
        # 这里的假设是：信息是互补的，直接叠加即可
        # Both
        # 因为经过了清理, 这里h_topo出现None其实是出了故障
        if h_topo is None: return None
        return h_topo + h_geom

class ConcatFusion(BaseFusion):
    """
    简单拼接
    Usage 'both': cat([h_topo, h_geom]) -> 2D
    Usage 'h_topo or h_geom': h_topo or h_geom -> D
    """
    def __init__(self, stream_usage='both', **kwargs):
        super(ConcatFusion, self).__init__(stream_usage)
    
    def forward(self, h_topo, h_geom, **kwargs):
        h_topo, h_geom = self._clean_inputs(h_topo, h_geom)
        
        if self.stream_usage == 'topo':
            return h_topo
        if self.stream_usage == 'geom':
            return h_geom
            
        if h_topo is None: return None
        return torch.cat([h_topo, h_geom], dim=-1)

class ConcatFusionDeep(BaseFusion):
    """
    简单拼接后深度融合
    h = MLP(cat(h_a, h_b))
    """
    def __init__(self, input_dim, output_dim, dropout=0.0, stream_usage='both', **kwargs):
        super(ConcatFusionDeep, self).__init__(stream_usage)
        # 定义一个concat用的MLP
        # 根据 usage 计算 MLP 输入维度
        if self.stream_usage == 'both':
            enc_dim = input_dim * 2
        else:
            enc_dim = input_dim # 单流情况下，MLP 也可以作为一种非线性变换层存在
            
        self.mlp = MLP(input_dim=enc_dim,
                       hidden_dim=enc_dim, # 保持宽度
                       output_dim=output_dim,
                       num_layers=2,
                       norm_layer=None, 
                       bias=True,
                       dropout=dropout,
                       act_layer=nn.LeakyReLU,
                       negative_slope=0.02,
                       inplace=True)
        
    def forward(self, h_topo, h_geom, **kwargs):
        h_topo, h_geom = self._clean_inputs(h_topo, h_geom)
        
        if self.stream_usage == 'topo':
            if h_topo is None: return None
            feat = h_topo
        elif self.stream_usage == 'geom':
            if h_geom is None: return None
            feat = h_geom
        else:
            if h_topo is None: return None
            feat = torch.cat([h_topo, h_geom], dim=-1)
            
        return self.mlp(feat)
        

class PoolingConcatFusion(BaseFusion):
    """
    融合时会将全局池化特征加入
    支持单流或双流的池化增强。
    
    参数:
    - pooling_method: 'max', 'mean', 'both' (控制池化类型)
    - stream_usage: 'topo', 'geom', 'both' (控制使用哪些流)
    
    维度计算示例 (D=128):
    - Usage='both', Pool='both':
      [Topo, Geom, TopoMax, TopoMean, GeomMax, GeomMean] -> 6 * D -> MLP -> Out
    - Usage='topo', Pool='max':
      [Topo, TopoMax] -> 2 * D -> MLP -> Out
    """
    def __init__(self, input_dim, output_dim, dropout=0.0, 
                 pooling_method='both', stream_usage='both', **kwargs):
        super(PoolingConcatFusion, self).__init__(stream_usage)

        self.pooling_method = pooling_method.lower()

        # 计算拼接后的总维度

        # 1. 基础特征维度 (Node Level)
        num_streams = 2 if self.stream_usage == 'both' else 1
        base_dim = input_dim * num_streams

        # 2. 池化特征维度 (Graph Level Broadcast)
        num_stats = 0
        if self.pooling_method in ['max', 'mean']:
            num_stats = 1
        elif self.pooling_method == 'both':
            num_stats = 2
        
        pool_dim = input_dim * num_stats * num_streams
        concat_dim = base_dim + pool_dim

        self.projector = MLP(input_dim=concat_dim,
                             hidden_dim=input_dim * 2, # 这里的隐层可以根据需要调整
                             output_dim=output_dim,
                             num_layers=2,
                             bias=True,
                             dropout=dropout,
                             act_layer=nn.LeakyReLU)

    def forward(self, h_topo, h_geom, batch=None, **kwargs):
        h_topo, h_geom = self._clean_inputs(h_topo, h_geom)
        
        # 确定主数据 (用于获取 device, batch_size 等)
        ref_feat = h_topo if h_topo is not None else h_geom
        if ref_feat is None: return None
        
        if batch is None:
            batch = torch.zeros(ref_feat.size(0), dtype=torch.long, device=ref_feat.device)

        concat_list = []
        
        # --- 收集特征 ---
        
        # 1. Topo Stream
        if self.stream_usage in ['topo', 'both']:
            concat_list.append(h_topo)
            # Pooling
            if self.pooling_method in ['max', 'both']:
                concat_list.append(global_max_pool(h_topo, batch)[batch])
            if self.pooling_method in ['mean', 'both']:
                concat_list.append(global_mean_pool(h_topo, batch)[batch])
                
        # 2. Geom Stream
        if self.stream_usage in ['geom', 'both']:
            concat_list.append(h_geom)
            # Pooling
            if self.pooling_method in ['max', 'both']:
                concat_list.append(global_max_pool(h_geom, batch)[batch])
            if self.pooling_method in ['mean', 'both']:
                concat_list.append(global_mean_pool(h_geom, batch)[batch])
        
        # 3. 融合
        combined = torch.cat(concat_list, dim=-1)
        return self.projector(combined)
        
class Identity(BaseFusion):
    """
    仅用于控制输出流, 原输出流不做处理
    """
    def __init__(self, stream_usage='topo', **kwargs):
        super(Identity, self).__init__(stream_usage)

    def forward(self, h_topo, h_geom, **kwargs):
        h_topo, h_geom = self._clean_inputs(h_topo, h_geom)
        # 流控制
        if self.stream_usage == 'topo':
            return h_topo
        if self.stream_usage == 'geom':
            return h_geom
        if self.stream_usage == 'both':
            raise ValueError("Identity cannot process 'both'.")

def build_final_fusion_layer(fusion_method, feature_dim, dropout=0.0, **kwargs):
    """
    [融合层工厂]
    method: 'sum', 'concat', 'concat_deep', 'pooling'

    :param fusion_method (str): 'sum', 'concat', 'concat_deep', 'pooling'
    :param feature_dim (int): 
    :param dropout (float): 
    :param kwargs: 包含 stream_usage, pooling_method 等
    """
    if fusion_method is None or fusion_method == 'None':
        return None, 0 # 这里返回 0 或特殊值，外层需处理
        
    ftype = fusion_method.lower()
    # 获得流控制参数
    usage = kwargs.get('stream_usage', 'both')

    # 确定输出维度是否被投影回 feature_dim
    # 大多数 Deep/Pooling 方法都会投影回 feature_dim
    # Sum 不变
    # Concat 翻倍 (如果是 Both)
    
    if ftype == 'sum':
        # Sum 不改变维度，无论单双流
        return SumFusion(stream_usage=usage), feature_dim
    
    elif ftype == 'concat':
        # 若双流输出则为2倍输入参数, 若仅单流输出则为原参数
        num_streams = 2 if usage == 'both' else 1
        return ConcatFusion(stream_usage=usage), feature_dim * num_streams
    
    elif ftype == 'concat_deep':
        return ConcatFusionDeep(input_dim=feature_dim, 
                                output_dim=feature_dim, 
                                dropout=dropout,
                                stream_usage=usage), feature_dim
        
    elif ftype == 'pooling':
        p_method = kwargs.get('pooling_method', 'both')
        return PoolingConcatFusion(input_dim=feature_dim,
                                   output_dim=feature_dim,
                                   dropout=dropout, 
                                   pooling_method=p_method,
                                   stream_usage=usage), feature_dim
    
    elif ftype == 'identity':
        return Identity(stream_usage=usage), feature_dim
        
    else:
        raise ValueError(f"Unknown fusion type: {fusion_method}")
    






        