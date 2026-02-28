import torch
import torch.nn as nn
from models.fusion.cross_attention import CrossGraphAttentionLayer
from models.fusion.cross_attention import CrossGatedFusionLayer, CrossGatedFusionLightLayer
from models.fusion.FiLM import FiLMFusionLayer
from models.layers.basic import fc_block, MLP

class BaseInterStreamFusion(nn.Module):
    """
    [中间层融合基类]
    职责:
    1. 解析融合方向 (dir): '^', 'v', 'x'
    2. 数据完整性检查 (防御性编程)
    3. 统一的 forward 模板，子类只需实现具体的计算逻辑
    """
    def __init__(self, fusion_dir='x'):
        super(BaseInterStreamFusion, self).__init__()
        # 归一化方向字符
        self.fusion_dir = fusion_dir.lower() if fusion_dir else 'x'
        
        if self.fusion_dir not in ['^', 'v', 'x']:
            raise ValueError(f"[Fusion] Invalid direction: {self.fusion_dir}. Must be '^', 'v', or 'x'.")

    def forward(self, h_topo, h_geom, **kwargs):
        """
        统一入口，不要重写此方法 (除非有特殊需求)
        """
        # 1. 数据检查
        if h_topo is None or h_geom is None:
            # 中间层融合通常要求双流都存在。如果缺一，通常意味着无法进行交互。
            # 策略：直接透传，不做融合 (Identity)
            return h_topo, h_geom

        # 2. 根据方向分发逻辑
        
        # Case A: 向上 (^): Geom 指导 Topo
        if self.fusion_dir == '^':
            # 计算 Topo 的新特征 (Geom 不变)
            h_topo_new = self.fusion_forward(target=h_topo, source=h_geom, direction='up', **kwargs)
            return h_topo_new, h_geom

        # Case B: 向下 (v): Topo 指导 Geom
        elif self.fusion_dir == 'v':
            # 计算 Geom 的新特征 (Topo 不变)
            h_geom_new = self.fusion_forward(target=h_geom, source=h_topo, direction='down', **kwargs)
            return h_topo, h_geom_new

        # Case C: 双向 (x): 互相指导
        else: # 'x'
            # 这里的实现取决于子类是否支持高效的双向计算
            # 默认实现：调用两次单向 (如果算法是对称的)
            # 子类可以重写 _fusion_bidirectional 来优化
            h_topo_new, h_geom_new = self.fusion_bidirectional(h_topo, h_geom, **kwargs)
            return h_topo_new, h_geom_new

    def fusion_forward(self, target, source, direction, **kwargs):
        """
        [子类必须实现] 单向融合逻辑
        Target = Function(Target, Source)
        """
        raise NotImplementedError

    def fusion_bidirectional(self, h_topo, h_geom, **kwargs):
        """
        [子类可选实现] 双向融合逻辑
        默认：分别调用两次 fusion_forward
        """
        # 注意：这里默认是“同时更新”，即计算 h_topo_new 时用的是旧 h_geom
        h_topo_new = self.fusion_forward(target=h_topo, source=h_geom, direction='up', **kwargs)
        h_geom_new = self.fusion_forward(target=h_geom, source=h_topo, direction='down', **kwargs)
        return h_topo_new, h_geom_new
    

class SumFusion(BaseInterStreamFusion):
    """
    简单相加融合
    Dir '^': h_topo = h_topo + h_geom
    Dir 'v': h_geom = h_geom + h_topo
    Dir 'x': 双方都加
    """
    def __init__(self, fusion_dir='x', **kwargs):
        super(SumFusion, self).__init__(fusion_dir)
    
    def fusion_forward(self, target, source, direction, **kwargs):
        return target + source
    

class ConcatFusion(BaseInterStreamFusion):
    """
    拼接融合 + 线性映射
    """
    def __init__(self, feature_dim, dropout=0.0, fusion_dir='x', **kwargs):
        super(ConcatFusion, self).__init__(fusion_dir)
        
        # 定义一个简单的 MLP 用于降维融合
        # 你的 ConcatFusionDeep 也可以直接套用这个模式
        
        self.fc_block = fc_block(input_dim=feature_dim*2,
                                 output_dim=feature_dim,
                                 norm_layer=nn.LayerNorm,
                                 act_layer=nn.LeakyReLU,
                                 negative_slope=0.02, 
                                 inplace=True,
                                 dropout=dropout, # Use dropout from args
                                 bias=False
                                 )
        
    def fusion_forward(self, target, source, direction, **kwargs):
        # 拼接
        cat_feat = torch.cat([target, source], dim=-1)
        # 投影
        return self.fc_block(cat_feat)
    

class ConcatDeepFusion(BaseInterStreamFusion):
    def __init__(self, feature_dim, dropout=0.0, fusion_dir='x', **kwargs):
        super(ConcatDeepFusion, self).__init__(fusion_dir)
        # 定义一个concat用的MLP
        self.mlp_cat = MLP(input_dim=feature_dim*2,
                           hidden_dim=feature_dim*4,
                           output_dim=feature_dim,
                           num_layers=3,
                           norm_layer=None,
                           bias=False,
                           dropout=dropout,
                           norm_layer_2=nn.LayerNorm,
                           act_layer=nn.LeakyReLU,
                           negative_slope = 0.01,
                           inplace = True)
        
    def fusion_forward(self, target, source, direction, **kwargs):
        # 拼接
        cat_feat = torch.cat([target, source], dim=-1)
        # 投影
        return self.mlp_cat(cat_feat)


class CrossGatedFusion(BaseInterStreamFusion):
    """
    [适配器] 将 CrossGatedFusionLayer (数学算子) 组装成支持流向控制的层。
    
    优化策略:
    1. 根据 fusion_dir 按需实例化，不浪费显存。
    2. 双向模式下 ('x')，实例化两个独立的层，保证参数不共享。
    """
    def __init__(self, feature_dim, dropout=0.0, fusion_dir='x'):
        super(CrossGatedFusion, self).__init__(fusion_dir)
        
        self.feature_dim = feature_dim
        self.dropout = dropout
        
        # 预定义为 None，防止调用不存在的属性报错
        self.fusion_g2t = None # Geom -> Topo (Up)
        self.fusion_t2g = None # Topo -> Geom (Down)

        # === 按需实例化 ===
        
        # Case 1: 向上 或 双向 -> 需要实例化 g2t
        if self.fusion_dir in ['^', 'x']:
            self.fusion_g2t = CrossGatedFusionLayer(feature_dim, dropout)
            
        # Case 2: 向下 或 双向 -> 需要实例化 t2g
        if self.fusion_dir in ['v', 'x']:
            self.fusion_t2g = CrossGatedFusionLayer(feature_dim, dropout)

    def fusion_forward(self, target, source, direction, **kwargs):
        """
        单向调用逻辑
        """
        if direction == 'up':
            # Geom(Source) 指导 Topo(Target)
            # 只有在 init 中创建了 fusion_g2t 才能调用，否则说明逻辑/配置有误
            if self.fusion_g2t is None:
                raise RuntimeError("Requesting 'up' fusion but fusion_g2t was not initialized. Check config 'dir'.")
            return self.fusion_g2t(h_query=target, h_key_value=source)
            
        elif direction == 'down':
            # Topo(Source) 指导 Geom(Target)
            if self.fusion_t2g is None:
                raise RuntimeError("Requesting 'down' fusion but fusion_t2g was not initialized. Check config 'dir'.")
            return self.fusion_t2g(h_query=target, h_key_value=source)
        
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def fusion_bidirectional(self, h_topo, h_geom, **kwargs):
        """
        双向调用逻辑 (仅当 dir='x' 时会被基类调用)
        """
        # 此时 self.fusion_g2t 和 self.fusion_t2g 肯定都非空
        
        # Topo update (Geom -> Topo)
        h_topo_new = self.fusion_g2t(h_query=h_topo, h_key_value=h_geom)
        
        # Geom update (Topo -> Geom)
        h_geom_new = self.fusion_t2g(h_query=h_geom, h_key_value=h_topo)
        
        return h_topo_new, h_geom_new


class CrossGatedLightFusion(BaseInterStreamFusion):
    """
    [适配器] 将 CrossGatedFusionLayer (数学算子) 组装成支持流向控制的层。
    
    优化策略:
    1. 根据 fusion_dir 按需实例化，不浪费显存。
    2. 双向模式下 ('x')，实例化两个独立的层，保证参数不共享。
    """
    def __init__(self, feature_dim, dropout=0.0, fusion_dir='x'):
        super(CrossGatedLightFusion, self).__init__(fusion_dir)
        
        self.feature_dim = feature_dim
        self.dropout = dropout
        
        # 预定义为 None，防止调用不存在的属性报错
        self.fusion_g2t = None # Geom -> Topo (Up)
        self.fusion_t2g = None # Topo -> Geom (Down)

        # === 按需实例化 ===
        
        # Case 1: 向上 或 双向 -> 需要实例化 g2t
        if self.fusion_dir in ['^', 'x']:
            self.fusion_g2t = CrossGatedFusionLightLayer(feature_dim, dropout)
            
        # Case 2: 向下 或 双向 -> 需要实例化 t2g
        if self.fusion_dir in ['v', 'x']:
            self.fusion_t2g = CrossGatedFusionLightLayer(feature_dim, dropout)

    def fusion_forward(self, target, source, direction, **kwargs):
        """
        单向调用逻辑
        """
        if direction == 'up':
            # Geom(Source) 指导 Topo(Target)
            # 只有在 init 中创建了 fusion_g2t 才能调用，否则说明逻辑/配置有误
            if self.fusion_g2t is None:
                raise RuntimeError("Requesting 'up' fusion but fusion_g2t was not initialized. Check config 'dir'.")
            return self.fusion_g2t(h_query=target, h_key_value=source)
            
        elif direction == 'down':
            # Topo(Source) 指导 Geom(Target)
            if self.fusion_t2g is None:
                raise RuntimeError("Requesting 'down' fusion but fusion_t2g was not initialized. Check config 'dir'.")
            return self.fusion_t2g(h_query=target, h_key_value=source)
        
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def fusion_bidirectional(self, h_topo, h_geom, **kwargs):
        """
        双向调用逻辑 (仅当 dir='x' 时会被基类调用)
        """
        # 此时 self.fusion_g2t 和 self.fusion_t2g 肯定都非空
        
        # Topo update (Geom -> Topo)
        h_topo_new = self.fusion_g2t(h_query=h_topo, h_key_value=h_geom)
        
        # Geom update (Topo -> Geom)
        h_geom_new = self.fusion_t2g(h_query=h_geom, h_key_value=h_topo)
        
        return h_topo_new, h_geom_new


class CrossAttnFusion(BaseInterStreamFusion):
    """
    基于Cross attention方法
    """
    def __init__(self, feature_dim, num_heads=4, dropout=0.0, fusion_dir='x', **kwargs):
        super(CrossAttnFusion, self).__init__(fusion_dir)
        # 初始化参数
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # 预定义设置为None
        self.fusion_g2t = None # Geom -> Topo (Up)
        self.fusion_t2g = None # Topo -> Geom (Down)

        # Case 1: 向上 或 双向 -> 需要实例化 g2t
        if self.fusion_dir in ['^', 'x']:
            self.fusion_g2t = CrossGraphAttentionLayer(feature_dim, num_heads, dropout)
        
        # Case 2: 向下 或 双向 -> 需要实例化 t2g
        if self.fusion_dir in ['v', 'x']:
            self.fusion_t2g = CrossGraphAttentionLayer(feature_dim, num_heads, dropout)
    
    def fusion_forward(self, target, source, direction, **kwargs):
        """
        单向调用逻辑
        """
        if direction == 'up':
            # Geom(Source) 指导 Topo(Target)
            # 只有在 init 中创建了 fusion_g2t 才能调用，否则说明逻辑/配置有误
            if self.fusion_g2t is None:
                raise RuntimeError("Requesting 'up' fusion but fusion_g2t was not initialized. Check config 'dir'.")
            return self.fusion_g2t(h_query=target, h_key_value=source)
            
        elif direction == 'down':
            # Topo(Source) 指导 Geom(Target)
            if self.fusion_t2g is None:
                raise RuntimeError("Requesting 'down' fusion but fusion_t2g was not initialized. Check config 'dir'.")
            return self.fusion_t2g(h_query=target, h_key_value=source)
        
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    def fusion_bidirectional(self, h_topo, h_geom, **kwargs):
        """
        双向调用逻辑 (仅当 dir='x' 时会被基类调用)
        """
        # 此时 self.fusion_g2t 和 self.fusion_t2g 肯定都非空
        
        # Topo update (Geom -> Topo)
        h_topo_new = self.fusion_g2t(h_query=h_topo, h_key_value=h_geom)
        
        # Geom update (Topo -> Geom)
        h_geom_new = self.fusion_t2g(h_query=h_geom, h_key_value=h_topo)
        
        return h_topo_new, h_geom_new

    
class FiLMFusion(BaseInterStreamFusion):
    """
    基于Cross attention方法
    """
    def __init__(self, feature_dim, fusion_dir='x', **kwargs):
        super(FiLMFusion, self).__init__(fusion_dir)

        # 初始化参数
        self.feature_dim = feature_dim

        # 预定义设置为None
        self.fusion_g2t = None # Geom -> Topo (Up)
        self.fusion_t2g = None # Topo -> Geom (Down)

        # Case 1: 向上 或 双向 -> 需要实例化 g2t
        if self.fusion_dir in ['^', 'x']:
            self.fusion_g2t = FiLMFusionLayer(feature_dim)
        
        # Case 2: 向下 或 双向 -> 需要实例化 t2g
        if self.fusion_dir in ['v', 'x']:
            self.fusion_t2g = FiLMFusionLayer(feature_dim)
    
    def fusion_forward(self, target, source, direction, **kwargs):
        """
        单向调用逻辑
        """
        if direction == 'up':
            # Geom(Source) 指导 Topo(Target)
            # 只有在 init 中创建了 fusion_g2t 才能调用，否则说明逻辑/配置有误
            if self.fusion_g2t is None:
                raise RuntimeError("Requesting 'up' fusion but fusion_g2t was not initialized. Check config 'dir'.")
            return self.fusion_g2t(h_content=target, h_style=source)
            
        elif direction == 'down':
            # Topo(Source) 指导 Geom(Target)
            if self.fusion_t2g is None:
                raise RuntimeError("Requesting 'down' fusion but fusion_t2g was not initialized. Check config 'dir'.")
            return self.fusion_t2g(h_content=target, h_style=source)
        
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    def fusion_bidirectional(self, h_topo, h_geom, **kwargs):
        """
        双向调用逻辑 (仅当 dir='x' 时会被基类调用)
        """
        # 此时 self.fusion_g2t 和 self.fusion_t2g 肯定都非空
        
        # Topo update (Geom -> Topo)
        h_topo_new = self.fusion_g2t(h_content=h_topo, h_style=h_geom)
        
        # Geom update (Topo -> Geom)
        h_geom_new = self.fusion_t2g(h_content=h_geom, h_style=h_topo)

        return h_topo_new, h_geom_new
        
    
def build_fusion_layer(fusion_method, feature_dim, dropout=0.0, **kwargs):
    """
    [融合层工厂]
    method: 'cross_attn', 'cross_gated', 'sum', 'concat'

    :param fusion_method:
    :param feature_dim:
    :param dropout:
    """
    if fusion_method is None:
        return None
        
    ftype = fusion_method.lower()
    
    if ftype == 'cross_attn':
        return CrossAttnFusion(feature_dim, num_heads=4, dropout=dropout)
    
    elif ftype == 'cross_gated':
        return CrossGatedFusion(feature_dim, dropout=dropout)
    
    elif ftype == 'cross_gated_light':
        return CrossGatedLightFusion(feature_dim, dropout=dropout)
    
    elif ftype == 'sum':
        return SumFusion()
        
    elif ftype == 'concat':
        return ConcatFusion(feature_dim, dropout=dropout)
    
    elif ftype == 'concat_deep':
        return ConcatDeepFusion(feature_dim, dropout=dropout)
    
    elif ftype == 'film':
        return FiLMFusion(feature_dim)
        
    else:
        raise ValueError(f"Unknown fusion type: {fusion_method}")








    


    


    

    

