import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMFusionLayer(nn.Module):
    def __init__(self, feature_dim, residual=True):
        super().__init__()
        # 构建从特征生成调制参数方法
        self.gamma_net = nn.Linear(feature_dim, feature_dim)
        self.beta_net = nn.Linear(feature_dim, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim) # FiLM 通常配合 BN 使用

        # 残差
        self.residual = residual

        # 权重初始化 (会有助于训练)
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化策略: Zero Initialization
        将 Gamma 和 Beta 的初始值设为 0。
        
        效果:
        Modulation = 0 * Content + 0 = 0
        Output = Content + 0 = Content (Identity Mapping)
        
        这保证了在训练初期，融合层不会破坏预训练好的或原始的单流特征。
        """
        # 初始化 Gamma 生成器: 输出为 0
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.zeros_(self.gamma_net.bias)
        
        # 初始化 Beta 生成器: 输出为 0
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)
    
    def forward(self, h_content, h_style): 

        # h_content = 被指导数据
        # h_style   = 指导数据
        
        gamma = self.gamma_net(h_style) # + 1.0
        beta = self.beta_net(h_style)

        # 首先对输入采用BN
        # 将 content 拉回标准正态分布, 使其对缩放和平移敏感
        h_content_norm = self.bn(h_content)
        
        # 仿射调制
        h_modulated = gamma * h_content_norm + beta
        
        # 残差 + 归一化
        # 残差连接
        if self.residual:
            # 加上残差, 保留原始空间信息
            h_out = h_content + h_modulated
        else:
            h_out = h_modulated
        # 对于经典的FiLM方法一般最后是不做BN的
        # h_out = self.bn(h_out)
        return h_out