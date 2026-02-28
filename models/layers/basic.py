import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# 基础卷积块
# 用于处理 UV 网格 (2D) 和 边采样点序列 (1D)
###############################################################################

def conv1d_block(in_channels, out_channels, kernel_size=3, padding=1, bias=False,
                 norm_layer=nn.BatchNorm1d, act_layer=nn.LeakyReLU, **act_args):
    """
    1D 卷积块: -> Conv1d -> BN -> Act_function ->

    :param in_channels (int): 输入通道数
    :param out_channels (int): 输出通道数
    :param kernel_size (int): 卷积核尺寸 (默认 3)
    :param padding (int): 填充值 (默认 (kernel_size)//2 )
    :param bias (bool): 激活函数偏置 (默认 False)
    :param norm_layer (nn.modual): 激活函数类 (默认 nn.BatchNorm1d)
    :param act_layer (nn.modual): 激活函数类类型 (例如 nn.LeakyReLU, nn.GELU, nn.SiLU). 如果为 None, 则不使用激活函数
    :param **act_args (any): 传递给激活函数的关键字参数
    """
    # 如果有 Norm 层 (BN/IN/GN)，卷积层通常不需要 bias (因为 Norm 层有 shift)
    # 如果没有 Norm 层，卷积层通常需要 bias 来拟合数据
    if bias is None:
        bias = (norm_layer is None)

    # 构建基础层
    layers = [
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
    ]

    # 动态添加归一化层
    if norm_layer is not None:
        # 注意: 某些 Norm (如 LayerNorm) 可能参数定义不同，
        # 但 BN, InstanceNorm, GroupNorm(需特殊处理) 接口大致兼容。
        # 这里假设是标准的 BN 或 IN
        layers.append(norm_layer(out_channels))

    # 动态实例化激活函数
    if act_layer is not None:
        layers.append(act_layer(**act_args))

    return nn.Sequential(*layers)

def conv2d_block(in_channels, out_channels, kernel_size=3, padding=1, bias=False, 
                 norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU, **act_args):
    """
    2D 卷积块: -> Conv2d -> BN -> Act_function ->

    :param in_channels (int): 输入通道数
    :param out_channels (int): 输出通道数
    :param kernel_size (int): 卷积核尺寸 (默认 3)
    :param padding (int): 填充值 (默认 (kernel_size)//2 )
    :param bias (bool): 激活函数偏置 (默认 False)
    :param norm_layer (nn.modual): 激活函数类 (默认 nn.BatchNorm2d)
    :param act_layer (nn.modual): 激活函数类类型 (例如 nn.LeakyReLU, nn.GELU, nn.SiLU). 如果为 None, 则不使用激活函数
    :param **act_args (any): 传递给激活函数的关键字参数
    """
    # 如果有 Norm 层 (BN/IN/GN)，卷积层通常不需要 bias (因为 Norm 层有 shift)
    # 如果没有 Norm 层，卷积层通常需要 bias 来拟合数据
    if bias is None:
        bias = (norm_layer is None)

    # 构建基础层
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
    ]

    # 动态添加归一化层
    if norm_layer is not None:
        # 注意: 某些 Norm (如 LayerNorm) 可能参数定义不同，
        # 但 BN, InstanceNorm, GroupNorm(需特殊处理) 接口大致兼容。
        # 这里假设是标准的 BN 或 IN
        layers.append(norm_layer(out_channels))

    # 动态实例化激活函数
    if act_layer is not None:
        layers.append(act_layer(**act_args))

    return nn.Sequential(*layers)


###############################################################################
# 全连接层 (FC) - 通用模块
###############################################################################

def fc_block(input_dim, output_dim, bias=False, dropout=0.0, 
             norm_layer=nn.BatchNorm1d, norm_layer_2=None, act_layer=nn.LeakyReLU, **act_args):
    """
    全连接块: -> Linear -> BN -> Act_function -> Dropout -> LN -> 

    :param input_dim (int): 输入维度
    :param output_dim (int): 输出维度
    :param norm_layer (nn.modual): 前置归一化类 (默认 nn.BatchNorm1d)
    :param act_layer (nn.modual): 激活函数类 (默认 nn.LeakyReLU)
    :param bias (bool): 激活函数偏置 (默认 False)
    :param **act_args (any): 激活函数参数 (例如 negative_slope=0.2, inplace=True)
    :param dropout (float): dropout概率 (默认 0.0)
    :param norm_layer_2 (nn.modual): 后置归一化类 (默认None, 一般后置归一化为 nn.LayerNorm)
    :return fc_block (nn.Sequential)
    """
    layers = []
    # 一般来说, 若进行BN是不需要bias的
    # 而若是单纯的linear, 则一般是需要进行bias的
    if bias is None:
        bias = (norm_layer is None)

    # 构造线性层
    layers.append(nn.Linear(input_dim, output_dim, bias=bias))
    # 归一化
    if norm_layer is not None:
        layers.append(norm_layer(output_dim))
    # 激活层
    if act_layer is not None:
        layers.append(act_layer(**act_args))
    # dropout
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    if norm_layer_2 is not None:
        layers.append(norm_layer_2(output_dim))
    return nn.Sequential(*layers)


###############################################################################
# 多层感知机 (MLP) - 通用模块
###############################################################################

class MLP_old(nn.Module):
    """
    --已过时--
    高度可配置的 MLP 模块
    支持指定层数、归一化方式、激活函数
    """
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 num_layers=2, 
                 norm_layer=nn.BatchNorm1d,
                 act_layer=nn.ReLU,
                 dropout=0.0,
                 bias=False):
        super(MLP, self).__init__()
        
        self.num_layers = num_layers
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.act = act_layer()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if num_layers < 1:
            raise ValueError("MLP num_layers must be >= 1")

        # 构建层
        if num_layers == 1:
            self.linears.append(nn.Linear(input_dim, output_dim, bias=bias))
        else:
            # 输入层
            self.linears.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            self.norms.append(norm_layer(hidden_dim))
            
            # 隐藏层
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
                self.norms.append(norm_layer(hidden_dim))
            
            # 输出层
            self.linears.append(nn.Linear(hidden_dim, output_dim, bias=bias))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.num_layers == 1:
            return self.linears[0](x)
        
        h = x
        for i in range(self.num_layers - 1):
            h = self.linears[i](h)
            h = self.norms[i](h)
            h = self.act(h)
            h = self.dropout(h)
            
        return self.linears[-1](h)

class MLP(nn.Module):
    """
    高度可配置的 MLP 模块

    :param input_dim (int): 输入层维度
    :param hidden_dim (int): 隐藏层维度
    :param output_dim (int): 输出层维度
    :param num_layers (int): 层数 (默认 2, 为隐藏层+输出层数量)
    :param norm_layer (nn.modual): 归一化方法 (默认 nn.BatchNorm1d )
    :param bias (bool): 激活函数偏置 (默认 False)
    :param dropout (float): dropout概率 (默认 0.0)
    :param norm_layer_2 (nn.modual): dropout 后的归一化方法 (一般为LN, 默认为None不开启)
    :param act_layer (nn.modual): 激活函数类类型 (例如 nn.LeakyReLU, nn.GELU, nn.SiLU). 如果为 None, 则不使用激活函数
    :param **act_args (any): 传递给激活函数的关键字参数
    """
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 num_layers=2, 
                 norm_layer=nn.BatchNorm1d,
                 bias=False,
                 dropout=0.0,
                 norm_layer_2 = None,
                 act_layer=nn.ReLU,
                 **act_args):
        super(MLP, self).__init__()
        
        # 一般来说, 若进行BN是不需要bias的
        # 而若没有进行BN, 则一般是需要进行bias的
        if bias is None:
            bias = (norm_layer is None)

        if num_layers < 1:
            raise ValueError("MLP num_layers must be >= 1")

        # --- 构建隐藏层部分 ---
        # 如果只有1层，hidden_layers 为空
        hidden_layers = []
        
        if num_layers > 1:
            # 输入层 (Input -> Hidden)
            hidden_layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                hidden_layers.append(norm_layer(hidden_dim))
            # 动态实例化激活函数
            if act_layer is not None:
                hidden_layers.append(act_layer(**act_args)) 
            if dropout > 0:
                hidden_layers.append(nn.Dropout(dropout))
            if norm_layer_2 is not None:
                hidden_layers.append(norm_layer_2(hidden_dim))
            
            # 中间层循环 (Hidden -> Hidden)
            for _ in range(num_layers - 2):
                hidden_layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
                if norm_layer is not None:
                    hidden_layers.append(norm_layer(hidden_dim))
                if act_layer is not None:
                    hidden_layers.append(act_layer(**act_args)) # 每次都是新实例
                if dropout > 0:
                    hidden_layers.append(nn.Dropout(dropout))
                if norm_layer_2 is not None:
                    hidden_layers.append(norm_layer_2(hidden_dim))
        
        # 将所有隐藏层打包成一个 Sequential
        self.hidden_layers = nn.Sequential(*hidden_layers)

        # --- 构建输出层部分 ---
        # 如果 num_layers > 1，输入是 hidden_dim；否则输入是 input_dim
        # 一般来说, 最后一层不做BN, 那一般是需要开启bias的
        last_in_dim = hidden_dim if num_layers > 1 else input_dim
        self.output_layer = nn.Linear(last_in_dim, output_dim, bias=True)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 这里的 nonlinearity 可以写成 'relu' 或 'leaky_relu'，
            # 虽然不一定完全匹配传入的 act_layer，但对大多数情况是通用的稳健选择
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 逻辑极其简单：先过隐藏层(如果有), 再过输出层
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
    

        





