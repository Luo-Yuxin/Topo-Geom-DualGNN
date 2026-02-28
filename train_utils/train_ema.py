import torch
from copy import deepcopy



class ModelEMA:
    """
    Model Exponential Moving Average (EMA) with Warmup and State Caching.
    
    Optimization:
    - [状态缓存]: 在 __init__ 时缓存 (EMA参数, 原模型参数) 的引用对。
      这避免了在每次 update 时调用 .state_dict() 创建巨大字典和进行字符串匹配的开销。
    - [In-place操作]: 使用 .mul_() 和 .add_() 进行原地更新，减少显存临时分配。
    """
    def __init__(self, model, decay=0.9999, device=None):
        # 创建模型的深拷贝作为 EMA 模型 (Shadow Model)
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        self.num_updates = 0

        # [优化关键] 分离 Parameter 和 Buffer 的引用
        # 只要传入 update 的 model 还是同一个对象(训练中通常如此), 这就非常高效
        # 为了解决EMA与BN的协作问题, 我们对可训练参数(Weights/Biases)进行 EMA 衰减
        # 但对 Buffer(BN 统计量) 直接从源模型 复制(Copy, )不进行衰减

        # 1. 收集 Parameters (需要 EMA 平滑的权重、偏置)
        # zip 依赖于 PyTorch 遍历顺序的确定性，这在同一个模型结构中是保证的
        self.param_refs = []
        for ema_p, src_p in zip(self.module.parameters(), model.parameters()):
            if src_p.requires_grad: # 只追踪可训练参数
                self.param_refs.append((ema_p, src_p))
            
        # 2. 收集 Buffers (不需要梯度的统计量，如 BN running_mean/var)
        self.buffer_refs = []
        for ema_b, src_b in zip(self.module.buffers(), model.buffers()):
            self.buffer_refs.append((ema_b, src_b))
            
        if self.device is not None:
            self.module.to(device=device)

    def update(self, model=None):
        """
        Update EMA parameters.
        model: 这里的 model 参数主要为了保持接口兼容性。
               实际更新时，我们使用初始化时缓存的 self.param_refs 直接访问原模型参数。
        """
        self.num_updates += 1
        
        # 动态 Decay (Warmup)
        # 训练初期 decay 较小(快速跟进), 后期较大(稳定平滑)
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        with torch.no_grad():
            # 1. 更新 Parameters: 使用 EMA 衰减
            for ema_v, model_v in self.param_refs:
                # 如果设备不同，需要搬运 (通常都在 GPU 上)
                if self.device is not None and model_v.device != self.device:
                    model_v = model_v.to(device=self.device)
                # 使用原地操作 (In-place) 节省显存和带宽
                # 公式: ema_v = decay * ema_v + (1-decay) * model_v
                # 变换为: ema_v *= decay; ema_v += (1-decay) * model_v
                ema_v.mul_(decay).add_(model_v, alpha=1 - decay)

            # 2. 更新 Buffers: 使用 Direct Copy (避免双重平均)
            for ema_v, model_v in self.buffer_refs:
                if self.device is not None and model_v.device != self.device:
                    model_v = model_v.to(device=self.device)
                # 直接同步源模型的统计量
                ema_v.copy_(model_v)
               
    def set(self, model=None):
        """强制重置 EMA 模型"""
        self.num_updates = 0
        with torch.no_grad():
            # 重置所有参数和Buffer
            for ema_v, model_v in self.param_refs + self.buffer_refs:
                if self.device is not None and model_v.device != self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(model_v)