import torch
import torch.nn as nn

class MultiTaskLossWrapper(nn.Module):
    """
    [自动加权损失函数 - 通用版]
    基于不确定性 (Uncertainty Weighting)
    论文: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (CVPR 2018)
    
    原理: L_total = sum( 0.5 * exp(-log_var_i) * L_i + 0.5 * log_var_i )
    """
    def __init__(self, task_num, init_val=0.0, log_var_limit=(-2, 5)):
        """
        Args:
            task_num (int): 任务数量 (例如 2 或 3)
            init_val (float): log_var 的初始值。默认为 0.0 (对应权重 0.5)。
        """
        super(MultiTaskLossWrapper, self).__init__()
        self.log_var_limit = log_var_limit
        self.task_num = task_num
        # log_vars 是一个形状为 [task_num] 的可学习向量
        self.log_vars = nn.Parameter(torch.zeros(task_num) + init_val)

    def forward(self, *losses):
        """
        输入: 任意数量的 loss (scalar tensors)
        用法: criterion(loss_a, loss_b, ...)
        注意: 输入 loss 数量必须等于初始化时的 task_num
        """
        if len(losses) != self.task_num:
            raise ValueError(f"Expected {self.task_num} losses, but got {len(losses)}")
        
        total_loss = 0.0
        for i, loss in enumerate(losses):
            # log_var_limit为log_var所允许的范围
            # 防止 log_var 走向负无穷 (导致 loss 为负数且不稳定)
            # 防止 log_var 走向正无穷 (导致忽略该任务)
            if len(self.log_var_limit) == 2:
                self.log_var_min = min(self.log_var_limit)
                self.log_var_max = max(self.log_var_limit)
                # precision = 1 / (2 * sigma^2) = 0.5 * exp(-log_var)
                precision = 0.5 * torch.exp(-(self.log_vars[i].clamp(min=self.log_var_min, max=self.log_var_max)))
            else:
                precision = 0.5 * torch.exp(-self.log_vars[i])
            
            # L_total += precision * L_i + log(sigma)
            # log(sigma) = 0.5 * log(sigma^2) = 0.5 * log_var
            total_loss += precision * loss + 0.5 * self.log_vars[i]
        
        return total_loss

    def get_weights(self):
        """返回当前的权重值 (列表)，用于日志记录"""
        weights = []
        with torch.no_grad():
            if len(self.log_var_limit) == 2:
                clamped_log_vars = torch.clamp(self.log_vars, min=self.log_var_min, max=self.log_var_max)
                for val in clamped_log_vars:
                    # 记录时同样由于 clamp 逻辑, 最好显示 clamp 后的值
                    weights.append((0.5 * torch.exp(-val)).item())
            else:
                weights.append((0.5 * torch.exp(-val)).item())
        return weights