import os
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib
try:
    matplotlib.use('Agg')
except:
    pass
import matplotlib.pyplot as plt
import itertools

# --- 引入 TorchMetrics 核心组件 ---
from torchmetrics import MetricCollection, ConfusionMatrix
from torchmetrics.classification import (
    MulticlassAccuracy, 
    MulticlassJaccardIndex, 
    MulticlassF1Score,
    MulticlassRecall,
    BinaryAccuracy, 
    BinaryF1Score, 
    BinaryJaccardIndex
)
from torch_geometric.utils import to_dense_adj, degree

# 确保导入的是修复后的 Dataset
from dataset.step_dataset import StepDataset
from config_train import CONFIG

# [新增] worker_init_fn 确保多进程数据加载的随机性也是确定的
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed):
    """
    固定所有随机种子，确保实验可复现。
    """
    # 1. Python random
    random.seed(seed)
    # 2. Numpy
    np.random.seed(seed)
    # 3. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果使用多GPU
    # 4. 保证 CuDNN 确定性 (会牺牲一点点性能，但保证可复现)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已固定为: {seed}")

def count_parameters(model):
    """统计并打印模型参数信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*40}")
    print(f"模型参数统计:")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    print(f"{'='*40}\n")
    return trainable_params

def compute_pna_degrees(dataset):
    """
    [PNA 专用] 计算数据集的度分布直方图。
    PNA 需要这个统计信息来初始化 Scalers (Amplification/Attenuation)。
    """
    print("正在计算数据集度分布 (用于 PNA)...")
    
    # 统计最大度
    max_degree_topo = 0
    max_degree_geom = 0
    
    # 第一次遍历: 找最大度
    # 仅使用部分样本估算即可，或者遍历全部
    for i in range(min(len(dataset), 500)): 
        data = dataset[i]
        d_topo = degree(data.edge_index_topo[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree_topo = max(max_degree_topo, d_topo.max().item())
        
        d_geom = degree(data.edge_index_geom[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree_geom = max(max_degree_geom, d_geom.max().item())

    # 初始化直方图
    deg_hist_topo = torch.zeros(max_degree_topo + 1, dtype=torch.long)
    deg_hist_geom = torch.zeros(max_degree_geom + 1, dtype=torch.long)

    # 第二次遍历: 填充直方图
    for i in range(min(len(dataset), 500)):
        data = dataset[i]
        d_topo = degree(data.edge_index_topo[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg_hist_topo += torch.bincount(d_topo, minlength=deg_hist_topo.numel())
        
        d_geom = degree(data.edge_index_geom[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg_hist_geom += torch.bincount(d_geom, minlength=deg_hist_geom.numel())

    return deg_hist_topo, deg_hist_geom

def check_has_pna(defs):
    """
    递归检查定义字典中是否包含 'pna'
    """
    if not defs: return False
    for params in defs.values():
        # 兼容两种格式: {'0': 'pna'} 或 {'0': {'gnn': 'pna'}}
        if isinstance(params, str):
            if params == 'pna': return True
        elif isinstance(params, dict):
            if params.get('gnn') == 'pna': return True
    return False

class MetricTracker(nn.Module):
    """
    统一管理所有任务的指标，复刻 inst_trainer.py 的 Global Accumulation 逻辑。
    包含:
    1. 语义分割: Acc, mIoU, Macro-F1
    2. 实例分割: Binary Acc, F1, IoU
    3. 底面预测: Binary Acc, F1, IoU
    """
    def __init__(self, num_classes, device, tasks=('sem', 'inst', 'bot')):
        super().__init__()
        self.device = device
        # 如果传入的是单个字符串 (例如 'sem')，set() 会将其拆解为 {'s','e','m'}
        # 因此需要先将其转为列表或保持为元组
        if isinstance(tasks, str):
            self.tasks = {tasks}
        else:
            self.tasks = set(tasks)

        print(f"MetricTracker initialized for tasks: {self.tasks}")

        # 定义指标构建工厂 (延迟初始化，只有需要的才会被创建)
        metrics_factory = {
            'sem': lambda: MetricCollection({
                # 1. 总体准确率 (Global Accuracy: Correct / Total)
                'sem_acc': MulticlassAccuracy(num_classes=num_classes),
                # 2. Mean IoU
                'sem_miou': MulticlassJaccardIndex(num_classes=num_classes, average='macro'),
                # 3. Macro F1
                'sem_f1': MulticlassF1Score(num_classes=num_classes, average='macro'),
                # 4. Macro Recall
                # 计算公式: (1/K) * sum(TP_i / (TP_i + FN_i))
                'sem_mrecall': MulticlassRecall(num_classes=num_classes, average='macro'),
                
                # 5. Per-class Recall
                # 计算公式: TP_i / (TP_i + FN_i) for each class i
                # average=None 会返回一个向量 [C]，compute时会自动展开
                # 'sem_class_recall': MulticlassRecall(num_classes=num_classes, average=None) 
            }),
            'inst': lambda: MetricCollection({
                'inst_acc': BinaryAccuracy(),
                'inst_f1': BinaryF1Score(),
                'inst_iou': BinaryJaccardIndex()
            }),
            'bot': lambda: MetricCollection({
                'bot_acc': BinaryAccuracy(),
                'bot_f1': BinaryF1Score(),
                'bot_iou': BinaryJaccardIndex()
            })
        }

        # 动态创建 ModuleDict
        self.train_metrics = nn.ModuleDict()
        self.val_metrics = nn.ModuleDict()
        self.test_metrics = nn.ModuleDict()
        
        for task_name in self.tasks:
            if task_name in metrics_factory:
                self.train_metrics[task_name] = metrics_factory[task_name]().clone(prefix='Train_').to(device)
                self.val_metrics[task_name] = metrics_factory[task_name]().clone(prefix='Val_').to(device)
                self.test_metrics[task_name] = metrics_factory[task_name]().clone(prefix='Test_').to(device)
        
        # 混淆矩阵 (仅当包含 sem 任务时才初始化)
        if 'sem' in self.tasks:
            self.train_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
            self.val_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
            self.test_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
        else:
            self.train_confmat = None
            self.val_confmat = None
            self.test_confmat = None

    def update(self, phase, preds, targets):
        """
        通用更新接口
        phase: 'train' or 'val' or 'test'
        preds: 字典 {'sem':..., 'inst':..., 'bot':...}
        targets: 字典 {'sem':..., 'inst':..., 'bot':...}
        """

        if phase == 'train':
            metrics = self.train_metrics
            confmat = self.train_confmat
        elif phase == 'val':
            metrics = self.val_metrics
            confmat = self.val_confmat
        else:
            metrics = self.test_metrics
            confmat = self.test_confmat
        
        # 1. Semantic
        if 'sem' in self.tasks and 'sem' in preds:
            # preds['sem'] 是 logits [B, C], 需要 argmax，或者 torchmetrics 支持 logits
            # MulticlassAccuracy 支持 logits，但也支持 index。这里统一传 index 更安全
            if preds['sem'].dim() > 1 and preds['sem'].shape[1] > 1:
                p_sem = preds['sem'].argmax(dim=1)
            else:
                p_sem = preds['sem']
            # MetricCollection.update 会自动分发给内部的 Acc, mIoU, F1
            metrics['sem'].update(p_sem, targets['sem'])
            
            # 独立更新混淆矩阵
            confmat.update(p_sem, targets['sem'])

        # 2. Instance
        if 'inst' in self.tasks and 'inst' in preds:
            # Instance 通常输出 logits，需要 sigmoid
            p_inst_logits = preds['inst']
            p_inst_prob = torch.sigmoid(p_inst_logits)
            
            # Flatten 后的 1D 概率向量完全适用于二分类指标计算
            metrics['inst'].update(p_inst_prob, targets['inst'])

        # 3. Bottom
        if 'bot' in self.tasks and 'bot' in preds:
            p_bot_logits = preds['bot']
            p_bot_prob = torch.sigmoid(p_bot_logits)
            metrics['bot'].update(p_bot_prob, targets['bot'])

    def compute(self, phase):
        """计算当前 Epoch 的所有全局指标"""
        if phase == 'train':
            metrics = self.train_metrics
            confmat = self.train_confmat
        elif phase == 'val':
            metrics = self.val_metrics
            confmat = self.val_confmat
        else:
            metrics = self.test_metrics
            confmat = self.test_confmat
        
        results = {}
        # 遍历 sem, inst, bot
        for key in metrics:
            # collection.compute() 会返回一个字典 {prefix_acc: val, prefix_miou: val, ...}
            res = metrics[key].compute()
            results.update(res) # 自动带上前缀 (Train_sem_acc 等)
            # # [核心改进] 自动解包向量指标
            # for k, v in res.items():
            #     if v.numel() > 1: # 如果是向量 (如 per-class acc)
            #         # 将其拆解为 k_0, k_1, ...
            #         # 这样 TensorBoard 就能把它们当作标量记录下来
            #         for class_idx, val in enumerate(v):
            #             results[f"{k}_{class_idx}"] = val
            #     else:
            #         results[k] = v
        
        # 计算混淆矩阵并返回 (用于绘图)
        cm = confmat.compute()
        if confmat is not None:
            cm = confmat.compute()
            
        return results, cm

    def reset(self, phase):
        """重置状态"""
        if phase == 'train':
            metrics = self.train_metrics
            confmat = self.train_confmat
        elif phase == 'val':
            metrics = self.val_metrics
            confmat = self.val_confmat
        else:
            metrics = self.test_metrics
            confmat = self.test_confmat
        for key in metrics:
            metrics[key].reset()
            
        confmat.reset()

def plot_confusion_matrix(cm, class_names=None):
    """绘制混淆矩阵"""
    if cm is None: return None # 安全检查

    if isinstance(cm, torch.Tensor):
        cm = cm.cpu().numpy()
    
    # 归一化 (Row-Normalized)
    row_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = cm.astype('float') / (row_sum + 1e-6)
    
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Normalized Confusion Matrix (Recall)',
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' 
    thresh = cm_norm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm_norm[i, j] > 0.01: 
            ax.text(j, i, format(cm_norm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=8)
            
    fig.tight_layout()
    return fig