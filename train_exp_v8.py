import os
# [Debug] 开启同步模式，让报错指向真正的行数
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
# [新增] 引入 SWA 工具
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.cuda.amp import autocast, GradScaler # [新增] 混合精度训练
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from train_utils.exp_recorder import ExperimentRecorder
from train_utils.train_utils_func import (
    seed_worker, 
    set_seed, 
    count_parameters, 
    compute_pna_degrees, 
    check_has_pna,
    MetricTracker,          # [核心] 引入新的指标管理器
    plot_confusion_matrix
)
from train_utils.train_ema import ModelEMA
from train_utils.multi_task_loss import MultiTaskLossWrapper

from dataset.step_dataset import StepDataset
from models.dual_stream_net import DualStreamNet
from config_train import CONFIG

def log_metrics_to_tensorboard(writer, phase, results, epoch):
    """
    辅助函数：将扁平的指标字典转换为 TensorBoard 的分层结构
    keys in results: 'Train_sem_acc', 'Val_inst_f1', etc.
    Target: 'Semantic/Train_Acc', 'Instance/Val_F1'
    """
    for k, v in results.items():
        parts = k.split('_') # e.g., ['Train', 'sem', 'acc']
        if len(parts) < 3:
            writer.add_scalar(f"Others/{k}", v, epoch)
            continue
            
        phase_prefix = parts[0] # Train / Val
        task_type = parts[1]    # sem / inst / bot
        metric_name = parts[2]  # acc / miou / f1 / iou
        
        
        # 映射任务名称到分组
        if task_type == 'sem': group = 'Semantic'
        elif task_type == 'inst': group = 'Instance'
        elif task_type == 'bot': group = 'Bottom'
        else: group = 'Others'
        
        # 映射指标名称 (大写美化)
        metric_display = metric_name.replace('miou', 'mIoU').replace('iou', 'IoU').upper() if len(metric_name) < 3 else metric_name.capitalize()
        if metric_name == 'miou': metric_display = 'mIoU'
        
        tag = f"{group}/{phase_prefix}_{metric_display}"
        writer.add_scalar(tag, v, epoch)

# --- 主训练函数 ---

def train():
    set_seed(CONFIG['seed'])
    print(f"Using device: {CONFIG['device']}")

    # [开启 TF32 加速
    # Ampere 架构 (RTX 30系, A100) 及以上支持 TF32，能显著加速 FP32 矩阵乘法
    if torch.cuda.is_available() and CONFIG['device'] != 'cpu':
        try:
            # 获取计算能力 (major, minor)
            gpu_cap = torch.cuda.get_device_capability(CONFIG['device'])
            # Compute Capability >= 8.0 (Ampere) 支持 TF32
            if gpu_cap[0] >= 8:
                torch.set_float32_matmul_precision("high")
                print(f"  [加速] 检测到 Ampere+ 架构 GPU (Compute {gpu_cap[0]}.{gpu_cap[1]}), TF32 加速已开启")
            else:
                print(f"  [提示] 当前 GPU (Compute {gpu_cap[0]}.{gpu_cap[1]}) 不支持 TF32, 将使用标准 FP32 精度")
        except AttributeError:
            # 老版本 PyTorch 可能没有这个属性，忽略
            pass


    # --- Config Auto-Fix ---
    num_classes = CONFIG['model'].get('num_classes', 25)
    
    # --- Init TensorBoard ---
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    comment = f"_dim{CONFIG['model']['hidden_dim']}_lr{CONFIG['lr']}"
    log_dir = os.path.join('runs', current_time + comment)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard log directory: {log_dir}")

    # Dataset Init ...
    dataset = StepDataset(root=CONFIG['data_root'], 
                          raw_dir_name=CONFIG['raw_dir_name'],
                          label_dir_name=CONFIG['label_dir_name'],
                          processed_dir_name=CONFIG['processed_dir_name'],
                          uv_sample_num=CONFIG['uv_sample'],
                          force_process=False)
    
    if len(dataset) == 0: return
    
    # Auto-config dims
    sample = dataset[0]
    CONFIG['model']['topo_node_in'] = sample.x_topo.shape[1]
    CONFIG['model']['topo_edge_in'] = sample.edge_attr_topo.shape[-1]
    CONFIG['model']['geom_node_in'] = sample.x_geom.shape[1]
    CONFIG['model']['geom_edge_in'] = sample.edge_attr_geom.shape[1]
    
    # PNA Config
    use_pna_topo = check_has_pna(CONFIG['model'].get('topo_defs', {}))
    use_pna_geom = check_has_pna(CONFIG['model'].get('geom_defs', {}))
    if use_pna_topo or use_pna_geom:
        deg_topo, deg_geom = compute_pna_degrees(dataset)
        CONFIG['model']['topo_deg'] = deg_topo
        CONFIG['model']['geom_deg'] = deg_geom
    else:
        CONFIG['model']['topo_deg'] = None
        CONFIG['model']['geom_deg'] = None
    
    # Split
    total_len = len(dataset)
    train_len = int(total_len * CONFIG['split_ratio'][0])
    val_len = int(total_len * CONFIG['split_ratio'][1])
    test_len = total_len - train_len - val_len
    
    generator = torch.Generator().manual_seed(CONFIG['seed'])
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len], generator=generator
    )

    loader_args = {
        'batch_size': CONFIG['batch_size'],
        'num_workers': CONFIG['num_workers'],
        'pin_memory': True,
        'worker_init_fn': seed_worker,
        'generator': generator,
        'persistent_workers': (CONFIG['num_workers'] > 0)
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)
    
    # --- Model & Opt ---
    model = DualStreamNet(CONFIG['model']).to(CONFIG['device'])
    # [新增] 多任务损失
    if CONFIG.get('use_mtl', False):
        # 判断哪些模块参与了计算
        active_task_count = 0
        actived_task = []
        if CONFIG['lambda_sem'] > 0: 
            active_task_count += 1
            actived_task.append('sem')
        if CONFIG['lambda_inst'] > 0: 
            active_task_count += 1
            actived_task.append('inst')
        if CONFIG['lambda_bottom'] > 0: 
            active_task_count += 1 
            actived_task.append('bottom')
        # 初始化 Wrapper
        mtl_loss_wrapper = MultiTaskLossWrapper(task_num=active_task_count, log_var_limit=CONFIG.get('log_var_limit', (-2, 5))).to(CONFIG['device'])
        print(f"MultiTaskLossWrapper is actived on: {actived_task}")
        # 优化器需要同时优化模型参数与权重参数
        # L2正则化的意义是抑制模型参数变得过大, 以将参数向0拉
        # 由于mtl中的log_vars 代表的是对数方差, 物理意义上它应该自由变化以平衡权重, 不应该被强行拉向 0
        # 因此我们不能为mtl设置 weight_decay
        param_groups = [
            {'params': model.parameters(), 'weight_decay': CONFIG['weight_decay']},
            {'params': mtl_loss_wrapper.parameters(), 'weight_decay': 0.0}
        ]
        optimizer = optim.AdamW(param_groups, lr=CONFIG['lr'])
    else:
        # 不使用mtl
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])


    # =========================================================================
    # [核心新增] SWA 的严谨初始化 (提前至 Scheduler 之前以联动学习率)
    # =========================================================================
    use_swa = CONFIG.get('use_swa', False)
    swa_model = None
    swa_scheduler = None
    swa_start = int(CONFIG['epochs'] * 0.75) # 默认最后 25% 的 epoch 开启 SWA
    
    if use_swa:
        print(f"启用 SWA (Stochastic Weight Averaging) | Start Epoch: {swa_start}")
        swa_model = AveragedModel(model)
        # 【SWA核心 1】必须使用 SWALR 作为平均阶段的调度器。
        swa_lr = CONFIG.get('swa_lr', CONFIG['lr'] * 0.1)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        print(f"  -> SWA 专属学习率设为: {swa_lr}")
    else:
        swa_lr = 0.0



    # 添加 Learning Rate Scheduler (CosineAnnealingLR)
    scheduler = None
    scheduler = None
    if CONFIG.get('use_lrs', False):
        if CONFIG.get('use_warmup', False):
            # [策略一实现] 动态压缩 OneCycleLR 周期
            cycle_epochs = swa_start if use_swa else CONFIG['epochs']
            div_factor = CONFIG.get('warmup_start_div_factor', 25.0)
            
            if use_swa:
                # 数学推导: min_lr = max_lr / (div_factor * final_div_factor)
                # 令 min_lr = swa_lr，则 final_div_factor = max_lr / (div_factor * swa_lr)
                final_div_factor = float(CONFIG['lr']) / (div_factor * swa_lr)
                print(f"  -> [SWA 联动] OneCycleLR 周期压缩至 {cycle_epochs} Epochs")
                print(f"  -> [SWA 联动] 动态计算 final_div_factor 为 {final_div_factor:.4f}，确保平滑衔接")
            else:
                final_div_factor = CONFIG.get('warmup_final_div_factor', 10000.0)

            scheduler = OneCycleLR(
                                    optimizer, 
                                    max_lr=CONFIG['lr'], 
                                    epochs=cycle_epochs, 
                                    steps_per_epoch=len(train_loader), 
                                    pct_start=CONFIG.get('warmup_rate', 0.3), 
                                    div_factor=div_factor, 
                                    final_div_factor=final_div_factor
                                    )
        else:
            # CosineAnnealingLR (Per Epoch update)
            cycle_epochs = swa_start if use_swa else CONFIG['epochs']
            eta_min = swa_lr if use_swa else 0
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cycle_epochs, eta_min=eta_min)
            if use_swa:
                print(f"  -> [SWA 联动] CosineAnnealingLR 周期压缩至 {cycle_epochs} Epochs，底线 LR 设为 {eta_min}")


    # 添加混合精度训练 Scaler
    scaler = GradScaler()
    model_params_info = count_parameters(model)

    # EMA 初始化
    use_ema = CONFIG.get('use_ema', False)
    ema_decay = CONFIG.get('ema_decay', 0.999)
    ema = None
    if use_ema:
        print(f"启用 EMA 训练 (Decay: {ema_decay}, with Dynamic Warmup & State Caching)")
        ema = ModelEMA(model, decay=ema_decay, device=CONFIG['device'])
    
    # Losses
    criterion_sem = nn.CrossEntropyLoss()
    criterion_inst = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([18.8]).to(CONFIG['device'])) # 19.0
    criterion_bottom = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.32]).to(CONFIG['device']))# 9.3
    
    # 初始化指标跟踪器
    # 这将自动管理 Train 和 Val 的所有指标状态 (Acc, F1, IoU, mIoU)
    tracker = MetricTracker(num_classes=num_classes, device=CONFIG['device'])

    best_composite_score = -float('inf') # 使用综合评分
    best_val_record = 0.0
    best_model_path = os.path.join('checkpoints', 'best_model.pth')
    last_model_path = os.path.join('checkpoints', 'last_model.pth') 
    swa_model_path = os.path.join('checkpoints', 'swa_model.pth') # [新增] 保存 SWA 专用路径
    os.makedirs('checkpoints', exist_ok=True)
    
    last_epoch_stats = {}
    last_val_stats = {}

    for epoch in range(CONFIG['epochs']):
        model.train()
        # 重置训练指标
        tracker.reset('train')
        train_loss = 0.0

        # [新增] 用于记录当前 Epoch 所有 Batch 的梯度范数
        epoch_grad_norms = []
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for batch_idx, batch in pbar:
            batch = batch.to(CONFIG['device'])
            
            # Data Check
            if batch.y.max() >= num_classes:
                raise ValueError(f"Data Error: Found label {batch.y.max().item()} >= {num_classes}")

            # Forward
            optimizer.zero_grad()

            # 使用 Autocast 进行前向传播
            with autocast():
                sem_logits, inst_matrix, inst_mask, bottom_logits = model(batch)
            
                # --- Prepare Ground Truth for Instance ---
                max_nodes = inst_matrix.size(1)
                gt_matrix = to_dense_adj(batch.y_instance_edge_index, batch.batch, max_num_nodes=max_nodes)
            
                # --- Loss Calculation ---
                loss_sem = criterion_sem(sem_logits, batch.y)
                loss_bottom = 0.0
                if hasattr(batch, 'y_bottom'):
                    loss_bottom = criterion_bottom(bottom_logits.squeeze(1), batch.y_bottom.float())
                    
                
                loss_inst = 0.0
                if inst_mask is not None:
                    inst_mask_bool = inst_mask.bool()
                    loss_inst = criterion_inst(inst_matrix[inst_mask_bool], gt_matrix[inst_mask_bool])
                else:
                    loss_inst = criterion_inst(inst_matrix, gt_matrix)
                
                # [新增] 多任务损失
                if CONFIG.get('use_mtl', False):
                    # 初始化列表以收集所有激活任务的 Loss
                    losses_to_weight = []
                    # 判断哪些模块参与了计算
                    if CONFIG['lambda_sem'] > 0: losses_to_weight.append(loss_sem)
                    if CONFIG['lambda_inst'] > 0: losses_to_weight.append(loss_inst)
                    if CONFIG['lambda_bottom'] > 0: losses_to_weight.append(loss_bottom)
                    # 传入 Wrapper 自动计算总 Loss
                    # 使用 * 解包列表
                    loss = mtl_loss_wrapper(*losses_to_weight)
                else:
                    # 不使用mtl
                    loss = CONFIG['lambda_sem'] * loss_sem + \
                        CONFIG['lambda_inst'] * loss_inst + \
                        CONFIG['lambda_bottom'] * loss_bottom
                
            # PyTorch ≥ 1.1.0: 先调用 optimizer.step(), 再调用 scheduler.step()
            # 因此增加修复, 当检测到scaler变小, 才调用scheduler.step()
            # 因为scaler变小意味着当前梯度出现了INF或NAN, 优化器会跳过当前这一步
            # 记录 Scale 更新前的值, 用于检测变动
            scale_before = scaler.get_scale()

            # 使用 Scaler 进行反向传播和步进
            scaler.scale(loss).backward()
            # if CONFIG['grad_clip']: 
            #     scaler.unscale_(optimizer)
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # [核心修改] 捕获并记录梯度范数
            current_grad_norm = 0.0
            if CONFIG['grad_clip']: 
                # unscale_ 必须显式调用，才能计算出真实的梯度范数
                scaler.unscale_(optimizer)
                # clip_grad_norm_ 返回的是裁切前的 Total Norm
                norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG.get('grad_clip_num', 1.0))
                if isinstance(norm_tensor, torch.Tensor):
                    current_grad_norm = norm_tensor.item()
                else:
                    current_grad_norm = norm_tensor
                
                epoch_grad_norms.append(current_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()

            # 记录 Scale 更新后的值, 用于检测变动
            scale_after = scaler.get_scale()
            
            # 判断 Optimizer 是否被跳过
            # 如果 Scale 减小了 (scale_after < scale_before)，说明刚刚发生了 skip
            optimizer_was_skipped = (scale_after < scale_before)

            # 【SWA核心 2-A】如果是基于 Batch 的调度器 (OneCycleLR) 
            # 当进入 SWA 阶段时，停止原调度器的步进，交由 Epoch 级别的 SWALR 处理
            if CONFIG.get('use_lrs', False) and CONFIG.get('use_warmup', False):
                if not optimizer_was_skipped:
                    # 如果还没到 swa_start，正常步进；否则冻结原调度器
                    if not (use_swa and epoch >= swa_start):
                        scheduler.step()

            # EMA 更新
            if ema is not None:
                ema.update(model)

            train_loss += loss.item()

            # --- [核心] 更新指标 (Update Phase) ---
            # 准备预测字典和标签字典
            preds_dict = {'sem': sem_logits}
            targets_dict = {'sem': batch.y}
            
            # Instance (masking logic included in tracker or handled here?)
            # 为了简单，我们传 masked 后的数据给 tracker，或者传全量
            # 考虑到 BinaryAccuracy 等处理多维输入会自动 flatten，我们直接传 masked 的部分最准确
            if inst_mask is not None:
                preds_dict['inst'] = inst_matrix[inst_mask_bool].flatten()
                targets_dict['inst'] = gt_matrix[inst_mask_bool].flatten().float()
            else:
                preds_dict['inst'] = inst_matrix.flatten()
                targets_dict['inst'] = gt_matrix.flatten().float()
                
            if hasattr(batch, 'y_bottom'):
                preds_dict['bot'] = bottom_logits.squeeze(1)
                targets_dict['bot'] = batch.y_bottom
            
            tracker.update('train', preds_dict, targets_dict)

            # 显示当前学习率
            # current_lr = optimizer.param_groups[0]['lr']
            # pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'LR': f"{current_lr:.6f}"})
        
        # 【SWA核心 2-B】在 Epoch 结束时判断 SWA 更新
        if use_swa and epoch >= swa_start:
            # 累加模型参数 (数学平均)
            swa_model.update_parameters(model)
            # 使用 SWA 调度器保持恒定的较高学习率
            swa_scheduler.step()
        elif CONFIG.get('use_lrs', False) and not CONFIG.get('use_warmup', False):
            # 否则如果是普通的 CosineAnnealingLR，正常步进
            scheduler.step()
        
        # --- End of Epoch (Compute Phase) ---
        train_results, _ = tracker.compute('train') # 计算全局指标
        avg_loss = train_loss / len(train_loader)   

        # [新增] 计算并记录梯度统计信息
        if len(epoch_grad_norms) > 0:
            avg_grad_norm = np.mean(epoch_grad_norms)
            max_grad_norm = np.max(epoch_grad_norms)
            # 添加到 results 中，方便 log_metrics_to_tensorboard 处理
            train_results['Grad_Norm_Avg'] = avg_grad_norm
            train_results['Grad_Norm_Max'] = max_grad_norm
        
        # [新增] MultiTaskLossWrapper 损失监控
        if CONFIG.get('use_mtl', False):
            current_weights = mtl_loss_wrapper.get_weights()
            # 假设 losses_to_weight 的顺序是 [sem, inst, bot]
            # 注意要和你的 append 顺序对应
            task_names = []
            if CONFIG['lambda_sem'] != 0: task_names.append('Sem')
            if CONFIG['lambda_inst'] != 0: task_names.append('Inst')
            if CONFIG['lambda_bottom'] != 0: task_names.append('Bot')

            for name, w in zip(task_names, current_weights):
                writer.add_scalar(f"MTL_Weights/{name}", w, epoch)
                print(f"  [MTL] {name} Weight: {w:.4f}")

        # Logging Train
        writer.add_scalar('Loss/Train', avg_loss, epoch)
        # 记录学习率 (取最后一个 batch 的 LR 即可，因为 step 很密集)
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
        log_metrics_to_tensorboard(writer, 'Train', train_results, epoch)
            
        last_epoch_stats = {k: v.item() for k, v in train_results.items()}
        last_epoch_stats['loss'] = avg_loss
        
        # --- Validation ---
        val_model = ema.module if ema else model
        val_model.eval()
        tracker.reset('val')
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(CONFIG['device'])

                # Val 开启 autocast 以节省显存
                with autocast():
                    sem, inst, mask, bot = val_model(batch)
                
                    # GT Prep
                    max_n = inst.size(1)
                    gt_adj = to_dense_adj(batch.y_instance_edge_index, batch.batch, max_num_nodes=max_n)
                    
                    # --- Loss Calculation ---
                    loss_sem = criterion_sem(sem, batch.y)
                    
                    loss_bottom = 0.0
                    if hasattr(batch, 'y_bottom'):
                        loss_bottom = criterion_bottom(bot.squeeze(1), batch.y_bottom.float())
                    
                    loss_inst = 0.0
                    if mask is not None:
                        inst_mask_bool = mask.bool()
                        loss_inst = criterion_inst(inst[inst_mask_bool], gt_adj[inst_mask_bool])
                    else:
                        loss_inst = criterion_inst(inst, gt_adj)
                        
                    batch_loss = CONFIG['lambda_sem'] * loss_sem + \
                                CONFIG['lambda_inst'] * loss_inst + \
                                CONFIG['lambda_bottom'] * loss_bottom
                    val_loss += batch_loss.item()

                # Prepare Inputs for Tracker
                p_dict = {'sem': sem}
                t_dict = {'sem': batch.y}
                
                if mask is not None:
                    mask_b = mask.bool()
                    p_dict['inst'] = inst[mask_b].flatten()
                    t_dict['inst'] = gt_adj[mask_b].flatten().float()
                else:
                    p_dict['inst'] = inst.flatten()
                    t_dict['inst'] = gt_adj.flatten().float()
                    
                if hasattr(batch, 'y_bottom'):
                    p_dict['bot'] = bot.squeeze(1)
                    t_dict['bot'] = batch.y_bottom
                
                tracker.update('val', p_dict, t_dict)
        
        # Compute Val
        val_results, val_cm = tracker.compute('val')
        avg_val_loss = val_loss / len(val_loader)
        
        # Logging Val
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        log_metrics_to_tensorboard(writer, 'Val', val_results, epoch)
            
        fig = plot_confusion_matrix(val_cm, class_names=[str(i) for i in range(num_classes)])
        writer.add_figure('ConfusionMatrix/Val', fig, epoch)
        plt.close(fig)

        val_inst_f1 = val_results['Val_inst_f1']
        val_bot_iou = val_results['Val_bot_iou']

        print(f"[Epoch {epoch}] Train mIoU: {train_results['Train_sem_miou']:.2%} | Val mIoU: {val_results['Val_sem_miou']:.2%} \
              | Inst F1: {val_inst_f1:.2%} | Bot IoU: {val_bot_iou:.2%}")

        # --- Check Best ---
        # composite_score = val_results['Val_sem_miou'] + val_results['Val_inst_f1'] + val_results['Val_bot_iou']
        composite_score = val_results['Val_sem_miou'] + val_results['Val_inst_f1'] 
        
        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_val_record = val_results['Val_sem_acc'].item()
            torch.save(val_model.state_dict(), best_model_path)
            
        last_val_stats = {k: v.item() for k, v in val_results.items()}
        last_val_stats['loss'] = avg_val_loss
        last_val_stats['composite_score'] = composite_score.item()

    # --- End Training ---
    torch.save(val_model.state_dict(), last_model_path)
    # =========================================================================
    # [核心新增] 训练结束：进行 SWA 的 Batch Normalization 重计算
    # =========================================================================
    if use_swa:
        print("\n" + "="*20 + " SWA BATCH NORM UPDATE " + "="*20)
        print("正在进行最重要的一步: 使用训练集数据刷新 SWA 模型的 BN 层统计量...")
        # 【SWA核心 3】强制使用原数据的分布特征对 SWA 模型进行标定
        # PyTorch 的 update_bn 会将 loader 遍历一遍，更新所有 BN 均值和方差
        update_bn(train_loader, swa_model, device=CONFIG['device'])
        
        # 保存 SWA 最终模型。AveragedModel 包装在 '.module' 中，取出保存以保持键名一致
        torch.save(swa_model.module.state_dict(), swa_model_path)
        print(f"✅ SWA 模型标定完成，并已保存至: {swa_model_path}")
    writer.close()
    
    # Testing
    print("\n" + "="*20 + " TESTING PHASE " + "="*20)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    tracker.reset('test') 
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(CONFIG['device'])

            # Test 开启 autocast 以节省显存
            with autocast():
                sem, inst, mask, bot = model(batch)
            
                max_n = inst.size(1)
                gt_adj = to_dense_adj(batch.y_instance_edge_index, batch.batch, max_num_nodes=max_n)

            p_dict = {'sem': sem}
            t_dict = {'sem': batch.y}
            if mask is not None:
                p_dict['inst'] = inst[mask.bool()].flatten()
                t_dict['inst'] = gt_adj[mask.bool()].flatten().float()
            else:
                p_dict['inst'] = inst.flatten()
                t_dict['inst'] = gt_adj.flatten().float()
            if hasattr(batch, 'y_bottom'):
                p_dict['bot'] = bot.squeeze(1)
                t_dict['bot'] = batch.y_bottom
            
            tracker.update('test', p_dict, t_dict)
    
    test_results, test_cm = tracker.compute('test')
    print(f"Test Results: sem-Acc: {test_results['Test_sem_acc']:.2%} | sem-mIoU: {test_results['Test_sem_miou']:.2%}")
    print(f"Test Results: inst-Acc: {test_results['Test_inst_acc']:.2%} | inst-F-1: {test_results['Test_inst_f1']:.2%}")
    print(f"Test Results: bot-Acc: {test_results['Test_bot_acc']:.2%} | bot-IoU: {test_results['Test_bot_iou']:.2%}")
    test_stats = {k: v.item() for k, v in test_results.items()}

    # Testing SWA
    if use_swa:
        print("\n" + "="*20 + " TESTING PHASE " + "="*20)
        model.load_state_dict(torch.load(swa_model_path))
        model.eval()
        tracker.reset('test') 
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing SWA Model"):
                batch = batch.to(CONFIG['device'])

                # Test 开启 autocast 以节省显存
                with autocast():
                    sem, inst, mask, bot = model(batch)
                
                    max_n = inst.size(1)
                    gt_adj = to_dense_adj(batch.y_instance_edge_index, batch.batch, max_num_nodes=max_n)

                p_dict = {'sem': sem}
                t_dict = {'sem': batch.y}
                if mask is not None:
                    p_dict['inst'] = inst[mask.bool()].flatten()
                    t_dict['inst'] = gt_adj[mask.bool()].flatten().float()
                else:
                    p_dict['inst'] = inst.flatten()
                    t_dict['inst'] = gt_adj.flatten().float()
                if hasattr(batch, 'y_bottom'):
                    p_dict['bot'] = bot.squeeze(1)
                    t_dict['bot'] = batch.y_bottom
                
                tracker.update('test', p_dict, t_dict)

        swa_test_results, swa_test_cm = tracker.compute('test')
        print(f"SWA Model Results: sem-Acc: {swa_test_results['Test_sem_acc']:.2%} | sem-mIoU: {swa_test_results['Test_sem_miou']:.2%}")
        print(f"SWA Model Results: inst-Acc: {swa_test_results['Test_inst_acc']:.2%} | inst-F-1: {swa_test_results['Test_inst_f1']:.2%}")
        print(f"SWA Model Results: bot-Acc: {swa_test_results['Test_bot_acc']:.2%} | bot-IoU: {swa_test_results['Test_bot_iou']:.2%}")
        
        # 将 SWA 的成绩合并入记录表
        swa_test_stats = {k: v.item() for k, v in swa_test_results.items()}
        # 打包swa参数
        other_model_params = {"SWA-test-sem-acc": swa_model_path}
    else:
        swa_test_stats = None
        other_model_params = {}

    # Save Log
    recorder = ExperimentRecorder(log_root='training_log')
    recorder.save_experiment_v2(
        config=CONFIG,
        timestamp = current_time,
        model_params_info=model_params_info,
        epoch_stats=last_epoch_stats,
        val_stats=last_val_stats,
        test_stats=test_stats,
        other_stats=swa_test_stats,
        best_model_path=best_model_path,
        last_model_path=last_model_path,
        other_model_path=other_model_params,
        best_val_acc_record=best_val_record)

if __name__ == '__main__':
    train()