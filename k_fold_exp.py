import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
# [新增] 引入 SWA 工具
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from train_utils.exp_recorder import ExperimentRecorder
from train_utils.train_utils_func import (
    seed_worker, 
    set_seed, 
    count_parameters, 
    compute_pna_degrees, 
    check_has_pna,
    MetricTracker
)
from train_utils.train_ema import ModelEMA
from train_utils.multi_task_loss import MultiTaskLossWrapper

from dataset.step_dataset import StepDataset
from models.dual_stream_net import DualStreamNet
from config_train import CONFIG

# --- K-Fold 配置 ---
K_FOLDS = 5

def train_kfold():
    # 1. 基础设置
    set_seed(CONFIG['seed'])
    print(f"Using device: {CONFIG['device']}")
    
    # TF32 加速设置
    if torch.cuda.is_available() and CONFIG['device'] != 'cpu':
        try:
            gpu_cap = torch.cuda.get_device_capability(CONFIG['device'])
            if gpu_cap[0] >= 8:
                torch.set_float32_matmul_precision("high")
                print(f"  [加速] TF32 加速已开启 (Compute {gpu_cap[0]}.{gpu_cap[1]})")
        except: pass

    # 2. 准备目录结构
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    exp_name = f"{current_time}_dim{CONFIG['model']['hidden_dim']}_K{K_FOLDS}"
    base_save_dir = os.path.join('k_fold', exp_name)
    os.makedirs(base_save_dir, exist_ok=True)
    
    # 保存 Config
    with open(os.path.join(base_save_dir, 'config.json'), 'w') as f:
        json.dump(CONFIG, f, indent=4, default=str)
    
    print(f"K-Fold Experiment Dir: {base_save_dir}")

    # 3. 数据集初始化 (只初始化一次)
    dataset = StepDataset(root=CONFIG['data_root'], 
                          raw_dir_name=CONFIG['raw_dir_name'],
                          label_dir_name=CONFIG['label_dir_name'],
                          processed_dir_name=CONFIG['processed_dir_name'],
                          uv_sample_num=CONFIG['uv_sample'],
                          force_process=False)
    
    if len(dataset) == 0: return

    # 自动配置维度
    sample = dataset[0]
    CONFIG['model']['topo_node_in'] = sample.x_topo.shape[1]
    CONFIG['model']['topo_edge_in'] = sample.edge_attr_topo.shape[-1]
    CONFIG['model']['geom_node_in'] = sample.x_geom.shape[1]
    CONFIG['model']['geom_edge_in'] = sample.edge_attr_geom.shape[1]
    num_classes = CONFIG['model'].get('num_classes', 25)

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

    # 4. K-Fold 循环
    # 使用 shuffle=True 确保数据被打乱
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=CONFIG['seed'])
    
    # 用于存储每一折的最终测试结果
    fold_metrics = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"\n{'='*20} Starting Fold {fold+1}/{K_FOLDS} {'='*20}")
        
        # --- [A] 每一折的数据加载器 ---
        train_subsampler = torch.utils.data.Subset(dataset, train_ids)
        val_subsampler = torch.utils.data.Subset(dataset, val_ids)
        
        loader_args = {
            'batch_size': CONFIG['batch_size'],
            'num_workers': CONFIG['num_workers'],
            'pin_memory': True,
            'worker_init_fn': seed_worker,
            'persistent_workers': (CONFIG['num_workers'] > 0)
        }
        
        train_loader = DataLoader(train_subsampler, shuffle=True, **loader_args)
        # 验证集也作为本折的测试集
        val_loader = DataLoader(val_subsampler, shuffle=False, **loader_args)

        # --- [B] 每一折的文件路径 ---
        fold_dir = os.path.join(base_save_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        best_model_path = os.path.join(fold_dir, 'best_model.pth')
        swa_model_path = os.path.join(fold_dir, 'swa_model.pth') # [新增] SWA 保存路径
        
        # TensorBoard (每折一个独立目录)
        writer = SummaryWriter(log_dir=os.path.join('runs', f"{exp_name}_fold{fold}"))

        # --- [C] 重新初始化模型与组件 ---
        # 必须重新实例化，否则会继承上一折的权重
        model = DualStreamNet(CONFIG['model']).to(CONFIG['device'])
        
        # MTL 初始化
        mtl_loss_wrapper = None
        if CONFIG.get('use_mtl', False):
            active_task_count = 0
            if CONFIG['lambda_sem'] > 0: active_task_count += 1
            if CONFIG['lambda_inst'] > 0: active_task_count += 1
            if CONFIG['lambda_bottom'] > 0: active_task_count += 1
            
            mtl_loss_wrapper = MultiTaskLossWrapper(task_num=active_task_count, log_var_limit=CONFIG.get('log_var_limit', (-2, 5))).to(CONFIG['device'])
            
            # [重要] 保持参数分组逻辑
            param_groups = [
                {'params': model.parameters(), 'weight_decay': CONFIG['weight_decay']},
                {'params': mtl_loss_wrapper.parameters(), 'weight_decay': 0.0}
            ]
            optimizer = optim.AdamW(param_groups, lr=CONFIG['lr'])
        else:
            optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])


        # =========================================================================
        # [核心新增] SWA 的严谨初始化 (提前至 Scheduler 之前以联动学习率)
        # =========================================================================
        use_swa = CONFIG.get('use_swa', False)
        swa_model = None
        swa_scheduler = None
        swa_start = CONFIG.get('swa_start_epoch', int(CONFIG['epochs'] * 0.75))
        
        if use_swa:
            print(f"  -> 启用 SWA | Start Epoch: {swa_start}")
            swa_model = AveragedModel(model)
            swa_lr = CONFIG.get('swa_lr', CONFIG['lr'] * 0.1)
            swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
            print(f"  -> SWA 专属学习率设为: {swa_lr}")
        else:
            swa_lr = 0.0


        # Scheduler
        # scheduler = None
        # if CONFIG.get('use_lrs', False):
        #     if CONFIG.get('use_warmup', False):
        #         scheduler = OneCycleLR(optimizer, max_lr=CONFIG['lr'], epochs=CONFIG['epochs'], 
        #                                steps_per_epoch=len(train_loader), 
        #                                pct_start=CONFIG.get('warmup_rate', 0.3),
        #                                div_factor=25.0, final_div_factor=10000.0)
        #     else:
        #         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=0)

        # scaler = GradScaler()
        scheduler = None
        if CONFIG.get('use_lrs', False):
            if CONFIG.get('use_warmup', False):
                # [策略一] 动态压缩周期与计算终点
                cycle_epochs = swa_start if use_swa else CONFIG['epochs']
                div_factor = CONFIG.get('warmup_start_div_factor', 25.0)
                
                if use_swa:
                    final_div_factor = float(CONFIG['lr']) / (div_factor * swa_lr)
                else:
                    final_div_factor = CONFIG.get('warmup_final_div_factor', 10000.0)

                scheduler = OneCycleLR(optimizer, max_lr=CONFIG['lr'], epochs=cycle_epochs, 
                                       steps_per_epoch=len(train_loader), 
                                       pct_start=CONFIG.get('warmup_rate', 0.3),
                                       div_factor=div_factor, final_div_factor=final_div_factor)
            else:
                cycle_epochs = swa_start if use_swa else CONFIG['epochs']
                eta_min = swa_lr if use_swa else 0
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cycle_epochs, eta_min=eta_min)

        scaler = GradScaler()
        
        # EMA
        use_ema = CONFIG.get('use_ema', False)
        if use_ema:
            ema = ModelEMA(model, decay=CONFIG.get('ema_decay', 0.999), device=CONFIG['device']) 
            print(f"启用 EMA 训练 (Decay: {CONFIG.get('ema_decay', 0.999)}, with Dynamic Warmup & State Caching)")
        else:
            None

        # Criterion & Tracker
        criterion_sem = nn.CrossEntropyLoss()
        criterion_inst = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([19.0]).to(CONFIG['device']))
        criterion_bottom = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.3]).to(CONFIG['device']))
        
        tracker = MetricTracker(num_classes=num_classes, device=CONFIG['device'])

        # --- [D] 训练循环 ---
        best_composite_score = -float('inf')
        
        for epoch in range(CONFIG['epochs']):
            model.train()
            tracker.reset('train')
            train_loss = 0.0
            
            # Train Loop
            pbar = tqdm(train_loader, desc=f"Fold {fold} | Epoch {epoch+1}", leave=False)
            for batch in pbar:
                batch = batch.to(CONFIG['device'])
                optimizer.zero_grad()
                
                with autocast():
                    sem_logits, inst_matrix, inst_mask, bottom_logits = model(batch)
                    
                    # GT Prep
                    max_nodes = inst_matrix.size(1)
                    gt_matrix = to_dense_adj(batch.y_instance_edge_index, batch.batch, max_num_nodes=max_nodes)

                    # Loss
                    loss_sem = criterion_sem(sem_logits, batch.y)
                    
                    loss_bottom = torch.tensor(0.0, device=CONFIG['device'])
                    if hasattr(batch, 'y_bottom'):
                        loss_bottom = criterion_bottom(bottom_logits.squeeze(1), batch.y_bottom.float())

                    loss_inst = torch.tensor(0.0, device=CONFIG['device'])
                    if inst_mask is not None:
                        loss_inst = criterion_inst(inst_matrix[inst_mask.bool()], gt_matrix[inst_mask.bool()])
                    else:
                        loss_inst = criterion_inst(inst_matrix, gt_matrix)
                    
                    # Aggregate Loss
                    if mtl_loss_wrapper:
                        losses_to_weight = []
                        if CONFIG['lambda_sem'] > 0: losses_to_weight.append(loss_sem)
                        if CONFIG['lambda_inst'] > 0: losses_to_weight.append(loss_inst)
                        if CONFIG['lambda_bottom'] > 0: losses_to_weight.append(loss_bottom)
                        loss = mtl_loss_wrapper(*losses_to_weight)
                    else:
                        loss = CONFIG['lambda_sem']*loss_sem + CONFIG['lambda_inst']*loss_inst + CONFIG['lambda_bottom']*loss_bottom

                # Backward
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                
                if CONFIG['grad_clip']:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG.get('grad_clip_num', 1.0))
                
                scaler.step(optimizer)
                scaler.update()
                
                # Check skip & Scheduler
                scale_after = scaler.get_scale()
                optimizer_was_skipped = (scale_after < scale_before)
                
                # [核心] 如果是 based on batch 的调度器，限制在 SWA 开始前步进
                if scheduler and CONFIG.get('use_warmup', False) and not optimizer_was_skipped:
                    if not (use_swa and epoch >= swa_start):
                        scheduler.step()
                
                if ema: ema.update(model)
                train_loss += loss.item()

                # Update Tracker
                preds_dict = {'sem': sem_logits}
                targets_dict = {'sem': batch.y}
                if inst_mask is not None:
                    preds_dict['inst'] = inst_matrix[inst_mask.bool()].flatten()
                    targets_dict['inst'] = gt_matrix[inst_mask.bool()].flatten().float()
                else:
                    preds_dict['inst'] = inst_matrix.flatten()
                    targets_dict['inst'] = gt_matrix.flatten().float()
                if hasattr(batch, 'y_bottom'):
                    preds_dict['bot'] = bottom_logits.squeeze(1)
                    targets_dict['bot'] = batch.y_bottom
                
                tracker.update('train', preds_dict, targets_dict)
            
            # [核心] Epoch 结束时的调度与 SWA 权重融合
            if use_swa and epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            elif scheduler and not CONFIG.get('use_warmup', False):
                # 对应非 OneCycleLR 的 CosineAnnealing
                scheduler.step()

            # --- Validation (此折的验证集) ---
            val_model = ema.module if ema else model
            val_model.eval()
            tracker.reset('val')
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(CONFIG['device'])
                    with autocast():
                        sem, inst, mask, bot = val_model(batch)
                        
                        # Just Tracker Update (Loss calculation omitted for brevity as requested)
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
                        
                        tracker.update('val', p_dict, t_dict)
            
            # Compute Metrics
            train_res, _ = tracker.compute('train')
            val_res, _ = tracker.compute('val')
            
            # Simple TensorBoard Log (Reduced)
            writer.add_scalar('Loss/Train', train_loss / len(train_loader), epoch)
            writer.add_scalar('Metric/Train_mIoU', train_res['Train_sem_miou'], epoch)
            writer.add_scalar('Metric/Val_mIoU', val_res['Val_sem_miou'], epoch)
            writer.add_scalar('Metric/Val_Inst_F1', val_res['Val_inst_f1'], epoch)

            # [核心修改] Check Best - 动态计算综合得分
            # 只计算被 CONFIG 激活的任务的指标
            composite_score = 0.0
            if CONFIG['lambda_sem'] > 0:
                composite_score += val_res['Val_sem_miou']
            if CONFIG['lambda_inst'] > 0:
                composite_score += val_res['Val_inst_f1']
            if CONFIG['lambda_bottom'] > -1:
                composite_score += val_res['Val_bot_iou']

            # 防止空 score (虽然理论上不可能，除非所有 lambda 都是 0)
            if composite_score == 0.0 and CONFIG['lambda_sem'] == 0 and CONFIG['lambda_inst'] == 0:
                pass # 仅 bottom 为 0 的情况或者全 0

            if composite_score > best_composite_score:
                best_composite_score = composite_score
                torch.save(val_model.state_dict(), best_model_path)
        
        # --- [E] Fold Finish: Evaluation ---
        print(f"Fold {fold} Finished. Best Score: {best_composite_score:.4f}")
        
        # Load Best Model for Final Report
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        tracker.reset('test') # Use 'test' key for final fold result
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(CONFIG['device'])
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
        
        fold_final_res, _ = tracker.compute('test')
        # Convert tensors to float for JSON serialization
        fold_final_res_clean = {k: v.item() for k, v in fold_final_res.items()}
        
        # 2. SWA 模型的定标与最终测试
        if use_swa:
            print("  -> 正在使用训练集更新 SWA 模型的 BN 统计量...")
            update_bn(train_loader, swa_model, device=CONFIG['device'])
            torch.save(swa_model.module.state_dict(), swa_model_path)
            
            swa_model.eval()
            tracker.reset('test')
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(CONFIG['device'])
                    with autocast():
                        sem, inst, mask, bot = swa_model(batch)
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
            
            swa_fold_final_res, _ = tracker.compute('test')
            # 将 SWA 的成绩挂载在原本的 json 字典里，前缀加上 SWA_
            for k, v in swa_fold_final_res.items():
                fold_final_res_clean[f"SWA_{k}"] = v.item()

        # Save Fold Results
        with open(os.path.join(fold_dir, 'fold_result.json'), 'w') as f:
            json.dump(fold_final_res_clean, f, indent=4)
            
        fold_metrics.append(fold_final_res_clean)
        writer.close()

    # 5. 汇总 K-Fold 结果
    print("\n" + "="*20 + " K-FOLD SUMMARY " + "="*20)
    summary = {}
    metric_keys = fold_metrics[0].keys()
    
    for key in metric_keys:
        values = [m[key] for m in fold_metrics]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    
    # 打印核心指标
    # 打印核心指标
    print(f"Standard Model:")
    print(f"Average Sem-acc: {summary['Test_sem_acc_mean']:.2%} ± {summary['Test_sem_acc_std']:.4%}")
    print(f"Average Sem-mIoU: {summary['Test_sem_miou_mean']:.2%} ± {summary['Test_sem_miou_std']:.4%}")
    print(f"Average Inst-acc: {summary['Test_inst_acc_mean']:.2%} ± {summary['Test_inst_acc_std']:.4%}")
    print(f"Average Inst-F1: {summary['Test_inst_f1_mean']:.2%} ± {summary['Test_inst_f1_std']:.4%}")
    print(f"Average bot-acc: {summary['Test_bot_acc_mean']:.2%} ± {summary['Test_bot_acc_std']:.4%}")
    print(f"Average bot-Iou: {summary['Test_bot_iou_mean']:.2%} ± {summary['Test_bot_iou_std']:.4%}")
    if use_swa:
        print(f"SWA Model:")
        print(f"Average Sem-acc: {summary.get('SWA_Test_sem_acc_mean', 0.0):.2%} ± {summary.get('SWA_Test_sem_acc_std', 0.0):.4%}")
        print(f"Average Sem-mIoU: {summary.get('SWA_Test_sem_miou_mean', 0.0):.2%} ± {summary.get('SWA_Test_sem_miou_std', 0.0):.4%}")
        print(f"Average Inst-acc: {summary.get('SWA_Test_inst_acc_mean', 0.0):.2%} ± {summary.get('SWA_Test_inst_acc_std', 0.0):.4%}")
        print(f"Average Inst-F1: {summary.get('SWA_Test_inst_f1_mean', 0.0):.2%} ± {summary.get('SWA_Test_inst_f1_std', 0.0):.4%}")
        print(f"Average bot-acc: {summary.get('SWA_Test_bot_acc_mean', 0.0):.2%} ± {summary.get('SWA_Test_bot_acc_std', 0.0):.4%}")
        print(f"Average bot-Iou: {summary.get('SWA_Test_bot_iou_mean', 0.0):.2%} ± {summary.get('SWA_Test_bot_iou_std', 0.0):.4%}")

    with open(os.path.join(base_save_dir, 'summary_results.json'), 'w') as f:
        json.dump(summary, f, indent=4)

if __name__ == '__main__':
    train_kfold()