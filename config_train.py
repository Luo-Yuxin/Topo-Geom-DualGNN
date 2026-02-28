import torch
import torch.nn as nn
# --- 训练配置 ---
COMMON_CONFIG = {
    'data_root': './data',
    'raw_dir_name': 'raw_steps_f', 
    'label_dir_name': 'raw_labels_f',
    'processed_dir_name': 'processed_f', 
    'uv_sample': (5, 5, 5),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'seed': 42,
    'split_ratio': [0.7, 0.15, 0.15],
    'check_batch': False,
    
    'batch_size': 256, 
    'epochs': 100,
    'lr': 0.005,
    'weight_decay': 0.01,

    'use_lrs' : True,
    'use_warmup' :True,
    'warmup_rate' : 0.25,
    'warmup_start_div_factor' : 25.0, # start lr = lr / warmup_start_div_factor
    'warmup_final_div_factor' : 10000.0, # final lr = lr / warmup_final_div_factor
    
    'grad_clip': True,
    'grad_clip_num': 1.5,

    'use_mtl' : False,
    'log_var_limit' : (-2, 5),

    'lambda_sem': 1.0,
    'lambda_inst': 2, 
    'lambda_bottom': 0.5,
    
    'use_ema' : True,
    'ema_decay' : 0.999,

    'use_swa' : False,
    'swa_start_epoch': 100,    # 建议值：在最后 25% 的时间启动
    'swa_lr': 0.0005,         # 建议值：max_lr (0.005) 的 10%

}

# 切片方法, 可以用于选择原特征中的部分特征  
SLICE_CONFIG = {
    'geom_node_feat_indices': None, # [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], # 示例
    'geom_edge_feat_indices': None,
    'topo_node_feat_indices': None,
    'topo_edge_feat_indices': None 
}

MODEL_CONFIG = {
    'hidden_dim': 64,                   # 隐藏层维度
    'dropout': 0.20,                     # Dropout
    'drop_edge_topo': 0.0,
    'drop_edge_geom': 0.20,
    'num_classes': 25,                  # 语义类别数
    'priority_relation' : [4, 3, 2, 1, 1], # 几何关系权重

    # 1. 开启双流
    'topo_enable' : True,
    'geom_enable' : True,

    # 2. 定义 GNN
    # GNN 编码器类型: 'gated_gcn', 'gated_gcn_light', 'pna', 'sage', 'gat', 'deepergcn', 'identity'
    'topo_defs' : {'0': {'gnn': 'pna'},
                   '1': {'gnn': 'pna'},
                   '2': {'gnn': 'pna'},
                   '3': {'gnn': 'pna'}},

    'geom_defs' : {'0': {'gnn': 'pna'},
                   '1': {'gnn': 'pna'},
                   '2': {'gnn': 'pna'},
                   '3': {'gnn': 'pna'}},

    # 是否开启GNN后处理增强
    'use_GNN_post' : False,

    # 3. 定义融合
    # 融合算法: 'cross_attn', 'cross_gated', 'cross_gated_light', 'sum', 'concat', 'film'
    'fusion_defs' : {'0': {'method': 'film', 'dir': '^'},
                     '1': {'method': 'film', 'dir': '^'},
                     '2': {'method': 'film', 'dir': '^'}},

    # 4. 执行流 (GNNs)
    # 0, 1, 2 为层编号, 当出现重复的时候说明复用对应编号的层(即共享参数)
    # ">" 先融合后GNN, "<" 先GNN后融合, 
    # "x" 先后两次融合, "=" 不进行融合, 
    'topo_flow' : (0, 1, 2, 3),
    'geom_flow' : (0, 1, 2, 3),
    'fusion_flow': ('0>', '1>', '2>', 'None='),

    # 最终融合头配置 (Final Fusion)
    #   method: 'sum', 'concat', 'concat_deep', 'pooling', 'identity'
    #   stream_usage: 'both', 'topo', 'geom'
    #   pooling_method: 'max', 'mean', 'both' (仅 pooling 模式有效)
    'final_fusion': {
        'method': 'concat',       # 使用池化增强融合
        'stream_usage': 'both',    # 使用双流数据
        # 'pooling_method': 'max'    # 仅使用最大池化上下文
    }, 

    # 解码头配置(decoder)
    # 'sem' : 'mlp_multi_class'
    # 'inst': 'inner_product_head', 'bilinear_head', 'mlp_concat_head'
    # 'bot' : 'mlp_bin_class'
    # 在'bilinear_head'下, 我们可以选择: 'none', 'soft', 'hard'
    'decoder_type' : {'sem' :'mlp_multi_class',
                      'inst':'bilinear_head',
                      'bot': 'mlp_bin_class',
                      'symmetric_model': 'soft'},

    **SLICE_CONFIG # 注入切片配置 
}

# 最终导出的 CONFIG
CONFIG = COMMON_CONFIG.copy()
CONFIG['model'] = MODEL_CONFIG
