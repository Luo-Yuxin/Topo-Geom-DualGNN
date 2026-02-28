from config_train import CONFIG

def check_config(config):
    """
    规范化 Config: 检查必要参数, 并自动补全关闭的流为 None
    """
    # 获取模型参数
    model_config = config['model']
    topo_enable = model_config.get('topo_enable', False)
    geom_enable = model_config.get('geom_enable', False)

    topo_defs = model_config.get('topo_defs', None)
    topo_flow = model_config.get('topo_flow', None)

    geom_defs = model_config.get('geom_defs', None)
    geom_flow = model_config.get('geom_flow', None)
    
    fusion_defs = model_config.get('fusion_defs', None)
    fusion_flow = model_config.get('fusion_flow', None)
    # 事实上, 在DualGNN的设计中, 'final_fusion'必须要存在, 这涉及单流状态下本身可能存在的Global拼接
    final_fusion = model_config.get('final_fusion', {})
    # 规范性检查
    if topo_enable:
        # 检查topo定义状态
        if topo_defs == None or len(topo_defs) == 0:
            raise ValueError("'topo_defs' in config.model is missing or incorrect. \
                                example: {'0':{'gnn':'identity'}}")
        if topo_flow == None or len(topo_flow) == 0:
            raise ValueError("'topo_flow' in config.model is missing or incorrect. \
                                example: (0, 1, 2, 3)")
    if geom_enable:
        # 检查geom定义状态
        if geom_defs == None or len(geom_defs) == 0:
            raise ValueError("'geom_defs' in config.model is missing or incorrect. \
                                example: {'0':{'gnn':'identity'}}")
        if geom_flow == None or len(geom_flow) == 0:
            raise ValueError("'geom_flow' in config.model is missing or incorrect. \
                                example: (0, 1, 2, 3)")
    if topo_enable and geom_enable:
        # 两流长度必须相等
        if len(topo_flow) != len(geom_flow):
            raise ValueError("'topo_flow' and 'geom_flow' in config.model \
                                are not of equal length")
        # 检查fusion定义状态
        if fusion_defs == None or len(fusion_defs) == 0:
            raise ValueError("'fusion_defs' in config.model is missing or incorrect. \
                                example: {'0': {'method': 'concat'}}")
        if fusion_flow == None or len(fusion_flow) != len(topo_flow):
            raise ValueError("'fusion_flow' in config.model is missing or not of equal length. \
                                example: ('0<', '0=', '0=', '0=')")
    
    # 关闭部分通道时初始化用不到的通道
    if not topo_enable and not geom_enable:
        raise ValueError(f"All flows had been shut down")
    if not topo_enable:
        model_config['topo_defs'] = {'0':{'gnn':None}}
        model_config['fusion_defs'] = {'0':{'method':None}}
        model_config['topo_flow'] = tuple([None] * len(geom_flow))
        model_config['fusion_flow'] = tuple([None] * len(geom_flow))
        # 将final_fusion更换输出
        final_fusion['stream_usage'] = 'geom'
        model_config['final_fusion'] = final_fusion
    if not geom_enable:
        model_config['geom_defs'] = {'0':{'gnn':None}}
        model_config['fusion_defs'] = {'0':{'method':None}}
        model_config['geom_flow'] = tuple([None] * len(topo_flow))
        model_config['fusion_flow'] = tuple([None] * len(topo_flow))
        # 将final_fusion更换输出
        final_fusion['stream_usage'] = 'topo'
        model_config['final_fusion'] = final_fusion
    
    # 优化器调度预警
    # 当使用衰减时
    if config.get('use_lrs', False):
        # 检查学习率
        lr = config.get('lr', 1e-4)
        if not config.get('use_warmup', False):
            # 不使用OneCycleLR
            # 过高的学习率衰减仅由余弦衰减控制, 可能会造成崩溃
            if lr > 1e-3:
                print(f"Train Warning: learning rate {lr} may too high to olny use CosineAnnealingLR, \
                      try to use warm_up or reduce the learning rate.")
        else:
            # 当使用OneCycleLR
            epochs = config.get('epochs', 100)
            warmup_rate = config.get('warmup_rate', 0.3)
            warmup_start_div_factor = config.get('warmup_start_div_factor', 25.0)
            warmup_final_div_factor = config.get('warmup_final_div_factor', 10000.0)
            # 学习率上升期平均学习率斜率计算
            lr_rise_rate = (lr - lr/warmup_start_div_factor)/(epochs * warmup_rate)
            if lr_rise_rate > 2e-3:
                print(f"Train Warning: learning rate rise rate {lr_rise_rate} may too high, \
                      try to rise the warmup_rate or reduce the learning rate.")
    

            

