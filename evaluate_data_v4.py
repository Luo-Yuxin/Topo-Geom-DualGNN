import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg') # 设置后端，防止无GUI环境报错
import matplotlib.pyplot as plt
import pandas as pd
from torch_geometric.utils import to_networkx
import networkx as nx

# --- 环境设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# 尝试加载 dataset 模块路径
possible_paths = [
    os.path.join(current_dir, 'dataset'),
    os.path.join(current_dir, 'datasets'),
    current_dir
]
for path in possible_paths:
    if os.path.exists(path):
        sys.path.append(path)

# 解决 pickle 反序列化路径问题 (Monkey Patch)
try:
    from dataset.step_dataset import StepDataset
    import dataset.step_dataset as step_dataset_module
except ImportError:
    try:
        from dataset.step_dataset import StepDataset
        import dataset.step_dataset as step_dataset_module
    except ImportError as e:
        # Fallback: 假设 step_dataset 在当前目录
        raise ValueError(f"{e}")

sys.modules['step_dataset'] = step_dataset_module
# -------------------------------------------------------

def plot_distribution(data_list, title, xlabel, ylabel, save_path, color='skyblue'):
    """
    [新增] 绘制分布直方图辅助函数
    """
    if not data_list:
        return
    
    plt.figure(figsize=(10, 6))
    # 自动决定 bins 数量
    bins = min(50, len(set(data_list))) if len(set(data_list)) > 0 else 10
    
    plt.hist(data_list, bins=bins, color=color, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.5)
    
    # 标注一些关键统计线
    mean_val = np.mean(data_list)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')
    plt.legend()
    
    plt.savefig(save_path)
    plt.close()
    print(f"图片已保存至: {save_path}")

def calculate_graph_stats(name, node_counts, avg_degrees):
    """
    [新增] 计算并打印图统计指标
    """
    if not node_counts:
        print(f"{name} 无有效数据")
        return

    # 转换为 numpy 数组方便计算
    nodes = np.array(node_counts)
    degs = np.array(avg_degrees)

    print(f"\n     {name} 统计指标:")
    print(f"     • 节点数量 (Nodes):")
    print(f"       - Max: {np.max(nodes)}")
    print(f"       - Min: {np.min(nodes)}")
    print(f"       - Avg: {np.mean(nodes):.2f}")
    
    print(f"     • 平均节点度 (Avg Degree = Edge/Node):")
    print(f"       - Max Avg Degree: {np.max(degs):.4f}")
    print(f"       - Min Avg Degree: {np.min(degs):.4f}")
    print(f"       - Avg Avg Degree: {np.mean(degs):.4f}")

def analyze_dataset(data_root, raw_dir, processed_dir, label_dir):
    print("="*60)
    print(f"数据检查报告 (Deep Data Inspection)")
    print(f"数据源: {data_root}/{processed_dir}")
    print("="*60)

    try:
        dataset = StepDataset(root=data_root, 
                              raw_dir_name=raw_dir,
                              label_dir_name=label_dir,
                              processed_dir_name=processed_dir,
                              force_process=False)
    except Exception as e:
        print(f"严重错误: Dataset 加载失败 - {e}")
        return

    total_files = len(dataset)
    print(f"样本总数: {total_files}")

    # --- 统计容器 ---
    stats = {
        'error_files': [],
        
        # 极值追踪器 (Area - Dim 0)
        'area_max': {'val': -float('inf'), 'file': None, 'node_idx': -1},
        'area_min': {'val': float('inf'), 'file': None, 'node_idx': -1},
        
        # 负面积记录
        'negative_area_files': [],

        # 语义标签追踪
        'max_sem_label': {'val': -1, 'file': None},
        'sem_label_counts': Counter(),

        # 其他统计
        'large_val_geom': [],
        'large_val_topo': [],
        'zero_feat_geom': [],
        'zero_feat_topo_node': [],
        'geom_feat_accum': None,
        'disconnected': [],
        'isolated': [],

        # [新增] 图结构统计容器
        'topo_stats': {
            'node_counts': [],
            'avg_degrees': []  # E / N
        },
        'geom_stats': {
            'node_counts': [],
            'avg_degrees': []  # E / N
        }
    }

    # 阈值设定
    LARGE_VALUE_THRESHOLD = 1000.0
    ZERO_TOLERANCE = 1e-6          

    print("\n 正在逐个扫描文件特征...")
    
    for i in tqdm(range(total_files)):
        try:
            data = dataset.get(i)
            raw_fname = dataset.raw_file_names[i]
            num_nodes = data.num_nodes
            
            if num_nodes == 0:
                stats['error_files'].append(f"{raw_fname} (Empty Graph)")
                continue
            
            # =========================================================
            # [新增] 1. 收集 Topo Graph 统计信息
            # =========================================================
            if hasattr(data, 'x_topo') and data.x_topo is not None:
                # 节点数: x_topo 的第0维
                n_topo = data.x_topo.size(0)
                # 边数: edge_index_topo 的第1维
                e_topo = 0
                if hasattr(data, 'edge_index_topo') and data.edge_index_topo is not None:
                    e_topo = data.edge_index_topo.size(1)
                
                # 平均度 = 边数 / 节点数 (遵循你的定义)
                avg_deg_topo = e_topo / n_topo if n_topo > 0 else 0
                
                stats['topo_stats']['node_counts'].append(n_topo)
                stats['topo_stats']['avg_degrees'].append(avg_deg_topo)

            # =========================================================
            # [新增] 2. 收集 Geom Graph 统计信息
            # =========================================================
            if hasattr(data, 'x_geom') and data.x_geom is not None:
                n_geom = data.x_geom.size(0)
                e_geom = 0
                if hasattr(data, 'edge_index_geom') and data.edge_index_geom is not None:
                    e_geom = data.edge_index_geom.size(1)
                
                avg_deg_geom = e_geom / n_geom if n_geom > 0 else 0
                
                stats['geom_stats']['node_counts'].append(n_geom)
                stats['geom_stats']['avg_degrees'].append(avg_deg_geom)

            # =========================================================
            # 以下为原有逻辑 (保持不变)
            # =========================================================

            # -------------------------------------------------
            # 几何特征 (x_geom) 深度检查 & 极值追踪
            # -------------------------------------------------
            if hasattr(data, 'x_geom') and data.x_geom is not None:
                x_geom = data.x_geom
                # --- 面积极值追踪 (假设面积在第0维) ---
                areas = x_geom[:, 0]
                
                local_max_val, local_max_idx = torch.max(areas, dim=0)
                if local_max_val > stats['area_max']['val']:
                    stats['area_max']['val'] = local_max_val.item()
                    stats['area_max']['file'] = raw_fname
                    stats['area_max']['node_idx'] = local_max_idx.item()
                
                local_min_val, local_min_idx = torch.min(areas, dim=0)
                if local_min_val < stats['area_min']['val']:
                    stats['area_min']['val'] = local_min_val.item()
                    stats['area_min']['file'] = raw_fname
                    stats['area_min']['node_idx'] = local_min_idx.item()

                if (areas < 0).any():
                    neg_indices = (areas < 0).nonzero(as_tuple=True)[0].tolist()
                    stats['negative_area_files'].append({
                        'file': raw_fname,
                        'values': areas[neg_indices].tolist(),
                        'indices': neg_indices
                    })

                if stats['geom_feat_accum'] is None:
                    stats['geom_feat_accum'] = {
                        'sum': torch.zeros(x_geom.shape[1]),
                        'sq_sum': torch.zeros(x_geom.shape[1]),
                        'count': 0,
                        'max': torch.full((x_geom.shape[1],), -float('inf')),
                        'min': torch.full((x_geom.shape[1],), float('inf'))
                    }
                
                stats['geom_feat_accum']['sum'] += x_geom.sum(dim=0)
                stats['geom_feat_accum']['sq_sum'] += (x_geom ** 2).sum(dim=0)
                stats['geom_feat_accum']['count'] += x_geom.shape[0]
                stats['geom_feat_accum']['max'] = torch.maximum(stats['geom_feat_accum']['max'], x_geom.max(dim=0)[0])
                stats['geom_feat_accum']['min'] = torch.minimum(stats['geom_feat_accum']['min'], x_geom.min(dim=0)[0])

                max_val_per_node = x_geom.abs().max(dim=1)[0]
                if (max_val_per_node > LARGE_VALUE_THRESHOLD).any():
                    max_val = x_geom.abs().max().item()
                    bad_indices = (max_val_per_node > LARGE_VALUE_THRESHOLD).nonzero(as_tuple=True)[0].tolist()
                    stats['large_val_geom'].append({
                        'file': raw_fname,
                        'max_val': max_val,
                        'count': len(bad_indices),
                        'indices': bad_indices[:5]
                    })

                geom_sums = x_geom.abs().sum(dim=1)
                if (geom_sums < ZERO_TOLERANCE).any():
                    zero_indices = (geom_sums < ZERO_TOLERANCE).nonzero(as_tuple=True)[0].tolist()
                    stats['zero_feat_geom'].append({
                        'file': raw_fname,
                        'indices': zero_indices
                    })

            # -------------------------------------------------
            # 拓扑特征 (x_topo) 深度检查
            # -------------------------------------------------
            if hasattr(data, 'x_topo') and data.x_topo is not None:
                topo_sums = data.x_topo.reshape(num_nodes, -1).abs().sum(dim=1)
                if (topo_sums < ZERO_TOLERANCE).any():
                    zero_indices = (topo_sums < ZERO_TOLERANCE).nonzero(as_tuple=True)[0].tolist()
                    stats['zero_feat_topo_node'].append({
                        'file': raw_fname,
                        'indices': zero_indices
                    })

            # -------------------------------------------------
            # 语义标签 (y) 检查
            # -------------------------------------------------
            if hasattr(data, 'y') and data.y is not None:
                y_np = data.y.cpu().numpy()
                stats['sem_label_counts'].update(y_np)
                local_max_label = y_np.max()
                if local_max_label > stats['max_sem_label']['val']:
                    stats['max_sem_label']['val'] = int(local_max_label)
                    stats['max_sem_label']['file'] = raw_fname

            # -------------------------------------------------
            # 图连通性检查
            # -------------------------------------------------
            if data.num_edges > 0:
                G = to_networkx(data, to_undirected=True)
                if not nx.is_connected(G):
                    stats['disconnected'].append(raw_fname)
                if nx.number_of_isolates(G) > 0:
                    stats['isolated'].append(raw_fname)

        except Exception as e:
            stats['error_files'].append(f"Index {i}: {str(e)}")

    # --- 生成最终报告 ---
    print("\n" + "="*60)
    print("检测结果报告 (Diagnostic Report)")
    print("="*60)

    # 1. 面积极值定位 (核心需求)
    print("\n[1] 面积特征 (Dim 0) 极值定位:")
    print("-" * 60)
    
    max_info = stats['area_max']
    print(f" 全局最大面积: {max_info['val']:.4f}")
    print(f"   - 文件: {max_info['file']}")
    print(f"   - 节点索引: {max_info['node_idx']}")
    print("-" * 30)
    
    min_info = stats['area_min']
    print(f"全局最小面积: {min_info['val']:.4f}")
    print(f"   - 文件: {min_info['file']}")
    print(f"   - 节点索引: {min_info['node_idx']}")
    print("-" * 60)

    # 2. 负面积警告
    if len(stats['negative_area_files']) > 0:
        print(f"\n 发现 {len(stats['negative_area_files'])} 个文件包含负面积 (Negative Area)!")
        for item in stats['negative_area_files'][:5]:
            print(f"  - 文件: {item['file']}")
            print(f"    值: {item['values']}")
            print(f"    索引: {item['indices']}")
    else:
        print("\n 未发现负面积特征。")

    # 3. 几何特征分布统计
    if stats['geom_feat_accum'] is not None:
        print("\n[2] 几何特征统计 (x_geom Distribution):")
        print("-" * 85)
        print(f"{'Dim':<5} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Std':<10} | {'Status'}")
        print("-" * 85)
        
        count = stats['geom_feat_accum']['count']
        sums = stats['geom_feat_accum']['sum']
        sq_sums = stats['geom_feat_accum']['sq_sum']
        mins = stats['geom_feat_accum']['min']
        maxs = stats['geom_feat_accum']['max']
        
        means = sums / count
        stds = torch.sqrt((sq_sums / count) - (means ** 2) + 1e-6)
        
        for dim in range(len(means)):
            status = " OVERFLOW" if maxs[dim] > 1000 else (" Large" if maxs[dim] > 100 else " Normal")
            print(f"{dim:<5} | {mins[dim]:<10.2f} | {maxs[dim]:<10.2f} | {means[dim]:<10.2f} | {stds[dim]:<10.2f} | {status}")
        print("-" * 85)

    # 4. 标签统计
    print("\n[3] 语义标签统计 (Semantic Labels):")
    print("-" * 60)
    max_label_idx = stats['max_sem_label']['val']
    
    print(f"  全局最大标签索引 (Max Label Index): {max_label_idx}")
    print(f"   - 出现文件: {stats['max_sem_label']['file']}")
    print(f"    建议 num_classes 设置至少为: {max_label_idx + 1}")

    print("\n 类别分布详情 (Class Distribution):")
    print(f"   {'Class ID':<10} | {'Count':<10} | {'Percentage':<12} | {'Status'}")
    print("-" * 65)
    
    total_labels = sum(stats['sem_label_counts'].values())
    for cls_id in range(max_label_idx + 1):
        count = stats['sem_label_counts'].get(cls_id, 0)
        percentage = (count / total_labels) * 100 if total_labels > 0 else 0.0
        status = ""
        if count == 0:
            status = " ZERO SAMPLE (从未出现)"
        elif percentage < 0.1:
            status = " Rare (<0.1%)"
        elif percentage < 1.0:
            status = " Uncommon (<1%)"
        else:
            status = " Normal"
        print(f"   {cls_id:<10} | {count:<10} | {percentage:<11.2f}% | {status}")
    print("-" * 65)
    print(f"   总计样本 (Nodes): {total_labels}")

    # 5. 异常文件汇总
    print(f"\n[4] 异常文件统计:")
    print(f"  - 包含极大值的几何文件数 (> {LARGE_VALUE_THRESHOLD}): {len(stats['large_val_geom'])}")
    print(f"  - 存在零特征的几何节点数: {len(stats['zero_feat_geom'])}")
    print(f"  - 存在零特征的拓扑节点数 (采样失败): {len(stats['zero_feat_topo_node'])}")
    print(f"  - 结构断裂的图数量: {len(stats['disconnected'])}")

    # =========================================================
    # [新增] 6. 图结构统计与可视化
    # =========================================================
    print(f"\n[5]  图结构统计 (Graph Structure Statistics):")
    print("-" * 60)
    
    # 准备保存目录
    analysis_dir = os.path.join(data_root, 'analysis_results')
    os.makedirs(analysis_dir, exist_ok=True)
    print(f"    统计图表将保存至: {analysis_dir}")

    # --- Topo Graph ---
    calculate_graph_stats("Topo Graph", 
                          stats['topo_stats']['node_counts'], 
                          stats['topo_stats']['avg_degrees'])
    
    # 绘制 Topo 节点分布
    plot_distribution(stats['topo_stats']['node_counts'],
                      title='Topo Graph Node Count Distribution',
                      xlabel='Number of Nodes',
                      ylabel='Number of Files (Graphs)',
                      save_path=os.path.join(analysis_dir, 'topo_node_dist.png'),
                      color='skyblue')
    
    # 绘制 Topo 平均度分布
    plot_distribution(stats['topo_stats']['avg_degrees'],
                      title='Topo Graph Average Degree Distribution (Edge/Node)',
                      xlabel='Average Degree',
                      ylabel='Number of Files (Graphs)',
                      save_path=os.path.join(analysis_dir, 'topo_avg_degree_dist.png'),
                      color='steelblue')

    # --- Geom Graph ---
    calculate_graph_stats("Geom Graph", 
                          stats['geom_stats']['node_counts'], 
                          stats['geom_stats']['avg_degrees'])

    # 绘制 Geom 节点分布
    plot_distribution(stats['geom_stats']['node_counts'],
                      title='Geom Graph Node Count Distribution',
                      xlabel='Number of Nodes',
                      ylabel='Number of Files (Graphs)',
                      save_path=os.path.join(analysis_dir, 'geom_node_dist.png'),
                      color='lightgreen')
    
    # 绘制 Geom 平均度分布
    plot_distribution(stats['geom_stats']['avg_degrees'],
                      title='Geom Graph Average Degree Distribution (Edge/Node)',
                      xlabel='Average Degree',
                      ylabel='Number of Files (Graphs)',
                      save_path=os.path.join(analysis_dir, 'geom_avg_degree_dist.png'),
                      color='forestgreen')
    
    print("\n 所有统计分析完成!")

if __name__ == "__main__":
    # 配置
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
    RAW_DIR = 'raw_steps_f'
    LABEL_DIR = 'raw_labels_f'
    PROCESSED_DIR = 'processed_f' # 你的全量数据文件夹
    
    analyze_dataset(DATA_ROOT, RAW_DIR, PROCESSED_DIR, LABEL_DIR)