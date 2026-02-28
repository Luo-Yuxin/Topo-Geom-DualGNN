import os
import sys
import torch
import time

# --- 关键：添加项目根目录到 sys.path，解决模块导入问题 ---
# 获取当前脚本所在目录 (datasets/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (MFR_DualGNN/)
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 python 路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 现在可以正常导入了
from step_dataset import StepDataset

def test_dataset_conversion():
    # 1. 设置路径
    # 假设 data 文件夹在项目根目录下
    # 结构: MFR_DualGNN/data/raw_steps/xxx.step
    root_dir = os.path.join(project_root, "data")
    
    # 确保文件夹存在
    raw_path = os.path.join(root_dir, "raw_steps_s")
    # label_path = os.path.join(root_dir, "labels")
    # processed_path = os.path.join(root_dir, "processed")
    
    if not os.path.exists(raw_path):
        print(f"错误: 找不到原始数据目录: {raw_path}")
        print("请确保将 STEP 文件和 JSON 文件放入该目录。")
        return

    # 2. 初始化数据集
    # num_workers=4 适合一般测试，工作站上可设为 16
    print("正在初始化 StepDataset...")
    start_time = time.time()
    
    dataset = StepDataset(
        root=root_dir, 
        raw_dir_name='raw_steps_s', 
        label_dir_name='labels_s',
        processed_dir_name='processed_s',
        force_process=True,  # 强制重新处理以测试流程
        num_workers=6,       # 并行核心数
        uv_sample_num=(5, 5, 5)
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n处理完成! 总耗时: {duration:.2f} 秒")
    print(f"数据集大小: {len(dataset)}")

    # 3. 检查数据质量
    if len(dataset) > 0:
        print("\n--- 样本检查 (第1个样本) ---")
        data = dataset[0]
        print(data)
        
        # 检查关键属性
        print(f"节点数: {data.num_nodes}")
        print(f"拓扑节点特征 (x_topo): {data.x_topo.shape if hasattr(data, 'x_topo') else 'Missing'}")
        print(f"拓扑边 (edge_index_topo): {data.edge_index_topo.shape if hasattr(data, 'edge_index_topo') else 'Missing'}")
        print(f"几何边 (edge_index_geom): {data.edge_index_geom.shape if hasattr(data, 'edge_index_geom') else 'Missing'}")
        
        # 检查标签
        print(f"语义标签 (y): {data.y.shape if hasattr(data, 'y') else 'None'}")
        print(f"底面标签 (y_bottom): {data.y_bottom.shape if hasattr(data, 'y_bottom') else 'None'}")
        
        # 检查实例矩阵 (稀疏)
        if hasattr(data, 'y_instance_edge_index'):
            num_inst_edges = data.y_instance_edge_index.shape[1]
            print(f"实例相关性边数 (Sparse): {num_inst_edges}")
            # 简单验证: 数量应 <= N*N
        else:
            print("实例标签 (y_instance_edge_index): None")
            
        print("\n测试通过! 数据转换流程正常。")
    else:
        print("\n警告: 数据集为空，请检查 raw_steps 目录下是否有 .step 文件. ")

if __name__ == "__main__":
    # Windows 下多进程必须的保护
    test_dataset_conversion()