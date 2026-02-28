import os
import torch
import time
from dataset.step_dataset import StepDataset

# 这个脚本应该放置在项目根目录 (MFR_DualGNN/) 下运行
# 运行命令: python generate_data.py

def generate():
    # 1. 自动定位根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, "data")
    
    print(f"项目根目录: {project_root}")
    print(f"数据根目录: {data_root}")
    
    # 2. 配置参数 (根据需要修改)
    config = {
        'root': data_root,
        'raw_dir_name': 'raw_steps_f',      # 输入文件夹
        'label_dir_name': 'labels_f',       # 标签文件夹
        'processed_dir_name': 'processed_f',# 输出文件夹
        'uv_sample_num': (5, 5, 5),
        'force_process': True,              # 是否强制重新生成
        'num_workers': 4,                   # 并行核心数

        'shape_norm_method' : 'bbox',       # 几何体缩放方法
        'shape_norm_param' : 100.0,         # 几何体缩放方法参数
        'use_log_area' : True,             # 几何体面面积是否对数化
        'use_log_linear' : False            # 几何体相关线性参数是否对数化
    }
    
    print("="*40)
    print("开始生成数据集...")
    print("="*40)
    
    start_time = time.time()
    
    # 初始化 Dataset 会触发 process()
    dataset = StepDataset(**config)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n数据集生成完毕!")
    print(f"总耗时: {duration:.2f} 秒")
    print(f"有效样本数: {len(dataset)}")
    
    if len(dataset) > 0:
        print("\n[检查] 第一个样本数据结构:")
        print(dataset[0])

if __name__ == "__main__":
    # Windows 多进程保护
    generate()