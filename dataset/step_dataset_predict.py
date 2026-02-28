import os
import os.path as osp
import torch
import glob
import sys
# 假设你的目录结构允许这样导入，如果报错请调整 sys.path
sys.path.append("..") 
from preprocessing.build_graph import read_step_file, build_geom_graph, build_topo_graph
from preprocessing.converter import DualStreamGraphConverter
from torch_geometric.data import Dataset, Batch

# 复用你在 step_dataset.py 中定义的 StepData 类
# 如果它在另一个文件中，请 import 它；如果就在脚本里，直接定义即可
from torch_geometric.data import Data

class StepData(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'y_instance_edge_index':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

def single_step_to_data(step_path, 
                        uv_sample_num=(5, 5, 5),
                        shape_norm_method='bbox',
                        shape_norm_param=100.0,
                        use_log_area=False, 
                        use_log_linear=False):
    """
    核心函数：将单个未标记的 STEP 文件转换为 PyG Data 对象。
    """
    raw_filename = osp.basename(step_path)
    
    # 1. 读取 STEP 几何
    try:
        shape = read_step_file(step_path)
    except Exception as e:
        print(f"Error reading {raw_filename}: {e}")
        return None

    # 2. 构建拓扑图 (G_topo)
    # 注意：参数必须与训练时完全一致
    topo_graph = build_topo_graph(shape, sample_num=uv_sample_num, estimate=False, 
                                  method=shape_norm_method, param=shape_norm_param)
    
    # 3. 构建几何图 (G_geom)
    # 注意：参数必须与训练时完全一致
    geom_graph, _, _ = build_geom_graph(shape, estimate=False, 
                                        method=shape_norm_method, param=shape_norm_param, 
                                        use_log_area=use_log_area,
                                        use_log_linear=use_log_linear)
    
    # 4. 转换为 PyG Data (不加载 Label)
    converter = DualStreamGraphConverter()
    
    # 这里不需要 extra_attributes，因为预测时没有 y, y_instance, y_bottom
    base_data = converter.convert(topo_graph, geom_graph, extra_attributes={})
    
    # 5. 封装为自定义 StepData
    data = StepData.from_dict(base_data.to_dict())
    
    # 补充必要属性
    num_nodes = topo_graph.number_of_nodes()
    data.num_nodes = num_nodes
    
    # 保存文件名，方便预测后知道结果对应哪个文件
    data.filename = raw_filename
    
    return data

class StepInferenceDataset(Dataset):
    """
    用于批量预测的数据集类。
    不像训练集需要预处理保存为 .pt，这里支持直接按需读取 .step 文件（On-the-fly）。
    如果数据量巨大，建议还是先离线转存为 .pt，逻辑参考 step_dataset.py。
    """
    def __init__(self, step_dir, extensions=['.step', '.stp'], **kwargs):
        super().__init__()
        self.step_dir = step_dir
        self.files = []
        for ext in extensions:
            self.files.extend(glob.glob(osp.join(step_dir, f'*{ext}')))
        self.files.sort()
        
        # 保存预处理参数
        self.process_kwargs = kwargs

    def len(self):
        return len(self.files)

    def get(self, idx):
        step_path = self.files[idx]
        data = single_step_to_data(step_path, **self.process_kwargs)
        if data is None:
            # 如果读取失败，返回一个空对象或者由 DataLoader 的 collate_fn 处理
            # 这里简单处理，实际工程中可能需要更健壮的错误处理
            return Data() 
        return data

# --- 使用示例 (Usage Examples) ---

if __name__ == "__main__":
    # 假设参数，请根据你实际训练时的配置修改
    config = {
        "uv_sample_num": (5, 5, 5),
        "shape_norm_method": "bbox",
        "shape_norm_param": 100.0,
        "use_log_area": False,
        "use_log_linear": False
    }

    print("--- 场景 1: 单个文件预测 ---")
    test_file = "path/to/your/test_file.step"
    
    # 1. 生成数据
    # 注意：这里的 data 是单个图，没有 batch 维度
    data = single_step_to_data(test_file, **config)
    
    if data:
        print(f"成功转换: {data.filename}")
        print(f"节点数: {data.num_nodes}")
        # print(data) -> StepData(x=[N, F], edge_index=[2, E], ...)

        # 2. 模拟放入网络
        # 重要：PyG 网络通常期望 Batch 格式，即使只有一个图
        # 手动添加 Batch 维度
        batch_data = Batch.from_data_list([data])
        
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model.to(device)
        # batch_data = batch_data.to(device)
        # model.eval()
        # with torch.no_grad():
        #     pred = model(batch_data)
        #     print("预测完成")

    print("\n--- 场景 2: 文件夹批量预测 ---")
    folder_path = "path/to/step_folder"
    
    # 1. 创建数据集
    dataset = StepInferenceDataset(folder_path, **config)
    print(f"文件夹中共有 {len(dataset)} 个文件")

    if len(dataset) > 0:
        # 2. 创建 DataLoader
        from torch_geometric.loader import DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        # 3. 循环预测
        for batch in loader:
            # batch.filename 是一个列表，包含了当前 batch 中每个图对应的文件名
            print(f"正在处理 Batch，包含文件: {batch.filename}")
            
            # batch = batch.to(device)
            # preds = model(batch)
            pass