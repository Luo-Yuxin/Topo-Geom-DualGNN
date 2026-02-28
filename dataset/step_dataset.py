import os
import os.path as osp
import glob
import json
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 导入你的预处理模块
import sys
sys.path.append("..") 
from preprocessing.build_graph import read_step_file, build_geom_graph, build_topo_graph
from preprocessing.converter import DualStreamGraphConverter

# --- 定义一个自定义 Data 类 ---
class StepData(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        # 告诉 PyG: 当合并 Batch 时，'y_instance_edge_index' 需要根据 'num_nodes' 进行累加
        if key == 'y_instance_edge_index':
            return self.num_nodes
        # 其他标准字段 (edge_index 等) 使用默认行为
        return super().__inc__(key, value, *args, **kwargs)

# --- 辅助函数：定义在类外，以便多进程序列化 (Pickling) ---

def load_labels_from_json(json_path, num_nodes):
    """
    从 JSON 文件加载标签。独立函数，方便 worker 调用。
    """

    if not osp.exists(json_path):
        # 仅在调试时打印，避免多进程下 stdout 混乱，或者可以 logging
        return None, None, None
    # 读取对应地址的json文件
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError:
            return None, None, None
    # 初始化三类标签
    y_semantic = None
    y_instance_adj = None
    y_bottom = None
    # 用于判断json中的数据是否合法
    # 即整体存储在一个列表里面, 列表长度大于1且列表中嵌套的列表中储存的值大于2
    if isinstance(raw_data, list) and len(raw_data) > 0 and len(raw_data[0]) >= 2:
        # 即读取标签字典
        data_dict = raw_data[0][1]
    else:
        return None, None, None

    # 2. 语义分割标签
    if "seg" in data_dict:
        # 读取语义分割标签字典
        seg_dict = data_dict["seg"]
        # 根据传递来的图节点数初始化节点标签矩阵
        y_semantic = np.zeros(num_nodes, dtype=np.int64)
        # 从字典中读取值
        if num_nodes == len(seg_dict):
            for face_idx_str, label_val in seg_dict.items():
                # 为保证字典中面节点数目等同于由几何体构建的图节点数目
                idx = int(face_idx_str)
                y_semantic[idx] = int(label_val)      
        else:
            raise ValueError("Seg_num in Json is {}, but graph nodes_num is {}, " \
            "Json file is {}".format(len(seg_dict), num_nodes, osp.basename(json_path)))
                
    
    # 3. 实例分割标签 (Correlation Matrix -> Sparse Edge Index)
    if "inst" in data_dict:
        inst_matrix = np.array(data_dict["inst"])
        # 若维度不匹配报错
        if inst_matrix.shape == (num_nodes, num_nodes):
            rows, cols = np.where(inst_matrix > 0)
            y_instance_adj = np.vstack([rows, cols])
        else:
            raise ValueError("Inst_num in Json is {}, but graph nodes_num is {}, " \
            "Json file is {}".format(inst_matrix.shape, num_nodes, osp.basename(json_path)))

    # 4. 底面标签
    if "bottom" in data_dict:
        bottom_dict = data_dict["bottom"]
        # 若维度不匹配则报错
        if len(bottom_dict) == num_nodes:
            y_bottom = np.zeros(num_nodes, dtype=np.int64)
            for face_idx_str, label_val in bottom_dict.items():
                # 为保证字典中面节点数目等同于由几何体构建的图节点数目
                idx = int(face_idx_str)
                y_bottom[idx] = int(label_val)
        else:
            raise ValueError("Bottom_num in Json is {}, but graph nodes_num is {}, " \
            "Json file is {}".format(len(bottom_dict), num_nodes, osp.basename(json_path)))
            

    return y_semantic, y_instance_adj, y_bottom

def process_single_step_file(args):
    """
    单个文件处理的 Worker 函数。
    Args:
        args: tuple (step_path, json_path, out_path, uv_sample_num)
    Returns:
        tuple (success_bool, message_or_error)
    """
    # 从参数元组中获得各个参数值
    step_path, json_path, out_path, uv_sample_num, shape_norm_method, shape_norm_param, use_log_area, use_log_linear = args
    # 获得该step文件的文件名
    raw_filename = osp.basename(step_path)
    
    try:
        # 1. 读取 STEP 几何
        shape = read_step_file(step_path)
        
        # 2. 构建拓扑图 (G_topo)
        topo_graph = build_topo_graph(shape, sample_num=uv_sample_num, estimate=False, 
                                      method=shape_norm_method, param=shape_norm_param)
        
        # 3. 构建几何图 (G_geom)
        geom_graph, _, _ = build_geom_graph(shape, estimate=False, 
                                            method=shape_norm_method, param=shape_norm_param, 
                                            use_log_area=use_log_area,
                                            use_log_linear=use_log_linear)
        
        # 4. 读取 JSON 标签
        # 利用networkX中的方法
        num_nodes = topo_graph.number_of_nodes()
        y_semantic_label, y_instance_adj, y_bottom_label = load_labels_from_json(json_path, num_nodes)

        # 初始化标签字典
        extra_attributes = {}
        # 将标签数据向字典中填充
        if y_semantic_label is not None:
            extra_attributes['y'] = (torch.long, y_semantic_label)
        else:
            raise ValueError("y_semantic_label is None in {}".format(osp.basename(json_path)))
        if y_instance_adj is not None:
            # 实例矩阵通常作为稀疏边索引存储
            extra_attributes['y_instance_edge_index'] = (torch.long, y_instance_adj)
        else:
            raise ValueError("y_instance_edge_index is None in {}".format(osp.basename(json_path)))
        if y_bottom_label is not None:
            extra_attributes['y_bottom'] = (torch.long, y_bottom_label)
        else:
            raise ValueError("y_bottom is None in {}".format(osp.basename(json_path)))
        
        # 5. 转换为 PyG Data
        converter = DualStreamGraphConverter()
        base_data = converter.convert(topo_graph, geom_graph, extra_attributes=extra_attributes)
            
        # 7. 保存
        # 我们将 base_data 的属性复制到 StepData 中
        # data = StepData()
        # for key, value in base_data:
        #     setattr(data, key, value)
        data = StepData.from_dict(base_data.to_dict())

        # 确保 num_nodes 存在，这对于 __inc__ 很重要
        if not hasattr(data, 'num_nodes') or data.num_nodes is None:
            # 直接从networkX中读取节点数目即可
            # data.num_nodes = base_data.x_geom.shape[0] if hasattr(base_data, 'x_geom') else num_nodes
            data.num_nodes = num_nodes

        torch.save(data, out_path)
        return True, raw_filename
        
    except Exception as e:
        return False, f"{raw_filename}: {str(e)}"

# --- Dataset 类 ---

class StepDataset(Dataset):
    """
    自定义 PyG 数据集，用于加载 STEP 文件并生成双流图数据。
    支持并行处理。
    """
    def __init__(self, root, 
                 raw_dir_name='raw_steps', 
                 label_dir_name='raw_labels',
                 processed_dir_name='processed', 
                 transform=None, pre_transform=None, 
                 uv_sample_num=(5, 5, 5),
                 force_process=False,
                 num_workers=4,

                 shape_norm_method='bbox',
                 shape_norm_param=100.0,
                 use_log_area=False, 
                 use_log_linear=False
                 ): 
        """
        :param root: 数据集根目录
        :param raw_dir_name: 存放 .step/.stp 文件的文件夹名
        :param label_dir_name: 存放 .json 文件的文件夹名
        :param processed_dir_name: 存放处理后 .pt 文件的文件夹名
        :param uv_sample_num: UV 采样参数
        :param force_process: 是否强制重新处理
        :param num_workers: 并行处理的核心数，默认为 4
        """
        self.raw_dir_name = raw_dir_name
        self.label_dir_name = label_dir_name
        self.processed_dir_name = processed_dir_name
        self.uv_sample_num = uv_sample_num
        self.force_process = force_process
        self.num_workers = num_workers # 保存核心数设置
        # 几何体与参数缩放
        self.shape_norm_method = shape_norm_method
        self.shape_norm_param = shape_norm_param
        self.use_log_area = use_log_area
        self.use_log_linear = use_log_linear


        # 缓存文件列表，避免重复 glob
        self._cached_raw_files = None
        # 存储最终有效的 .pt 文件路径
        self.valid_file_paths = []
        # 定义清单文件路径
        self.manifest_path = osp.join(root, processed_dir_name, 'manifest.json')

        # 在调用PyG父类前根据情况将数据清理掉
        if self.force_process:
            self._clear_processed_files(root, processed_dir_name)
        
        super().__init__(root, transform, pre_transform)
        # 初始化完成后，优先从清单加载，否则扫描并生成清单
        self._refresh_file_list()

    def _refresh_file_list(self):
        """
        建立有效文件索引
        策略：
        1. 尝试读取 manifest.json
        2. 如果失败或不存在，扫描 processed 目录下的 .pt 文件
        3. 生成新的 manifest.json
        """
        if not osp.exists(self.processed_dir):
            self.valid_file_paths = []
            return

        # 1. 尝试从清单加载 (极速模式)
        if osp.exists(self.manifest_path) and not self.force_process:
            print(f"正在从清单加载文件列表: {self.manifest_path}")
            try:
                with open(self.manifest_path, 'r') as f:
                    valid_filenames = json.load(f)
                
                # 重构完整路径
                self.valid_file_paths = [osp.join(self.processed_dir, f) for f in valid_filenames]
                print(f"加载完成: 共 {len(self.valid_file_paths)} 个有效样本 (来自文本文件).")
                return
            except Exception as e:
                print(f"读取清单文件失败 ({e})，转为重新扫描...")

        # 2. 扫描硬盘 (首次运行模式)
        print("正在扫描 processed 目录以生成有效文件清单...")
        files = glob.glob(osp.join(self.processed_dir, 'data_*.pt'))
        files.sort() # 确保确定性顺序
        
        self.valid_file_paths = files
        print(f"扫描完成: 共找到 {len(self.valid_file_paths)} 个有效样本.")

        # 3. 保存清单 (为下次启动加速)
        try:
            valid_filenames = [osp.basename(f) for f in self.valid_file_paths]
            with open(self.manifest_path, 'w') as f:
                json.dump(valid_filenames, f, indent=2)
            print(f"已生成新的清单文件: {self.manifest_path}")
        except Exception as e:
            print(f"警告: 无法保存清单文件: {e}")


    def _clear_processed_files(self, root, processed_dir_name):
        """清理已处理的文件，强制重新生成"""
        processed_dir = osp.join(root, processed_dir_name)
        if osp.exists(processed_dir):
            print(f"检测到强制更新请求, 正在清理 {processed_dir} 下的旧文件...")
            # 仅删除以 data_ 开头的 .pt 文件, 避免误删其他文件
            files = glob.glob(osp.join(processed_dir, 'data_*.pt'))
            for f in files:
                try:
                    os.remove(f)
                except OSError as e:
                    print(f"无法删除文件 {f}: {e}")

    @property
    def raw_file_names(self):
        # 使用缓存，避免每次属性访问都扫描硬盘
        if self._cached_raw_files is not None:
            return self._cached_raw_files
        # 查找 raw 目录下的所有 step 文件
        step_files = glob.glob(osp.join(self.raw_dir, '*.step')) + \
                     glob.glob(osp.join(self.raw_dir, '*.stp'))
        # 进行排序, 以保证顺序的一致性
        step_files.sort()

        self._cached_raw_files = [osp.basename(f) for f in step_files]
        return self._cached_raw_files
    

    @property
    def processed_file_names(self):
        # [关键修复] 如果清单文件存在且不强制更新，以清单为准
        # 这防止 PyG 发现文件缺失（处理失败的样本）而触发 process()
        if osp.exists(self.manifest_path) and not self.force_process:
            try:
                with open(self.manifest_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass

        # 如果没有清单（首次运行）或强制更新，默认期望所有原始文件都应被处理
        # PyG 会对比这里返回的列表和实际存在的文件，如果有缺失，就会调用 process()
        return [f'data_{osp.splitext(f)[0]}.pt' for f in self.raw_file_names]
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.raw_dir_name)
    
    @property
    def label_dir(self) -> str:
        return osp.join(self.root, self.label_dir_name)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.processed_dir_name)

    def process(self):
        """
        并行处理逻辑
        """

        # 这里的 self.raw_file_names 会触发一次 glob，但后续不会再触发
        all_raw_files = self.raw_file_names
        
        if len(all_raw_files) == 0:
            print("未找到原始数据文件。")
            return

        print(f"扫描完成，共找到 {len(all_raw_files)} 个原始文件。正在准备任务...")

        # 准备任务列表
        tasks = []
        
        print("正在扫描文件列表...")
        for idx, raw_filename in enumerate(self.raw_file_names):
            # 直接通过字符串生成输出路径
            out_filename = f'data_{osp.splitext(raw_filename)[0]}.pt'
            out_path = osp.join(self.processed_dir, out_filename)
            
            # 如果不强制处理且文件已存在，则跳过
            if not self.force_process and osp.exists(out_path):
                continue
                
            step_path = osp.join(self.raw_dir, raw_filename)
            json_filename = osp.splitext(raw_filename)[0] + ".json"
            json_path = osp.join(self.label_dir, json_filename)
            
            # 将任务参数打包
            tasks.append((step_path, json_path, out_path, self.uv_sample_num, 
                          self.shape_norm_method, self.shape_norm_param, 
                          self.use_log_area, self.use_log_linear))

        if len(tasks) == 0:
            print("所有数据已处理完毕，无需转换。")
            return

        print(f"准备开始处理 {len(tasks)} 个文件，使用 {self.num_workers} 个核心并行计算...")
        
        # 使用 multiprocessing.Pool 进行并行处理
        # 建议 num_workers 不要超过 CPU 物理核心数
        with Pool(processes=self.num_workers) as pool:
            # 使用 imap_unordered 可以让进度条更平滑地更新
            # chunksize 设置稍大一点可以减少进程间通信开销
            chunk_size = max(1, len(tasks) // (self.num_workers * 256))
            results = list(tqdm(pool.imap(process_single_step_file, tasks, chunksize=chunk_size), 
                                total=len(tasks), unit="file"))

        # 统计结果
        success_count = sum(1 for r in results if r[0])
        failures = [r[1] for r in results if not r[0]]
        
        print(f"\n处理完成: 成功 {success_count} 个, 失败 {len(failures)} 个")
        if failures:
            print("失败文件列表 (前10个):")
            for msg in failures[:10]:
                print(msg)
            if len(failures) > 10:
                print("...")

    # len() 依赖有效文件清单，O(1) 复杂度
    def len(self):
        return len(self.valid_file_paths)
    # get() 依赖有效文件清单，O(1) 复杂度
    def get(self, idx):
        return torch.load(self.valid_file_paths[idx])
        
