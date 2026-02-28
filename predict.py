import os
# [Debug] 开启同步模式，让报错指向真正的行数
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import json
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from torch_geometric.data import Dataset, Batch
from torch_geometric.utils import to_dense_adj

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from scipy.sparse import csr_matrix, coo_matrix, issparse
from scipy.sparse.csgraph import connected_components
from collections import defaultdict, Counter

from dataset.step_dataset_predict import single_step_to_data
from models.dual_stream_net import DualStreamNet


class FeatureParser_old:
    """
    加工特征解析器
    用于将神经网络的原始输出 (Sem, Inst Matrix, Bot) 解析为结构化的加工特征实例。
    """
    def __init__(self, background_label=None):
        """
        :param background_label: 如果存在背景类(非特征面), 可在此指定ID, 解析时会自动过滤。
        """
        self.background_label = background_label

    def parse(self, sem_pred, inst_pred_matrix, bot_pred):
        """
        解析预测结果
        
        :param sem_pred: [N] 语义分割预测结果 (int array)
        :param inst_pred_matrix: [N, N] 实例分割邻接矩阵。
                                 支持类型: 
                                 1. numpy.ndarray (bool 或 int, 0/1)
                                 2. scipy.sparse matrix
                                 注意: 矩阵应当是对称的, 表示面之间的连通性。
        :param bot_pred: [N] 底面预测结果 (0/1 array)
        :return: 结构化的大字典 result_dict
        """
        # 1. 基础信息获取
        num_faces = len(sem_pred)
        
        # 2. 构建图并求解连通分量 (Instance Clustering)
        # 优化: 直接利用网络输出的 N*N 矩阵，跳过转换为 pairs 的步骤
        
        adj_matrix = None
        
        if issparse(inst_pred_matrix):
            # Case A: 输入已经是 scipy 稀疏矩阵
            adj_matrix = inst_pred_matrix
        elif isinstance(inst_pred_matrix, np.ndarray):
            # Case B: 输入是 Numpy 密集矩阵 (N*N Bool/Int)
            # csr_matrix 会自动忽略 0/False 元素，高效构建稀疏结构
            adj_matrix = csr_matrix(inst_pred_matrix)
        else:
            raise TypeError(f"不支持的输入类型: {type(inst_pred_matrix)}。请传入 numpy array 或 scipy sparse matrix。")

        # 维度安全检查
        if adj_matrix.shape[0] != num_faces:
             raise ValueError(f"维度不匹配: sem_pred 有 {num_faces} 个面, 但 inst_pred_matrix 形状为 {adj_matrix.shape}")

        # connected_components 返回:
        # n_components: 连通分量个数
        # labels: [N] 数组, labels[i] 表示第 i 个面所属的组件ID
        # directed=False 表示视作无向图 (A连B 等同于 B连A)
        n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)

        # 3. 按组件ID将面聚合
        instances_raw = defaultdict(list)
        for face_id, comp_id in enumerate(labels):
            instances_raw[comp_id].append(face_id)

        # 4. 构建结构化输出
        # 格式: { feature_type_id: [ {instance_dict}, ... ] }
        final_output = defaultdict(list)

        for comp_id, faces_list in instances_raw.items():
            # 将 list 转为 tuple，方便后续处理
            faces_tuple = tuple(sorted(faces_list))

            # --- A. 确定特征类型 (Majority Voting) ---
            # 获取该实例所有面的预测类别
            current_sem_preds = sem_pred[faces_list]
            
            if len(current_sem_preds) > 0:
                # 统计众数 (出现最多的类别)
                pred_type = Counter(current_sem_preds).most_common(1)[0][0]
            else:
                continue

            # 如果是背景类，则跳过
            if self.background_label is not None and pred_type == self.background_label:
                continue

            # --- B. 确定加工底面 (Intersection) ---
            # 在当前实例的面中，寻找 bot_pred == 1 的面
            # 这里的 bot_pred 也是对应所有面的数组
            bot_faces = [fid for fid in faces_list if bot_pred[fid] == 1]
            bot_tuple = tuple(sorted(bot_faces))

            # --- C. 组装实例字典 ---
            instance_dict = {
                'feature_type': int(pred_type),      # 特征类别
                'feature_composition': faces_tuple,  # 构成面
                'bot': bot_tuple                     # 加工底面
            }

            final_output[int(pred_type)].append(instance_dict)

        return dict(final_output)



class FeatureParser:
    """
    加工特征解析器 (V2 - 语义一致性增强版)
    改进策略: 
    不再仅仅依赖 Inst 进行聚类，而是引入 "语义一致性约束"。
    两个面必须满足: (Inst预测相连) AND (Sem预测类别相同) 才能被归为一个实例。
    """
    def __init__(self, background_label=None):
        """
        :param background_label: 如果存在背景类(非特征面), 可在此指定ID, 解析时会自动过滤。
        """
        self.background_label = background_label

    def parse(self, sem_pred, inst_pred_matrix, bot_pred):
        """
        解析预测结果
        
        :param sem_pred: [N] 语义分割预测结果 (int array)
        :param inst_pred_matrix: [N, N] 实例分割邻接矩阵 (Numpy 或 Sparse)
        :param bot_pred: [N] 底面预测结果 (0/1 array)
        :return: 结构化的大字典 result_dict
        """
        num_faces = len(sem_pred)
        
        # 1. 标准化输入矩阵为 CO0 格式 (Coordinate format)
        # COO 格式非常适合进行基于坐标 (row, col) 的快速过滤
        if issparse(inst_pred_matrix):
            # 转换为 coo_matrix 以方便获取 row, col 数组
            adj_coo = inst_pred_matrix.tocoo()
        elif isinstance(inst_pred_matrix, np.ndarray):
            # Numpy 密集矩阵 -> 稀疏矩阵
            adj_coo = coo_matrix(inst_pred_matrix)
        else:
            raise TypeError(f"不支持的输入类型: {type(inst_pred_matrix)}")

        # 维度检查
        if adj_coo.shape[0] != num_faces:
             raise ValueError(f"维度不匹配: sem_pred({num_faces}) vs inst_matrix{adj_coo.shape}")

        # 2. [核心改进] 应用语义一致性过滤 (Semantic Consistency Filtering)
        # 获取所有预测存在连接的边 (u, v)
        rows = adj_coo.row
        cols = adj_coo.col
        data = adj_coo.data

        # 向量化检查: 只有当 sem_pred[u] == sem_pred[v] 时，连接才有效
        # sem_pred[rows] 获取每条边起点的类别
        # sem_pred[cols] 获取每条边终点的类别
        consistency_mask = (sem_pred[rows] == sem_pred[cols])

        # 过滤边: 只保留 mask 为 True 的边
        new_rows = rows[consistency_mask]
        new_cols = cols[consistency_mask]
        new_data = data[consistency_mask]

        # 重建经过语义清洗的邻接矩阵 (CSR格式适合连通分量计算)
        # shape 必须保持 (N, N) 以防某些节点变得孤立
        refined_adj_matrix = csr_matrix(
            (new_data, (new_rows, new_cols)), 
            shape=(num_faces, num_faces)
        )

        # 3. 求解连通分量 (Instance Clustering)
        # 现在，不同类别的特征即使 Inst 预测相连，也会因为 Sem 不同而被切断
        n_components, labels = connected_components(
            csgraph=refined_adj_matrix, 
            directed=False, 
            return_labels=True
        )

        # 4. 按组件ID聚合 (与之前逻辑一致)
        instances_raw = defaultdict(list)
        for face_id, comp_id in enumerate(labels):
            instances_raw[comp_id].append(face_id)

        # 5. 构建结构化输出
        final_output = defaultdict(list)

        for comp_id, faces_list in instances_raw.items():
            faces_tuple = tuple(sorted(faces_list))
            
            # --- A. 确定特征类型 ---
            # 由于我们强制了语义一致性，理论上一个实例内的所有面 sem 应该是一样的。
            # 但考虑到孤立点或自环的情况，还是取一下第一个面的 sem 即可。
            # (为了代码极其稳健，这里依然保留众数投票，防止极端异常)
            current_sem_preds = sem_pred[faces_list]
            if len(current_sem_preds) == 0: continue
            
            # 实际上现在 current_sem_preds 里面应该全是同一个值 (除非该连通分量是单点且无自环)
            pred_type = Counter(current_sem_preds).most_common(1)[0][0]

            # 背景类过滤
            if self.background_label is not None and pred_type == self.background_label:
                continue
            
            # 噪点过滤 (可选): 如果只有1个面且不是特定的单面特征，可以视情况过滤
            # if len(faces_list) < 2: continue 

            # --- B. 确定加工底面 ---
            bot_faces = [fid for fid in faces_list if bot_pred[fid] == 1]
            bot_tuple = tuple(sorted(bot_faces))

            # --- C. 组装 ---
            instance_dict = {
                'feature_type': int(pred_type),
                'feature_composition': faces_tuple,
                'bot': bot_tuple
            }

            final_output[int(pred_type)].append(instance_dict)

        return dict(final_output)




class Predict:

    def __init__(self, model_param_path, config, config_path=None):

        # 模型参数地址
        self.model_param_path = model_param_path
        # 模型原始设定字典地址
        self.config_path = config_path
        # 模型原始设定字典
        self.config = self._get_config(config_path) if config_path is not None else config
        config_add = config_add = {
        "shape_norm_method": "bbox",
        "shape_norm_param": 100.0,
        "use_log_area": True,
        "use_log_linear": False,
        "background_label":24
    }

        self.config = {**self.config, **config_add}
        # 构造模型
        self.model = self._load_model()
        # 初始化解析器
        self.parser = FeatureParser(self.config.get("background_label", 25))
        # inst判断参数
        self.inst_thres = 0.5
        # bot判断参数
        self.bot_thres = 0.5
    
    def _get_config(self):
        """
        用于从config文件的地址中解析config
        """
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_model(self):
        """
        根据模型参数以及模型的原始设定构建网络
        """
        # 读取device
        device = self.config['device']
        # 加载模型
        model_pred = DualStreamNet(self.config["model"])
        model_pred.to(device)
        model_pred.load_state_dict(torch.load(self.model_param_path))
        
        # 将模型设置为评估模式
        return model_pred.eval()
    
    def _trans_single_step_to_data(self, step_path):
        """
        用于从step路径读取并将其转化为data
        """
        uv_sample_num = self.config['uv_sample']
        shape_norm_method = self.config['shape_norm_method']
        shape_norm_param = self.config['shape_norm_param']
        use_log_area = self.config['use_log_area']
        use_log_linear = self.config['use_log_linear']

        step_data = single_step_to_data(step_path=step_path,
                                        uv_sample_num=uv_sample_num,
                                        shape_norm_method=shape_norm_method,
                                        shape_norm_param=shape_norm_param,
                                        use_log_area=use_log_area,
                                        use_log_linear=use_log_linear)
        if step_data:
            batch_data = Batch.from_data_list([step_data])
            batch_data.to(self.config['device'])

            return batch_data
        else:
            raise ValueError(f"Failed to trans: {step_path}, single_step_to_data return None")
        
    def predict_main_single(self, step_path):
        """
        用于预测单一文件
        """
        # 获得转化的data
        batch_data = self._trans_single_step_to_data(step_path)
        # 预测
        with torch.no_grad():
            # # 开启 autocast 可以加速并节省显存 (需与 device 一致)
            # device_type = 'cuda' if 'cuda' in str(self.config['device']) else 'cpu'
            # # cpu autocast 需要 PyTorch 1.10+，如果是 cuda 则通常没问题
            # if device_type == 'cuda':
            #     with autocast():
            #         pred = self.model(batch_data)
            # else:
            #     pred = self.model(batch_data)
            pred = self.model(batch_data)
        
        if pred:
            return self._pred_interpreter(pred, batch_data)
        else:
            raise ValueError(f"Failed to predict: {step_path}, DualGNN return None")
        
    def _inst_reshape(self, inst_np):
        """
        inst_reshape 的 Docstring
        
        :param self: 说明
        """
        total_num = inst_np.size
        n = int(np.sqrt(total_num))
        # 计算n并校验是否为完全平方数
        # 即校验inst是否为方阵
        if n * n != total_num:
            raise ValueError(f"The length of the array {total_num} is not a perfect square and cannot be converted to a square matrix. ")
        
        return inst_np.reshape(n, n)
    
    def _inst_extract(self, inst_np):
        """
        _inst_extract 的 Docstring
        
        :param self: 说明
        :param inst_np: 说明
        """
        # 将其转化为方阵
        inst_mat = self._inst_reshape(inst_np)
        # np.where返回True的行/列索引
        rows, cols = np.where(inst_mat)
        # 返回二维数组
        corr_pairs = np.stack([rows, cols], axis=1)

        return corr_pairs


    def _pred_interpreter(self, pred, batch_data):
        """
        用于解释预测的结果, 以将其转化为符合需求的类型
        """
        # 获得预测结果
        sem_out, inst_matrix, inst_mask, bottom_out = pred
        # sem 通常是 Logits [Num_nodes, Num_classes], 需要取 argmax 得到类别
        sem = torch.argmax(sem_out, dim=1).cpu().numpy()
        # 构建预测结果字典
        p_dict = {'sem': sem}

        # 将inst的预测结果写入
        if inst_mask is not None:
            mask_bool = inst_mask.bool()
            inst_logits = inst_matrix[mask_bool].sigmoid()
            inst = inst_logits > self.inst_thres
            inst = inst.cpu().numpy().astype('int32')
            p_dict['inst'] = self._inst_reshape(inst)
        else:
            inst_logits = inst_matrix.sigmoid()
            inst = inst_logits > self.inst_thres
            inst = inst.cpu().numpy().astype('int32')
            p_dict['inst'] = self._inst_reshape(inst)

        # 将bottom的预测结果写入
        bottom_logits = bottom_out.sigmoid().flatten()
        bottom = bottom_logits > self.bot_thres
        bottom = bottom.cpu().numpy().astype('int32')

        p_dict['bot'] = bottom

        print(p_dict['sem'])

        # 进一步解释
        result = self.parser.parse(p_dict['sem'], p_dict['inst'], p_dict['bot'])

        return result



if __name__ == '__main__':

    step_path = r"E:\PYTHON\Final_MFR\MFR_DualGNN_Performace_test\MFR_DualGNN_MFTRCAD\data\steps_new\20240116_231044_55_result.step"
    model_param_path = r"E:\PYTHON\Final_MFR\MFR_DualGNN_Performace_test\MFR_DualGNN_MFTRCAD\training_log\Feb13_16-11-31_88.09\best_val_sem_acc-88.10.pth"
    config_path = r"E:\PYTHON\Final_MFR\MFR_DualGNN_Performace_test\MFR_DualGNN_MFTRCAD\training_log\Feb13_16-11-31_88.09\config.json"

    # 读取config文件
    with open(config_path, "r", encoding="utf-8") as f:
        config_raw = json.load(f)
    
    # config中需补充的内容
    config_add = {
        "shape_norm_method": "bbox",
        "shape_norm_param": 100.0,
        "use_log_area": True,
        "use_log_linear": False,
        "background_label":25
    }

    config = {**config_raw, **config_add}

    predict_model = Predict(model_param_path, config=config)

    pred_result = predict_model.predict_main_single(step_path)

    print(pred_result)