import math
import functools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import os
import sys

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.GeomAbs import GeomAbs_SurfaceType
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.TopExp import topexp
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Pnt
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepGProp import brepgprop

from occwl.uvgrid import ugrid, uvgrid
from occwl.edge import Edge
from occwl.face import Face

# 获取当前脚本所在目录 (datasets/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (MFR_DualGNN/)
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 python 路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.geom_embedding import Geom_embedding_face
from preprocessing.position_face import Position_Descriptor_Surface, Surface_Relationship_Analyzer


def read_step_file(file_path):
    """
    读取 STEP 文件并返回几何形状。

    :param file_path: STEP 文件的路径
    :return: 几何形状对象
    """
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(file_path)
    if status == 1:
        step_reader.TransferRoots()
        shape = step_reader.Shape()
        
        return shape
    else:
        raise ValueError("Error reading STEP file")
    
@functools.lru_cache(maxsize=1, typed=False)
def step_topo_mapping(shape, mapped_topo_type):
    """
    读取shape获取该对象的映射关系
    给一个内存管理器, 用于存储映射关系
    
    param shape: TopoDS_Solid型的shape
    param mapped_topo_type: 映射的拓扑类型
    return: 映射关系字典
    """
    # 新建映射关系字典
    mapping_dict = {}
    # 将shape转为拓扑几何解析类
    explorer = TopologyExplorer(shape)

    # 获取映射的拓扑类型
    if mapped_topo_type == "edge":
        # 获取边列表
        edges = list(explorer.edges())
        # 创建映射关系
        edge_count = 0
        for edge in edges:
            mapping_dict[edge.HashCode(999999)] = edge_count
            edge_count = edge_count + 1
        
    elif mapped_topo_type == "face":
        # 获取面列表
        faces = list(explorer.faces())
        # 创建映射关系
        face_count = 0
        for face in faces:
            mapping_dict[face.HashCode(999999)] = face_count
            face_count = face_count + 1
    else:
        raise ValueError("The param: mapped_topo_type is illegal")

def get_combinations(lst):
    """
    读取一个列表内的元素并返回由其中所有元素两两排列组合形成的新列表。

    :param list 列表
    :return: list 两两组合后的多维列表
    """
    # 使用itertools.combinations生成所有两两组合
    combinations = list(itertools.combinations(lst, 2))
    return combinations

def _get_edge_endpoints(edge):
    """辅助函数: 获取边的起点和终点坐标(几何意义上的, 考虑了Orientation)"""
    # BRep_Tool.Curve 不考虑 Edge 的 Orientation
    # topexp.FirstVertex 和 LastVertex 考虑了 Orientation
    v_start = topexp.FirstVertex(edge)
    v_end = topexp.LastVertex(edge)
    
    p_start = BRep_Tool.Pnt(v_start)
    p_end = BRep_Tool.Pnt(v_end)
    
    return p_start, p_end

def _get_shape_bbox_param(shape):
    """
    _get_shape_bbox_param 的 Docstring
    
    :param shape: Topo_DS_shape
    :return: 包围盒对角线坐标值元组, 包围盒中心坐标, 包围盒对角线长度
    """
    # 实例化包围盒类
    bbox = Bnd_Box()
    # 计算原始shape的轴对齐包围盒, 并将结果填充到bbox对象中
    brepbndlib_Add(shape, bbox)
    # 获得包围盒对角线坐标
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    # 检查包围盒是否有效
    if bbox.IsVoid():
        raise ValueError("This shape is Void.")
    
    # 计算包围盒尺寸与中心
    current_diag = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2 + (zmax - zmin)**2)
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    center_z = (zmin + zmax) / 2.0
    current_center = gp_Pnt(center_x, center_y, center_z)

    # 整理返回值
    bbox_diag_pnt = (xmin, ymin, zmin, xmax, ymax, zmax)
    return bbox_diag_pnt, current_center, current_diag
        
def _get_surface_area_statistics(shape):
    """
    _get_surface_area_statistics 的 Docstring
    
    :param shape: 原几何体的Topo_DS_shape
    :return tuple: (面积最大值, 面积最小值, 面积均值, 面积标准差)
    """
    # 按照OCC的遍历顺序遍历面并计算面积
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    props = GProp_GProps() # 实例化一次，重复使用，减少内存开销
    
    has_faces = False
    face_area = []
    face_id = 0
    # 构建面迭代器以统计最大面
    # 该方法不需要将面全都储存在函数中, 因此效果会高一些
    while explorer.More():
        face = explorer.Current()
        face_id = face_id + 1
        # 计算当前面属性
        try:
            brepgprop.SurfaceProperties(face, props)
            current_area = props.Mass()
            face_area.append(current_area)
            has_faces = True
        except Exception as e:
            print(f"Get surface area error: {e}")
        explorer.Next()
    # 若几何体中不存在面, 则报错
    if not has_faces:
        raise ValueError("Normalize Shape Error: Input shape has no faces.")
    # 将面面积列表转化为np数组
    area_arr = arr = np.asarray(face_area, dtype=np.float32)
    # 计算并返回统计学参数
    # 最大值 / 最小值 / 平均值 / 均方差
    return (np.max(arr).item(), np.min(arr).item(), np.mean(arr).item(), np.std(arr, ddof=0).item())

def _normalize_shape_bbox(shape, target_diagonal=100):
    """
    将几何体缩放至标准尺寸
    
    :param shape: 原始 TopoDS_Shape
    :param target_diagonal: 目标对角线比例
    :return shape: 缩放后的 TopoDS_Shape
    """
    # 实例化包围盒类
    bbox = Bnd_Box()
    # 计算shape的轴对齐包围盒, 并将结果填充到bbox对象中
    brepbndlib_Add(shape, bbox)
    # 获得包围盒对角线坐标
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    # 获得包围盒对角线长度
    current_diag = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2 + (zmax - zmin)**2)
    # 防止除以零或处理极小物体
    if current_diag < 1e-4:
        print(f"警告: 模型极小 (diag={current_diag}), 跳过缩放")
        return shape
    # 计算缩放比例
    scale_factor = target_diagonal / current_diag
    # 若当前几何体对角线尺寸与目标尺寸一致, 则跳过处理
    if abs(scale_factor - 1.0) < 0.1:
        return shape
    # 计算当前几何体的中心坐标
    # 缩放将以其中心坐标为原点
    center = gp_Pnt((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
    trsf = gp_Trsf()
    trsf.SetScale(center, scale_factor)
    # 执行变换
    transformer = BRepBuilderAPI_Transform(shape, trsf)
    return transformer.Shape()

def _normalize_shape_max_area(shape, target_max_area=100):
    """
    将几何体通过将最大面缩放至标准面积来实现整体缩放
    
    :param shape: 原始 TopoDS_Shape
    :param max_area: 缩放后的最大面面积
    :return shape: 缩放后的TopoDS_Shape
    """
    # 按照OCC的遍历顺序遍历面并计算面积
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    props = GProp_GProps() # 实例化一次，重复使用，减少内存开销
    
    max_area_original = 0.0
    has_faces = False
    # 构建面迭代器以统计最大面
    # 该方法不需要将面全都储存在函数中, 因此效果会高一些
    while explorer.More():
        face = explorer.Current()
        # 计算当前面属性
        brepgprop.SurfaceProperties(face, props)
        current_area = props.Mass()
        
        if current_area > max_area_original:
            max_area_original = current_area
        
        has_faces = True
        explorer.Next()
    # 若几何体中不存在面, 则报错
    if not has_faces:
        raise ValueError("Normalize Shape Error: Input shape has no faces.")
    # 对于最大面积仍是很小的面, 同样报错, 这说明该几何体大概存在退化情况
    if max_area_original < 1e-9:
        raise ValueError(f"Normalize Shape Error: Max face area is too small ({max_area_original}).")
    # 计算缩放比例, 面积比是长度比的平方, 因此长度缩放因子是面积比的平方根
    scale_factor = math.sqrt(target_max_area / max_area_original)
    # 若比例已经很接近, 则直接返回以节省计算
    if abs(scale_factor - 1.0) < 1e-6:
        return shape
    
    # 获得几何体中心以给予缩放
    # 实例化包围盒类
    bbox = Bnd_Box()
    # 计算shape的轴对齐包围盒, 并将结果填充到bbox对象中
    brepbndlib_Add(shape, bbox)
    # 若包围盒为空, 则报错
    if bbox.IsVoid():
         raise ValueError("Normalize Shape Error: Bounding box is void.")
    # 获得包围盒对角线坐标
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    # 计算当前几何体的中心坐标
    # 缩放将以其中心坐标为原点
    center = gp_Pnt((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
    trsf = gp_Trsf()
    trsf.SetScale(center, scale_factor)
    # 执行变换
    transformer = BRepBuilderAPI_Transform(shape, trsf)
    return transformer.Shape()

def _normalize_shape_min_area(shape, target_min_area=(math.e - 1.0)):
    """
    将几何体通过将最小面缩放至标准面积来实现整体缩放
    
    :param shape: 原始 TopoDS_Shape
    :param target_min_area: 缩放后的最小面面积
    :return shape: 缩放后的TopoDS_Shape
    """
    # 按照OCC的遍历顺序遍历面并计算面积
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    props = GProp_GProps() # 实例化一次，重复使用，减少内存开销
    
    min_area_original = 1e12
    has_faces = False
    # 构建面迭代器以统计最大面
    # 该方法不需要将面全都储存在函数中, 因此效果会高一些
    while explorer.More():
        face = explorer.Current()
        # 计算当前面属性
        brepgprop.SurfaceProperties(face, props)
        current_area = props.Mass()
        
        if current_area < min_area_original:
            min_area_original = current_area
        
        has_faces = True
        explorer.Next()
    # 若几何体中不存在面, 则报错
    if not has_faces:
        raise ValueError("Normalize Shape Error: Input shape has no faces.")
    # 对于最大面积仍是很小的面, 同样报错, 这说明该几何体大概存在退化情况
    if min_area_original < 1e-9:
        raise ValueError(f"Normalize Shape Error: Max face area is too small ({min_area_original}).")
    # 计算缩放比例, 面积比是长度比的平方, 因此长度缩放因子是面积比的平方根
    scale_factor = math.sqrt(target_min_area / min_area_original)
    # 若比例已经很接近, 则直接返回以节省计算
    if abs(scale_factor - 1.0) < 1e-6:
        return shape
    
    # 获得几何体中心以给予缩放
    # 实例化包围盒类
    bbox = Bnd_Box()
    # 计算shape的轴对齐包围盒, 并将结果填充到bbox对象中
    brepbndlib_Add(shape, bbox)
    # 若包围盒为空, 则报错
    if bbox.IsVoid():
         raise ValueError("Normalize Shape Error: Bounding box is void.")
    # 获得包围盒对角线坐标
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    # 计算当前几何体的中心坐标
    # 缩放将以其中心坐标为原点
    center = gp_Pnt((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
    trsf = gp_Trsf()
    trsf.SetScale(center, scale_factor)
    # 执行变换
    transformer = BRepBuilderAPI_Transform(shape, trsf)
    return transformer.Shape()

def normalize_shape_old(shape, method='bbox', param=None):
    """
    --该方法已过时--
    normalize_shape 用于统一整合几何体尺寸变换
    
    :param shape: 说明
    :param method: 说明
    :param param: 说明
    """
    # 无参数即按默认参数处理
    if param == None:
        if method == 'bbox':
            return _normalize_shape_bbox(shape)
        elif method == 'max_area':
            return _normalize_shape_max_area(shape)
        elif method == 'min_area':
            return _normalize_shape_min_area(shape)
        elif method == 'original':
            return shape
        else:
            raise ValueError(f"{method} is not defined in normalize_shape")
    else:
        if method == 'bbox':
            return _normalize_shape_bbox(shape, param)
        elif method == 'max_area':
            return _normalize_shape_max_area(shape, param)
        elif method == 'min_area':
            return _normalize_shape_min_area(shape, param)
        elif method == 'original':
            return shape
        else:
            raise ValueError(f"{method} is not defined in normalize_shape")

# 该函数用于控制对几何体的缩放
# original / bbox / max_area / min_area
def normalize_shape(shape, method='bbox', param=None):
    """
    将几何体按指定方法缩放致标准尺寸, 并将其包围盒中心平移至原点

    :param shape: 说明
    :param method: 说明
    :param param: 说明
    """
    # 参数预定义
    predefined_param = {'original': 1.0,
                        'bbox': 100.0,
                        'max_area': 100,
                        'min_area': math.e - 1.0}
    if param == None:
        param = predefined_param[method]
    # 计算对角线尺寸与中心坐标
    _, bbox_center, bbox_diag = _get_shape_bbox_param(shape)
    # 构建变换矩阵
    trsf_translate = gp_Trsf()
    # 执行平行矩阵
    vec_to_origin = gp_Vec(bbox_center, gp_Pnt(0, 0, 0))
    # 将当前几何体包围盒中心调整到坐标原点
    trsf_translate.SetTranslation(vec_to_origin)

    # 计算缩放值
    scale_factor = 1.0
    if method == 'bbox':
        if bbox_diag < 1e-6:
            print(f"警告: 模型极小 (diag={bbox_diag}), 跳过缩放")
            return shape
        # 计算缩放比例
        scale_factor = param / bbox_diag
    elif method == 'max_area':
        # 获得几何体面积统计学数据
        area_statistics = _get_surface_area_statistics(shape)
        # 计算缩放比例
        if area_statistics[0] < 1e-9:
            raise ValueError(f"ERROR: 最大面面积极小 ({area_statistics[0]})")
        else:
            scale_factor = math.sqrt(param / area_statistics[0])
    elif method == 'min_area':
        # 获得几何体面积统计学数据
        area_statistics = _get_surface_area_statistics(shape)
        # 计算缩放比例
        if area_statistics[1] < 1e-9:
            raise ValueError(f"ERROR: 最大面面积极小 ({area_statistics[1]})")
        else:
            scale_factor = math.sqrt(param / area_statistics[1])
    elif method == 'original':
        scale_factor = 1.0
    else:
        pass

    # 构建缩放变换矩阵
    trsf_scale = gp_Trsf()
    trsf_scale.SetScale(gp_Pnt(0, 0, 0), scale_factor)

    # 计算整体变换矩阵
    final_trsf = trsf_scale.Multiplied(trsf_translate)
    # 执行变换
    # 检查整体变换矩阵是否为单位阵
    is_identity = (abs(final_trsf.ScaleFactor() - 1.0) < 1e-6) and \
                  (max(abs(final_trsf.TranslationPart().X()), 
                       abs(final_trsf.TranslationPart().Y()), 
                       abs(final_trsf.TranslationPart().Z())) < 1e-6)
    # 即变换阵并非单位阵, 执行变换
    if not is_identity:
        try:
            transformer = BRepBuilderAPI_Transform(shape, final_trsf, True)
            # Copy=False 表示尽可能修改原对象(虽然 shape 本身不可变，但这通常更高效)
            # 不过为了安全，BRepBuilderAPI_Transform 默认会生成新的 Shape
            return transformer.Shape()
        except Exception as e:
            print(f"变换失败: {e}")
            return shape
        
    return shape

def validate_shape_integrity(shape):
    """
    检查几何体是否健康
    1. 如果有任何面面积为负或极小，视为模型损坏
    
    :return: True if valid, raises ValueError if invalid
    """
    # 判断参数初始化
    min_area=0.0

    # 初始化健康标签
    integrity = True
    # 遍历几何面
    # 按照OCC的遍历顺序遍历面并计算面积
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    # 实例化一次，重复使用，减少内存开销
    props = GProp_GProps()
    # 构建面迭代器以计算面积
    # 该方法不需要将面全都储存在函数中, 因此效果会高一些
    # 初始化计数器
    idx = 0
    while explorer.More():
        face = explorer.Current()

        # 1. 若存在面面积为极小, 视为模型损坏
        # 尝试对面面积进行判断
        try:
            brepgprop_SurfaceProperties(face, props)
            area = props.Mass()
            if area <= min_area:
                integrity = False
                raise ValueError(f"Invalid Face detected at index {idx}: Area = {area} (<= {min_area})")
        except Exception as e:
            raise ValueError(f"Geometry error at index {idx}: {str(e)}")
        # 计数器累加
        idx = idx + 1

        explorer.Next()

    return integrity

def _sort_edges_geometrically(edge_list):
    """
    [增强版] 将一组无序的边排序成几何连续的链，并确定每条边的采样方向。
    
    策略升级：
    不再只构建一条链就停止, 而是提取所有可能的连通分量(Chains),
    并返回总几何长度最长的那一条链
    这有效解决了: 
    1. 两个半圆柱拼接导致的双缝合线问题(自动选择其中一条)
    2. 只有部分边能连接的情况(自动抛弃短的噪声链)
    
    返回:
        List[Tuple(TopoDS_Edge, bool)]: 最长链的排序结果
    """
    if not edge_list:
        return []
    
    # 使用 Edge 包装器计算长度的辅助函数
    def get_len(e):
        try: return Edge(e).length()
        except: return 0.0

    # 待处理池
    edge_pool = list(edge_list)
    
    # 存储所有发现的链：List of (total_length, chain_list)
    all_chains = []
    # 第一循环, 每次循环将产生一个连通分量
    # 在最外层的while循环里面, 我们构建了一个新的分量 current_chain
    # 并在其中增添了边列表中的第一条边作为种子边。之后我们进入了第二个while循环中.
    # 在其中根据初始的参数, 我们在for循环中寻找能够与种子边首尾相连的边.
    # 当for循环结束后, 我们找到了根据第一条边为种子的下一条边链, 而在判定连接阈值里面将该边清除
    # 并根据方向记录该边的末尾点(即下一个遍历的起点).
    # 第二个while循环再次开始, 根据新的点寻找列表中最近的边, 直到没有找到(found_next = False).
    # 此时第二个while终止, 开始第一个while的下一次循环.
    # 之后的第一条边由于是从剩下的列表中选择的第一位, 因此不会再出现之前已经分析过的边
    # 相当于从剩下的边中再次排序, 直到edge_pool为空.
    current_chain_dict = {}
    while edge_pool:
        # --- 开始构建一个新的连通分量 ---
        current_chain = []
        current_chain_length = 0.0
        
        # 种子边
        # 每选择一个种子后将该种子从候选列表中删除以避免被重复选择
        first_edge = edge_pool.pop(0)
        edeg_length = get_len(first_edge)
        # 将种子边存入连通分量中
        current_chain.append({'edge': first_edge, 'reverse': False})
        # 将种子边长度累加致链长度
        current_chain_length = current_chain_length + edeg_length
        
        # 获取当前链头尾点坐标
        p_first_start, p_first_end = _get_edge_endpoints(first_edge)
        # 当前链首坐标
        chain_head_pnt = p_first_start
        # 当前链尾坐标
        chain_tail_pnt = p_first_end
        
        # 贪心生长 (Grow)
        # 将距离首尾点最近的其他边计入链, 并判断其方向与种子链是否相同
        while edge_pool:
            # 初始化新循环标签
            found_next = False
            # 最合适边索引(默认-1即当不存在最合适边时判断式可以根据此跳出)
            best_candidate_idx = -1
            # 最合适的边方向性
            best_connect_type = None 
            # 初始化最近距离
            min_dist = 1e9
            
            # 在剩余池中寻找邻居
            for i, candidate in enumerate(edge_pool):
                # 获得候选边的首末点
                cand_start, cand_end = _get_edge_endpoints(candidate)
                
                # 计算 4 种连接距离
                d_tail_start = chain_tail_pnt.Distance(cand_start) # 尾接头 (正)
                d_tail_end = chain_tail_pnt.Distance(cand_end)     # 尾接尾 (反)
                d_start_end = chain_head_pnt.Distance(cand_end)    # 头接尾 (正)
                d_start_start = chain_head_pnt.Distance(cand_start) # 头接头 (反)
                # 得到其中最小值
                local_min = min(d_tail_start, d_tail_end, d_start_end, d_start_start)
                # 通过记录当前侯选边距离链首尾的最近距离
                # 来迭代找出距离链首尾最近的侯选边
                # 并判断是从链首连接还是从链尾连接
                if local_min < min_dist:
                    min_dist = local_min
                    best_candidate_idx = i
                    if local_min == d_tail_start: best_connect_type = 'tail_positive'
                    elif local_min == d_tail_end: best_connect_type = 'tail_negative'
                    elif local_min == d_start_end: best_connect_type = 'head_positive'
                    elif local_min == d_start_start: best_connect_type = 'head_negative'
            
            # 判定连接阈值
            if min_dist < 1e-3 and best_candidate_idx != -1:
                # 将可以加入链中的边从候选边列表中删除
                candidate = edge_pool.pop(best_candidate_idx)
                # 获得候选边长度
                cand_len = get_len(candidate)
                # 获得候选边的首尾坐标点
                cand_start, cand_end = _get_edge_endpoints(candidate)
                # 将根据连接形式更新当前链的首尾坐标
                if best_connect_type == 'tail_positive':
                    # 对于尾接头, 则以候选边的尾点更新链尾坐标
                    # (chain)*---<---* *---<---*(candidate)
                    current_chain.append({'edge': candidate, 'reverse': False})
                    chain_tail_pnt = cand_end
                elif best_connect_type == 'tail_negative':
                    # 对于尾接尾, 则以候选边的头点更新链尾坐标
                    # (chain)*---<---* *--->---*(candidate)
                    current_chain.append({'edge': candidate, 'reverse': True})
                    chain_tail_pnt = cand_start
                elif best_connect_type == 'head_positive':
                    # 对于头接尾, 则以候选点的头点更新链头坐标
                    # (candidate)*---<---* *---<---*(chain)
                    current_chain.insert(0, {'edge': candidate, 'reverse': False})
                    chain_head_pnt = cand_start
                elif best_connect_type == 'head_negative':
                    # 对于头接头, 则以候选点的尾点更新链头坐标
                    # (candidate)*--->---* *---<---*(chain)
                    current_chain.insert(0, {'edge': candidate, 'reverse': True})
                    chain_head_pnt = cand_end
                # 将筛选出来的边长度累加致总长度
                current_chain_length = current_chain_length + cand_len
                # 由于成功延长的链, 继续在剩下的边中尝试寻找
                found_next = True
            
            # 如果这一轮没找到邻居, 说明当前链断了(或者是闭环了)
            if not found_next:
                break
        
        # 保存当前链链长以及链本身
        all_chains.append((current_chain_length, current_chain))
    
    # --- 筛选最长链 ---
    if not all_chains:
        # 对于完全不存在链, 这基本不可能, 因为至少会包含种子链
        return []
    
    # 按长度降序排列，取第一个
    all_chains.sort(key=lambda x: x[0], reverse=True)
    best_chain = all_chains[0][1]
    
    return best_chain 

def uv_sample(graph, sample_num = (5, 5, 5)):
    """
    "--已过时--" 用于给拓扑图中的面与边实现UV采样

    :param graph: graph, 一个networkX无向图
    :param sample_num: Tuple型变量, 其中记录了边T采样点数, 面UV采样点数

    :return graph: 经填充了采样信息的networkX无向图
    """
    feature_shape_edge = (sample_num[0], 6)
    # 对图中的拓扑面进行采样
    # 向构建好的拓扑图中增添特征信息
    for node_idx in graph.nodes:
        # 获得图中当前节点的拓扑面
        topo_face = graph.nodes[node_idx]["face"]
        # 将面类型转化为occwl中的面类型
        occwl_face = Face(topo_face)
        # 计算该拓扑面的UV网格
        points = uvgrid(
            occwl_face, method="point", 
            num_u=sample_num[1], 
            num_v=sample_num[2]
        )
        normals = uvgrid(
            occwl_face, method="normal",
            num_u=sample_num[1],
            num_v=sample_num[2]
        )
        visibility_status = uvgrid(
            occwl_face, method="visibility_status",
            num_u=sample_num[1],
            num_v=sample_num[2]
        )
        # 根据可见性构建mask掩码
        # 0: Inside, 1: Outside, 2: On boundary
        mask = np.logical_or(visibility_status == 0, visibility_status == 2).astype(np.float32)
        # 将这些采样信息拼接在一起构成拓扑图中面的特征
        feat_topo_face = np.concatenate((points, normals, mask), axis=-1)
        # 向图中填充该特征信息
        graph.nodes[node_idx]["sample"] = feat_topo_face
    
    # 对图中的边进行采样
    # 并向构建好的拓扑图中增添特征信息
    # 初始化退化边计数器
    degenerate_edge_count = 0
    for edge_idx in graph.edges:
        topo_edge = graph.edges[edge_idx]["edge"]
        occwl_edge = Edge(topo_edge)
        # 再次检测是否出现了退化边
        if occwl_edge.has_curve():
            # 对正常边的处理
            try:
                points = ugrid(occwl_edge, method="point", num_u=sample_num[0])
                tangents = ugrid(occwl_edge, method="tangent", num_u=sample_num[0])
                feat_topo_edge = np.concatenate((points, tangents), axis=-1)
            except Exception as e:
                print(f"处理有效边 {edge_idx} 时出错: {str(e)}")
                # 如果处理失败，使用零特征
                feat_topo_edge = np.zeros(feature_shape_edge)
                degenerate_edge_count += 1
        else:
            # 对于退化边，创建全零特征 2024/12/29日更新,前面既然已经筛掉不正常的边了,这里也不必保留
            feat_topo_edge = np.zeros(feature_shape_edge)
            degenerate_edge_count += 1
            pass
        # 确保特征是numpy数组并且形状正确
        feat_topo_edge = np.asarray(feat_topo_edge, dtype=np.float32)
        if feat_topo_edge.shape != feature_shape_edge:
            print(f"警告：节点 {node_idx} 的特征形状不正确: {feat_topo_edge.shape}, 期望: {feature_shape_edge}")
            feat_topo_edge = np.zeros(feature_shape_edge, dtype=np.float32)
        # 向边特征中补充
        graph.edges[edge_idx]["sample"] = feat_topo_edge
    # 若发现退化边则发出一个提醒
    if degenerate_edge_count > 0:
        print(f"UV采样提示: 检测到 {degenerate_edge_count} 条退化边或采样失败边，已用零向量填充。")
    
    # 待补充完毕后返回图
    return graph

def uv_sample_2_old(graph, sample_num=(5, 5, 5)):
    """
    "--已过时--" 深度优化的 UV 采样函数
    能够处理单条边和多段边列表，保证输出特征维度一致。
    """
    feature_shape_edge = (sample_num[0], 6)
    
    # --- 1. 面采样 (保持不变) ---
    for node_idx in graph.nodes:
        if "sample" in graph.nodes[node_idx]: continue # 避免重复采样
        
        topo_face = graph.nodes[node_idx]["face"]
        occwl_face = Face(topo_face)
        
        try:
            points = uvgrid(occwl_face, method="point", num_u=sample_num[1], num_v=sample_num[2])
            normals = uvgrid(occwl_face, method="normal", num_u=sample_num[1], num_v=sample_num[2])
            visibility_status = uvgrid(occwl_face, method="visibility_status", num_u=sample_num[1], num_v=sample_num[2])
            mask = np.logical_or(visibility_status == 0, visibility_status == 2).astype(np.float32)
            feat_topo_face = np.concatenate((points, normals, mask), axis=-1)
        except Exception as e:
            print(f"面采样失败 Node {node_idx}: {e}")
            feat_topo_face = np.zeros((sample_num[1], sample_num[2], 7), dtype=np.float32)

        graph.nodes[node_idx]["sample"] = feat_topo_face
    
    # --- 2. 边采样 (深度优化) ---
    degenerate_edge_count = 0
    target_k = sample_num[0]

    for edge_idx in graph.edges:
        edge_data = graph.edges[edge_idx].get("edge")
        
        # 兼容性处理：如果图是旧方法构建的，edge_data 只是一个 TopoDS_Edge
        # 如果是新方法 build_topo_graph_2，edge_data 是 [TopoDS_Edge, ...] 列表
        edge_list = []
        if isinstance(edge_data, list):
            edge_list = edge_data
        else:
            edge_list = [edge_data]

        # 过滤无效边并转换为 occwl Edge
        valid_occwl_edges = []
        total_length = 0.0
        
        for e in edge_list:
            occ_e = Edge(e)
            if occ_e.has_curve():
                try:
                    l = occ_e.length()
                    if l > 1e-6: # 忽略极短的噪点边
                        valid_occwl_edges.append((occ_e, l))
                        total_length += l
                except:
                    pass
        
        if not valid_occwl_edges:
            # 全是退化边
            graph.edges[edge_idx]["sample"] = np.zeros(feature_shape_edge, dtype=np.float32)
            degenerate_edge_count += 1
            continue

        # --- 多段边特征融合策略 ---
        # 策略：收集所有边的点云，按长度加权密度，最后统一重采样到 target_k
        
        # 1. 对边进行几何排序 (尝试连成链)
        # 如果只有一条边，无需排序
        if len(valid_occwl_edges) > 1:
            # 解压 edge 对象进行排序
            raw_edges = [x[0].topods_shape() for x in valid_occwl_edges]
            sorted_raw = _sort_edges_geometrically(raw_edges)
            # 重新封装为 occwl 并找回长度
            # (这里为了性能简化，假设排序后重新计算长度)
            sorted_occwl = []
            for raw in sorted_raw:
                oe = Edge(raw)
                sorted_occwl.append((oe, oe.length()))
            valid_occwl_edges = sorted_occwl

        # 2. 采集高密度点云
        # 为了保证重采样精度，我们采集比目标多 3 倍的点
        oversample_factor = 3
        total_samples_pool = target_k * oversample_factor
        
        raw_points = []
        raw_tangents = []

        for occ_e, length in valid_occwl_edges:
            # 根据长度比例分配采样点数
            ratio = length / total_length
            # 至少采 2 个点以保证首尾
            n_samples = max(2, int(total_samples_pool * ratio))
            
            try:
                # 采样点和切线
                p = ugrid(occ_e, method="point", num_u=n_samples)
                t = ugrid(occ_e, method="tangent", num_u=n_samples)
                raw_points.append(p)
                raw_tangents.append(t)
            except:
                continue

        if not raw_points:
            graph.edges[edge_idx]["sample"] = np.zeros(feature_shape_edge, dtype=np.float32)
            continue

        # 拼接所有点
        all_points = np.concatenate(raw_points, axis=0)     # (N_total, 3)
        all_tangents = np.concatenate(raw_tangents, axis=0) # (N_total, 3)

        # 3. 统一重采样 (Resampling)
        # 如果点数不够 (极少情况)，直接插值；如果多了，降采样
        current_n = all_points.shape[0]
        
        if current_n == target_k:
            final_points = all_points
            final_tangents = all_tangents
        else:
            # 简单的线性索引重采样
            # 这种方法假设点在路径上大致均匀分布(这是由上面的按长度分配保证的)
            indices = np.linspace(0, current_n - 1, target_k)
            
            # 对 indices 进行取整用于取值，或者进行线性插值
            # 这里使用简单的取整索引，对于密集点云足够精确
            indices_floor = np.floor(indices).astype(int)
            indices_ceil = np.ceil(indices).astype(int)
            weights = (indices - indices_floor).reshape(-1, 1)
            
            # 点插值
            p0 = all_points[indices_floor]
            p1 = all_points[indices_ceil]
            final_points = p0 * (1 - weights) + p1 * weights
            
            # 切线插值 (注意归一化)
            t0 = all_tangents[indices_floor]
            t1 = all_tangents[indices_ceil]
            final_tangents = t0 * (1 - weights) + t1 * weights
            # 重新归一化切线
            norms = np.linalg.norm(final_tangents, axis=1, keepdims=True)
            norms[norms < 1e-6] = 1 # 避免除零
            final_tangents = final_tangents / norms

        # 4. 组合最终特征
        feat_topo_edge = np.concatenate((final_points, final_tangents), axis=-1)
        
        # 再次确保形状
        if feat_topo_edge.shape != feature_shape_edge:
             feat_topo_edge = np.zeros(feature_shape_edge, dtype=np.float32)
             
        graph.edges[edge_idx]["sample"] = feat_topo_edge

    if degenerate_edge_count > 0:
        print(f"UV采样提示: 检测到 {degenerate_edge_count} 条边采样异常，已用零向量填充。")
    
    return graph

def uv_sample_2(graph, sample_num=(5, 5, 5), output=False):
    """
    深度优化的 UV 采样函数
    能够处理单条边和多段边列表，保证输出特征维度一致
    面->(M, N, C)
    边->(N, C)
    """
    feature_shape_edge = (sample_num[0], 6)
    target_k = sample_num[0]
    # 若导出采样点数据
    # 初始化采集列表
    if output:point_list = []

    # --- 1. 面采样 (保持不变) ---
    for node_idx in graph.nodes:
        if "sample" in graph.nodes[node_idx]: 
            continue
        
        topo_face = graph.nodes[node_idx]["face"]
        occwl_face = Face(topo_face)
        
        try:
            # 采样面上点的坐标
            try:
                points = uvgrid(occwl_face, method="point", num_u=sample_num[1], num_v=sample_num[2])
            except Exception as e:
                print(f"Points sample error: {e}")
            # 采样面上点的法坐标
            try:
                normals = uvgrid(occwl_face, method="normal", num_u=sample_num[1], num_v=sample_num[2])
            except Exception as e:
                print(f"Normals sample error: {e}")
            # 采样面上的可见性
            try:
                visibility_status = uvgrid(occwl_face, method="visibility_status", num_u=sample_num[1], num_v=sample_num[2])
                mask = np.logical_or(visibility_status == 0, visibility_status == 2).astype(np.float32)
                feat_topo_face = np.concatenate((points, normals, mask), axis=-1)
            except Exception as e:
                print(f"Visibility sample error: {e}")
        except Exception as e:
            feat_topo_face = np.zeros((sample_num[1], sample_num[2], 7), dtype=np.float32)
        # 保存采样点
        if output: point_list.append(points.reshape(-1, 3))
        graph.nodes[node_idx]["sample"] = feat_topo_face
    # 点整合
    if output: 
        points_list_all = np.vstack(point_list)
        save_path = 'points_cloud_face.npy'
        np.save(save_path, points_list_all)
    # --- 2. 边采样 (深度优化) ---
    degenerate_edge_count = 0

    for edge_idx in graph.edges:
        edge_data = graph.edges[edge_idx].get("edge")
        
        # 兼容性处理
        edge_list = []
        if isinstance(edge_data, list):
            edge_list = edge_data
        else:
            edge_list = [edge_data]

        # 预筛选有效边
        valid_raw_edges = []
        for e in edge_list:
            occwl_e = Edge(e)
            if occwl_e.has_curve():
                try:
                    edge_length = occwl_e.length()
                    if edge_length > 1e-6:
                        valid_raw_edges.append(e) # 暂存 TopoDS_Edge
                        
                except:
                    pass
        
        if not valid_raw_edges:
            graph.edges[edge_idx]["sample"] = np.zeros(feature_shape_edge, dtype=np.float32)
            degenerate_edge_count += 1
            continue
        
        # --- 单边处理策略 ---
        final_points = None
        final_tangents = None

        # 1. 几何排序与定向 (Sorting & Orientation)
        # sorted_chain 结构: [{'edge': TopoDS_Edge, 'reverse': bool}, ...]
        # 将边进行筛选与链接
        sorted_chain = _sort_edges_geometrically(valid_raw_edges)

        # [情况 A]: 只有一条有效边, 直接使用 occwl 高效采样
        if len(sorted_chain) == 1:
            raw_e = sorted_chain[0]['edge']
            occwl_e = Edge(raw_e)
            try:
                # 直接按目标点数采样
                p = ugrid(occwl_e, method="point", num_u=target_k)
                t = ugrid(occwl_e, method="tangent", num_u=target_k)
                # [关键]: 即使是单条边，也必须检查 Orientation
                # occwl/OCC 默认按参数 t 增加方向采样。
                # 如果边是 Reversed，说明拓扑方向与几何方向相反，我们需要翻转结果以符合拓扑逻辑。
                if raw_e.Orientation() == TopAbs_REVERSED:
                    p = p[::-1]
                    t = t[::-1] * -1
                
                final_points = p
                final_tangents = t
            except:
                # 降级处理 (虽然很少发生)
                graph.edges[edge_idx]["sample"] = np.zeros(feature_shape_edge, dtype=np.float32)
                continue
        # --- 多段边特征融合策略 ---
        # [情况 B]: 多段边, 使用复杂缝合逻辑
        else: 
            # 计算总长度
            total_length = 0.0
            # 计算总长度用于分配采样点
            for item in sorted_chain:
                occwl_e = Edge(item['edge'])
                l = occwl_e.length()
                item['length'] = l
                item['occwl_edge'] = occwl_e
                total_length += l

            # 2. 采集高密度点云
            oversample_factor = 3
            total_samples_pool = target_k * oversample_factor
            
            merged_points = []
            merged_tangents = []

            for i, item in enumerate(sorted_chain):
                occwl_e = item['occwl_edge']
                length = item['length']
                need_reverse = item['reverse']
                
                # 比例分配点数
                if total_length > 0:
                    ratio = length / total_length
                else:
                    ratio = 1.0 / len(sorted_chain)
                
                n_samples = max(2, int(total_samples_pool * ratio))
                
                try:
                    # 采样
                    p = ugrid(occwl_e, method="point", num_u=n_samples)
                    t = ugrid(occwl_e, method="tangent", num_u=n_samples)
                    
                    # [关键修正] 根据排序结果翻转方向
                    if need_reverse:
                        p = p[::-1]
                        t = t[::-1] * -1 # 切线也要反向
                    
                    # [去重策略]
                    # 如果这不是第一段，去掉首点（因为它理应与上一段的尾点重合）
                    if i > 0:
                        p = p[1:]
                        t = t[1:]
                    
                    merged_points.append(p)
                    merged_tangents.append(t)
                    
                except:
                    continue

            if not merged_points:
                graph.edges[edge_idx]["sample"] = np.zeros(feature_shape_edge, dtype=np.float32)
                continue

            # 拼接所有点
            all_points = np.concatenate(merged_points, axis=0)     
            all_tangents = np.concatenate(merged_tangents, axis=0) 

            # 3. 统一重采样 (Resampling)
            current_n = all_points.shape[0]
            
            if current_n == target_k:
                final_points = all_points
                final_tangents = all_tangents
            elif current_n < target_k:
                # 如果点数不够(很少见，除非边极短)，进行简单的线性插值上采样
                indices = np.linspace(0, current_n - 1, target_k)
                indices_floor = np.floor(indices).astype(int)
                indices_ceil = np.ceil(indices).astype(int)
                weights = (indices - indices_floor).reshape(-1, 1)
                
                p0 = all_points[indices_floor]
                p1 = all_points[indices_ceil]
                final_points = p0 * (1 - weights) + p1 * weights
                
                t0 = all_tangents[indices_floor]
                t1 = all_tangents[indices_ceil]
                final_tangents = t0 * (1 - weights) + t1 * weights
                
            else:
                # 降采样
                indices = np.linspace(0, current_n - 1, target_k)
                indices_round = np.round(indices).astype(int)
                # 确保不越界
                indices_round = np.clip(indices_round, 0, current_n - 1)
                
                final_points = all_points[indices_round]
                final_tangents = all_tangents[indices_round]
                
            # 归一化切线
            norms = np.linalg.norm(final_tangents, axis=1, keepdims=True)
            norms[norms < 1e-6] = 1 
            final_tangents = final_tangents / norms

        # 4. 组合最终特征
        feat_topo_edge = np.concatenate((final_points, final_tangents), axis=-1)
        # print(feat_topo_edge.shape)
        
        if feat_topo_edge.shape != feature_shape_edge:
            feat_topo_edge = np.zeros(feature_shape_edge, dtype=np.float32)
            
        graph.edges[edge_idx]["sample"] = feat_topo_edge

    if degenerate_edge_count > 0:
        print(f"UV采样提示: 检测到 {degenerate_edge_count} 条边采样异常，已用零向量填充。")
    
    return graph

def build_geom_graph(shape, tolerance=1e-6, angular_tolerance=5.0, estimate=False, **kwargs):
    """
    构建几何关系图
    """
    # 对几何体进行校验
    validate_shape_integrity(shape)
    # 对几何体执行缩放
    #shape = normalize_shape_min_area(shape, target_min_area=math.e)
    shape = normalize_shape(shape, 
                            method=kwargs.get('shape_norm_method', 'bbox'), 
                            param=kwargs.get('shape_norm_param', None))

    # 按照OCC的遍历顺序获得面列表
    explorer = TopologyExplorer(shape)
    topo_faces = list(explorer.faces())
    # 获得面的索引与hash值映射字典
    # mapping_dict = step_topo_mapping(shape, 'face')

    # ------------------------------------------------
    # 用于提取面的通用几何信息, 信息将被存储于一个2D列表中
    # 提取面通用几何信息
    # 选择需要提取的通用几何信息
    selected_props = ['area', 
                      'loop_count', 
                      'shape_index_mean', 
                      'shape_index_var', 
                      'principal_axes_lengths', 
                      'gyradius', 
                      'eccentricity']
    # 实例化通用几何信息提取类
    embedder = Geom_embedding_face(selected_props, 
                                   use_log_area=kwargs.get('use_log_area', False), 
                                   use_log_linear=kwargs.get('use_log_linear', False))
    node_feats_2d = []
    for face in topo_faces:
        # 获得通用的几何信息以及面类型的独热编码
        shared_props, one_hot = embedder.embedding(face)
        # 预先将通用特征值与独热编码拼接起来
        feat_concat = np.concatenate([shared_props, one_hot])
        node_feats_2d.append({
            'shared_props': shared_props,
            'one_hot': one_hot,
            'feat_concat': feat_concat
        })

    #------------------------------------------------
    # 用于提取面之间的几何关系信息, 信息将被存储于一个3D列表中
    # 预提取descriptors
    descriptors = [Position_Descriptor_Surface(face).get_position_signature() for face in topo_faces]
    # 构建几何关系列表矩阵
    n = len(topo_faces)
    relations_3d = [[None for _ in range(n)] for _ in range(n)]
    # 初始化几何关系获取方法
    calculator_geom = Surface_Relationship_Analyzer(shape=shape)
    # 几何关系重要度字典
    priority = {'coplanar': 4, 'tangent': 3, 'coaxial': 2, 'perpendicular': 1, 'parallel': 1}
    # 距离归一化方法
    # TODO: 由于距离函数有可能返回None, 因此此处须待距离函数优化完毕后进行

    # 使用循环填充关系列表
    for i in range(n):
        # 由于不判断自己与自己的关系, 因此关系矩阵为一个去除主对角线的上三角矩阵
        for j in range(i+1, n):
            relation = calculator_geom.analyze_relationship(descriptors[i], descriptors[j])
            # 遍历priority的每个键k, 拼接成is_+k的键名, 从relation中取值
            # 若取值为True(或存在且为真), 则将k加入active_keys
            active_keys = [k for k in priority if relation.get(f'is_{k}', False)]
            # 如果active_keys中有值的话
            if active_keys:
                # 将根据几何关系重要度字典中选出优先级最高的作为主关系
                primary = max(active_keys, key=lambda k: priority[k])
                # 初始化一个用于表示几何关系的独热向量
                one_hot_relation = np.zeros(len(priority), dtype=np.float32)
                # 为每种关系分配一个数组索引, 以方便后续给one_hot_relation赋值
                idx_map = {k: idx for idx, k in enumerate(priority.keys())}
                # 根据
                for k in active_keys:
                    idx = idx_map[k]
                    # one_hot_relation[idx] = priority[k] / sum(priority[ak] for ak in active_keys)  # 加权复合
                    one_hot_relation[idx] = priority[k]
                # 将关系字典进行复制, 以便于后续的调试
                relation_dict = relation.copy()
                relation_dict['one_hot_relation'] = one_hot_relation
                relation_dict['primary_relation'] = primary
                relations_3d[i][j] = relation_dict
                relations_3d[j][i] = relation_dict.copy()  # 对称复制
            else:
                relations_3d[i][j] = None
                relations_3d[j][i] = None
    
    # 构建NetworkX图
    graph = nx.Graph()
    # 添加节点
    for i in range(n):
        # 向节点内添加各类信息
        # **node_feats_2d[i] 将会将字典进行解包
        graph.add_node(i, **node_feats_2d[i], descriptor=descriptors[i], 
                       type_str=str(GeomAbs_SurfaceType(descriptors[i]['surface_type'])))
        
    # 连接节点的边
    for i in range(n):
        for j in range(i+1, n):
            relation_dict = relations_3d[i][j]
            if relation_dict is not None:
                graph.add_edge(i, j, 
                               one_hot_relation = relation_dict['one_hot_relation'],
                               dist=relation_dict.get('distance', None),
                               primary_relation = relation_dict['primary_relation'])
    
    # 是否开启评价
    # 若开启则返回评价列表，若关闭则返回图
    if estimate:
        estimate_list = []
        # 新建评价指标数据列表
        # 获得step文件中的面数量
        estimate_list.append(len(topo_faces))
        # 获得图的节点与边的数量
        estimate_list.append(graph.number_of_nodes())
        estimate_list.append(graph.number_of_edges())

        # 输出当前实体中的面与边的数量
        print(f"实体中面数量: {estimate_list[0]}")
        # 输出当前图中的节点与边的数量
        print(f"图中节点数量: {estimate_list[1]}")
        print(f"图中边的数量: {estimate_list[2]}")
        # return estimate_list


    return graph, relations_3d, node_feats_2d

def build_topo_graph_old(shape, sample_num = (5, 5, 5), self_loops=False, estimate=False):
    """
    "--已过时--" 读取 shape 并返回由shape所构成的FAG图。

    :param shape: TopoDS_Solid型的shape
    :param sample_num: Tuple型变量, 其中记录了边T采样点数, 面UV采样点数
    :param self_loops: Bool型变量, 用于在构建图时考虑是否引入自环
    :param estimate: Bool型变量, 用于决定是否打印图信息用于调试
    :return graph: 由shape构成的FAG图
    """

    # 新建networks无向图
    graph = nx.Graph()

    # 获得面列表
    explorer = TopologyExplorer(shape)
    # 生成面列表
    faces = list(explorer.faces())
    # 生成边列表, 并利用列表推导式去除退化边
    edges = [edge for edge in list(explorer.edges()) if BRepAdaptor_Curve(edge).Is3DCurve()]

    # 创建FAG的face节点
    face_count = 0
    # 创建face的索引字典
    face_dict = {}
    for face in faces:
        # 新增face节点
        graph.add_node(face_count, face=face)
        face_dict[face.HashCode(999999)] = face_count
        face_count = face_count + 1
    
    # 处理面邻接关系
    for edge in edges:
        # 获取该边的邻接面
        connected_faces = list(explorer.faces_from_edge(edge))
        # 从邻接面中构建面节点中的连接
        if len(connected_faces) == 1 and not self_loops:
            # 不允许自环的话，
            continue
        # 若允许存在seam边形成的自环，则这里self_loops应该为True
        elif len(connected_faces) == 1 and ShapeAnalysis_Edge().IsSeam(edge, connected_faces[0]) and self_loops:
            graph.add_edge(face_dict[(connected_faces[0]).HashCode(999999)], 
                           face_dict[(connected_faces[0]).HashCode(999999)],
                           edge=edge)
        else:
            # 从邻接面列表中获取面
            # 将获得的面列表两两组合，这是为了解决出现一个边被多个面共用的情况，不过实际中这种情况往往不存在
            combinated_faces = get_combinations(connected_faces)
            for couple_faces in combinated_faces:
                # 双向连接
                graph.add_edge(face_dict[(couple_faces[0]).HashCode(999999)], 
                            face_dict[(couple_faces[1]).HashCode(999999)],
                            edge=edge)
                
                # graph.add_edge(face_dict[(couple_faces[1]).HashCode(999999)], 
                #             face_dict[(couple_faces[0]).HashCode(999999)],
                #             edge=edge)

    # 向图中补充采样信息
    sampled_graph = uv_sample(graph, sample_num)

    # 是否开启评价
    # 若开启则返回评价列表，若关闭则返回图
    if estimate:
        estimate_list = []
        # 新建评价指标数据列表
        # 获得step文件中的面数量
        estimate_list.append(len(faces))
        # 获得step文件中的边数量
        estimate_list.append(len(edges))
        # 获得图的节点与边的数量
        estimate_list.append(sampled_graph.number_of_nodes())
        estimate_list.append(sampled_graph.number_of_edges())

        # 输出当前实体中的面与边的数量
        print(f"实体中面数量: {estimate_list[0]}")
        print(f"实体中边数量: {estimate_list[1]}")
        # 输出当前图中的节点与边的数量
        print(f"图中节点数量: {estimate_list[2]}")
        print(f"图中边的数量: {estimate_list[3]}")
        # return estimate_list
    
    return sampled_graph

def build_topo_graph(shape, sample_num=(5, 5, 5), self_loops=False, estimate=False, **kwargs):
    """
    [新版] 构建拓扑图方法 2
    
    特点：
    1. 使用 Surface_Relationship_Analyzer 的 adjacency_map 构建图
    2. 支持多边连接(Multi-edge Connection): 将连接两个面的所有边聚合成一个列表存储
    3. 鲁棒性更强，适用于 B 样条曲面分割导致的多段边情况
    """
    # 对几何体进行校验
    validate_shape_integrity(shape)
    # 对几何体执行缩放
    shape = normalize_shape(shape, 
                            method=kwargs.get('method', 'bbox'), 
                            param=kwargs.get('param', None))
    
    # 初始化分析器，构建邻接映射
    # 这步会自动处理"多边连接"问题，将所有连接边归拢到列表
    analyzer = Surface_Relationship_Analyzer(shape=shape, self_loops=self_loops)
    adjacency_map = analyzer.adjacency_map # 结构: (id1, id2) -> [edge1, edge2...]
    
    graph = nx.Graph()
    
    # 添加节点
    # 注意：analyzer.topo_face_list 是按遍历顺序存储的，其索引即为 ID
    for idx, face in enumerate(analyzer.topo_face_list):
        graph.add_node(idx, face=face)
        
    # 基于邻接表添加常规边 (包含多段边聚合)
    for (id1, id2), edge_list in adjacency_map.items():
        # 直接将 edge_list 存入，不再是单条 edge
        graph.add_edge(id1, id2, edge=edge_list)

    # 执行深度优化的采样
    # uv_sample 现在会自动识别 edge 是列表还是单对象，并执行加权重采样
    # sampled_graph = uv_sample(graph, sample_num)
    try:
        sampled_graph = uv_sample_2(graph, sample_num)
    except Exception as e:
        print(f"{e}")
    
    # 评价输出
    if estimate:
        print("=== Build Topo Graph 2 Estimate ===")
        print(f"实体中面数量: {len(analyzer.topo_face_list)}")
        print(f"图中节点数量: {sampled_graph.number_of_nodes()}")
        print(f"图中边的数量: {sampled_graph.number_of_edges()}")
        
        # 统计多段边的情况
        multi_segment_count = 0
        max_segments = 0
        for u, v, data in sampled_graph.edges(data=True):
            edges = data['edge']
            if isinstance(edges, list):
                count = len(edges)
                if count > 1:
                    multi_segment_count += 1
                if count > max_segments:
                    max_segments = count
        print(f"多段边连接数: {multi_segment_count}")
        print(f"最大单连接边数: {max_segments}")

    return sampled_graph

# 图的可视化
def visualize_graph(G, label_dict=None):
    """
    使用 matplotlib 和 networkx 可视化拓扑图。

    :param G: 拓扑图(networkx 图对象)
    """
    pos = nx.spring_layout(G)  # 计算节点位置
    # pos = nx.random_layout(G)

    if label_dict == None:

        #node_labels = {i: f"{i} & {G.nodes[i]['edge']}" for i in G.nodes()}  # 设置节点标签
        node_labels = {i: f"{i} " for i in G.nodes()}  # 设置节点标签
        #edge_labels = {(i, j): f"{i}->{j}" for i, j in G.edges()}  # 设置边标签
        
    else:
        labels = {i: f"{label_dict[str(i)]}" for i in G.nodes()}

    # 绘制节点
    nx.draw_networkx_nodes(
        G, pos,
        node_size=300, # 设置节点大小
        node_color='#66ccff', # 设置节点颜色
        alpha=1, # 设置透明度
        linewidths=1.2, # 设置描边线宽
        edgecolors='#555555', # 设置描边颜色
        node_shape='o' # 设置节点形状
    )
    
    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, 
                            labels=node_labels, 
                            font_size=8, 
                            font_color='black',
                            font_weight='light', 
                            font_family='sans-serif',
                            horizontalalignment='center',
                            verticalalignment='center')
    
    # 绘制边
    nx.draw_networkx_edges(
        G, pos,
        width=0.5,  # 设置边的粗细
        arrows=False,  # 显示箭头
        style='solid',  # 设置边样式
        alpha=0.5, # 设置透明度
        #arrowstyle='-|>', #  箭头样式
        #arrowsize=10  # 设置箭头大小
    )
    
    # 绘制边标签
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=4, font_family='sans-serif')
    
    plt.title("Topology Graph of STEP Model Edges")
    plt.show()


if __name__ == "__main__":
    # filename = r"F:\MFR\FR_Net\visualize\Test_Net\self_test\StpToBin_test\20221121_154647_0.step"
    # filename = "F:\\MFR\\FR_Net\\Test_step\\OCC_Test.STEP"
    # filename = "F:\\MFR\\FR_Net\\Test_step\\cube.STEP"
    # filename = "F:\\MFR\\FR_Net\\Test_step\\ball.STEP"
    # filename = "F:\\MFR\\FR_Net\\Test_step\\CA6140.STEP"
    # filename = "F:\\MFR\\FR_Net\\visualize\\Test_Net\\step_creat\\cylinder.step"
    # filename = "F:\\MFR\\FR_Net\\visualize\\Test_Net\\step_creat\\ball.step"
    # filename = "F:\\MFR\\FR_Net\\visualize\\Test_Net\\Q\\20221121_154647_5.step"
    # filename = r"F:\MFR\FR_Net\visualize\Test_Net\Q\step.step"
    # filename = "F:\\MFR\\FR_Net\\Test_step\\AAB_2.STEP"
    # filename = r"F:\MFR\EAGNet\test_data\steps\20221121_154647_11.step"

    # filename = r"F:\MFR\Final_MFR\MFR_DualGNN_Performace_test\MFTRCAD\MFR_DualGNN_MFTRCAD\data\steps\20240125_003844_577_result.step"
    filename = "F:\\MFR\\FR_Net\\Test_step\\t_tube.STEP"
    shape = read_step_file(filename)
    sample_num = (5, 3, 3)
    
    graph, _, _ = build_geom_graph(shape, estimate=True)
    # graph = build_topo_graph(shape, self_loops=False, estimate=False)
    # print(graph.nodes[1])
    
    # print(graph)
    visualize_graph(graph)