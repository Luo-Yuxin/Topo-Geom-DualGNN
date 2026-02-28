import numpy as np

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_SurfaceType
from OCC.Core.BRepGProp import BRepGProp_Face, brepgprop_SurfaceProperties
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_WIRE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.GeomLProp import GeomLProp_SLProps

from OCC.Core.gp import gp_Vec, gp_Pnt, gp_Dir
from OCC.Core.gp import gp_Pln, gp_Cylinder, gp_Cone, gp_Sphere, gp_Torus
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Ax3
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace

import torch
import torch.nn as nn
import math

class Geom_embedding_face(nn.Module):
    """
    一个用于实现面几何信息嵌入的类
    """
    def __init__(
            self,
            selected_props = ['area', 
                              'loop_count', 
                              'shape_index_mean', 
                              'shape_index_var', 
                              'principal_axes_lengths', 
                              'gyradius', 
                              'eccentricity'],
            use_log_area = False,   # 控制是否对面积求对数
            log_area_shift = 1.0,   # 面积对数平移量
            use_log_linear = False, # 控制是否对线性尺寸(长度、半径)求对数
            log_linear_shift = 1.0  # 线性尺寸对数平移量
            ):
        
        self.face_geom_selected_props = selected_props
        # 面积对数缩放
        self.use_log_area = use_log_area
        self.log_area_shift = log_area_shift
        # 线性尺寸相关配置
        self.use_log_linear = use_log_linear
        self.log_linear_shift = log_linear_shift

        
        super().__init__()

    def embedding(self, topo_face):

        one_hot = self._get_type_encoding(topo_face)

        shared_props = self._get_shared_props(topo_face, selected_props=self.face_geom_selected_props)

        return shared_props, one_hot

    def _get_type_encoding(self, topo_face):
        """
        获取B-Rep面的类型编码, 将使用one-hot向量

        :param topo_face (TopoDS_Face): pyocc中的面对象

        :return np.ndarray: 11维one-hot向量, 对应GeomAbs_SurfaceType的11种类型
        """
        # 创建适配器以查询面类型
        adaptor = BRepAdaptor_Surface(topo_face)
        type_id = adaptor.GetType()  # 返回GeomAbs_SurfaceType枚举值(int)

        # 定义类型到索引的映射(基于枚举顺序)
        type_to_index = {
            GeomAbs_SurfaceType.GeomAbs_Plane: 0,
            GeomAbs_SurfaceType.GeomAbs_Cylinder: 1,
            GeomAbs_SurfaceType.GeomAbs_Cone: 2,
            GeomAbs_SurfaceType.GeomAbs_Sphere: 3,
            GeomAbs_SurfaceType.GeomAbs_Torus: 4,
            GeomAbs_SurfaceType.GeomAbs_BezierSurface: 5,
            GeomAbs_SurfaceType.GeomAbs_BSplineSurface: 6,
            GeomAbs_SurfaceType.GeomAbs_SurfaceOfRevolution: 7,
            GeomAbs_SurfaceType.GeomAbs_SurfaceOfExtrusion: 8,
            GeomAbs_SurfaceType.GeomAbs_OffsetSurface: 9,
            GeomAbs_SurfaceType.GeomAbs_OtherSurface: 10
        }

        # 获取索引, 如果类型未知, fallback到OtherSurface
        index = type_to_index.get(type_id, 10)  # 默认10 for Other

        # 创建one-hot向量
        one_hot = np.zeros(11, dtype=np.float32)
        one_hot[index] = 1.0

        return one_hot
    
    def _log_transform(self, raw_value, params):
        """
        对数值进行 对数 变换
        支持标量和 numpy 数组
        """
        # 解包参数
        shift_x = params[0]
        shift_y = params[1]
        min_value = -1e-9
        try:
            # 处理 numpy 数组
            if isinstance(raw_value, np.ndarray):
                # 防止数值为负
                min_raw_value = np.min(raw_value)
                if min_raw_value < min_value:
                    raise ValueError(f"value is minor than min_value: {min_raw_value}")
                else:
                    # 加上平移量
                    value_shifted = raw_value + shift_x
                    return np.log(value_shifted) + shift_y
            else:
                # 处理标量
                if raw_value < min_value:
                    raise ValueError(f"value is minor than min_value: {raw_value}")
                else:
                    value_shifted = raw_value + shift_x
                    # 对于长度属性，如果是 0 (比如退化面)，返回 log(min) 或者 0 可能更合适
                    return math.log(value_shifted) + shift_y
        except Exception as e:
            print(f"Log transform error: {e}, value={raw_value}")
            return raw_value if isinstance(raw_value, np.ndarray) else 0.0

    def _linear_transform(self, raw_value, params):
        """
        _linear_transform 用于对数据做线性变换
        
        :param raw_value: 原数据
        :param param: 线性变换参数
        """

        # 解包参数
        param_k = params[0]
        param_b = params[1]
        try:
            transformed_value = param_k * raw_value + param_b
            return transformed_value
        except Exception as e:
            print(f"Linear transform error: {e}, value={raw_value}")
            return raw_value

    def _non_negative_norm_with_log(self, raw_value, params):
        """
        _non_negative_norm_with_log 用于将非负向量做归一化并拼接上其放缩系数的对数值
        
        :param raw_value: 原始数据
        :param params: 对数参数
        :return array: 经归一化并拼接后的向量
        """
        if np.all(raw_value==0):
            return np.zeros(4)
        # 获得最大值
        max_value = raw_value.max()
        if max_value < 1e-9:
            return np.zeros(4)
        # 对列表进行非负归一化
        value_norm = raw_value / max_value
        # 提取最大值的对数
        log_max_value = self._log_transform(max_value, params)
        # 将该因子拼接上去
        return np.concatenate([value_norm, [log_max_value]])

    def _number_project(self, method, raw_value, params, enable):
        """
        _number_project 用于对数据进行映射变换
        比如做对数映射变换
        
        :param method: 映射变换方法
        :param raw_value: 待映射数据
        :return value_projected: 完成映射数据
        """
        if enable:
            if method == 'origion':
                return raw_value
            elif method == 'log':
                return self._log_transform(raw_value, params)
            elif method == 'linear':
                return self._linear_transform(raw_value, params)
            elif method == 'norm_log':
                return self._non_negative_norm_with_log(raw_value, params)
            else:
                raise ValueError(f"{method} is not in number_project")
        else:
            return raw_value

    def _get_shared_props(self, face, selected_props=None):
        """
        获取B-Rep面的共享几何属性, 使用NumPy数组输出
        """
        # 定义所有可用属性及其计算函数（字典便于扩展）
        _prop_functions = {
            'area': lambda: self._number_project('log', self._compute_area(face), 
                                                 (self.log_area_shift, 0.0), self.use_log_area),  # 1D, 规范化
            # 'perimeter': lambda: self._compute_perimeter(face),  # 1D
            'loop_count': lambda: self._compute_loop_count(face),  # 1D, 无需规范
            'shape_index_mean': lambda: self._compute_shape_index_stats(face)[0],  # 1D
            'shape_index_var': lambda: self._compute_shape_index_stats(face)[1],  # 1D
            # 'compactness': lambda: self._compute_compactness(face),  # 1D, 不变
            'principal_axes_lengths': lambda: self._number_project('log', self._compute_principal_axes_lengths(face),
                                                                   (self.log_linear_shift, 0.0), self.use_log_linear),  # 3D, 排序 max/mid/min
            'gyradius': lambda: self._number_project('log', self._compute_gyradius(face),
                                                     (self.log_linear_shift, 0.0), self.use_log_linear),  # 1D
            'eccentricity': lambda: self._compute_eccentricity(face),  # 1D, 不变
        }

        # 默认所有，如果指定则过滤
        if selected_props is None:
            selected_props = list(_prop_functions.keys())
        else:
            # 验证选择的属性有效
            invalid = set(selected_props) - set(_prop_functions)
            if invalid:
                raise ValueError(f"Invalid properties: {invalid}")
        
            # 计算选定属性
            props_list = []
            for prop in selected_props:
                value = _prop_functions[prop]()
                if isinstance(value, (int, float)):
                    props_list.append([value])
                else:
                    props_list.append(value)
            
            # Concat并返回NumPy数组
            all_props = np.concatenate(props_list).astype(np.float32)
            return all_props
        
    # 辅助函数：计算面积
    def _compute_area(self, face):
        props = GProp_GProps()
        try:
            brepgprop_SurfaceProperties(face, props)
            return props.Mass()
        except Exception as e:
            print(f"[Warning] Failed to compute face area (Degenerated Face?): {e}")
            return 
    
    # 辅助函数：计算周长（所有边长度和）
    def _compute_perimeter(self, face):
        pass
    
    # 辅助函数：边界环数
    def _compute_loop_count(self, face):
        wire_count = 0
        explorer = TopExp_Explorer(face, TopAbs_WIRE)
        while explorer.More():
            wire_count += 1
            explorer.Next()
        return wire_count
    
    # 辅助函数：形状指数均值/方差（采样计算）
    def _compute_shape_index_stats(self, face, grid_size=7):
        try:
            geom_surface = BRep_Tool.Surface(face)
            uv_points = self._sample_UV_points(face=face, grid_size=grid_size)

            shape_indices = []
            
            for u, v in uv_points:
                try:
                    # 使用 GeomLProp_SLProps 计算曲面属性（推荐方法）
                    # 这种方法更稳定，专门用于计算曲面微分几何属性
                    props = GeomLProp_SLProps(geom_surface, u, v, 2, 1e-6)
                    
                    # 检查是否计算成功
                    if not props.IsCurvatureDefined():
                        continue
                        
                    # 获取法向量
                    normal = props.Normal()
                    if gp_Vec(normal).Magnitude() < 1e-10:
                        continue
                        
                    # 获取主曲率
                    max_curvature, min_curvature = props.MaxCurvature(), props.MinCurvature()
                    # 如果两个曲率都是无穷大或未定义，跳过
                    if abs(max_curvature) > 1e10 or abs(min_curvature) > 1e10:
                        continue
                        
                    # 计算形状指数（Shape Index）
                    if abs(max_curvature - min_curvature) < 1e-10:
                        # 球面/平面情况
                        si = 0.0 if abs(max_curvature) < 1e-10 else (1.0 if max_curvature > 0 else -1.0)
                    else:
                        si = (2 / np.pi) * np.arctan((max_curvature + min_curvature) / (max_curvature - min_curvature))
                    
                    shape_indices.append(si)
                    
                except Exception as e:
                    # 跳过这个采样点，继续处理下一个
                    continue
            
            # 处理无有效采样点的情况
            if not shape_indices:
                return 0.0, 0.0
        
            # 返回均值和方差
            return float(np.mean(shape_indices)), float(np.var(shape_indices))
        
        except Exception as e:
            print(f"计算形状指数统计时出错: {e}")
            return 0.0, 0.0
    
    # 辅助函数：紧致度
    def _compute_compactness(self, face):
        pass

    # 辅助函数：主轴长度（PCA）
    def _compute_principal_axes_lengths(self, face, grid_size=10):  # grid_size=10 ≈100点
        points = self._sample_points(face, grid_size)
        if len(points) < 3:
            return np.zeros(3)
        # 中心化点云（提升不变性）
        mean = np.mean(points, axis=0)
        centered_points = points - mean
        cov = np.cov(centered_points.T)
        eigenvalues = np.linalg.eigvals(cov)
        # 处理负特征值（数值误差）
        eigenvalues = np.maximum(eigenvalues, 0)
        lengths = np.sqrt(np.sort(eigenvalues)[::-1])  # max, mid, min
        return lengths
    
    def _sample_points(self, face, grid_size=10):
        adaptor = BRepAdaptor_Surface(face)
        u_min, u_max = adaptor.FirstUParameter(), adaptor.LastUParameter()
        v_min, v_max = adaptor.FirstVParameter(), adaptor.LastVParameter()
        points = []
        
        # 均匀网格
        u_grid = np.linspace(u_min, u_max, grid_size)
        v_grid = np.linspace(v_min, v_max, grid_size)
        for u in u_grid:
            for v in v_grid:
                try:
                    p = adaptor.Value(u, v).Coord()  # (x,y,z)
                    points.append(p)
                except Exception:
                    pass  # 忽略无效UV点（e.g.,  trimming曲线外）
        
        return np.array(points) if points else np.empty((0, 3))
    
    def _sample_UV_points(self, face, grid_size=10):
        adaptor = BRepAdaptor_Surface(face)
        # 获取参数范围 - 使用正确的方法
        u_min, u_max = adaptor.FirstUParameter(), adaptor.LastUParameter()
        v_min, v_max = adaptor.FirstVParameter(), adaptor.LastVParameter()
        # 检查参数范围是否有效
        if u_max <= u_min or v_max <= v_min:
            # 如果参数范围无效，使用默认值
            u_min, u_max, v_min, v_max = 0.0, 1.0, 0.0, 1.0

        # 生成grid_size个等间距的u值（包含u_min和u_max）
        u_samples = np.linspace(u_min, u_max, grid_size)
        # 生成grid_size个等间距的v值（包含v_min和v_max）
        v_samples = np.linspace(v_min, v_max, grid_size)
        # 生成网格点的UV组合（笛卡尔积）
        u_grid, v_grid = np.meshgrid(u_samples, v_samples, indexing='ij')
        uv_points = np.stack([u_grid.ravel(), v_grid.ravel()], axis=1)

        return uv_points

    
    # 辅助函数：回转半径（平均惯性矩）
    def _compute_gyradius(self, face):
        props = GProp_GProps()
        brepgprop_SurfaceProperties(face, props)
        area = props.Mass()  # 面积
        if area == 0:
            return 0.0
        # 平均回转半径：用xx+yy+zz的平均
        Ixx = props.MomentOfInertia(gp_Ax1(props.CentreOfMass(), gp_Dir(1,0,0)))
        Iyy = props.MomentOfInertia(gp_Ax1(props.CentreOfMass(), gp_Dir(0,1,0)))
        Izz = props.MomentOfInertia(gp_Ax1(props.CentreOfMass(), gp_Dir(0,0,1)))
        I_avg = (Ixx + Iyy + Izz) / 3
        return math.sqrt(I_avg / area)

    # 辅助函数：偏心率
    def _compute_eccentricity(self, face):
        axes_lengths = self._compute_principal_axes_lengths(face)
        if axes_lengths[0] == 0:
            return 0.0
        return math.sqrt(1 - (axes_lengths[2] / axes_lengths[0]) ** 2)  # min/max
    

def face_make(enum):
    """
    生成一个面
    """
    if enum == 1:
        face = gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        topo_face = BRepBuilderAPI_MakeFace(face, -5, 5, -5, 5).Face()
        return topo_face
    if enum == 2:
        face = gp_Cylinder(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 3.0)
        topo_face = BRepBuilderAPI_MakeFace(face, 0, 2*3.14159, 0, 10).Face()
        return topo_face
    if enum == 3:
        cone_ax3 = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        face = gp_Cone(cone_ax3, math.pi/6, 10.0)  # 半锥角π/6，从顶点(0,0,0)延伸到Z=10
        # 参数范围：角度0-2π，轴向0-10
        topo_face = BRepBuilderAPI_MakeFace(face, 0, 2*math.pi, 0, 10).Face()
        return topo_face
    if enum == 4:
        # 球面：球心在原点，半径5，覆盖整个球面（角度范围0-2π，0-π）
        sphere_ax3 = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        face = gp_Sphere(sphere_ax3, 5.0)
        topo_face = BRepBuilderAPI_MakeFace(face, 0, 2*math.pi, 0, math.pi).Face()
        return topo_face
    if enum == 5:
        # 圆环面：Z轴为旋转轴，中心在原点，主半径8，管半径2
        face = gp_Torus(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 8.0, 2.0)
        # 参数范围：旋转角度0-2π，管角度0-2π
        topo_face = BRepBuilderAPI_MakeFace(face, 0, 2*math.pi, 0, 2*math.pi).Face()
        return topo_face


if __name__ == "__main__":

    topo_face = face_make(2)

    selected_props = ['area', 
                      'loop_count', 
                      'shape_index_mean', 
                      'shape_index_var', 
                      'principal_axes_lengths', 
                      'gyradius', 
                      'eccentricity']

    embedder = Geom_embedding_face(selected_props)



    shared_props, one_hot = embedder.embedding(topo_face)

    print(shared_props)
