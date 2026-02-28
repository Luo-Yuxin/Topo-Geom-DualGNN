import numpy as np
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_SurfaceType
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Vec, gp_Ax1, gp_Lin
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
from OCC.Core.TopExp import topexp
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.GProp import GProp_GProps
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.BRep import BRep_Tool
from scipy.spatial import KDTree
import math

class Position_Descriptor_Surface:
    """曲面位置描述器"""
    
    def __init__(self, face):
        # 载入topoDS_Shape以及载入topoDS_Face
        # self.shape = shape
        self.face = face
        # 将拓扑面转化为BRep面
        self.surface_adaptor = BRepAdaptor_Surface(face)
        # 获得面类型
        self.surface_type = self.surface_adaptor.GetType()
        
    def get_position_signature(self):
        """
        获取位置特征签名 - 综合位置描述
        """
        signature = {
            'surface_id': self.face.HashCode(999999),
            'surface_type': self.surface_type,
            'reference_point': self.get_reference_point(),
            'principal_direction': self.get_principal_direction(),
            'bounding_box_center': self._get_bounding_box_center(),
            'surface_area': self._get_area_surface(),
            'radius': self._get_radius_surface()
        }
        return signature    
        
    def get_reference_point(self):
        """
        获取面的参考点

        对于平面则获得其包围盒中点在面上的投影
        对于圆柱面则获得其包围盒中点在轴上的投影
        对于圆锥面则获得其顶点
        对于球面则获得其球心
        对于圆环则获得其包围盒中点在轴上的投影
        对于回转面则获得其包围盒中点在轴上的投影
        对于拉伸面以及其他类型的面则获得其包围盒中点
        """
        
        if self.surface_type == GeomAbs_SurfaceType.GeomAbs_Plane:
            return self._get_reference_point_plane()
        elif self.surface_type == GeomAbs_SurfaceType.GeomAbs_Cylinder:
            return self._get_reference_point_cylinder()
        elif self.surface_type == GeomAbs_SurfaceType.GeomAbs_Cone:
            return self._get_reference_point_cone()
        elif self.surface_type == GeomAbs_SurfaceType.GeomAbs_Sphere:
            return self._get_reference_point_sphere()
        elif self.surface_type == GeomAbs_SurfaceType.GeomAbs_Torus:
            return self._get_reference_point_torus()
        elif self.surface_type == GeomAbs_SurfaceType.GeomAbs_SurfaceOfRevolution:
            return self._get_reference_point_revolution()
        else:
            return self._get_reference_point_generic()
    
    def get_principal_direction(self):
        """
        获取主方向向量
        
        对于球面来说, 一般是没有明确方向的
        """
        if self.surface_type == GeomAbs_SurfaceType.GeomAbs_Plane:
            plane = self.surface_adaptor.Plane()
            return (plane.Axis().Direction(), "Norm")
        elif self.surface_type == GeomAbs_SurfaceType.GeomAbs_Cylinder:
            cylinder = self.surface_adaptor.Cylinder()
            return (cylinder.Axis().Direction(), "Axis")
        elif self.surface_type == GeomAbs_SurfaceType.GeomAbs_Cone:
            cone = self.surface_adaptor.Cone()
            return (cone.Axis().Direction(), "Axis")
        elif self.surface_type == GeomAbs_SurfaceType.GeomAbs_Torus:
            torus = self.surface_adaptor.Torus()
            return (torus.Axis().Direction(), "Axis")
        elif self.surface_type == GeomAbs_SurfaceType.GeomAbs_SurfaceOfRevolution:
            return (self.surface_adaptor.AxeOfRevolution().Direction(), "Axis")
        else:
            return (None, None)
   
    def _get_reference_point_plane(self):
        """平面的参考点 - 使用平面上距离原点最近的点"""
        plane = self.surface_adaptor.Plane()
        plane_origin = plane.Location()
        
        # 获取面的边界框中心作为参考点
        bbox_center = self._get_bounding_box_center()
        
        # 将边界框中心投影到平面上
        normal = plane.Axis().Direction()
        vec_to_plane = gp_Vec(bbox_center, plane_origin)
        dot_product = vec_to_plane.Dot(gp_Vec(normal))
        projected_point = gp_Pnt(gp_Vec(bbox_center.XYZ()).Added(gp_Vec(normal).Multiplied(dot_product)).XYZ())

        
        return projected_point
    
    def _get_reference_point_cylinder(self):
        """圆柱面的参考点 - 使用轴线上距离边界框中心最近的点"""
        cylinder = self.surface_adaptor.Cylinder()
        axis = cylinder.Axis()
        axis_location = axis.Location()
        axis_direction = axis.Direction()
        
        bbox_center = self._get_bounding_box_center()
        
        # 计算边界框中心到轴线的投影点
        vec_to_axis = gp_Vec(axis_location, bbox_center)
        dot_product = vec_to_axis.Dot(gp_Vec(axis_direction))
        reference_point = gp_Pnt(gp_Vec(axis_location.XYZ()).Added(gp_Vec(axis_direction).Multiplied(dot_product)).XYZ())
        
        return reference_point
    
    def _get_reference_point_cone(self):
        """圆锥面的参考点 - 使用顶点"""
        cone = self.surface_adaptor.Cone()
        apex = cone.Apex()
        return apex
    
    def _get_reference_point_sphere(self):
        """球面的参考点 - 使用球心"""
        sphere = self.surface_adaptor.Sphere()
        center = sphere.Location()
        return center
    
    def _get_reference_point_torus(self):
        """圆环面的参考点 - 使用环中心"""
        torus = self.surface_adaptor.Torus()
        axis = torus.Axis()
        center = axis.Location()
        return center
    
    def _get_reference_point_revolution(self):
        """
        回转面的参考点 - 包围盒中心在轴上投影
        """
        axis = self.surface_adaptor.AxeOfRevolution()
        axis_location = axis.Location()
        axis_direction = axis.Direction()

        bbox_center = self._get_bounding_box_center()

        # 计算包围盒中心到轴线的投影点
        vec_to_axis = gp_Vec(axis_location, bbox_center)
        dot_product = vec_to_axis.Dot(gp_Vec(axis_direction))
        reference_point = gp_Pnt(gp_Vec(axis_location.XYZ()).Added(gp_Vec(axis_direction).Multiplied(dot_product)).XYZ())
        
        return reference_point

    def _get_reference_point_generic(self):
        """通用曲面的参考点 - 使用边界框中心"""
        return self._get_bounding_box_center()
    
    def _get_bounding_box_center(self):
        """
        获取边界框中心
        """
        bbox = Bnd_Box()
        brepbndlib_Add(self.face, bbox)
        x_min, y_min, z_min, x_max, y_max, z_max = bbox.Get()
        
        center = gp_Pnt(
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            (z_min + z_max) / 2
        )
        return center
    
    def _get_area_surface(self):
        """
        获取表面积
        """
        props = GProp_GProps()
        brepgprop_SurfaceProperties(self.face, props)
        return props.Mass()
    
    def _get_radius_surface(self):
        """
        获得曲面的半径
        """
        if self.surface_type == GeomAbs_SurfaceType.GeomAbs_Plane:
            # 平面没有半径
            return (0.0, 0.0)
        elif self.surface_type == GeomAbs_SurfaceType.GeomAbs_Cylinder:
            # 圆柱面的半径即为圆柱面本身半径
            return (self.surface_adaptor.Cylinder().Radius(), self.surface_adaptor.Cylinder().Radius())
        elif self.surface_type == GeomAbs_SurfaceType.GeomAbs_Cone:
            # 圆锥面则存在两个半径, 即上端与下端的半径
            # 这需要结合其UV参数来确定, 在圆锥面上, U为角度参数, V为轴向参数
            surface_cone = self.surface_adaptor.Cone()
            # 获得圆锥面的轴
            axis_cone = surface_cone.Axis()
            # 获得圆锥面的顶点
            vertex = surface_cone.Apex()
            # 获得圆锥面的半锥角
            semi_angle = surface_cone.SemiAngle()
            # 获得圆锥面的UV范围
            u_min, u_max = self.surface_adaptor.FirstUParameter(), self.surface_adaptor.LastUParameter()
            v_min, v_max = self.surface_adaptor.FirstVParameter(), self.surface_adaptor.LastVParameter()

            # 根据V值确定最终半径
            # 距离顶点最近/最远的轴向距离乘以半锥角的正切值即为半径
            radius_1 = abs(v_min) * math.tan(semi_angle)
            radius_2 = abs(v_max) * math.tan(semi_angle)

            return (radius_1, radius_2, semi_angle)
        elif self.surface_type == GeomAbs_SurfaceType.GeomAbs_Sphere:
            # 球面的半径即为球面本身的半径
            return (self.surface_adaptor.Sphere().Radius(), self.surface_adaptor.Sphere().Radius())
        elif self.surface_type == GeomAbs_SurfaceType.GeomAbs_Torus:
            # 对于圆环面, 其有一个大径和一个小径
            return (self.surface_adaptor.Torus().MajorRadius(), self.surface_adaptor.Torus().MinorRadius())
        elif self.surface_type == GeomAbs_SurfaceType.GeomAbs_SurfaceOfRevolution:
            # 对于回转曲面, 可能需要分析其最终形成的面型来判断其半径
            # 比如平行与回转轴的line, 则形成的是圆柱面, 最终还是line到轴线的距离
            # 而要是不平行的line, 则形成圆锥面, 要是圆弧, 则是环, 甚至是球
            # 这部分难度较大, 或许可以考虑使用面两端的边的半径
            # 考虑两端的边的半径
            axis_revolution = self.surface_adaptor.AxeOfRevolution()
            # gp_Ax1并没有距离算法
            # 因此我们将其转化为gp_Lin来进行点到直线距离计算
            axis_line = gp_Lin(axis_revolution)
            # 回转母线
            basis_curve = self.surface_adaptor.BasisCurve()
            # 获得母线上端点坐标
            v_min = self.surface_adaptor.FirstVParameter()
            v_max = self.surface_adaptor.LastVParameter()
            pnt_min = basis_curve.Value(v_min)
            pnt_max = basis_curve.Value(v_max)
            # 计算端点到轴的距离
            radius_1 = axis_line.Distance(pnt_min)
            #radius_2 = axis_line.Direction(pnt_max)
            radius_2 = axis_line.Distance(pnt_max)

            return (radius_1, radius_2)
        else:
            return (None)

class Surface_Relationship_Analyzer:
    """
    曲面关系分析器
    统一了判断类型
    """

    def __init__(self, shape, self_loops=False, tolerance=1e-6, angular_tolerance=1.0):

        # 获得几何体形状类
        self.shape = shape
        # 实例化拓扑解析器
        self.explorer = TopologyExplorer(self.shape)
        # 获得几何体拓扑面列表
        self.topo_face_list = list(self.explorer.faces())
        self.topo_edge_list = list(self.explorer.edges())
        # 获得几何体面边序列ID
        # self.face_mapping = self._step_topo_mapping(self.shape, "face")
        # self.edge_mapping = self._step_topo_mapping(self.shape, "edge")
        self.face_mapping = self._build_mapping_from_list(self.topo_face_list)
        self.edge_mapping = self._build_mapping_from_list(self.topo_edge_list)
        # 设置是否开启自环
        self.self_loops = self_loops
        # 获得几何体拓扑连接性列表
        self.adjacency_map = self._build_adjacency_lookup()
        # 设置判断容差
        self.tolerance = tolerance
        # 设置角度容差
        self.angular_tolerance = math.radians(angular_tolerance)

        # 因为对关系做了统一的描述
        # 不再需要分类型判断了
    
    def analyze_relationship(self, desc_face_1, desc_face_2):
        """
        分析两个面之间的关系
        """
        # 定义合法面元组
        valid_tuple = (0, 1, 2, 3, 4, 7)
        # 初始化关系字典
        results = {}
        # 检测两面中是否存在暂无法处理的非法面
        # 若存在非法曲面, 则将关系定义为None
        if (self._other_surface_filter(valid_tuple, desc_face_1['surface_type']) and
            self._other_surface_filter(valid_tuple, desc_face_2['surface_type'])):

            # 分析面之间的姿态信息
            is_parallel, is_perpendicular, angle_to_direction = self._orientation_relation_face(desc_face_1, desc_face_2)
            # 分析面之间的位置信息
            distance = self._distance_relation_face(desc_face_1, desc_face_2)
            # 分析面之间的同轴关系, 仅限于为Axis类型方向的面
            is_coaxial = self._coaxial_relation_face(desc_face_1, desc_face_2)
            # 分析面之间的G0/G1连续问题, 这里只考虑潜在的连续关系, 不要求实际会发生接触
            is_tangent = self._Continuity_relation_face(desc_face_1, desc_face_2)
            # 分析面与面之间是否属于同一个面
            is_coplanar = self._coplanar_relation_face(desc_face_1, desc_face_2)

            # 更新输出字典
            results.update({
                'is_parallel': is_parallel,
                'is_perpendicular': is_perpendicular,
                'is_coaxial': is_coaxial,
                'is_tangent': is_tangent,
                'is_coplanar': is_coplanar,
                'angle_to_direction': angle_to_direction,
                'distance': distance
            })
        # 对于包含其他类型曲面, 我们仅考虑存在相切关系
        else:
            is_parallel = False
            is_perpendicular = False
            is_coaxial = False
            is_tangent = self._Continuity_relation_face(desc_face_1, desc_face_2)
            is_coplanar = False
            angle_to_direction = False
            distance = self._distance_relation_face(desc_face_1, desc_face_2)

            # 更新输出字典
            results.update({
                'is_parallel': is_parallel,
                'is_perpendicular': is_perpendicular,
                'is_coaxial': is_coaxial,
                'is_tangent': is_tangent,
                'is_coplanar': is_coplanar,
                'angle_to_direction': angle_to_direction,
                'distance': distance
            })

        return results
    
    def _other_surface_filter(self, valid_tuple, type_desc_face):
        """
        部分曲面关系暂无法分析与其他面的几何关系
        因此需要剔除
        """
        # 遍历面类型, 若面类型不在有效面类型元组中则返回False
        valid_or_not = False
        for valid_type in valid_tuple:
            if type_desc_face == valid_type:
                valid_or_not = True
        
        return valid_or_not
    
    def _step_topo_mapping(self, shape, mapped_topo_type):
        """
        读取shape获取该对象的映射关系
        给一个内存管理器, 用于存储映射关系
        目前可被_build_mapping_from_list所替代

        :param shape: TopoDS_Solid型的shape
        :param mapped_topo_type: 映射的拓扑类型
        :return: 映射关系字典
        """

        # 新建映射关系字典
        mapping_dict = {}
        # 获取shape的拓扑探测器
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
        
        return mapping_dict
    
    def _build_mapping_from_list(self, topo_list):
        """
        生成hash->id的映射关系
        
        :param topo_list: 面或边等列表

        :return mapping_dict: hash(999999)->id
        """
        # 初始化映射字典
        mapping_dict = {}
        # 通过枚举值进行遍历
        for idx, item in enumerate(topo_list):
            mapping_dict[item.HashCode(999999)] = idx
        return mapping_dict

    def _build_adjacency_lookup(self):
        """
        [修正版] 构建 (面ID_1, 面ID_2) -> [Edge_1, Edge_2, ...] 的查找字典
        
        改进点：
        1. 使用 TopologyExplorer 代替底层 Map 手动遍历，提高稳定性。
        2. 引入 Is3DCurve() 过滤，剔除退化边，与 build_topo_graph 逻辑保持一致。
        3. 更鲁棒的迭代器处理。
        """
        # 初始化查找表
        adjacency_map = {}

        # 初始化边分析工具
        sa_edge = ShapeAnalysis_Edge()

        # 1. 获取所有边，并过滤掉退化边 (匹配你的 build_topo_graph 逻辑)
        # BRepAdaptor_Curve(edge).Is3DCurve() 用于判断边是否具有三维几何特征
        valid_edges = [edge for edge in self.topo_edge_list if BRepAdaptor_Curve(edge).Is3DCurve()]

        # 2. 遍历有效边，查找共用该边的面
        for edge in valid_edges:
            # 获取共用该边的面生成器 -> 转列表
            connected_faces = list(self.explorer.faces_from_edge(edge))
            
            # 仅处理连接了两个面的情况 (常规邻接)
            # 如果需要处理 seam 边 (自环)，可以在这里通过 len(connected_faces) == 1 判断
            if len(connected_faces) == 2:
                face_1 = connected_faces[0]
                face_2 = connected_faces[1]

                # 获取面ID (确保使用方括号访问字典)
                try:
                    id_1 = self.face_mapping[face_1.HashCode(999999)]
                    id_2 = self.face_mapping[face_2.HashCode(999999)]
                except KeyError:
                    # 极少数情况下，Explorer 遍历出的面可能与 face_mapping 初始化时的面哈希不一致
                    # 这种情况通常可以忽略，或者打印警告
                    continue
                
                if id_1 != -1 and id_2 != -1:
                    # 利用排序元组构造唯一 Key (避免 (1,2) 和 (2,1) 重复)
                    if id_1 < id_2:
                        key = (id_1, id_2)
                    else:
                        key = (id_2, id_1)

                    # 将其保存入字典
                    if key not in adjacency_map:
                        adjacency_map[key] = []
                    adjacency_map[key].append(edge)
            # 针对出现了自环边的情况
            elif len(connected_faces) == 1 and self.self_loops:
                face_1 = connected_faces[0]
                # 用于二次检查是否为衔接边
                if sa_edge.IsSeam(edge, face_1):
                    try:
                        id_1 = self.face_mapping[face_1.HashCode(999999)]
                    except KeyError:
                        # 极少数情况下，Explorer 遍历出的面可能与 face_mapping 初始化时的面哈希不一致
                        # 这种情况通常可以忽略，或者打印警告
                        continue
                    if id_1 != -1:
                        key = (id_1, id_1)

                        # 将其保存入字典
                        if key not in adjacency_map:
                            adjacency_map[key] = []
                        adjacency_map[key].append(edge)

        
        return adjacency_map

    def _get_normal_at_uv(self, face, u, v):
        """
        计算面在指定 UV 处的实际法线(考虑拓扑方向)
        """
        # 1. 计算几何属性
        surf = BRepAdaptor_Surface(face, True) # True 限制在 Face 边界内
        props = BRepLProp_SLProps(surf, u, v, 1, 1e-6) # 1阶导数即可

        if not props.IsNormalDefined():
            # 极少数情况(如奇点)，可能无法计算法线
            return None

        # 2. 获取几何法线
        normal = props.Normal() # gp_Dir

        # 3. 根据拓扑方向修正法线
        # 这是你之前困惑的关键点：必须根据 Orientation 翻转几何法线
        if face.Orientation() == TopAbs_REVERSED:
            normal.Reverse()
            
        return normal

    def _dihedral_angle_at_shared_edge(self, edge, face1, face2):
        """
        计算该边的两个邻接面在该边参数中点处的二面角。
        策略: 
        获取 Edge 在全局坐标系下的中点 (Mid Point in 3D)
        将该 3D 点分别投影到两个面上, 获取确切的 UV 坐标
        增加对 Face Location (局部坐标系) 的支持
        :param edge: 共享边
        :param face1: 邻接面1
        :param face2: 邻接面2
        :return: float 弧度制下的夹角 (0 - pi)
        """
        # 1. 获取 Edge 在全局坐标系下的中点
        # BRepAdaptor_Curve 会自动处理 Edge 的 Location，返回的是 Global 坐标
        edge_adaptor = BRepAdaptor_Curve(edge)
        t_min = edge_adaptor.FirstParameter()
        t_max = edge_adaptor.LastParameter()
        mid_param = (t_min + t_max) / 2.0
        global_pnt = edge_adaptor.Value(mid_param)

        # 2. 定义一个内部辅助函数：将 3D 点投影到 Face 上获取 UV
        def get_accurate_uv(face, p_global):
            # A. 获取几何曲面 (Geometry)
            geom_surf = BRep_Tool.Surface(face)
            
            # B. 处理坐标变换 (Critical Step!)
            # face.Location() 记录了面的位移/旋转
            # 我们必须把全局点 p_global 逆变换回面的局部坐标系 p_local
            # 否则 GeomAPI_ProjectPointOnSurf 投影结果是错的
            loc = face.Location()
            if not loc.IsIdentity():
                p_local = p_global.Transformed(loc.Inverted().Transformation())
            else:
                p_local = p_global
            
            # C. 执行投影
            # GeomAPI_ProjectPointOnSurf 极其稳健，能找到最近点
            projector = GeomAPI_ProjectPointOnSurf(p_local, geom_surf)
            
            if projector.NbPoints() > 0:
                u, v = projector.LowerDistanceParameters()
                return u, v
            else:
                # 理论上不应发生，除非 Edge 严重偏离 Face (模型破损)
                return 0.0, 0.0

        # 3. 分别投影获取准确的 UV
        # 这种方式比 pcurve.Value(t) 慢一点，但对于相切判断绝对准确
        u1, v1 = get_accurate_uv(face1, global_pnt)
        u2, v2 = get_accurate_uv(face2, global_pnt)

        # 4. 计算法线
        n1 = self._get_normal_at_uv(face1, u1, v1)
        n2 = self._get_normal_at_uv(face2, u2, v2)

        if n1 is None or n2 is None:
            return 0.0 

        # 5. 计算夹角
        dot_val = n1.Dot(n2)
        
        # Clamp 防止浮点误差导致的 acos(1.0000001) 崩溃
        if dot_val > 1.0: dot_val = 1.0
        elif dot_val < -1.0: dot_val = -1.0

        angle = math.acos(dot_val)
        
        return angle

    def _is_G1_continue(self, edge, face1, face2, tol_deg=1.0, debug_flag=False):
        """
        判断是否 G1 连续
        """
        angle = self._dihedral_angle_at_shared_edge(edge, face1, face2)
        # 转换为角度便于理解
        angle_deg = math.degrees(angle)

        if debug_flag:
            print(angle)
            print("***")

        # 判定逻辑：
        # 如果两个面是平滑连接的，法线夹角应该接近 0 (或者 180，取决于法线是指向实体内还是外)
        # 对于标准的 Solid（法线均指向外），平滑连接处的法线夹角应接近 0。
        
        # 如果你的逻辑是 angle 接近 0 为平滑：
        if angle_deg < tol_deg:
            return True
            
        # 还有一种情况：对于 seam edge（闭合圆柱侧面），法线是重合的，angle 为 0
        
        return False

    def _orientation_relation_face(self, desc_face_1, desc_face_2):
        """
        用来判断面与面之间的空间姿态关系
        平行/垂直/倾斜角
        """
        # 调试用
        # print(desc_face_1['surface_type'])
        # print(desc_face_2['surface_type'])
        # print("***")

        # 获取方向向量以及方向向量类型信息
        dir_face_1, dir_type_1 = desc_face_1['principal_direction']
        dir_face_2, dir_type_2 = desc_face_2['principal_direction']

        # 针对不同的方向向量类型进行不同的判断
        if (dir_type_1 == None or dir_type_2 == None):
            # 如果不存在方向向量, 则说明没有姿态关系, 角度也为空值
            is_parallel = False
            is_perpendicular = False
            angle_to_direction = None
        elif (dir_type_1 == dir_type_2):
            # 即两个方向向量均为法向量或轴向量
            # 计算方向向量的夹角
            angle_12 = dir_face_1.Angle(dir_face_2)
            # 在这种情况下, 两法向量夹角为0或pi则表明具有平行关系
            if (angle_12 < self.angular_tolerance or abs(angle_12-math.pi) < self.angular_tolerance):
                is_parallel = True
                is_perpendicular = False
                angle_to_direction = angle_12
            elif (abs(angle_12 - math.pi/2.0) < self.angular_tolerance):
                # 若夹角为pi/2, 则认为其为垂直关系
                is_parallel = False
                is_perpendicular = True
                angle_to_direction = angle_12
            else:
                # 若其他条件均不满足, 则获得角度
                is_parallel = False
                is_perpendicular = False
                angle_to_direction = angle_12
        elif (dir_type_1 != dir_type_2):
            # 对于两者方向向量类型不符合
            # 计算方向向量的夹角
            angle_12 = dir_face_1.Angle(dir_face_2)
            # 在这种情况下, 两法向量夹角为0或pi则表明具有垂直关系
            if (angle_12 < self.angular_tolerance or abs(angle_12-math.pi) < self.angular_tolerance):
                is_parallel = False
                is_perpendicular = True
                angle_to_direction = angle_12
            elif (abs(angle_12 - math.pi/2.0) < self.angular_tolerance):
                # 若夹角为pi/2, 则认为其为平行关系
                is_parallel = True
                is_perpendicular = False
                angle_to_direction = angle_12
            else:
                # 若其他条件均不满足, 则获得角度
                is_parallel = False
                is_perpendicular = False
                angle_to_direction = angle_12
        # 返回结果
        return is_parallel, is_perpendicular, angle_to_direction

    def _distance_relation_face(self, desc_face_1, desc_face_2):
        """
        用来判断面与面之间的空间距离关系
        一般只对具有平行关系的面才有效
        但为了冗余起见, distance被设置成了一个具有两个值的元组
        对于平面来说, 其中分别记录了一面参考点到另一面的距离
        对于没有方向性的面, 比如球面, 则继承另一面的方向属性
        """
        # 获得各个面的方向向量信息以及参考点信息
        dir_face_1, dir_type_1 = desc_face_1['principal_direction']
        dir_face_2, dir_type_2 = desc_face_2['principal_direction']

        # 作为初步的尝试, 关于没有方向性的面我们只考虑球面
        # 首先判断是否为球面
        if desc_face_1['surface_type'] == desc_face_2['surface_type'] == GeomAbs_SurfaceType.GeomAbs_Sphere:
            # 若两个面均为球面, 则赋予一个属性
            dir_face_1, dir_type_1 = gp_Dir(0, 0, 1), 'Axis'
            dir_face_2, dir_type_2 = gp_Dir(0, 0, 1), 'Axis'
        elif desc_face_1['surface_type'] == GeomAbs_SurfaceType.GeomAbs_Sphere:
            # 若1面为球面
            # 球面将继承另一面的方向属性
            dir_face_1, dir_type_1 = dir_face_2, dir_type_2
        elif desc_face_2['surface_type'] == GeomAbs_SurfaceType.GeomAbs_Sphere:
            # 若2面为球面
            # 球面将继承另一面的方向属性
            dir_face_2, dir_type_2 = dir_face_1, dir_type_1
        else:
            # 若都不是球面
            # 则仍维持原状
            pass
        
        # 获得各个面的参考点信息
        ref_point_1 = desc_face_1['reference_point']
        ref_point_2 = desc_face_2['reference_point']
        # 获得从1点指向2点的向量
        vector_12 = gp_Vec(ref_point_1, ref_point_2)

        # 对于存在法向量的, 距离即将点连接向量投影到法向量上
        # 对于不存在法向量的, 需要将点连接向量与轴向方向向量做叉积
        if (dir_type_1 == 'Norm'):
            # 对于第一个面具有的是法向量方向
            if (dir_type_2 == 'Norm'):
                # 若两个都是法向量方向, 则距离1为其两参考点连接向量在法向量1上的投影
                # (在不平行下针对不同法向量投影机结果不同)
                # 距离2为参考点距离
                distance_1 = abs(vector_12.Dot(gp_Vec(dir_face_1)))
                distance_2 = vector_12.Magnitude()
            elif (dir_type_2 == 'Axis'):
                # 若第二面是轴方向, 则计算点在法向量的投影(因为法向量本身包含到面距离的相关性)
                # 因为对于轴来说, 在不平行时其距离为参考点到平面距离
                # 第二个距离与以前同步为参考点距离
                distance_1 = abs(vector_12.Dot(gp_Vec(dir_face_1)))
                distance_2 = vector_12.Magnitude()
            else:
                # 当面2不具备方向性, 我们将不进行计算(保留参考点距离)
                # 或许可以计算点到平面的距离?
                distance_1 = None
                distance_2 = vector_12.Magnitude()
        elif (dir_type_1 == 'Axis'):
            # 对于第一个面具有的是轴向量方向
            if (dir_type_2 == 'Norm'):
                # 若第二面是法方向, 则计算点在法向的投影
                # 第二个距离与以前同步为参考点距离
                distance_1 = abs(vector_12.Dot(gp_Vec(dir_face_2)))
                distance_2 = vector_12.Magnitude()
            elif (dir_type_2 == 'Axis'):
                # 若都是轴向向量, 则计算两轴线之间的距离
                # 使用pyocc的内置方法计算
                distance_1 = gp_Lin(ref_point_1, dir_face_1).Distance(gp_Lin(ref_point_2, dir_face_2))
                # 第二距离为参考点之间的距离
                distance_2 = vector_12.Magnitude()
            else:
                # 当面2不具备方向性, 我们将不进行计算
                # 或许可以计算点到轴线的距离?
                distance_1 = None
                distance_2 = vector_12.Magnitude()
        else:
            # 当面1不具备方向性, 我们将不进行计算
            distance_1 = None
            distance_2 = vector_12.Magnitude()

        return (distance_1, distance_2)

    def _coaxial_relation_face(self, desc_face_1, desc_face_2):
        """
        判断两面之间是否存在同轴关系
        仅限于Axis类型方向的面
        """
        # 获取方向向量以及方向向量类型信息
        dir_face_1, dir_type_1 = desc_face_1['principal_direction']
        dir_face_2, dir_type_2 = desc_face_2['principal_direction']
        
        # 作为初步的尝试, 关于没有方向性的面我们只考虑球面
        # 首先判断是否为球面
        if desc_face_1['surface_type'] == desc_face_2['surface_type'] == GeomAbs_SurfaceType.GeomAbs_Sphere:
            # 若两个面均为球面, 则赋予一个属性
            dir_face_1, dir_type_1 = gp_Dir(0, 0, 1), 'Axis'
            dir_face_2, dir_type_2 = gp_Dir(0, 0, 1), 'Axis'
        elif desc_face_1['surface_type'] == GeomAbs_SurfaceType.GeomAbs_Sphere:
            # 若1面为球面
            # 球面将继承另一面的方向属性
            dir_face_1, dir_type_1 = dir_face_2, dir_type_2
        elif desc_face_2['surface_type'] == GeomAbs_SurfaceType.GeomAbs_Sphere:
            # 若2面为球面
            # 球面将继承另一面的方向属性
            dir_face_2, dir_type_2 = dir_face_1, dir_type_1
        else:
            # 若都不是球面
            # 则仍维持原状
            pass

        # 获得参考点信息以及距离信息/ 目前来看只需要距离信息就足够了
        # ref_point_1 = desc_face_1['reference_point']
        # ref_point_2 = desc_face_2['reference_point']
        distance = self._distance_relation_face(desc_face_1, desc_face_2)

        # 初始化同轴标识
        is_coaxial = False

        if (dir_type_1 == 'Axis' and dir_type_2 == 'Axis'):
            # 对于都是轴方向, 首先判断其是否是平行的
            angle_12 = dir_face_1.Angle(dir_face_2)
            # 两法向量夹角为0或pi则表明具有平行关系
            if (angle_12 < self.angular_tolerance or abs(angle_12-math.pi) < self.angular_tolerance):
                # 若两轴线平行, 判断两轴线间的距离
                # 通过参考点以及方向向量构建空间直线/ 目前直接更改为通过距离函数计算
                # line_distance = gp_Lin(ref_point_1, dir_face_1).Distance(gp_Lin(ref_point_2, dir_face_2))
                if distance[0] < self.tolerance:
                    # 若两轴线重合, 即说明同轴
                    is_coaxial = True
                else:
                    # 轴线不重合
                    is_coaxial = False
            # 轴线不平行
            else:
                is_coaxial = False
        # 存在非轴方向
        else:
            is_coaxial = False

        # 返回判断结果
        return is_coaxial
    
    def _coplanar_relation_face(self, desc_face_1, desc_face_2):
        """
        判断两面之间是否是共面的
        """
        is_coplanar = False
        # 判断两个面之间是否是共面的则判断两个面相对应的几何形状参数描述是否是一致的且位置参数描述是对应的上的
        # 两个面首先得是一种类型
        if desc_face_1['surface_type'] == desc_face_2['surface_type']:
            # 对于平面
            if desc_face_1['surface_type'] == GeomAbs_SurfaceType.GeomAbs_Plane:
                # 要求平行且距离为0
                is_parallel, is_perpendicular, C = self._orientation_relation_face(desc_face_1, desc_face_2)
                distance = (self._distance_relation_face(desc_face_1, desc_face_2))[0]
                if is_parallel and np.abs(distance) < self.tolerance:
                    is_coplanar = True
                else:
                    is_coplanar = False
            # 对于圆柱面
            elif desc_face_1['surface_type'] == GeomAbs_SurfaceType.GeomAbs_Cylinder:
                # 要求同轴且半径相同
                is_coaxial = self._coaxial_relation_face(desc_face_1, desc_face_2)
                if is_coaxial and np.abs(np.subtract(desc_face_1['radius'][0], desc_face_2['radius'][0])) < self.tolerance:
                    is_coplanar = True
                else:
                    is_coplanar = False
            # 对于圆锥面
            elif desc_face_1['surface_type'] == GeomAbs_SurfaceType.GeomAbs_Cone:
                # 要求同轴且同顶点且同半角
                is_coaxial = self._coaxial_relation_face(desc_face_1, desc_face_2)
                point_1 = desc_face_1['reference_point']
                point_2 = desc_face_2['reference_point']
                point_distance = point_1.Distance(point_2)
                semi_angle_1 = desc_face_1['radius'][2]
                semi_angle_2 = desc_face_2['radius'][2]
                if (is_coaxial and np.abs(point_distance) < self.tolerance and 
                    np.abs(np.subtract(math.sin(semi_angle_1), math.sin(semi_angle_2))) < self.tolerance):
                    is_coplanar = True
                else:
                    is_coplanar = False
            # 对于圆环面
            elif desc_face_1['surface_type'] == GeomAbs_SurfaceType.GeomAbs_Torus:
                # 要求同轴同参考点且同大径与小径
                is_coaxial = self._coaxial_relation_face(desc_face_1, desc_face_2)
                point_1 = desc_face_1['reference_point']
                point_2 = desc_face_2['reference_point']
                point_distance = point_1.Distance(point_2)
                major_radius_sub = np.subtract(desc_face_1['radius'][0], desc_face_2['radius'][0])
                minor_radius_sub = np.subtract(desc_face_1['radius'][1], desc_face_2['radius'][1])
                if (is_coaxial and point_distance<self.tolerance and 
                    major_radius_sub<self.tolerance and minor_radius_sub<self.tolerance):
                    is_coplanar = True
                else:
                    is_coplanar = False
            # 对于球面：
            elif desc_face_1['surface_type'] == GeomAbs_SurfaceType.GeomAbs_Sphere:
                # 要求同参考点且同半径
                is_coaxial = self._coaxial_relation_face(desc_face_1, desc_face_2)
                if (is_coaxial and np.subtract(desc_face_1['radius'][0], desc_face_2['radius'][0])<self.tolerance):
                    is_coplanar = True
                else:
                    is_coplanar = False
            # 其他类型面待补充
            else:
                is_coplanar = False
        else:
            is_coplanar = False
        
        return is_coplanar

    def _Continuity_relation_face_by_topo(self, desc_face_1, desc_face_2):
        """
        判断拓扑连接上的相切, 主要是为了给非常规曲面使用
        """
        id_face_1 = self.face_mapping[desc_face_1['surface_id']]
        id_face_2 = self.face_mapping[desc_face_2['surface_id']]

        # 针对衔接边进行优化
        if id_face_1 == id_face_2:
            # 由于在几何关系图的构造过程中desc_face_1与desc_face_2就不相等
            # 因此这里更像是一种保护措施
            is_tangent = False
            return is_tangent
        
        # 构造查询 Key
        key = tuple(sorted((id_face_1, id_face_2)))

        # 检查是否相邻
        if key not in self.adjacency_map:
            # print("这两个面不相邻")
            return False # 或者返回 None
        shared_edges = self.adjacency_map[key]
        debug_flag = False

        # 遍历所有共享边进行判断
        # 只要有一条边不满足 G1，通常这两个面的关系就不能算作“平滑”
        # 初始化相切标签
        is_tangent = True
        for edge in shared_edges:
            if not self._is_G1_continue(edge, self.topo_face_list[id_face_1], self.topo_face_list[id_face_2], debug_flag=debug_flag):
                is_tangent = False
                break
        return is_tangent

    def _Continuity_relation_face_by_geom(self, desc_face_1, desc_face_2):
        """
        判断两面之间是否存在可能的G0连续关系
        """
        # 相切公式为:
        # distance == radius_1 + radius_2 or
        # distance == abs(radius_1 - radius_2)
        # 关于distance的定义, 在获取distance的部分有介绍
        # 对于存在Norm的面, 即参考点连线在norm上的投影长度
        # 除此之外的Axis面(对于球面来说也为Axis, 因为继承了或设定了)
        # 则为轴线之间的最短距离
        # 再说明一下为啥会出现两种radius, 这是为了给圆锥面以及圆环面这种存在大径和小径的面用的

        # 对于圆环面来说相切是一个十分麻烦的东西, 我们针对圆环面与其他面的相切情况做单独的判断

        # 首要的是获得两面的半径信息以及两面的距离信息
        distance = self._distance_relation_face(desc_face_1, desc_face_2)
        # 如果不存在距离信息就直接跳过
        if distance[0] is None:
            # 如果没有定义的距离（例如面不平行），几何相切通常不成立（除非是特殊曲面）
            # 或者跳过矩阵计算
            return False
        radius_1 = desc_face_1['radius']
        radius_2 = desc_face_2['radius']
        is_parallel, is_perpendicular, angle_12 = self._orientation_relation_face(desc_face_1, desc_face_2)

        # 初始化 is_tangent 为 False
        is_tangent = False
        
        # 根据是否存在圆环面来判断
        if (desc_face_1['surface_type'] == GeomAbs_SurfaceType.GeomAbs_Torus or 
            desc_face_2['surface_type'] == GeomAbs_SurfaceType.GeomAbs_Torus):
            # 第一个面是圆环面
            if desc_face_2['surface_type'] != GeomAbs_SurfaceType.GeomAbs_Torus:
                face_torus = desc_face_1
                face_any = desc_face_2
            # 第二个面是圆环面
            elif desc_face_1['surface_type'] != GeomAbs_SurfaceType.GeomAbs_Torus:
                face_torus = desc_face_2
                face_any = desc_face_1
            # 如果都是圆环面
            # 则同参考点, 同大小径的圆环面将被定义为相切
            else:
                face_torus = desc_face_1
                face_any = desc_face_2
                # 关于圆环面的相切问题, 会比较复杂, 具体原理参照笔记
                # 首先是圆环中轴重合但环形轴不重合的情况
                # 此时要求两圆环得同轴
                if self._coaxial_relation_face(face_torus, face_any):
                    # 外切判断
                    if np.abs((radius_2[0]-radius_1[0])**2 + 
                            distance[1]**2 - (radius_2[1]+radius_1[1])**2) < self.tolerance:
                        is_tangent = True
                    # 内切判断
                    elif np.abs((radius_2[0]-radius_1[0])**2 + 
                                distance[1]**2 - (radius_2[1]-radius_1[1])**2) < self.tolerance:
                        is_tangent = True
                    else:
                        is_tangent = False
                # 此时为环形轴重合, 中心轴不重合的情况
                else:
                    # 首先要求中心轴要相交或平行, 小径必须相等
                    if ((distance[0] < self.tolerance or is_parallel) and
                        np.abs(radius_1[1]-radius_2[1] < self.tolerance)):
                        # 对于轴平行的
                        if is_parallel:
                            # 若满足R1+R2-L=0 OR R2-R1-L=0
                            if (np.abs(radius_1[0]+radius_2[0]-distance[1])<self.tolerance or 
                                np.abs(np.abs(radius_1[0]-radius_2[0])-distance[1])<self.tolerance):
                                is_tangent = True
                            else:
                                is_tangent = False
                        # 如果满足三角形条件(即中心轴会相交)
                        # 如果根据余弦定理计算出的角度与轴夹角相同(或互补, 因为我们无法确定轴指向)
                        # 当然有一个前提是能够构成三角形(两边之和大于第三边)
                        else:
                            angle_acos = math.acos((distance[1]**2 - radius_1[0]**2 - radius_2[0]**2) / 
                                                (2*radius_1[0]*radius_2[0]+1e-6))
                            # 判断相等或互补
                            if (np.abs(angle_acos - angle_12) < self.tolerance or 
                                np.abs(angle_acos + angle_12 - np.pi) < self.tolerance):
                                is_tangent = True
                            else:
                                is_tangent = False
                    else:
                        is_tangent = False

            # 若另一面是平面
            if face_any['surface_type'] == GeomAbs_SurfaceType.GeomAbs_Plane:
                # 判断平面与圆环面相切, 首先是平面要与圆环面垂直
                # A, is_perpendicular, C = self._orientation_relation_face(face_torus, face_any)
                # 若确实是垂直的
                if is_perpendicular:
                    radius_torus_minor = face_torus['radius'][1]
                    # 垂直且距离为小径则说明相切
                    if np.abs(np.subtract(distance[0], radius_torus_minor)) < self.tolerance:
                        is_tangent = True
                    else:
                        is_tangent = False
                else:
                    # 若不垂直则不相切
                    is_tangent = False
            # 若另一面是圆柱面
            if face_any['surface_type'] == GeomAbs_SurfaceType.GeomAbs_Cylinder:
                # 判断圆柱面与圆环面是否平行或垂直
                # is_parallel, is_perpendicular, C = self._orientation_relation_face(face_torus, face_any)
                # 如果是平行的
                if is_parallel:
                    # 则要求圆柱与圆环是同轴的
                    is_coaxial = self._coaxial_relation_face(face_torus, face_any)
                    # 如果是同轴的
                    if is_coaxial:
                        # 如果圆柱半径为大径减小径或大径加小径
                        if (np.abs(np.subtract(face_any['radius'][0], face_torus['radius'][0]+face_torus['radius'][1])) < self.tolerance or
                            np.abs(np.subtract(face_any['radius'][0], face_torus['radius'][0]-face_torus['radius'][1])) < self.tolerance):
                            # 两面相切
                            is_tangent = True
                        else:
                            is_tangent = False
                    else:
                        is_tangent = False
                # 如果是垂直的
                elif is_perpendicular:
                    # 则要求距离为大径且圆柱半径与小径相同
                    if (np.abs(np.subtract(distance[0], face_torus['radius'][0])) < self.tolerance and
                        np.abs(np.subtract(face_any['radius'][0], face_torus['radius'][1])) < self.tolerance):
                        # 两面相切
                        is_tangent = True
                    else:
                        is_tangent = False
                else:
                    is_tangent = False
            # 若另一面是球面
            if face_any['surface_type'] == GeomAbs_SurfaceType.GeomAbs_Sphere:
                # 球面的半径等于圆环的小径且距离为圆环的大径
                if (np.abs(np.subtract(face_any['radius'][0], face_torus['radius'][1])) < self.tolerance and
                    np.abs(np.subtract(distance[0], face_torus['radius'][0])) < self.tolerance):
                    # 两面相切
                    is_tangent = True
                else:
                    is_tangent = False
        # 对于更普适的情况
        else:
            # 我们使用np矩阵计算代替循环遍历所有半径的计算
            # |r1_1, 1| * |   1,    1|
            # |r1_2, 1|   |r2_1, r2_2|
            calc_distance_1 = np.abs(np.dot(np.array([[radius_1[0], 1.0], [radius_1[1], 1.0]]), 
                                            np.array([[1.0, 1.0], [radius_2[0], radius_2[1]]])))
            calc_distance_2 = np.abs(np.dot(np.array([[radius_1[0], -1.0], [radius_1[1], -1.0]]), 
                                            np.array([[1.0, 1.0], [radius_2[0], radius_2[1]]])))
            # 我们通过将一个同等规模的值全为distance的矩阵减去计算得到的distance矩阵
            # 若结果中存在十分小的值, 则说明其中至少有一个半径上出现了可能的G0相切
            # 两类距离值的distance矩阵, 一个是沿向量距离, 一个是参考点距离
            distance_1 = np.dot(distance[0], np.ones((2, 2)))
            # distance_2 = np.dot(distance[1], np.ones((2, 2)))
            # 计算差值中是否存在小于容差值的情况
            # 获得bool矩阵
            sub_matrix_1 = np.abs(np.subtract(distance_1, calc_distance_1)) < self.tolerance
            sub_matrix_2 = np.abs(np.subtract(distance_1, calc_distance_2)) < self.tolerance
            # 确定bool矩阵中是否有true
            is_tangent = np.any(sub_matrix_1) or np.any(sub_matrix_2)
            # 我们目前认为相切的前置条件是存在平行关系
            if is_tangent:
                is_parallel, B, C = self._orientation_relation_face(desc_face_1, desc_face_2)
                # 若不平行
                if not is_parallel: 
                    # 由于平行关系对于球面并不适用, 因此位置关系对于球面均返回False
                    # 因此我们需要给球面单独设定一个规则
                    # 如果这些面里面有一个是球面
                    if (desc_face_1['surface_type'] == GeomAbs_SurfaceType.GeomAbs_Sphere or 
                        desc_face_2['surface_type'] == GeomAbs_SurfaceType.GeomAbs_Sphere):
                        # 如果其中有一个面是平面, 我们假定球面不会与平面相切(控制复杂度)
                        # 其实球面与平面相切并不会出复杂度的问题, 这大量出现在三个圆角相交的地方
                        # if (desc_face_1['surface_type'] == GeomAbs_SurfaceType.GeomAbs_Plane or 
                        #     desc_face_2['surface_type'] == GeomAbs_SurfaceType.GeomAbs_Plane):
                        #     is_tangent = False
                        # else:
                        #     is_tangent = True
                        is_tangent = True
                    # 若不平行且其中无球面, 则认为相切不成立
                    else:
                        is_tangent = False
                # 如果检验是平行的, 则认为是相切的
                else:
                    # 保持原 is_tangent 的状态
                    pass
            else:
                # 保持原 is_tangent 的状态
                pass

        return is_tangent

    def _Continuity_relation_face(self, desc_face_1, desc_face_2):
        """
        判断两面之间是否存在可能的连续关系
        """
        id_face_1 = self.face_mapping[desc_face_1['surface_id']]
        id_face_2 = self.face_mapping[desc_face_2['surface_id']]
        key = tuple(sorted((id_face_1, id_face_2)))
        # 若两面在拓扑上存在连接, 则以拓扑分析结果为准
        if key in self.adjacency_map:
            return self._Continuity_relation_face_by_topo(desc_face_1, desc_face_2)
        else:
            # 只有在不相邻时, 才使用几何推断
            return self._Continuity_relation_face_by_geom(desc_face_1, desc_face_2)
    
def demonstrate_position_descriptors():
    """演示位置描述器的使用"""
    from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Pln, gp_Cylinder, gp_Ax3
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    
    # 创建测试面
    plane1 = gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    plane_face1 = BRepBuilderAPI_MakeFace(plane1, -5, 5, -5, 5).Face()
    
    plane2 = gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    plane_face2 = BRepBuilderAPI_MakeFace(plane2, -5, 5, -5, 5).Face()
    
    cylinder = gp_Cylinder(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 3.0)
    cylinder_face = BRepBuilderAPI_MakeFace(cylinder, 0, 2*3.14159, 0, 10).Face()
    
    # 创建位置描述器
    plane_desc1 = Position_Descriptor_Surface(plane_face1)
    plane_desc2 = Position_Descriptor_Surface(plane_face2)
    cylinder_desc = Position_Descriptor_Surface(cylinder_face)
    
    # 获取位置签名
    plane1_sig = plane_desc1.get_position_signature()
    plane2_sig = plane_desc2.get_position_signature()
    cylinder_sig = cylinder_desc.get_position_signature()
    
    surface_analyzer = Surface_Relationship_Analyzer()
    relationship_results = surface_analyzer.analyze_relationship(plane1_sig, plane2_sig)

    print(relationship_results)



if __name__ == "__main__":
    demonstrate_position_descriptors()