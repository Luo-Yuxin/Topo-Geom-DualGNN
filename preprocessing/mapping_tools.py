import networkx as nx
import matplotlib.pyplot as plt
import itertools
import os
import functools
import json

from OCC.Display.SimpleGui import init_display
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRepTools import BRepTools_WireExplorer, breptools
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.gp import gp_Pnt
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_G1

# 生成面与边的hash值与面以及边索引的映射字典
@functools.lru_cache(maxsize=2)
def step_topo_mapping(shape, mapped_topo_type):
    """
    读取shape获取该对象的映射关系
    给一个内存管理器, 用于存储映射关系

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

# 获得面上环边的id字典
def step_wires_dict(shape, flag=0):
    """
    分类提取面的外环和内环。

    :param face: TopoDS_Face
    :param flag: 1: 返回外环, -1: 返回内环, 其他: 返回环边
    :return: 外环边id字典 或 内环边id字典 或 所有环边id字典
    """
    # 获取shape的拓扑探测器
    explorer = TopologyExplorer(shape)

    topo_face_list = list(explorer.faces())
    topo_face_id_dict = step_topo_mapping(shape, "face")
    topo_edge_id_dict = step_topo_mapping(shape, "edge")
    topo_face_wires_dict = {}
    # 遍历所有面以获得面环边信息
    for face in topo_face_list:
        # 获得面ID信息
        topo_face_id = topo_face_id_dict[face.HashCode(999999)]
        # 获得面环边信息
        wires = list(TopologyExplorer(face).wires())
        if flag == 1:  # 提取外环
            outer_wire = breptools.OuterWire(face)  # 获取外环
            if outer_wire.IsNull():
                raise ValueError("Outer wire is null. Face{} might be invalid.".format(topo_face_id))
            edges_id = [topo_edge_id_dict[edge.HashCode(999999)] for edge in TopologyExplorer(outer_wire).edges()]
        elif flag == -1:  # 提取内环
            edges_id = []
            outer_wire = breptools.OuterWire(face)
            for wire in wires:
                if wire.IsSame(outer_wire):  # 跳过外环
                    continue
                edges_id.extend([topo_edge_id_dict[edge.HashCode(999999)] for edge in TopologyExplorer(wire).edges()])
        else:   # 提取所有环
            for wire in wires:
                if wire.IsNull():
                    raise ValueError("Wire is null. Face{} might be invalid.".format(topo_face_id))
                edges_id.extend([topo_edge_id_dict[edge.HashCode(999999)] for edge in TopologyExplorer(wire).edges()])
        
        # 将边id转化为元组补充入字典中
        topo_face_wires_dict[topo_face_id] = tuple(edges_id)
    # 将得到的环边id字典返回
    return topo_face_wires_dict
