import sys
import os
import numpy as np
import networkx as nx

# --- 1. 后端配置 ---
from OCC.Display.backend import load_backend
load_backend("qt-pyside2")

# --- 2. PySide2 导入 ---
from PySide2.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, 
                             QLabel, QSplitter, QFrame, QCheckBox, QTextEdit, QProgressBar, QGroupBox)
from PySide2.QtCore import Qt, Signal, QTimer, QPoint, QSize
from PySide2.QtGui import QFont, QColor

# --- 3. OCC 导入 ---
import OCC.Display.qtDisplay as qtDisplay
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeEdge
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.BRep import BRep_Builder
from OCC.Core.AIS import AIS_Shape
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_RED, Quantity_NOC_BLUE, Quantity_NOC_GREEN, Quantity_NOC_YELLOW

# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- 4. 业务模块 ---
try:
    from preprocessing.build_graph import read_step_file, build_geom_graph, build_topo_graph
    from preprocessing.build_graph import normalize_shape 
except ImportError:
    sys.path.append(os.path.join(current_dir, '..'))
    from preprocessing.build_graph import read_step_file, build_geom_graph, build_topo_graph
    try:
        from preprocessing.build_graph import normalize_shape
    except ImportError:
        print("[Warning] Could not import normalize_shape. Visualization might be misaligned.")
        normalize_shape = lambda s, **k: s

class UVNetViewer(qtDisplay.qtViewer3d):
    """简单的 OCC 3D 视图控件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_initialized = False

    def InitDriver(self):
        super().InitDriver()
        if self._display:
            self._display.Context.SetPixelTolerance(10)
            self._is_initialized = True
            # 尝试设置背景色 (兼容性写法)
            try:
                if hasattr(self._display, 'SetBackgroundColor'):
                    self._display.SetBackgroundColor(1, 1, 1)
                elif hasattr(self._display, 'View'):
                    self._display.View.SetBackgroundColor(Quantity_Color(1, 1, 1, Quantity_TOC_RGB))
            except Exception:
                pass

class UVNetVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UVNet-Style Data Visualizer")
        self.resize(1280, 800)
        
        self.shape = None
        self.topo_graph = None
        
        self.ais_shape = None
        self.ais_points = None
        self.ais_vectors = None
        
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        self.canvas = UVNetViewer(self)
        self.canvas.InitDriver()
        self.display = self.canvas._display
        layout.addWidget(self.canvas, stretch=7)
        
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)
        layout.addWidget(right_panel, stretch=3)
        
        # 标题
        title = QLabel("控制面板")
        title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        right_layout.addWidget(title)
        
        # 加载按钮
        self.btn_load = QPushButton("加载 STEP 文件")
        self.btn_load.setFixedHeight(40)
        self.btn_load.clicked.connect(self.load_step)
        right_layout.addWidget(self.btn_load)
        
        # 归一化选项
        self.chk_normalize = QCheckBox("应用归一化 (Normalize)")
        self.chk_normalize.setChecked(True) 
        self.chk_normalize.setToolTip("加载时对几何体应用 normalize_shape，以匹配采样点位置")
        right_layout.addWidget(self.chk_normalize)

        # 几何体显示选项
        geom_group = QGroupBox("几何体显示")
        geom_layout = QVBoxLayout()
        self.chk_show_shape = QCheckBox("显示几何体 (Geometry)")
        self.chk_show_shape.setChecked(True)
        self.chk_show_shape.stateChanged.connect(self.refresh_view)
        geom_layout.addWidget(self.chk_show_shape)
        
        self.chk_transparent = QCheckBox("几何体透明 (Transparent)")
        self.chk_transparent.setChecked(True) 
        self.chk_transparent.stateChanged.connect(self.refresh_view)
        geom_layout.addWidget(self.chk_transparent)
        geom_group.setLayout(geom_layout)
        right_layout.addWidget(geom_group)
        
        # 采样显示选项
        sample_group = QGroupBox("采样数据显示")
        sample_layout = QVBoxLayout()
        
        # 面采样控制
        self.chk_show_face_points = QCheckBox("显示面采样点 (Face Points - Red)")
        self.chk_show_face_points.setChecked(True)
        self.chk_show_face_points.stateChanged.connect(self.refresh_view)
        sample_layout.addWidget(self.chk_show_face_points)
        
        # 边采样控制
        self.chk_show_edge_points = QCheckBox("显示边采样点 (Edge Points - Green)")
        self.chk_show_edge_points.setChecked(False) # 默认关闭，避免太乱
        self.chk_show_edge_points.stateChanged.connect(self.refresh_view)
        sample_layout.addWidget(self.chk_show_edge_points)
        
        # 向量控制
        self.chk_show_vectors = QCheckBox("显示向量 (法向/切向 - Blue/Yellow)")
        self.chk_show_vectors.setChecked(False)
        self.chk_show_vectors.stateChanged.connect(self.refresh_view)
        sample_layout.addWidget(self.chk_show_vectors)
        
        sample_group.setLayout(sample_layout)
        right_layout.addWidget(sample_group)
        
        # 日志输出
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.log_text)
        
        # 进度条
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        right_layout.addWidget(self.progress)

    def log(self, msg):
        self.log_text.append(msg)
        print(f"[Visualizer] {msg}") 
        QApplication.processEvents()

    def load_step(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open STEP", "", "STEP (*.stp *.step)")
        if not filename: return
        
        try:
            self.log(f"正在加载: {os.path.basename(filename)}...")
            self.shape = read_step_file(filename)
            
            # 应用归一化
            if self.chk_normalize.isChecked():
                self.log("正在执行几何归一化 (Normalize)...")
                self.shape = normalize_shape(self.shape)
            
            self.log("正在构建拓扑图 (采样中)...")
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
            QApplication.processEvents()
            
            self.topo_graph = build_topo_graph(self.shape, self_loops=False, estimate=False)
            
            self.progress.setVisible(False)
            self.log(f"图构建完成。节点数: {self.topo_graph.number_of_nodes()}, 边数: {self.topo_graph.number_of_edges()}")
            
            self.refresh_view()
            self.display.FitAll()
            
        except Exception as e:
            self.log(f"错误: {e}")
            import traceback
            traceback.print_exc()
            self.progress.setVisible(False)

    def refresh_view(self):
        """核心渲染逻辑"""
        if not self.shape: return
        
        self.display.EraseAll()
        
        # 1. 绘制几何体
        if self.chk_show_shape.isChecked():
            transparency = 0.8 if self.chk_transparent.isChecked() else 0.0
            self.display.DisplayShape(self.shape, transparency=transparency, update=False)
            
        # 2. 绘制采样数据
        show_face = self.chk_show_face_points.isChecked()
        show_edge = self.chk_show_edge_points.isChecked()
        show_vectors = self.chk_show_vectors.isChecked()
        
        if (show_face or show_edge or show_vectors) and self.topo_graph:
            self.draw_uvnet_features(show_face, show_edge, show_vectors)
            
        self.display.View.Update()

    def draw_uvnet_features(self, show_face, show_edge, show_vectors):
        builder = BRep_Builder()
        
        # 创建复合体容器
        comp_face_pts = TopoDS_Compound() # 面点 (红)
        comp_edge_pts = TopoDS_Compound() # 边点 (绿)
        comp_face_vec = TopoDS_Compound() # 面法向 (蓝)
        comp_edge_vec = TopoDS_Compound() # 边切向 (黄)
        
        builder.MakeCompound(comp_face_pts)
        builder.MakeCompound(comp_edge_pts)
        builder.MakeCompound(comp_face_vec)
        builder.MakeCompound(comp_edge_vec)
        
        has_face_pts = False
        has_edge_pts = False
        has_face_vec = False
        has_edge_vec = False
        
        total_face_pts = 0
        total_edge_pts = 0
        
        # -----------------------------------------------------
        # 1. 处理面节点 (Nodes) -> Face Samples [N, N, 7]
        # -----------------------------------------------------
        if show_face or show_vectors:
            # 探测键名
            face_sample_key = None
            if self.topo_graph.number_of_nodes() > 0:
                first_node = list(self.topo_graph.nodes(data=True))[0][1]
                for key in ['sample', 'x', 'feat']:
                    if key in first_node:
                        face_sample_key = key
                        break
            
            if face_sample_key:
                for nid, data in self.topo_graph.nodes(data=True):
                    samples = data.get(face_sample_key)
                    if samples is None: continue
                    
                    flat = samples.reshape(-1, samples.shape[-1])
                    for p in flat:
                        # Mask 检查
                        if p.shape[0] >= 7 and p[6] < 0.5: continue
                        
                        x, y, z = float(p[0]), float(p[1]), float(p[2])
                        pt_loc = gp_Pnt(x, y, z)
                        
                        if show_face:
                            v = BRepBuilderAPI_MakeVertex(pt_loc).Vertex()
                            builder.Add(comp_face_pts, v)
                            has_face_pts = True
                            total_face_pts += 1
                        
                        if show_vectors and p.shape[0] >= 6:
                            nx, ny, nz = float(p[3]), float(p[4]), float(p[5])
                            vec_len = 2.0 
                            pt_end = gp_Pnt(x + nx*vec_len, y + ny*vec_len, z + nz*vec_len)
                            edge = BRepBuilderAPI_MakeEdge(pt_loc, pt_end).Edge()
                            builder.Add(comp_face_vec, edge)
                            has_face_vec = True

        # -----------------------------------------------------
        # 2. 处理边 (Edges) -> Edge Samples [N, 6]
        # -----------------------------------------------------
        if show_edge or show_vectors:
            # 探测键名
            edge_sample_key = None
            if self.topo_graph.number_of_edges() > 0:
                first_edge = list(self.topo_graph.edges(data=True))[0][2]
                for key in ['sample', 'x', 'feat']:
                    if key in first_edge:
                        edge_sample_key = key
                        break
            
            if edge_sample_key:
                for u, v, data in self.topo_graph.edges(data=True):
                    samples = data.get(edge_sample_key)
                    if samples is None: continue
                    
                    # 边采样通常是 [N, 6] (x, y, z, tx, ty, tz)
                    # 有些实现可能是 [N, 3] (只有坐标)
                    flat = samples.reshape(-1, samples.shape[-1])
                    
                    for p in flat:
                        x, y, z = float(p[0]), float(p[1]), float(p[2])
                        pt_loc = gp_Pnt(x, y, z)
                        
                        if show_edge:
                            v = BRepBuilderAPI_MakeVertex(pt_loc).Vertex()
                            builder.Add(comp_edge_pts, v)
                            has_edge_pts = True
                            total_edge_pts += 1
                            
                        if show_vectors and p.shape[0] >= 6:
                            tx, ty, tz = float(p[3]), float(p[4]), float(p[5])
                            vec_len = 2.0 
                            pt_end = gp_Pnt(x + tx*vec_len, y + ty*vec_len, z + tz*vec_len)
                            edge = BRepBuilderAPI_MakeEdge(pt_loc, pt_end).Edge()
                            builder.Add(comp_edge_vec, edge)
                            has_edge_vec = True

        self.log(f"绘制统计: 面点={total_face_pts}, 边点={total_edge_pts}")

        # 渲染
        if has_face_pts and show_face:
            self.display.DisplayShape(comp_face_pts, color=Quantity_Color(Quantity_NOC_RED), update=False)
            
        if has_edge_pts and show_edge:
            self.display.DisplayShape(comp_edge_pts, color=Quantity_Color(Quantity_NOC_GREEN), update=False)
            
        if has_face_vec and show_vectors:
            self.display.DisplayShape(comp_face_vec, color=Quantity_Color(Quantity_NOC_BLUE), update=False)
            
        if has_edge_vec and show_vectors:
            self.display.DisplayShape(comp_edge_vec, color=Quantity_Color(Quantity_NOC_YELLOW), update=False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QApplication.font()
    font.setPointSize(10)
    font.setFamily("Microsoft YaHei")
    app.setFont(font)
    
    viewer = UVNetVisualizer()
    viewer.show()
    sys.exit(app.exec_())