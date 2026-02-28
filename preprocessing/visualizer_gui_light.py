import sys
import os
import math
import numpy as np
import networkx as nx

# --- 1. 后端配置 ---
from OCC.Display.backend import load_backend
load_backend("qt-pyside2")

# --- 2. Matplotlib 配置 ---
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams.update({'font.size': 10})

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# --- 3. PySide2 导入 ---
from PySide2.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTextEdit, QFileDialog, 
                             QLabel, QSplitter, QFrame, QGroupBox, 
                             QMessageBox, QStyleFactory)
from PySide2.QtCore import Qt, Signal, QTimer, QPoint, QSize
from PySide2.QtGui import QFont, QScreen, QWindow

# --- 4. OCC 导入 ---
import OCC.Display.qtDisplay as qtDisplay

# 获取当前脚本所在目录 (datasets/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (MFR_DualGNN/)
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 python 路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 5. 业务模块 ---
from preprocessing.build_graph import read_step_file, build_geom_graph, build_topo_graph

class GraphCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(GraphCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.graph = None
        self.pos = None

    def plot_graph(self, G, highlight_nodes=None, highlight_edges=None):
        self.axes.clear()
        if G is None:
            self.draw()
            return
        
        if self.graph != G or self.pos is None:
            self.graph = G
            try:
                self.pos = nx.spring_layout(G, k=0.6, iterations=60, seed=42)
            except:
                self.pos = nx.random_layout(G)

        node_colors = ['#1f78b4'] * len(G.nodes())
        node_sizes = [300] * len(G.nodes())

        if highlight_nodes:
            for idx, node in enumerate(G.nodes()):
                if node in highlight_nodes:
                    node_colors[idx] = '#ff7f0e'
                    node_sizes[idx] = 600
        
        nx.draw_networkx_nodes(G, self.pos, ax=self.axes, node_color=node_colors, node_size=node_sizes)
        # 增大图表字体
        nx.draw_networkx_labels(G, self.pos, ax=self.axes, font_color='white', font_size=10, font_weight='bold')
        nx.draw_networkx_edges(G, self.pos, ax=self.axes, edge_color='#cccccc', width=1.0, alpha=0.5)

        if highlight_edges:
            nx.draw_networkx_edges(G, self.pos, ax=self.axes, edgelist=highlight_edges, 
                                   edge_color='red', width=2.5)
        self.axes.axis('off')
        self.draw()

class MFRViewer(qtDisplay.qtViewer3d):
    """
    终极适配版 Viewer：基于实时窗口尺寸比率计算坐标
    """
    sig_selection_changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._is_initialized = False
        self._click_start_pos = QPoint(0, 0)
        self._debug_mode = False 

    def InitDriver(self):
        super().InitDriver()
        if self._display:
            self._display.Context.SetPixelTolerance(10)
            self._is_initialized = True
            
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._display and self._is_initialized:
            self._display.View.MustBeResized()

    def mousePressEvent(self, event):
        self._click_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        dist = (event.pos() - self._click_start_pos).manhattanLength()
        if event.button() == Qt.LeftButton and dist < 5:
            self._handle_selection(event.pos())

    def _handle_selection(self, pos):
        if not self._display: return
        
        qt_w = self.width()
        qt_h = self.height()
        
        try:
            occ_w, occ_h = self._display.View.Window().Size()
        except:
            dpr = self.devicePixelRatioF()
            occ_w, occ_h = int(qt_w * dpr), int(qt_h * dpr)
            
        ratio_x = occ_w / qt_w if qt_w > 0 else 1.0
        ratio_y = occ_h / qt_h if qt_h > 0 else 1.0
        
        x_logical = pos.x()
        y_logical = pos.y()
        
        x_physical = int(x_logical * ratio_x)
        y_physical = int(y_logical * ratio_y)
        
        if self._debug_mode:
            print(f"[Click] Qt:({x_logical},{y_logical}) -> OCC:({x_physical},{y_physical}) "
                  f"| Ratio: {ratio_x:.2f}")

        self._display.Select(x_physical, y_physical)
        selected_shapes = self._display.GetSelectedShapes()
        
        if selected_shapes:
            self.sig_selection_changed.emit(selected_shapes)

class MFRVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MFR Visualizer (Smart Scaling)")
        
        self.shape = None
        self.current_graph = None
        self.topo_graph = None
        self.geom_graph = None
        self.relations_3d_raw = None
        self.face_hash_map = {}
        self.selected_faces = []
        self.current_graph_mode = 0 
        
        self.init_ui()
        self.showMaximized()
        
        if self.windowHandle():
            self.windowHandle().screenChanged.connect(self.on_screen_changed)
            # 初始化时应用一次
            QTimer.singleShot(100, lambda: self.on_screen_changed(self.windowHandle().screen()))

        QTimer.singleShot(500, self.force_layout_ratio)

    def on_screen_changed(self, screen):
        """
        [核心修复] 当屏幕切换时，根据 DPR 智能调整字号
        """
        if not screen: return
        dpr = screen.devicePixelRatio()
        
        # 智能字号策略：
        # 如果 DPR < 1.5 (例如 1080p 屏幕被识别为 1.0)，说明 Qt 没有进行足够的缩放。
        # 此时我们将基础字号设大 (12pt)，手动补偿 Windows 的 125% 设置。
        # 如果 DPR >= 1.5 (例如 2K 屏幕)，Qt 已经放大了 2倍，我们使用标准字号 (10pt) 即可。
        if dpr < 1.5:
            target_pt = 15 
            icon_base = 32
        else:
            target_pt = 10
            icon_base = 24
            
        new_font = QFont("Microsoft YaHei", target_pt)
        new_font.setBold(True) # 按钮文字加粗更清晰
        
        # 应用到关键控件
        self.btn_load.setFont(new_font)
        self.btn_switch_graph.setFont(new_font)
        self.btn_clear.setFont(new_font)
        
        # 信息框也可以稍微调整
        info_font = QFont("Microsoft YaHei", target_pt)
        self.info_text.setFont(info_font)
        
        # 刷新工具栏图标大小 (保持清晰)
        self.mpl_toolbar.setIconSize(QSize(icon_base, icon_base))
        
        print(f"[Screen Changed] DPR: {dpr} -> Setting Font Size: {target_pt}pt")

    def force_layout_ratio(self):
        total_width = self.width()
        left_width = int(total_width * 0.7)
        self.splitter.setSizes([left_width, total_width - left_width])
        if hasattr(self, 'canvas_3d') and self.canvas_3d._display:
             self.canvas_3d._display.View.MustBeResized()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(self.splitter)

        # --- 左侧 ---
        self.canvas_3d = MFRViewer(self)
        self.canvas_3d.InitDriver()
        self.splitter.addWidget(self.canvas_3d)
        self.canvas_3d.sig_selection_changed.connect(self.on_select_face_signal)

        # --- 右侧 ---
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)
        self.splitter.addWidget(right_panel)
        
        # 按钮
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(10)
        
        # 初始字体 (会被 on_screen_changed 覆盖)
        base_font = QFont("Microsoft YaHei", 10)
        base_font.setBold(True)
        
        btn_height = 40 
        btn_style = """
            QPushButton {
                padding: 5px;
                background-color: #f0f0f0;
                border: 1px solid #c0c0c0;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #e0e0e0; }
            QPushButton:pressed { background-color: #d0d0d0; }
        """
        
        self.btn_load = QPushButton("加载 STEP")
        self.btn_load.setMinimumHeight(btn_height)
        self.btn_load.setFont(base_font)
        self.btn_load.setStyleSheet(btn_style)
        self.btn_load.clicked.connect(self.load_step_file)
        
        self.btn_switch_graph = QPushButton("几何关系图 (Geom)")
        self.btn_switch_graph.setMinimumHeight(btn_height)
        self.btn_switch_graph.setFont(base_font)
        self.btn_switch_graph.setStyleSheet("""
            QPushButton {
                padding: 5px;
                background-color: #e3f2fd;
                border: 1px solid #90caf9;
                border-radius: 4px;
                color: #0d47a1;
            }
            QPushButton:hover { background-color: #bbdefb; }
        """)
        self.btn_switch_graph.clicked.connect(self.toggle_graph_view)
        
        self.btn_clear = QPushButton("清除选择")
        self.btn_clear.setMinimumHeight(btn_height)
        self.btn_clear.setFont(base_font)
        self.btn_clear.setStyleSheet(btn_style)
        self.btn_clear.clicked.connect(self.clear_selection)
        
        ctrl_layout.addWidget(self.btn_load)
        ctrl_layout.addWidget(self.btn_switch_graph)
        ctrl_layout.addWidget(self.btn_clear)
        right_layout.addLayout(ctrl_layout)

        # 图表
        self.graph_canvas = GraphCanvas(self)
        self.mpl_toolbar = NavigationToolbar(self.graph_canvas, self)
        self.mpl_toolbar.setIconSize(QSize(24, 24))
        right_layout.addWidget(self.mpl_toolbar)
        right_layout.addWidget(self.graph_canvas)

        # 信息框
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFont(QFont("Microsoft YaHei", 15))
        self.info_text.setStyleSheet("QTextEdit { padding: 5px; }")
        
        right_layout.addWidget(self.info_text)

        self.display = self.canvas_3d._display
        self.display.SetSelectionModeFace()

    def load_step_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open STEP", "", "STEP (*.stp *.step)")
        if not filename: return
        try:
            self.info_text.setText("加载中...")
            QApplication.processEvents()
            self.shape = read_step_file(filename)
            self.display.EraseAll()
            self.display.DisplayShape(self.shape, update=True)
            self.display.SetSelectionModeFace()
            self.display.FitAll()
            self.geom_graph, self.relations_3d_raw, self.node_feats = build_geom_graph(self.shape, estimate=False)
            self.topo_graph = build_topo_graph(self.shape, self_loops=False, estimate=False)
            from OCC.Extend.TopologyUtils import TopologyExplorer
            explorer = TopologyExplorer(self.shape)
            self.face_hash_map = {f.HashCode(99999999): i for i, f in enumerate(explorer.faces())}
            self.info_text.setText(f"加载完成。面数量: {len(self.face_hash_map)}\n")
            
            # [关键修复] 重置图表状态，确保 draw() 被调用时认为是新图
            self.graph_canvas.graph = None 
            
            # [修复] 强制更新图显示
            self.update_visualization()
            
        except Exception as e:
            self.info_text.setText(f"Error: {e}")

    def toggle_graph_view(self):
        self.current_graph_mode = 1 - self.current_graph_mode
        if self.current_graph_mode == 0:
            self.btn_switch_graph.setText("几何关系图 (Geom)")
            self.current_graph = self.geom_graph
        else:
            self.btn_switch_graph.setText("拓扑关系图 (Topo)")
            self.current_graph = self.topo_graph
        self.update_visualization()

    def on_select_face_signal(self, selected_shapes):
        if not selected_shapes: return
        sel_shape = selected_shapes[0]
        found_index = -1
        sel_hash = sel_shape.HashCode(99999999)
        for i, face in enumerate(self.selected_faces):
            if face.HashCode(99999999) == sel_hash:
                found_index = i; break
        if found_index != -1: self.selected_faces.pop(found_index)
        else:
            self.selected_faces.append(sel_shape)
            if len(self.selected_faces) > 2: self.selected_faces.pop(0)
        self.update_info_panel()
        self.update_visualization()

    def clear_selection(self):
        self.selected_faces = []
        self.display.Context.ClearSelected(True)
        self.update_info_panel()
        self.update_visualization()

    def update_visualization(self):
        # 确保有图可画，如果当前模式对应的图还没生成（比如加载刚完成），尝试自动赋值
        if self.current_graph_mode == 0:
            self.current_graph = self.geom_graph
        else:
            self.current_graph = self.topo_graph
            
        # 如果还是没有（比如没加载文件），直接返回
        if self.current_graph is None: return

        highlight_nodes = []
        highlight_edges = []
        selected_ids = []
        for face in self.selected_faces:
            h_code = face.HashCode(99999999)
            if h_code in self.face_hash_map:
                nid = self.face_hash_map[h_code]
                selected_ids.append(nid); highlight_nodes.append(nid)
        if len(selected_ids) == 2:
            u, v = selected_ids[0], selected_ids[1]
            # 兼容有向图和无向图
            if self.current_graph.has_edge(u, v): highlight_edges.append((u, v))
            elif self.current_graph.is_directed() and self.current_graph.has_edge(v, u): highlight_edges.append((v, u))
            
        self.graph_canvas.plot_graph(self.current_graph, highlight_nodes, highlight_edges)

    def update_info_panel(self):
        if not self.selected_faces: self.info_text.setText("未选择面"); return
        txt = ""
        for i, face in enumerate(self.selected_faces):
            h_code = face.HashCode(99999999)
            nid = self.face_hash_map.get(h_code, -1)
            if nid != -1 and self.geom_graph.has_node(nid):
                data = self.geom_graph.nodes[nid]
                txt += f"面 ID: {nid}\n类型: {data.get('type_str')}\n"
                desc = data.get('descriptor', {})
                pnt = desc.get('reference_point')
                if hasattr(pnt, "X"): txt += f"Ref: ({pnt.X():.1f}, {pnt.Y():.1f}, {pnt.Z():.1f})\n"
                txt += "\n"
        if len(self.selected_faces) == 2:
             id1 = self.face_hash_map.get(self.selected_faces[0].HashCode(99999999))
             id2 = self.face_hash_map.get(self.selected_faces[1].HashCode(99999999))
             if id1 is not None and id2 is not None:
                r_min, r_max = min(id1, id2), max(id1, id2)
                if r_min < len(self.relations_3d_raw):
                    raw_rel = self.relations_3d_raw[r_min][r_max]
                    if raw_rel:
                        txt += f"关系: {raw_rel.get('primary_relation')}\n"
                        d = raw_rel.get('distance')
                        if d is not None:
                             val = d[0] if isinstance(d, tuple) else d
                             txt += f"距离: {val:.4f}\n"
        self.info_text.setText(txt)

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    
    # 默认使用大一点的字体作为基准
    font = QApplication.font()
    font.setPointSize(20) 
    font.setFamily("Microsoft YaHei")
    app.setFont(font)
    
    viewer = MFRVisualizer()
    sys.exit(app.exec_())