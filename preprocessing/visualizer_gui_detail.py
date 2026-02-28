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
        nx.draw_networkx_labels(G, self.pos, ax=self.axes, font_color='white', font_size=10, font_weight='bold')
        nx.draw_networkx_edges(G, self.pos, ax=self.axes, edge_color='#cccccc', width=1.0, alpha=0.5)

        if highlight_edges:
            nx.draw_networkx_edges(G, self.pos, ax=self.axes, edgelist=highlight_edges, 
                                   edge_color='red', width=2.5)
        self.axes.axis('off')
        self.draw()

class MFRViewer(qtDisplay.qtViewer3d):
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
            print(f"[Click] Ratio: {ratio_x:.2f}")

        self._display.Select(x_physical, y_physical)
        selected_shapes = self._display.GetSelectedShapes()
        
        if selected_shapes:
            self.sig_selection_changed.emit(selected_shapes)

class MFRVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MFR Visualizer (Feature Inspector)")
        
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
            QTimer.singleShot(100, lambda: self.on_screen_changed(self.windowHandle().screen()))

        QTimer.singleShot(500, self.force_layout_ratio)

    def on_screen_changed(self, screen):
        if not screen: return
        dpr = screen.devicePixelRatio()
        
        if dpr < 1.5:
            target_pt = 12 
            icon_base = 32
        else:
            target_pt = 10
            icon_base = 24
            
        new_font = QFont("Microsoft YaHei", target_pt)
        new_font.setBold(True) 
        
        self.btn_load.setFont(new_font)
        self.btn_switch_graph.setFont(new_font)
        self.btn_clear.setFont(new_font)
        
        info_font = QFont("Microsoft YaHei", target_pt)
        self.info_text.setFont(info_font)
        self.mpl_toolbar.setIconSize(QSize(icon_base, icon_base))

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

        # Left
        self.canvas_3d = MFRViewer(self)
        self.canvas_3d.InitDriver()
        self.splitter.addWidget(self.canvas_3d)
        self.canvas_3d.sig_selection_changed.connect(self.on_select_face_signal)

        # Right
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)
        self.splitter.addWidget(right_panel)
        
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(10)
        
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
        
        self.btn_switch_graph = QPushButton("当前视图: 几何关系图 (Geom)")
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

        self.graph_canvas = GraphCanvas(self)
        self.mpl_toolbar = NavigationToolbar(self.graph_canvas, self)
        self.mpl_toolbar.setIconSize(QSize(24, 24))
        right_layout.addWidget(self.mpl_toolbar)
        right_layout.addWidget(self.graph_canvas)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFont(QFont("Microsoft YaHei", 10))
        self.info_text.setStyleSheet("QTextEdit { padding: 5px; }")
        # [优化] 设置自动换行模式为 WidgetWidth，确保文本填满宽度
        self.info_text.setLineWrapMode(QTextEdit.WidgetWidth)
        
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
            
            # 构建图
            self.geom_graph, self.relations_3d_raw, self.node_feats = build_geom_graph(self.shape, estimate=False)
            self.topo_graph = build_topo_graph(self.shape, self_loops=False, estimate=False)
            
            from OCC.Extend.TopologyUtils import TopologyExplorer
            explorer = TopologyExplorer(self.shape)
            self.face_hash_map = {f.HashCode(99999999): i for i, f in enumerate(explorer.faces())}
            
            self.info_text.setText(f"加载完成。面数量: {len(self.face_hash_map)}\n")
            
            # 重置图表状态并强制刷新
            self.graph_canvas.graph = None 
            self.update_visualization()
            
        except Exception as e:
            self.info_text.setText(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def toggle_graph_view(self):
        self.current_graph_mode = 1 - self.current_graph_mode
        if self.current_graph_mode == 0:
            self.btn_switch_graph.setText("当前视图: 几何关系图 (Geom)")
            self.current_graph = self.geom_graph
        else:
            self.btn_switch_graph.setText("当前视图: 拓扑关系图 (Topo)")
            self.current_graph = self.topo_graph
        self.update_visualization()
        # 切换图类型时也刷新信息面板
        self.update_info_panel()

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
        if self.current_graph_mode == 0:
            self.current_graph = self.geom_graph
        else:
            self.current_graph = self.topo_graph
            
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
            if self.current_graph.has_edge(u, v): highlight_edges.append((u, v))
            elif self.current_graph.is_directed() and self.current_graph.has_edge(v, u): highlight_edges.append((v, u))
            
        self.graph_canvas.plot_graph(self.current_graph, highlight_nodes, highlight_edges)

    def update_info_panel(self):
        if not self.selected_faces: 
            self.info_text.setText("未选择面")
            return
        
        txt = ""
        # 显示当前模式提示
        mode_str = "几何关系图 (Geom)" if self.current_graph_mode == 0 else "拓扑关系图 (Topo)"
        txt += f"【当前信息模式: {mode_str}】\n"
        txt += "-" * 30 + "\n"

        # ==================== 单面信息 ====================
        for i, face in enumerate(self.selected_faces):
            h_code = face.HashCode(99999999)
            nid = self.face_hash_map.get(h_code, -1)
            
            txt += f"=== 面 {i+1} (ID: {nid}) ===\n"
            
            # 基础几何信息 (从 geom_graph 读取，因为它最全)
            if nid != -1 and self.geom_graph is not None and self.geom_graph.has_node(nid):
                geom_data = self.geom_graph.nodes[nid]
                txt += f"类型: {geom_data.get('type_str', 'Unknown')}\n"
                
                desc = geom_data.get('descriptor', {})
                pnt = desc.get('reference_point')
                if hasattr(pnt, "X"): 
                    txt += f"参考点: ({pnt.X():.1f}, {pnt.Y():.1f}, {pnt.Z():.1f})\n"

                # >>> 几何关系图模式特有信息 <<<
                if self.current_graph_mode == 0: 
                    # 1. shared_props
                    props = geom_data.get('shared_props')
                    if props is not None:
                        # [优化] 使用 max_line_width=1000 防止 numpy 过早换行，由 QTextEdit 自动换行
                        props_str = np.array2string(props, precision=4, separator=', ', suppress_small=True, max_line_width=1000)
                        txt += f"通用几何属性 (shared_props):\n{props_str}\n"
                    
                    # 2. one_hot
                    one_hot = geom_data.get('one_hot')
                    if one_hot is not None:
                        # 找到是 1 的索引位置，更直观
                        hot_idx = np.argmax(one_hot)
                        one_hot_str = np.array2string(one_hot, precision=1, separator=', ', suppress_small=True, max_line_width=1000)
                        txt += f"独热编码 (one_hot): Index {hot_idx}\n{one_hot_str}\n"
                        
                # >>> 拓扑关系图模式特有信息 <<<
                elif self.current_graph_mode == 1:
                    # 1. 通用几何特征 (需求：显示面的通用特征信息)
                    # 借助 geom_graph 的数据，因为是同一个面
                    props = geom_data.get('shared_props')
                    if props is not None:
                        props_str = np.array2string(props, precision=4, separator=', ', suppress_small=True, max_line_width=1000)
                        txt += f"通用几何属性:\n{props_str}\n"
                    
                    # 2. 采样点信息维度
                    if self.topo_graph and self.topo_graph.has_node(nid):
                        topo_data = self.topo_graph.nodes[nid]
                        sample = topo_data.get('sample')
                        if sample is not None:
                            txt += f"采样特征维度 (Sample Shape): {sample.shape}\n"
                        else:
                            txt += f"采样特征: None\n"
            else:
                txt += "节点信息未找到。\n"
            txt += "\n"

        # ==================== 双面/边信息 ====================
        if len(self.selected_faces) == 2:
            id1 = self.face_hash_map.get(self.selected_faces[0].HashCode(99999999))
            id2 = self.face_hash_map.get(self.selected_faces[1].HashCode(99999999))
            
            if id1 is not None and id2 is not None:
                txt += f"=== 连接关系 ({id1} <-> {id2}) ===\n"
                
                # >>> 几何关系图模式 <<<
                if self.current_graph_mode == 0: 
                    if self.geom_graph.has_edge(id1, id2):
                        edge_data = self.geom_graph.edges[id1, id2]
                        
                        # primary_relation
                        prim = edge_data.get('primary_relation')
                        txt += f"主要关系 (primary): {prim}\n"
                        
                        # one_hot_relation
                        oh_rel = edge_data.get('one_hot_relation')
                        if oh_rel is not None:
                            oh_rel_str = np.array2string(oh_rel, precision=4, separator=', ', suppress_small=True, max_line_width=1000)
                            txt += f"关系编码 (one_hot_relation):\n{oh_rel_str}\n"
                            
                        # 距离
                        d = edge_data.get('dist')
                        if d[0] is not None:
                             val = d[0] if isinstance(d, tuple) else d
                             txt += f"距离: {val:.4f}\n"
                        else:
                            txt += f"距离: {None}\n"
                    else:
                        txt += "这两个面在几何图中没有直接连边（可能无特定几何关系）。\n"
                        
                # >>> 拓扑关系图模式 <<<
                elif self.current_graph_mode == 1: 
                    if self.topo_graph and self.topo_graph.has_edge(id1, id2):
                        edge_data = self.topo_graph.edges[id1, id2]
                        
                        # 采样维度
                        sample = edge_data.get('sample')
                        if sample is not None:
                            txt += f"边采样特征维度: {sample.shape}\n"
                        else:
                            txt += f"边采样特征: None (可能为退化边)\n"
                    else:
                        txt += "这两个面在拓扑图中没有连边（即不相邻）。\n"

        self.info_text.setText(txt)

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    
    font = QApplication.font()
    font.setPointSize(20) 
    font.setFamily("Microsoft YaHei")
    app.setFont(font)
    
    viewer = MFRVisualizer()
    sys.exit(app.exec_())