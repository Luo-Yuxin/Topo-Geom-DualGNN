import sys
import os
import json
import time
import functools
from collections import defaultdict

# PySide2 Imports
from PySide2.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QMessageBox,
                             QLabel, QFrame, QDockWidget, QTreeWidget, QTreeWidgetItem,
                             QProgressBar, QSplitter, QGroupBox, QTextEdit)
from PySide2.QtCore import Qt, QThread, Signal
from PySide2.QtGui import QIcon, QColor

# Torch Imports (Needed for manual inference control in Worker)
import torch
from torch.cuda.amp import autocast

# PythonOCC Imports
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.AIS import AIS_Shape

# =============================================================================
# [CRITICAL FIX] Backend Loading Order
# 必须在导入 qtViewer3d 之前加载后端，否则会报错 ValueError: no backend has been imported yet
# =============================================================================
from OCC.Display.backend import load_backend
load_backend("qt-pyside2")
from OCC.Display.qtDisplay import qtViewer3d

# === 导入你的预测模块 ===
# 确保 predict.py 在同一目录下
try:
    from predict import Predict
except ImportError:
    print("错误: 未找到 predict.py，请确保该文件位于项目根目录。")
    sys.exit(1)

# =============================================================================
# 全局配置 (请在此处修改你的模型路径)
# =============================================================================
# 注意：请修改为你实际的权重文件和配置文件路径
MODEL_PARAM_PATH = r"k_fold\Feb14_17-01-42_dim64_K5\fold_2\best_model.pth"
CONFIG_PATH = r"k_fold\Feb14_17-01-42_dim64_K5\config.json"


class PredictionWorker(QThread):
    """
    后台预测线程，防止界面卡死
    """
    finished = Signal(dict, dict) # 修改 Signal，同时传递 结果数据 和 时间统计
    error = Signal(str)

    def __init__(self, step_path, model_path, config_path):
        super().__init__()
        self.step_path = step_path
        self.model_path = model_path
        self.config_path = config_path

    def run(self):
        try:
            # 总计时开始
            t_total_start = time.time()
            timing_stats = {}

            # 0. 初始化配置 (不算在推理耗时中，算作加载时间或忽略)
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"找不到配置文件: {self.config_path}")
            
            with open(self.config_path, "r", encoding="utf-8") as f:
                config_raw = json.load(f)
            
            config_add = {
                "shape_norm_method": "bbox",
                "shape_norm_param": 100.0,
                "use_log_area": True,
                "use_log_linear": False,
            }
            config = {**config_raw, **config_add}

            # 初始化 Predictor
            predictor = Predict(self.model_param_path, config=config)
            
            # --- 1. 前处理 (Pre-processing) ---
            t_pre_start = time.time()
            # 调用 Predict 类内部的转换函数
            batch_data = predictor._trans_single_step_to_data(self.step_path)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_pre_end = time.time()
            
            # --- 2. 网络推理 (Inference) ---
            t_infer_start = time.time()
            with torch.no_grad():
                # 判断是否使用 CUDA autocast
                device_type = 'cuda' if 'cuda' in str(predictor.config['device']) else 'cpu'
                if device_type == 'cuda':
                    with autocast():
                        pred = predictor.model(batch_data)
                else:
                    pred = predictor.model(batch_data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_infer_end = time.time()

            # --- 3. 后处理 (Post-processing) ---
            t_post_start = time.time()
            result = predictor._pred_interpreter(pred, batch_data)
            t_post_end = time.time()

            # 总计时结束
            t_total_end = time.time()

            # 收集统计信息 (单位: 秒 -> 毫秒)
            timing_stats['pre'] = (t_pre_end - t_pre_start) * 1000
            timing_stats['infer'] = (t_infer_end - t_infer_start) * 1000
            timing_stats['post'] = (t_post_end - t_post_start) * 1000
            timing_stats['total'] = (t_total_end - t_total_start) * 1000 # 这里包含了初始化的时间，或者你可以只加和上面三项

            # 如果希望 Total 仅为处理流之和：
            # timing_stats['total_pure'] = timing_stats['pre'] + timing_stats['infer'] + timing_stats['post']

            self.finished.emit(result, timing_stats)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
    
    @property
    def model_param_path(self):
        return self.model_path


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MFR-DualGNN Machining Feature Visualizer")
        self.resize(1400, 900)
        
        # 核心数据
        self.current_shape = None
        self.step_file_path = None
        self.faces_list = []  # 存储 TopoDS_Face 对象，索引对应 ID
        self.ais_context = None # 显式引用 Context
        
        # UI 初始化
        self.init_ui()
        
        # 尝试加载图标
        if os.path.exists("app.ico"):
            self.setWindowIcon(QIcon("app.ico"))

    def init_ui(self):
        """初始化界面布局"""
        # 主样式
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; }
            QDockWidget { font-weight: bold; }
            QPushButton {
                background-color: #2196F3; color: white; border: none;
                padding: 8px 15px; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:pressed { background-color: #0D47A1; }
            QTreeWidget { border: 1px solid #ccc; font-size: 14px; }
            QTreeWidget::item { padding: 5px; }
            QGroupBox { 
                border: 1px solid #ccc; 
                border-radius: 5px; 
                margin-top: 10px; 
                font-weight: bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QLabel#StatsLabel { font-family: Consolas, monospace; color: #333; }
        """)

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 1. 顶部工具栏
        toolbar = QFrame()
        toolbar.setFixedHeight(60)
        toolbar.setStyleSheet("background-color: white; border-radius: 5px;")
        tool_layout = QHBoxLayout(toolbar)
        
        btn_import = QPushButton("导入 STEP")
        btn_import.clicked.connect(self.import_step_file)
        btn_import.setStyleSheet("background-color: #66ccff;")
        
        btn_clear = QPushButton("清空工作区")
        btn_clear.clicked.connect(self.clear_session)
        btn_clear.setStyleSheet("""
            QPushButton { background-color: #FF5722; }
            QPushButton:hover { background-color: #E64A19; }
        """)
        
        btn_predict = QPushButton("特征识别 (Predict)")
        btn_predict.clicked.connect(self.start_prediction)
        btn_predict.setStyleSheet("background-color: #4CAF50;") # 绿色按钮
        
        btn_reset = QPushButton("重置视图")
        btn_reset.clicked.connect(self.reset_display)
        btn_reset.setStyleSheet("background-color: #999999;")

        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet("color: #666; margin-left: 10px;")

        tool_layout.addWidget(btn_import)
        tool_layout.addWidget(btn_clear)
        tool_layout.addWidget(btn_predict)
        tool_layout.addWidget(btn_reset)
        tool_layout.addWidget(self.lbl_status)
        tool_layout.addStretch()
        
        main_layout.addWidget(toolbar)

        # 2. 3D 视图区域
        self.canvas = qtViewer3d(central_widget)
        main_layout.addWidget(self.canvas)
        self.display = self.canvas._display
        self.ais_context = self.display.Context
        
        # 初始化 3D 环境
        self.display.View.SetBackgroundColor(Quantity_Color(0.2, 0.2, 0.2, Quantity_TOC_RGB))
        self.display.display_triedron()
        
        # 3. 侧边栏 (特征树 + 统计信息)
        self.dock = QDockWidget("结果面板", self)
        self.dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.dock.setMinimumWidth(320)
        
        # 创建 Dock 内部容器
        dock_content = QWidget()
        dock_layout = QVBoxLayout(dock_content)
        dock_layout.setContentsMargins(5, 5, 5, 5)

        # 3.1 特征树
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("特征结构 (Feature Hierarchy)")
        self.tree.itemClicked.connect(self.on_tree_item_clicked)
        dock_layout.addWidget(self.tree, stretch=7) # 树占 70% 高度
        
        # 3.2 统计信息面板
        self.stats_group = QGroupBox("性能统计 (Time Statistics)")
        stats_layout = QVBoxLayout(self.stats_group)
        
        self.lbl_stats = QLabel("暂无数据\n请运行特征识别...")
        self.lbl_stats.setObjectName("StatsLabel")
        self.lbl_stats.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        stats_layout.addWidget(self.lbl_stats)
        
        dock_layout.addWidget(self.stats_group, stretch=3) # 统计面板占 30% 高度
        
        self.dock.setWidget(dock_content)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)

    def clear_session(self):
        """清空当前会话"""
        self.current_shape = None
        self.step_file_path = None
        self.faces_list = []
        
        if self.display:
            self.display.EraseAll()
            self.ais_context.RemoveAll(True)
            self.display.View.FitAll()
            self.display.Repaint()
            
        self.tree.clear()
        self.lbl_stats.setText("暂无数据")
        self.lbl_status.setText("就绪 (工作区已清空)")

    def import_step_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 STEP 文件", "", "STEP Files (*.stp *.step)")
        if not path:
            return

        self.clear_session()

        try:
            self.lbl_status.setText(f"正在导入: {os.path.basename(path)}...")
            QApplication.processEvents()

            reader = STEPControl_Reader()
            status = reader.ReadFile(path)
            if status != IFSelect_RetDone:
                raise Exception("无法读取 STEP 文件")
            reader.TransferRoot()
            shape = reader.Shape()
            
            self.current_shape = shape
            self.step_file_path = path
            
            self.faces_list = []
            explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while explorer.More():
                self.faces_list.append(explorer.Current())
                explorer.Next()
            
            self.display.DisplayShape(shape, update=True)
            self.display.View.FitAll()
            
            self.lbl_status.setText(f"已导入: {os.path.basename(path)} (共 {len(self.faces_list)} 个面)")

        except Exception as e:
            QMessageBox.critical(self, "导入错误", str(e))
            self.lbl_status.setText("导入失败")

    def start_prediction(self):
        if not self.step_file_path:
            QMessageBox.warning(self, "警告", "请先导入 STEP 模型！")
            return

        if not os.path.exists(MODEL_PARAM_PATH) or not os.path.exists(CONFIG_PATH):
            QMessageBox.critical(self, "配置错误", 
                f"未找到模型权重或配置文件。\nModel: {MODEL_PARAM_PATH}\nConfig: {CONFIG_PATH}")
            return

        self.lbl_status.setText("正在进行特征识别...")
        self.tree.clear()
        
        # 占位符
        loading_item = QTreeWidgetItem(self.tree)
        loading_item.setText(0, "正在计算中...")
        self.lbl_stats.setText("计算中...")
        
        self.worker = PredictionWorker(self.step_file_path, MODEL_PARAM_PATH, CONFIG_PATH)
        self.worker.finished.connect(self.on_prediction_success)
        self.worker.error.connect(self.on_prediction_error)
        self.worker.start()

    def on_prediction_success(self, result_dict, timing_stats):
        """
        预测成功回调
        result_dict: 识别结果
        timing_stats: 耗时统计字典
        """
        self.lbl_status.setText("识别完成！")
        self.tree.clear()

        # 1. 更新统计面板
        stats_text = (
            f"前处理 (Pre) : {timing_stats['pre']:.2f} ms\n"
            f"推理 (Infer) : {timing_stats['infer']:.2f} ms\n"
            f"后处理 (Post): {timing_stats['post']:.2f} ms\n"
            f"--------------------------\n"
            f"总耗时 (Total): {timing_stats['total']:.2f} ms"
        )
        self.lbl_stats.setText(stats_text)

        # 2. 更新特征树
        sorted_types = sorted(result_dict.keys())
        total_instances = 0
        
        for f_type in sorted_types:
            instances = result_dict[f_type]
            total_instances += len(instances)
            
            type_item = QTreeWidgetItem(self.tree)
            type_item.setText(0, f"Feature Type {f_type} ({len(instances)} 个)")
            type_item.setExpanded(True)
            
            for idx, inst in enumerate(instances):
                inst_item = QTreeWidgetItem(type_item)
                inst_name = f"Type {f_type} - Instance {idx}"
                inst_item.setText(0, inst_name)
                inst_item.setData(0, Qt.UserRole, inst)
        
        if total_instances == 0:
            self.lbl_status.setText("识别完成，但未发现特征。")
            no_item = QTreeWidgetItem(self.tree)
            no_item.setText(0, "未识别到特征")

    def on_prediction_error(self, err_msg):
        self.lbl_status.setText("识别出错")
        self.tree.clear()
        self.lbl_stats.setText("计算失败")
        QMessageBox.critical(self, "识别错误", f"预测过程中发生错误:\n{err_msg}")

    def on_tree_item_clicked(self, item, column):
        inst_data = item.data(0, Qt.UserRole)
        comp_indices = set()
        bot_indices = set()

        if inst_data is not None:
            # Case A: 具体实例
            comp_indices = set(inst_data['feature_composition'])
            bot_indices = set(inst_data['bot'])
        else:
            # Case B: 特征类型组
            if item.childCount() > 0:
                print(f"Selecting Group: {item.text(0)}")
                for i in range(item.childCount()):
                    child_item = item.child(i)
                    child_data = child_item.data(0, Qt.UserRole)
                    if child_data:
                        comp_indices.update(child_data['feature_composition'])
                        bot_indices.update(child_data['bot'])
            else:
                return

        self.visualize_feature_instance(comp_indices, bot_indices)

    def visualize_feature_instance(self, comp_indices, bot_indices):
        self.ais_context.RemoveAll(True)
        
        col_bot = Quantity_Color(1.0, 0.0, 0.0, Quantity_TOC_RGB) # Red
        col_feat = Quantity_Color(0.0, 1.0, 0.0, Quantity_TOC_RGB) # Green
        col_trans = Quantity_Color(0.8, 0.8, 0.8, Quantity_TOC_RGB) # Gray
        
        for idx, face in enumerate(self.faces_list):
            ais_shape = AIS_Shape(face)
            
            if idx in bot_indices:
                ais_shape.SetColor(col_bot)
                ais_shape.SetWidth(2.0)
                self.ais_context.Display(ais_shape, False)
            elif idx in comp_indices:
                ais_shape.SetColor(col_feat)
                ais_shape.SetWidth(2.0)
                self.ais_context.Display(ais_shape, False)
            else:
                ais_shape.SetColor(col_trans)
                ais_shape.SetTransparency(0.6)
                self.ais_context.Display(ais_shape, False)
        
        self.ais_context.UpdateCurrentViewer()

    def reset_display(self):
        if self.current_shape:
            self.ais_context.RemoveAll(True)
            self.display.DisplayShape(self.current_shape, update=True)
            self.tree.clearSelection()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()