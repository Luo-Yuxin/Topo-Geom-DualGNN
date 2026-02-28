import os
import sys
import shutil
import logging
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# 核心 OpenCascade 库
try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    
    # 引入 TopologyUtils 用于高级拓扑检查 (来自 pythonocc-core)
    from OCC.Extend import TopologyUtils
except ImportError:
    print("错误: 未检测到 pythonocc-core 库。")
    print("请使用 conda 安装: conda install -c conda-forge pythonocc-core")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scan_result.log", encoding='utf-8', mode='w'),
    ]
)

class AdvancedValidator:
    """
    集成了基础几何检查、高级拓扑检查 (Manifold/Closed) 以及内腔检测
    """
    
    def load_step_file(self, file_path: str):
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(file_path))
        if status != IFSelect_RetDone:
            return None
        reader.TransferRoots()
        return reader.Shape()

    # --- 来自 TopologyChecker 的移植逻辑 (开始) ---
    
    def find_edges_from_wires(self, top_exp):
        edge_set = set()
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for edge in wire_exp.ordered_edges():
                edge_set.add(edge)
        return edge_set

    def find_edges_from_top_exp(self, top_exp):
        edge_set = set(top_exp.edges())
        return edge_set

    def check_closed(self, body):
        """
        检查是否封闭。
        逻辑：对比 Wire 中的边和拓扑遍历发现的边。如果有边不在 Wire 中，说明是游离边（非封闭）。
        """
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        edges_from_wires = self.find_edges_from_wires(top_exp)
        edges_from_top_exp = self.find_edges_from_top_exp(top_exp)
        missing_edges = edges_from_top_exp - edges_from_wires
        return len(missing_edges) == 0

    def check_manifold(self, top_exp):
        """
        检查是否流形 (Manifold)。
        非流形通常指一条边被超过 2 个面共享。
        """
        faces = set()
        for shell in top_exp.shells():
            for face in top_exp._loop_topo(TopAbs_FACE, shell):
                if face in faces:
                    return False # 同一个面出现在多个 Shell 中，或者结构混乱
                faces.add(face)
        return True

    def check_unique_coedges(self, top_exp):
        """
        检查 Co-edges 唯一性。
        用于检测复杂的拓扑环路错误。
        """
        coedge_set = set()
        for loop in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(loop)
            for coedge in wire_exp.ordered_edges():
                orientation = coedge.Orientation()
                tup = (coedge, orientation)
                if tup in coedge_set:
                    return False
                coedge_set.add(tup)
        return True

    # --- 来自 TopologyChecker 的移植逻辑 (结束) ---

    def is_solid(self, shape) -> bool:
        """必须包含至少一个 Solid"""
        return TopExp_Explorer(shape, TopAbs_SOLID).More()

    def has_internal_voids(self, shape) -> bool:
        """
        内腔检测 (原脚本逻辑)
        一个 Solid 如果包含 >1 个 Shell，通常意味着有内腔
        """
        solid_explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        while solid_explorer.More():
            solid = topods.Solid(solid_explorer.Current())
            shell_explorer = TopExp_Explorer(solid, TopAbs_SHELL)
            shell_count = 0
            while shell_explorer.More():
                shell_count += 1
                shell_explorer.Next()
            
            if shell_count > 1:
                return True
            solid_explorer.Next()
        return False

    def validate_all(self, shape):
        """
        执行所有检查，返回错误代码。优先级从严重到轻微。
        """
        # 1. 基础 BRep 检查 (底层数学定义)
        analyzer = BRepCheck_Analyzer(shape)
        if not analyzer.IsValid():
            return "error_geometry"

        # 2. 实体检查 (是否包含 Solid)
        if not self.is_solid(shape):
            return "error_not_solid"
        
        # 初始化 TopologyExplorer (OCC.Extend) 用于后续检查
        top_exp = TopologyUtils.TopologyExplorer(shape, ignore_orientation=True)
        if top_exp.number_of_faces() == 0:
            return "error_empty"

        # 3. 非流形检查 (Non-Manifold)
        if not self.check_manifold(top_exp):
            return "error_non_manifold"

        # 4. 封闭性检查 (Watertight / Closed)
        if not self.check_closed(shape):
            return "error_not_closed"

        # 5. Co-edges 检查
        if not self.check_unique_coedges(top_exp):
            return "error_topology_other"

        # 6. 内腔检查 (业务逻辑)
        if self.has_internal_voids(shape):
            return "error_void"

        return "valid"

def move_file_safe(src_path: Path, dest_folder: Path):
    if dest_folder:
        try:
            shutil.move(str(src_path), str(dest_folder / src_path.name))
        except Exception as e:
            logging.error(f"移动文件失败 {src_path.name}: {e}")

def process_single_file(args):
    """子进程任务"""
    file_path, output_dirs = args
    filename = file_path.name
    validator = AdvancedValidator()
    
    try:
        shape = validator.load_step_file(str(file_path))
        if not shape or shape.IsNull():
            return "error_read"

        result_code = validator.validate_all(shape)
        
        # 根据结果移动文件
        if result_code != "valid":
            dest = output_dirs.get(result_code)
            if dest:
                move_file_safe(file_path, dest)
                logging.warning(f"[{result_code}] {filename}")
        
        return result_code

    except Exception as e:
        logging.error(f"处理异常 {filename}: {e}")
        return "error_read"

def batch_process_advanced(input_dir: str, output_base_dir: str = None, num_works: int = 4):
    start_time = time.time()
    input_path = Path(input_dir)
    
    # 定义错误类型对应的目录
    dirs = {}
    if output_base_dir:
        base = Path(output_base_dir)
        dirs = {
            "error_geometry":       base / "01_Geometry_Invalid", # 底层几何坏了
            "error_not_solid":      base / "02_Not_Solid",        # 不是实体(片体)
            "error_non_manifold":   base / "03_Non_Manifold",     # 非流形(结构混乱)
            "error_not_closed":     base / "04_Not_Closed",       # 没封口/漏水
            "error_topology_other": base / "05_Topology_Bad",     # 其他拓扑问题
            "error_void":           base / "06_Internal_Voids",   # 有内腔
            "error_empty":          base / "00_Empty_Files"       # 空文件
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    # 收集文件 (Set去重)
    extensions = ['*.stp', '*.step', '*.STP', '*.STEP']
    files_set = set()
    for ext in extensions:
        files_set.update(input_path.glob(ext))
    files = list(files_set)

    total_files = len(files)
    if total_files == 0:
        print("未找到 STEP 文件。")
        return

    # 并行设置
    max_workers = num_works
    print(f"启动高级拓扑筛选 | 文件数: {total_files} | 进程数: {max_workers}")
    print("正在检查: 几何有效性 -> 实体 -> 流形 -> 封闭性 -> 拓扑环 -> 内腔")
    print("-" * 60)

    # 统计
    stats = {k: 0 for k in dirs.keys()}
    stats["valid"] = 0
    stats["error_read"] = 0

    tasks = [(f, dirs) for f in files]
    processed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single_file, task): task[0] for task in tasks}
        
        for future in as_completed(future_to_file):
            processed_count += 1
            try:
                res = future.result()
                if res in stats:
                    stats[res] += 1
                else:
                    stats["error_read"] += 1
            except Exception:
                stats["error_read"] += 1
            
            if processed_count % 5 == 0 or processed_count == total_files:
                percent = (processed_count / total_files) * 100
                sys.stdout.write(f"\r进度: [{processed_count}/{total_files}] {percent:.1f}% ")
                sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"\n" + "-" * 60)
    print(f"耗时: {elapsed:.2f}s | 速度: {total_files/elapsed:.2f} 文件/s")
    print("-" * 60)
    print(f"合格模型 (Perfect): {stats['valid']}")
    print("-" * 20)
    print(f"几何损坏 (BRep Check) : {stats['error_geometry']}")
    print(f"非实体 (Not Solid)    : {stats['error_not_solid']}")
    print(f"非流形 (Non-Manifold) : {stats['error_non_manifold']}")
    print(f"未封闭 (Open Shells)  : {stats['error_not_closed']}")
    print(f"拓扑环错误 (Loop Err) : {stats['error_topology_other']}")
    print(f"含内腔 (Internal Void): {stats['error_void']}")
    print(f"读取失败/其他         : {stats['error_read']}")
    print("-" * 60)

if __name__ == "__main__":
    # 配置 ===========================
    INPUT_FOLDER = r"data\raw_steps_f"
    OUTPUT_BASE_FOLDER = r"data\failed-step"
    num_works = 4
    # ===============================

    batch_process_advanced(INPUT_FOLDER, OUTPUT_BASE_FOLDER, num_works)