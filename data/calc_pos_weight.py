import os
import json
import glob
from tqdm import tqdm
import torch
from itertools import chain
from collections import Counter

# ================= 配置区域 =================
# 标签数据的根目录
# LABEL_DIR = os.path.join('data', 'raw_labels_f') # 请修改为你实际的路径
LABEL_DIR = r"data\labels_f"

# 文件扩展名
FILE_EXT = '*.json'

# 输出的统计结果文件路径
OUTPUT_JSON_PATH = 'class_weights_stats.json'
# ===========================================

def get_stats_from_json(file_path):
    """
    功能: 读取单个 JSON 文件，同时返回 inst 和 bottom 的统计数据。
    
    Returns:
        stats (dict): {
            'inst': (pos_count, neg_count),
            'bottom': (pos_count, neg_count)
        }
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        raise ValueError(f"JSON Load Error: {e}")

    # 1. 尝试定位包含数据的字典
    data_dict = None
    if isinstance(raw_data, list):
        if len(raw_data) > 0 and len(raw_data[0]) > 1:
             data_dict = raw_data[0][1]
    elif isinstance(raw_data, dict):
        data_dict = raw_data
        
    # 初始化结果容器
    result = {
        'inst': (0, 0),
        'bottom': (0, 0)
    }

    if data_dict is None:
        return result

    # 2. 统计 'inst' (实例分割邻接矩阵)
    if "inst" in data_dict:
        labels = data_dict["inst"]
        # labels 通常是二维邻接矩阵 [[0, 1], [1, 0]...]
        # chain(*labels) 将其展平为一维，高效统计
        counter = Counter(chain(*labels))
        neg = counter.get(0, 0)
        pos = counter.get(1, 0)
        result['inst'] = (pos, neg)

    # 3. 统计 'bottom' (底面预测)
    if "bottom" in data_dict:
        labels = data_dict["bottom"]
        neg = 0
        pos = 0
        
        # 兼容字典格式 {face_id: 0/1}
        if isinstance(labels, dict):
            # dict_values 转 list 后统计
            bottom_values = list(labels.values())
            neg = bottom_values.count(0)
            pos = bottom_values.count(1)
        # 兼容列表格式 [0, 1, 0...]
        elif isinstance(labels, list):
            neg = labels.count(0)
            pos = labels.count(1)
            
        result['bottom'] = (pos, neg)

    return result

def calculate_metrics(pos, neg):
    """辅助计算比率和权重"""
    total = pos + neg
    if total == 0:
        return 0.0, 0.0
    
    ratio = pos / total
    weight = 0.0
    if pos > 0:
        weight = neg / pos
        
    return ratio, weight

def print_report(name, stats):
    """打印单个任务的统计报告"""
    pos = stats['pos_count']
    neg = stats['neg_count']
    weight = stats['suggested_pos_weight']
    ratio = stats['pos_ratio']
    
    print(f"\n📈 [任务: {name}] 统计结果:")
    print("-" * 40)
    print(f"   ✅ 正样本 (1): {pos}")
    print(f"   ✅ 负样本 (0): {neg}")
    
    if pos + neg == 0:
        print("   ❌ 总样本为 0，跳过。")
        return

    print(f"   📊 正样本比例: {ratio:.4%}")

    if pos > 0:
        print(f"   💡 建议 pos_weight: {weight:.4f}")
        print(f"   📝 代码用法: criterion_{name} = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([{weight:.2f}]).to(device))")
    else:
        print("   ❌ 正样本为 0，无法计算权重。")

def main():
    print(f"🔍 开始扫描目录: {LABEL_DIR}")
    
    search_pattern = os.path.join(LABEL_DIR, FILE_EXT)
    files = glob.glob(search_pattern)
    
    if len(files) == 0:
        print(f"❌ 未找到任何 {FILE_EXT} 文件，请检查路径配置。")
        return

    print(f"📚 发现 {len(files)} 个文件，开始并行统计...")

    # 全局累加器
    total_stats = {
        'inst': {'pos': 0, 'neg': 0},
        'bottom': {'pos': 0, 'neg': 0}
    }
    
    error_files = []

    for file_path in tqdm(files, desc="Calculating"):
        try:
            file_stats = get_stats_from_json(file_path)
            
            # 累加 inst
            total_stats['inst']['pos'] += file_stats['inst'][0]
            total_stats['inst']['neg'] += file_stats['inst'][1]
            
            # 累加 bottom
            total_stats['bottom']['pos'] += file_stats['bottom'][0]
            total_stats['bottom']['neg'] += file_stats['bottom'][1]
            
        except Exception as e:
            error_files.append(f"{os.path.basename(file_path)}: {str(e)}")

    print("\n" + "="*60)
    print("📊 数据集类别权重报告 (Class Weight Report)")
    print("="*60)
    
    if error_files:
        print(f"⚠️  有 {len(error_files)} 个文件解析失败 (已跳过)")
    
    # --- 准备保存的数据结构 ---
    json_output = {}

    # 处理 Inst 数据
    inst_pos = total_stats['inst']['pos']
    inst_neg = total_stats['inst']['neg']
    inst_ratio, inst_weight = calculate_metrics(inst_pos, inst_neg)
    
    json_output['inst'] = {
        'pos_count': inst_pos,
        'neg_count': inst_neg,
        'total_count': inst_pos + inst_neg,
        'pos_ratio': inst_ratio,
        'suggested_pos_weight': inst_weight
    }
    
    # 处理 Bottom 数据
    bot_pos = total_stats['bottom']['pos']
    bot_neg = total_stats['bottom']['neg']
    bot_ratio, bot_weight = calculate_metrics(bot_pos, bot_neg)
    
    json_output['bottom'] = {
        'pos_count': bot_pos,
        'neg_count': bot_neg,
        'total_count': bot_pos + bot_neg,
        'pos_ratio': bot_ratio,
        'suggested_pos_weight': bot_weight
    }

    # --- 打印报告 ---
    print_report("Instance (inst)", json_output['inst'])
    print_report("Bottom (bottom)", json_output['bottom'])
    
    # --- 保存到 JSON 文件 ---
    try:
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=4, ensure_ascii=False)
        print("\n" + "="*60)
        print(f"💾 统计结果已保存至: {os.path.abspath(OUTPUT_JSON_PATH)}")
        print("="*60)
    except Exception as e:
        print(f"❌ 保存 JSON 文件失败: {e}")

if __name__ == "__main__":
    main()