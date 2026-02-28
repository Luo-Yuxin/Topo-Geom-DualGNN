import matplotlib.pyplot as plt
import numpy as np
import os

def plot_3d_scatter(points_2d_list, title="3D Scatter Plot", point_color='blue', point_size=20):
    """
    绘制三维散点图（核心逻辑保持不变）
    """
    # 1. 转换为numpy数组（自动适配二维列表/数组输入）
    points = np.asarray(points_2d_list, dtype=np.float32)
    
    # 输入格式校验
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"输入坐标格式错误！需为二维列表（每行x,y,z），当前输入维度：{points.shape}")
    
    # 2. 自动拆分x/y/z分量
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # 3. 创建3D画布
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 4. 绘制散点
    ax.scatter(x, y, z, c=point_color, s=point_size, alpha=0.8, edgecolors='black')
    
    # 5. 显式配置坐标系
    ax.set_xlabel('X Axis', fontsize=12, labelpad=10)
    ax.set_ylabel('Y Axis', fontsize=12, labelpad=10)
    ax.set_zlabel('Z Axis', fontsize=12, labelpad=10)
    
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 优化坐标轴范围
    def get_margin(arr):
        return (np.max(arr) - np.min(arr)) * 0.1 if np.max(arr) != np.min(arr) else 1.0

    x_margin = get_margin(x)
    y_margin = get_margin(y)
    z_margin = get_margin(z)
    
    ax.set_xlim(np.min(x)-x_margin, np.max(x)+x_margin)
    ax.set_ylim(np.min(y)-y_margin, np.max(y)+y_margin)
    ax.set_zlim(np.min(z)-z_margin, np.max(z)+z_margin)
    
    # 原点辅助线
    ax.plot([0, 0], [0, 0], [np.min(z)-z_margin, np.max(z)+z_margin], 
            color='red', linestyle=':', linewidth=1, label='Z')
    ax.plot([0, 0], [np.min(y)-y_margin, np.max(y)+y_margin], [0, 0], 
            color='green', linestyle=':', linewidth=1, label='Y')
    ax.plot([np.min(x)-x_margin, np.max(x)+x_margin], [0, 0], [0, 0], 
            color='blue', linestyle=':', linewidth=1, label='X')
    ax.legend(fontsize=10, loc='upper right')
    
    ax.set_title(title, fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

# ---------------------- 修改后的主程序 ----------------------
if __name__ == "__main__":
    # 指定要读取的文件名（需与第一步中保存的文件名一致）
    file_path = 'points_cloud_face.npy'
    
    if os.path.exists(file_path):
        try:
            print(f"正在加载文件: {file_path} ...")
            
            # 【核心修改】使用 np.load 读取数据
            loaded_points = np.load(file_path)
            
            print(f"加载成功！数据形状: {loaded_points.shape}")
            print("正在绘图...")
            
            plot_3d_scatter(
                points_2d_list=loaded_points,
                title=f"Visualization from {file_path}",
                point_color='orange',  # 你可以改成 'red', 'blue' 等
                point_size=50
            )
            
        except Exception as e:
            print(f"读取或绘图出错: {e}")
    else:
        print(f"错误: 找不到文件 '{file_path}'。请确保先运行了数据生成脚本。")