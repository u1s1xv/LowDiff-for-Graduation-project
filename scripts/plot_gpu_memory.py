import matplotlib.pyplot as plt
import numpy as np

# 数据设置 - 修改这些数值
# GPU内存使用情况（单位：GB）
# 请将以下数值修改为您的实际数据
original_memory = 6.97      # Original场景的GPU内存使用
without_offload_memory = 7.90  # w/o Offload场景的GPU内存使用
with_offload_memory =  8.95   # w/ Offload场景的GPU内存使用

# 创建图表
fig, ax = plt.subplots(figsize=(10, 6))

# 设置x轴位置
x_pos = np.arange(3)
width = 0.6

# 使用不同的颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色, 橙色, 绿色

# 数据数组
data = [original_memory, without_offload_memory, with_offload_memory]

# 绘制柱状图
bars = ax.bar(x_pos, data, width, color=colors, edgecolor='black', linewidth=1.5)

# 设置图表标签
ax.set_xlabel('Scenarios', fontsize=12, fontweight='bold')
ax.set_ylabel('GPU Memory (GB)', fontsize=12, fontweight='bold')
ax.set_title('GPU Memory Consumption', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['Original', 'w/o Offload', 'w/ Offload'], fontsize=11)

# 设置y轴范围：从0开始到12
ax.set_ylim(0, 12)
ax.set_yticks(range(0, 12, 2))  # 设置y轴刻度，从10到35，每5为一个间隔

# 在柱子上添加数值标签
for i, (bar, value) in enumerate(zip(bars, data)):
    height = bar.get_height()
    # 在柱子顶部显示数值
    ax.text(bar.get_x() + bar.get_width()/2., 
            height + 0.3,  # 在柱子上方显示
            f'{value:.1f} GB',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 添加网格线
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('gpu_memory_consumption.png', dpi=300, bbox_inches='tight')
print("图表已保存为 gpu_memory_consumption.png")

# 显示图表
plt.show()