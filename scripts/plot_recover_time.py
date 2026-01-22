import matplotlib.pyplot as plt
import numpy as np
import os

# 设置matplotlib后端，避免显示问题
plt.switch_backend('Agg')  # 使用非交互式后端，避免GUI问题

# 横坐标数据 - 检查点频率
checkpoint_frequencies = [5, 10, 25, 50]
x = np.arange(len(checkpoint_frequencies))  # 0, 1, 2, 3

# 纵坐标数据 - 恢复时间（单位：秒）
# 请在这里修改您的实际数据
recovery_times = {
    'Baseline': [0, 0, 0, 0.974],      # 请修改为实际数据
    'Naïve DC': [0, 0, 0, 1.197],      # 请修改为实际数据
    'LowDiff': [0, 0, 0, 33.977]      # 请修改为实际数据
}

# 不同策略的颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 柱状图宽度
bar_width = 0.2

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制每个策略的柱状图
for i, (strategy, times) in enumerate(recovery_times.items()):
    # 计算每个柱子的x轴位置
    offset = (i - 1.5) * bar_width
    bars = ax.bar(x + offset, times, bar_width, 
                  label=strategy, color=colors[i], edgecolor='black', 
                  linewidth=0.5, alpha=0.9)
    
    # 在柱子上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

# 设置图表属性
ax.set_xlabel('Checkpoint Frequency', fontsize=12, fontweight='bold')
ax.set_ylabel('Recovery Time (s)', fontsize=12, fontweight='bold')
ax.set_title('Recovery Time of Different Methods', fontsize=14, fontweight='bold', pad=15)

# 设置x轴刻度和标签
ax.set_xticks(x)
ax.set_xticklabels(checkpoint_frequencies)
ax.set_xlim(-0.5, len(x) - 0.5)

# 设置y轴范围
ax.set_ylim(0, 20)
ax.set_yticks(np.arange(0, 21, 2))
ax.set_yticklabels([f'{i}' for i in range(0, 21, 2)])

# 添加网格线
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

# 添加图例
ax.legend(title='Strategies', title_fontsize=11, fontsize=10, 
          loc='upper right', framealpha=0.9)

# 调整布局
plt.tight_layout()

# 保存图表到文件
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存为PNG
png_path = os.path.join(output_dir, "recovery_time_comparison.png")
plt.savefig(png_path, dpi=300, bbox_inches='tight')
print(f"图表已保存为: {png_path}")

# 可选：保存为PDF（矢量图，质量更高）
pdf_path = os.path.join(output_dir, "recovery_time_comparison.pdf")
plt.savefig(pdf_path, bbox_inches='tight')
print(f"图表已保存为: {pdf_path}")

# 可选：保存为SVG
svg_path = os.path.join(output_dir, "recovery_time_comparison.svg")
plt.savefig(svg_path, bbox_inches='tight')
print(f"图表已保存为: {svg_path}")

# 关闭图形，释放内存
plt.close(fig)

# 如果要在支持GUI的环境中显示图表，可以取消以下注释
# 注意：在服务器或无GUI环境中，可能需要注释掉这行
# plt.show()

print("图表生成完成！")