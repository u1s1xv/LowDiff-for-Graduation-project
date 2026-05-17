#!/usr/bin/env python3
"""
四组GPT-2训练日志的综合对比分析
对比不同压缩率(0, 0.001, 0.01, 0.1)的LowDiff策略效果
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_log(log_file):
    """解析日志文件"""
    epochs, batches, losses = [], [], []
    pattern = r'\[Epoch (\d+)/\d+\] Batch (\d+), Loss: ([\d.]+|nan)'
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(pattern, line)
            if match and match.group(3) != 'nan':
                epochs.append(int(match.group(1)))
                batches.append(int(match.group(2)))
                losses.append(float(match.group(3)))
    
    return epochs, batches, losses

def get_global_steps(epochs, batches):
    """计算全局步数"""
    batches_per_epoch = max([b for e, b in zip(epochs, batches) if e == 0]) + 1
    return [e * batches_per_epoch + b for e, b in zip(epochs, batches)]

def smooth_losses(losses, window=100):
    """平滑loss曲线"""
    if len(losses) <= window:
        return losses, list(range(len(losses)))
    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    return smoothed, list(range(window-1, len(losses)))

def plot_all_comparison(data_dict, save_dir):
    """绘制所有对比图"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    colors = {'baseline': '#2E86AB', 'ratio_0.001': '#F18F01', 
              'ratio_0.01': '#C73E1D', 'ratio_0.1': '#6A994E'}
    labels = {'baseline': 'Baseline (无压缩)', 'ratio_0.001': 'LowDiff (0.1%)', 
              'ratio_0.01': 'LowDiff (1%)', 'ratio_0.1': 'LowDiff (10%)'}
    
    # 1. 平滑曲线对比
    plt.figure(figsize=(15, 7))
    for key in ['baseline', 'ratio_0.001', 'ratio_0.01', 'ratio_0.1']:
        epochs, batches, losses = data_dict[key]
        steps = get_global_steps(epochs, batches)
        smoothed, smooth_idx = smooth_losses(losses)
        plt.plot([steps[i] for i in smooth_idx], smoothed, linewidth=2.5, 
                color=colors[key], label=labels[key], alpha=0.9)
    
    plt.xlabel('训练步数 (Global Steps)', fontsize=13)
    plt.ylabel('Loss', fontsize=13)
    plt.title('GPT-2训练Loss对比 - 不同压缩率的影响', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'all_comparison_smoothed.png', dpi=300, bbox_inches='tight')
    print(f"✓ 保存平滑对比曲线")
    plt.close()
    
    # 2. 按Epoch的平均Loss对比
    fig, ax = plt.subplots(figsize=(14, 7))
    x_pos = np.arange(10)
    width = 0.2
    
    for idx, key in enumerate(['baseline', 'ratio_0.001', 'ratio_0.01', 'ratio_0.1']):
        epochs, _, losses = data_dict[key]
        epoch_means = [np.mean([losses[i] for i, e in enumerate(epochs) if e == ep]) 
                      for ep in range(10)]
        ax.bar(x_pos + idx*width, epoch_means, width, label=labels[key], 
               color=colors[key], alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('平均Loss', fontsize=13)
    ax.set_title('各Epoch平均Loss对比', fontsize=15, fontweight='bold')
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels([f'E{i}' for i in range(10)])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_dir / 'epoch_mean_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ 保存Epoch平均Loss对比")
    plt.close()
    
    # 3. 统计表格
    stats = []
    for key in ['baseline', 'ratio_0.001', 'ratio_0.01', 'ratio_0.1']:
        _, _, losses = data_dict[key]
        stats.append({
            '配置': labels[key],
            '总步数': len(losses),
            '初始Loss': f"{losses[0]:.4f}",
            '最终Loss': f"{losses[-1]:.4f}",
            '最低Loss': f"{min(losses):.4f}",
            '平均Loss': f"{np.mean(losses):.4f}",
            '标准差': f"{np.std(losses):.4f}"
        })
    
    df = pd.DataFrame(stats)
    
    # 绘制统计表格
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                    cellLoc='center', loc='center',
                    colColours=['#E8E8E8']*len(df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('训练统计对比表', fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_dir / 'statistics_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ 保存统计表格")
    plt.close()
    
    # 4. 性能退化分析
    baseline_mean = np.mean(data_dict['baseline'][2])
    degradations = []
    
    print("\n" + "="*70)
    print("性能退化分析 (相对于Baseline)")
    print("="*70)
    
    for key, ratio_name in [('ratio_0.001', '0.1%'), ('ratio_0.01', '1%'), ('ratio_0.1', '10%')]:
        mean_loss = np.mean(data_dict[key][2])
        deg = ((mean_loss - baseline_mean) / baseline_mean) * 100
        degradations.append({'压缩率': ratio_name, '性能退化': f"{deg:+.2f}%"})
        print(f"压缩率 {ratio_name:>6}: 平均Loss {mean_loss:.4f}, 退化 {deg:+.2f}%")
    
    print("="*70)
    
    return df, degradations

def main():
    logs = {
        'baseline': Path(r"C:\Graduation project\LowDiff\baseline_lowdiff_20260212_151747.log"),
        'ratio_0.001': Path(r"C:\Graduation project\LowDiff\gpt_lowdiff_20260225_215131.log"),
        'ratio_0.01': Path(r"C:\Graduation project\LowDiff\gpt_lowdiff_20260213_022859.log"),
        'ratio_0.1': Path(r"C:\Graduation project\LowDiff\gpt_lowdiff_20260225_121819.log")
    }
    
    print("📖 正在读取所有日志文件...")
    data_dict = {}
    
    for key, log_path in logs.items():
        if not log_path.exists():
            print(f"❌ 找不到文件: {log_path}")
            return
        print(f"  - {key}: {log_path.name}")
        data_dict[key] = parse_log(log_path)
        print(f"    ✓ 解析了 {len(data_dict[key][2])} 个loss值")
    
    save_dir = logs['baseline'].parent / "plots_all_comparison"
    print(f"\n📊 正在生成综合对比图...")
    
    df, deg = plot_all_comparison(data_dict, save_dir)
    
    print(f"\n✅ 完成！所有图表已保存到: {save_dir}")

if __name__ == "__main__":
    main()

