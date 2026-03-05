#!/usr/bin/env python3
"""
梯度补偿机制效果分析
对比：Baseline vs 原始LowDiff vs 梯度补偿LowDiff
"""

import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_log(log_file):
    """解析日志文件"""
    epochs, losses = [], []
    pattern = r'\[Epoch (\d+)/\d+\] Batch \d+, Loss: ([\d.]+|nan)'
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(pattern, line)
            if match and match.group(2) != 'nan':
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    return epochs, losses

def smooth_losses(losses, window=100):
    """平滑loss曲线"""
    if len(losses) <= window:
        return losses
    return np.convolve(losses, np.ones(window)/window, mode='valid')

def analyze_stats(epochs, losses, label):
    """计算统计指标"""
    epoch_means = []
    for e in range(max(epochs) + 1):
        epoch_losses = [losses[i] for i, ep in enumerate(epochs) if ep == e]
        if epoch_losses:
            epoch_means.append(np.mean(epoch_losses))
    
    return {
        'label': label,
        'total_steps': len(losses),
        'num_epochs': len(epoch_means),
        'initial_loss': losses[0],
        'final_loss': losses[-1],
        'min_loss': np.min(losses),
        'mean_loss': np.mean(losses),
        'std_loss': np.std(losses),
        'reduction': losses[0] - losses[-1],
        'reduction_rate': (losses[0] - losses[-1]) / losses[0] * 100,
        'epoch_means': epoch_means
    }

def main():
    logs = {
        'Baseline': r"C:\Graduation project\LowDiff\baseline_lowdiff_20260227_131642.log",
        'LowDiff原始': r"C:\Graduation project\LowDiff\gpt_lowdiff_20260213_022859.log",
        'LowDiff+梯度补偿': r"C:\Graduation project\LowDiff\gpt_lowdiff_20260227_110015.log"
    }
    
    colors = {
        'Baseline': '#2E86AB',
        'LowDiff原始': '#C73E1D', 
        'LowDiff+梯度补偿': '#6A994E'
    }
    
    print("="*80)
    print("梯度补偿机制效果分析")
    print("="*80)
    
    # 1. 解析数据
    print("\n📖 正在解析训练日志...")
    data = {}
    for label, path in logs.items():
        log_path = Path(path)
        if not log_path.exists():
            print(f"❌ 找不到文件: {path}")
            return
        epochs, losses = parse_log(log_path)
        data[label] = {
            'epochs': epochs,
            'losses': np.array(losses),
            'stats': analyze_stats(epochs, losses, label)
        }
        print(f"  ✓ {label}: {len(losses)} 个loss值")
    
    # 2. 统计对比
    print("\n【关键统计指标对比】")
    print("-"*80)
    
    stats_list = []
    for label in logs.keys():
        s = data[label]['stats']
        stats_list.append({
            '配置': label,
            '总步数': s['total_steps'],
            '初始Loss': f"{s['initial_loss']:.4f}",
            '最终Loss': f"{s['final_loss']:.4f}",
            '最低Loss': f"{s['min_loss']:.4f}",
            '平均Loss': f"{s['mean_loss']:.4f}",
            '标准差': f"{s['std_loss']:.4f}",
            'Loss下降': f"{s['reduction']:.4f}",
            '下降率': f"{s['reduction_rate']:.1f}%"
        })
    
    df = pd.DataFrame(stats_list)
    print(df.to_string(index=False))
    
    # 3. 性能改进分析
    print("\n【梯度补偿机制效果评估】")
    print("-"*80)
    
    baseline_mean = data['Baseline']['stats']['mean_loss']
    original_mean = data['LowDiff原始']['stats']['mean_loss']
    compensated_mean = data['LowDiff+梯度补偿']['stats']['mean_loss']
    
    original_degradation = ((original_mean - baseline_mean) / baseline_mean) * 100
    compensated_degradation = ((compensated_mean - baseline_mean) / baseline_mean) * 100
    improvement = original_degradation - compensated_degradation
    
    print(f"\n(1) 相对于Baseline的性能退化:")
    print(f"  LowDiff原始:      {original_degradation:+.2f}%")
    print(f"  LowDiff+梯度补偿: {compensated_degradation:+.2f}%")
    print(f"  改进幅度:         {improvement:.2f}个百分点")
    
    baseline_final = data['Baseline']['stats']['final_loss']
    original_final = data['LowDiff原始']['stats']['final_loss']
    compensated_final = data['LowDiff+梯度补偿']['stats']['final_loss']
    
    print(f"\n(2) 最终Loss对比:")
    print(f"  Baseline:         {baseline_final:.4f}")
    print(f"  LowDiff原始:      {original_final:.4f} (高{original_final-baseline_final:.4f})")
    print(f"  LowDiff+梯度补偿: {compensated_final:.4f} (高{compensated_final-baseline_final:.4f})")
    
    if compensated_final < original_final:
        print(f"  ✓ 梯度补偿使最终Loss降低了 {original_final-compensated_final:.4f}")
    else:
        print(f"  ✗ 梯度补偿未能改善最终Loss")
    
    baseline_reduction = data['Baseline']['stats']['reduction']
    original_reduction = data['LowDiff原始']['stats']['reduction']
    compensated_reduction = data['LowDiff+梯度补偿']['stats']['reduction']
    
    print(f"\n(3) 收敛速度对比:")
    print(f"  Baseline:         下降{baseline_reduction:.4f} (100%)")
    print(f"  LowDiff原始:      下降{original_reduction:.4f} ({original_reduction/baseline_reduction*100:.1f}%)")
    print(f"  LowDiff+梯度补偿: 下降{compensated_reduction:.4f} ({compensated_reduction/baseline_reduction*100:.1f}%)")
    
    # 4. 可视化
    save_dir = Path(r"C:\Graduation project\LowDiff\plots_gradient_compensation")
    save_dir.mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 4.1 平滑loss曲线对比
    ax1 = fig.add_subplot(gs[0, :])
    for label in logs.keys():
        losses = data[label]['losses']
        smoothed = smooth_losses(losses, window=100)
        steps = np.arange(len(smoothed)) + 99
        ax1.plot(steps, smoothed, linewidth=2.5, color=colors[label], 
                label=label, alpha=0.9)
    ax1.set_xlabel('训练步数 (Global Steps)', fontsize=13)
    ax1.set_ylabel('Loss', fontsize=13)
    ax1.set_title('梯度补偿机制效果对比 - 训练Loss曲线', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 4.2 按Epoch的平均Loss
    ax2 = fig.add_subplot(gs[1, 0])
    max_epochs = max([len(data[label]['stats']['epoch_means']) for label in logs.keys()])
    for label in logs.keys():
        epoch_means = data[label]['stats']['epoch_means']
        ax2.plot(range(len(epoch_means)), epoch_means, 'o-', linewidth=2.5,
                color=colors[label], label=label, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('平均Loss', fontsize=12)
    ax2.set_title('各Epoch平均Loss对比', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 4.3 性能指标对比柱状图
    ax3 = fig.add_subplot(gs[1, 1])
    metrics = ['平均Loss', '最终Loss', '最低Loss']
    x = np.arange(len(logs))
    width = 0.25
    
    mean_vals = [data[label]['stats']['mean_loss'] for label in logs.keys()]
    final_vals = [data[label]['stats']['final_loss'] for label in logs.keys()]
    min_vals = [data[label]['stats']['min_loss'] for label in logs.keys()]
    
    ax3.bar(x - width, mean_vals, width, label='平均Loss', alpha=0.8, color='#4A90E2')
    ax3.bar(x, final_vals, width, label='最终Loss', alpha=0.8, color='#E94B3C')
    ax3.bar(x + width, min_vals, width, label='最低Loss', alpha=0.8, color='#50C878')
    
    ax3.set_xlabel('配置', fontsize=12)
    ax3.set_ylabel('Loss值', fontsize=12)
    ax3.set_title('关键指标对比', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(logs.keys(), fontsize=10)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(save_dir / 'gradient_compensation_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 对比图表已保存: {save_dir / 'gradient_compensation_analysis.png'}")
    
    # 5. 保存统计表格
    fig2, ax = plt.subplots(figsize=(15, 3))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center',
                    colColours=['#E8E8E8']*len(df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('梯度补偿机制统计对比表', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_dir / 'gradient_compensation_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ 统计表格已保存: {save_dir / 'gradient_compensation_table.png'}")
    
    # 6. 结论
    print("\n【结论】")
    print("-"*80)
    if improvement > 0:
        print(f"✅ 梯度补偿机制有效！性能退化减少了{improvement:.2f}个百分点")
    else:
        print(f"⚠️  梯度补偿机制效果不明显，性能退化仅改善{abs(improvement):.2f}个百分点")
    
    if compensated_final < original_final:
        improvement_pct = (original_final - compensated_final) / original_final * 100
        print(f"✅ 最终Loss改善了{improvement_pct:.1f}%")
    
    print("="*80)

if __name__ == "__main__":
    main()

