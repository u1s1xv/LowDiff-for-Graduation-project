#!/usr/bin/env python3
"""
GPT-2训练困惑度(Perplexity)分析
基于四组训练日志对比不同压缩率的影响
"""

import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_log(log_file):
    """解析日志文件，提取epoch和loss"""
    epochs, losses = [], []
    pattern = r'\[Epoch (\d+)/\d+\] Batch \d+, Loss: ([\d.]+|nan)'
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(pattern, line)
            if match and match.group(2) != 'nan':
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    return epochs, losses

def loss_to_perplexity(losses):
    """将loss转换为困惑度: perplexity = exp(loss)"""
    return np.exp(losses)

def smooth_curve(values, window=100):
    """平滑曲线"""
    if len(values) <= window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid')

def analyze_perplexity(epochs, perplexities, label):
    """分析困惑度统计"""
    epoch_means = []
    for e in range(10):
        epoch_ppl = [perplexities[i] for i, ep in enumerate(epochs) if ep == e]
        epoch_means.append(np.mean(epoch_ppl))
    
    return {
        'label': label,
        'epoch_means': epoch_means,
        'initial': perplexities[0],
        'final': perplexities[-1],
        'min': np.min(perplexities),
        'mean': np.mean(perplexities),
        'median': np.median(perplexities),
        'reduction': perplexities[0] - perplexities[-1],
        'reduction_rate': (perplexities[0] - perplexities[-1]) / perplexities[0] * 100
    }

def main():
    logs = {
        'Baseline': r"C:\Graduation project\LowDiff\baseline_lowdiff_20260212_151747.log",
        'LowDiff 0.1%': r"C:\Graduation project\LowDiff\gpt_lowdiff_20260225_215131.log",
        'LowDiff 1%': r"C:\Graduation project\LowDiff\gpt_lowdiff_20260213_022859.log",
        'LowDiff 10%': r"C:\Graduation project\LowDiff\gpt_lowdiff_20260225_121819.log"
    }
    
    colors = {'Baseline': '#2E86AB', 'LowDiff 0.1%': '#F18F01', 
              'LowDiff 1%': '#C73E1D', 'LowDiff 10%': '#6A994E'}
    
    print("="*80)
    print("GPT-2训练困惑度(Perplexity)分析报告")
    print("="*80)
    
    # 1. 解析数据并计算困惑度
    print("\n📖 正在解析日志并计算困惑度...")
    data = {}
    for label, path in logs.items():
        epochs, losses = parse_log(Path(path))
        perplexities = loss_to_perplexity(np.array(losses))
        data[label] = {
            'epochs': epochs,
            'losses': losses,
            'perplexities': perplexities,
            'stats': analyze_perplexity(epochs, perplexities, label)
        }
        print(f"  ✓ {label}: {len(losses)} 个数据点")
    
    # 2. 统计对比表
    print("\n【困惑度统计对比】")
    print("-"*80)
    stats_data = []
    for label in logs.keys():
        s = data[label]['stats']
        stats_data.append({
            '配置': label,
            '初始困惑度': f"{s['initial']:.2f}",
            '最终困惑度': f"{s['final']:.2f}",
            '最低困惑度': f"{s['min']:.2f}",
            '平均困惑度': f"{s['mean']:.2f}",
            '中位数': f"{s['median']:.2f}",
            '下降量': f"{s['reduction']:.2f}",
            '下降率': f"{s['reduction_rate']:.1f}%"
        })
    
    df = pd.DataFrame(stats_data)
    print(df.to_string(index=False))
    
    # 3. 困惑度与Loss的对应关系
    print("\n【困惑度与Loss的对应关系】")
    print("-"*80)
    print("公式: Perplexity = exp(Loss)")
    print("\n示例对应关系:")
    for loss_val in [1.0, 2.0, 3.0, 4.0, 5.0]:
        ppl = np.exp(loss_val)
        print(f"  Loss = {loss_val:.1f} → Perplexity = {ppl:.2f}")
    
    print("\n实际数据对应:")
    for label in logs.keys():
        s = data[label]['stats']
        avg_loss = np.mean(data[label]['losses'])
        avg_ppl = s['mean']
        print(f"  {label:15} 平均Loss = {avg_loss:.4f} → 平均Perplexity = {avg_ppl:.2f}")
    
    # 4. 收敛速度对比
    print("\n【收敛速度对比】")
    print("-"*80)
    baseline_reduction = data['Baseline']['stats']['reduction']
    print(f"{'配置':<15} {'困惑度下降':>12} {'相对Baseline':>15}")
    print("-"*80)
    for label in logs.keys():
        reduction = data[label]['stats']['reduction']
        relative = ((reduction - baseline_reduction) / baseline_reduction * 100) if label != 'Baseline' else 0
        print(f"{label:<15} {reduction:>12.2f} {relative:>14.1f}%")
    
    # 5. 可视化
    save_dir = Path(r"C:\Graduation project\LowDiff\plots_all_comparison")
    save_dir.mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 5.1 平滑困惑度曲线对比
    ax1 = fig.add_subplot(gs[0, :])
    for label in logs.keys():
        ppl = data[label]['perplexities']
        smoothed = smooth_curve(ppl, window=100)
        steps = np.arange(len(smoothed)) + 99
        ax1.plot(steps, smoothed, linewidth=2.5, color=colors[label], 
                label=label, alpha=0.9)
    ax1.set_xlabel('训练步数 (Global Steps)', fontsize=13)
    ax1.set_ylabel('困惑度 (Perplexity)', fontsize=13)
    ax1.set_title('GPT-2训练困惑度对比 - 不同压缩率的影响', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # 5.2 按Epoch的平均困惑度
    ax2 = fig.add_subplot(gs[1, 0])
    for label in logs.keys():
        epoch_means = data[label]['stats']['epoch_means']
        ax2.plot(range(10), epoch_means, 'o-', linewidth=2.5, 
                color=colors[label], label=label, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('平均困惑度', fontsize=12)
    ax2.set_title('各Epoch平均困惑度', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 5.3 困惑度分布箱型图
    ax3 = fig.add_subplot(gs[1, 1])
    ppl_data = [data[label]['perplexities'] for label in logs.keys()]
    bp = ax3.boxplot(ppl_data, labels=logs.keys(), patch_artist=True, showfliers=False)
    for patch, label in zip(bp['boxes'], logs.keys()):
        patch.set_facecolor(colors[label])
        patch.set_alpha(0.7)
    ax3.set_ylabel('困惑度', fontsize=12)
    ax3.set_title('困惑度分布箱型图', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='x', rotation=15)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 5.4 困惑度 vs Loss 散点图
    ax4 = fig.add_subplot(gs[2, 0])
    for label in logs.keys():
        losses = np.array(data[label]['losses'])
        ppl = data[label]['perplexities']
        # 采样以减少点数
        sample_idx = np.random.choice(len(losses), min(1000, len(losses)), replace=False)
        ax4.scatter(losses[sample_idx], ppl[sample_idx], alpha=0.3, 
                   s=10, color=colors[label], label=label)
    # 理论曲线
    loss_range = np.linspace(0, 6, 100)
    ax4.plot(loss_range, np.exp(loss_range), 'k--', linewidth=2, 
            label='理论: PPL=exp(Loss)', alpha=0.7)
    ax4.set_xlabel('Loss', fontsize=12)
    ax4.set_ylabel('困惑度 (Perplexity)', fontsize=12)
    ax4.set_title('困惑度与Loss的关系', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5.5 归一化收敛曲线
    ax5 = fig.add_subplot(gs[2, 1])
    for label in logs.keys():
        epoch_means = data[label]['stats']['epoch_means']
        initial = epoch_means[0]
        normalized = [(initial - ppl) / initial * 100 for ppl in epoch_means]
        ax5.plot(range(10), normalized, 'o-', linewidth=2.5, 
                color=colors[label], label=label, markersize=8)
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('困惑度相对改善 (%)', fontsize=12)
    ax5.set_title('归一化困惑度收敛效率', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    plt.savefig(save_dir / 'perplexity_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 困惑度分析图表已保存: {save_dir / 'perplexity_analysis.png'}")
    
    # 6. 保存统计表格图
    fig2, ax = plt.subplots(figsize=(14, 4))
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
    
    plt.title('困惑度统计对比表', fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_dir / 'perplexity_statistics_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ 困惑度统计表格已保存: {save_dir / 'perplexity_statistics_table.png'}")
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()

