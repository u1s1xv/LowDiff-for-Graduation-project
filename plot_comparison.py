#!/usr/bin/env python3
"""
对比绘制baseline和LowDiff策略的训练loss曲线
使用方法: python plot_comparison.py
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_log_file(log_file):
    """解析训练日志文件，提取epoch、batch和loss信息"""
    epochs = []
    batches = []
    losses = []
    
    pattern = r'\[Epoch (\d+)/\d+\] Batch (\d+), Loss: ([\d.]+|nan)'
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                batch = int(match.group(2))
                loss_str = match.group(3)
                
                if loss_str != 'nan':
                    loss = float(loss_str)
                    epochs.append(epoch)
                    batches.append(batch)
                    losses.append(loss)
    
    return epochs, batches, losses

def plot_comparison(baseline_data, lowdiff_data, save_dir):
    """绘制baseline和LowDiff的对比曲线"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    b_epochs, b_batches, b_losses = baseline_data
    l_epochs, l_batches, l_losses = lowdiff_data
    
    # 创建全局步数
    b_batches_per_epoch = max([b for e, b in zip(b_epochs, b_batches) if e == 0]) + 1
    b_global_steps = [e * b_batches_per_epoch + b for e, b in zip(b_epochs, b_batches)]
    
    l_batches_per_epoch = max([b for e, b in zip(l_epochs, l_batches) if e == 0]) + 1
    l_global_steps = [e * l_batches_per_epoch + b for e, b in zip(l_epochs, l_batches)]
    
    # 1. 对比平滑曲线
    window_size = 100
    if len(b_losses) > window_size and len(l_losses) > window_size:
        b_smoothed = np.convolve(b_losses, np.ones(window_size)/window_size, mode='valid')
        b_smoothed_steps = b_global_steps[window_size-1:]
        
        l_smoothed = np.convolve(l_losses, np.ones(window_size)/window_size, mode='valid')
        l_smoothed_steps = l_global_steps[window_size-1:]
        
        plt.figure(figsize=(14, 7))
        plt.plot(b_smoothed_steps, b_smoothed, linewidth=2.5, color='#2E86AB', 
                label='Baseline (无压缩)', alpha=0.9)
        plt.plot(l_smoothed_steps, l_smoothed, linewidth=2.5, color='#A23B72', 
                label='LowDiff (Top-K 1%)', alpha=0.9)
        
        plt.xlabel('训练步数 (Global Steps)', fontsize=13)
        plt.ylabel('Loss', fontsize=13)
        plt.title('GPT-2 训练Loss对比 - Baseline vs LowDiff', fontsize=15, fontweight='bold')
        plt.legend(fontsize=12, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'comparison_smoothed.png', dpi=300, bbox_inches='tight')
        print(f"✓ 保存平滑对比曲线: {save_dir / 'comparison_smoothed.png'}")
        plt.close()
    
    # 2. 对比原始曲线（半透明）
    plt.figure(figsize=(14, 7))
    plt.plot(b_global_steps, b_losses, alpha=0.4, linewidth=0.5, color='#2E86AB', label='Baseline')
    plt.plot(l_global_steps, l_losses, alpha=0.4, linewidth=0.5, color='#A23B72', label='LowDiff')
    
    plt.xlabel('训练步数 (Global Steps)', fontsize=13)
    plt.ylabel('Loss', fontsize=13)
    plt.title('GPT-2 训练Loss对比 (原始数据)', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'comparison_raw.png', dpi=300, bbox_inches='tight')
    print(f"✓ 保存原始对比曲线: {save_dir / 'comparison_raw.png'}")
    plt.close()
    
    # 3. 按Epoch对比（箱型图）
    b_epochs_unique = sorted(set(b_epochs))
    l_epochs_unique = sorted(set(l_epochs))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Baseline箱型图
    b_epoch_losses = []
    for epoch in b_epochs_unique:
        epoch_losses = [b_losses[i] for i, e in enumerate(b_epochs) if e == epoch]
        b_epoch_losses.append(epoch_losses)
    
    bp1 = axes[0].boxplot(b_epoch_losses, labels=[f'E{i}' for i in b_epochs_unique], 
                          patch_artist=True, showfliers=False)
    for patch in bp1['boxes']:
        patch.set_facecolor('#2E86AB')
        patch.set_alpha(0.7)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Baseline - 每个Epoch的Loss分布', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # LowDiff箱型图
    l_epoch_losses = []
    for epoch in l_epochs_unique:
        epoch_losses = [l_losses[i] for i, e in enumerate(l_epochs) if e == epoch]
        l_epoch_losses.append(epoch_losses)
    
    bp2 = axes[1].boxplot(l_epoch_losses, labels=[f'E{i}' for i in l_epochs_unique], 
                          patch_artist=True, showfliers=False)
    for patch in bp2['boxes']:
        patch.set_facecolor('#A23B72')
        patch.set_alpha(0.7)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('LowDiff - 每个Epoch的Loss分布', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'comparison_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"✓ 保存箱型图对比: {save_dir / 'comparison_boxplot.png'}")
    plt.close()
    
    # 4. 打印对比统计
    print("\n" + "="*60)
    print("训练对比统计 - Baseline vs LowDiff")
    print("="*60)
    print(f"{'指标':<20} {'Baseline':>15} {'LowDiff':>15} {'差值':>10}")
    print("-"*60)
    print(f"{'总训练步数':<20} {len(b_losses):>15} {len(l_losses):>15} {len(l_losses)-len(b_losses):>10}")
    print(f"{'初始Loss':<20} {b_losses[0]:>15.4f} {l_losses[0]:>15.4f} {l_losses[0]-b_losses[0]:>10.4f}")
    print(f"{'最终Loss':<20} {b_losses[-1]:>15.4f} {l_losses[-1]:>15.4f} {l_losses[-1]-b_losses[-1]:>10.4f}")
    print(f"{'最低Loss':<20} {min(b_losses):>15.4f} {min(l_losses):>15.4f} {min(l_losses)-min(b_losses):>10.4f}")
    print(f"{'平均Loss':<20} {np.mean(b_losses):>15.4f} {np.mean(l_losses):>15.4f} {np.mean(l_losses)-np.mean(b_losses):>10.4f}")
    print(f"{'Loss标准差':<20} {np.std(b_losses):>15.4f} {np.std(l_losses):>15.4f} {np.std(l_losses)-np.std(b_losses):>10.4f}")
    print("="*60)
    
    # 计算性能退化
    degradation = ((np.mean(l_losses) - np.mean(b_losses)) / np.mean(b_losses)) * 100
    print(f"\n📊 LowDiff性能退化: {degradation:+.2f}%")
    if degradation > 0:
        print(f"   （LowDiff平均loss比baseline高{abs(degradation):.2f}%）")
    else:
        print(f"   （LowDiff平均loss比baseline低{abs(degradation):.2f}%）")

def main():
    # 文件路径
    baseline_log = Path(r"C:\Graduation project\LowDiff\baseline_lowdiff_20260227_131642.log")
    lowdiff_log = Path(r"C:\Graduation project\LowDiff\gpt_lowdiff_20260227_110015.log")
    
    print("📖 正在读取日志文件...")
    
    if not baseline_log.exists():
        print(f"❌ 错误: 找不到baseline日志文件 {baseline_log}")
        return
    
    if not lowdiff_log.exists():
        print(f"❌ 错误: 找不到LowDiff日志文件 {lowdiff_log}")
        return
    
    print(f"  - Baseline: {baseline_log.name}")
    baseline_data = parse_log_file(baseline_log)
    print(f"    ✓ 解析了 {len(baseline_data[2])} 个loss值")
    
    print(f"  - LowDiff: {lowdiff_log.name}")
    lowdiff_data = parse_log_file(lowdiff_log)
    print(f"    ✓ 解析了 {len(lowdiff_data[2])} 个loss值")
    
    # 保存对比图
    save_dir = baseline_log.parent / "plots_comparison"
    print(f"\n📊 正在绘制对比图...")
    plot_comparison(baseline_data, lowdiff_data, save_dir)
    
    print(f"\n✅ 完成！所有对比图表已保存到: {save_dir}")

if __name__ == "__main__":
    main()

