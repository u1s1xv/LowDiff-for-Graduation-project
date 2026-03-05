#!/usr/bin/env python3
"""
绘制GPT-2在使用LowDiff策略时的训练loss曲线
使用方法: python plot_lowdiff_loss.py
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
    
    # 正则表达式匹配: [Epoch X/10] Batch Y, Loss: Z
    pattern = r'\[Epoch (\d+)/\d+\] Batch (\d+), Loss: ([\d.]+|nan)'
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                batch = int(match.group(2))
                loss_str = match.group(3)
                
                # 跳过nan值
                if loss_str != 'nan':
                    loss = float(loss_str)
                    epochs.append(epoch)
                    batches.append(batch)
                    losses.append(loss)
    
    return epochs, batches, losses

def plot_loss_curves(epochs, batches, losses, save_dir='.', title_suffix=''):
    """绘制训练loss曲线"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 创建全局步数
    max_epoch = max(epochs)
    batches_per_epoch = max([b for e, b in zip(epochs, batches) if e == 0]) + 1
    global_steps = [e * batches_per_epoch + b for e, b in zip(epochs, batches)]
    
    # 1. 绘制完整的训练曲线
    plt.figure(figsize=(12, 6))
    plt.plot(global_steps, losses, alpha=0.6, linewidth=0.5, color='#2E86AB')
    plt.xlabel('训练步数 (Global Steps)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'GPT-2 训练Loss曲线{title_suffix} (完整)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'lowdiff_loss_curve_full.png', dpi=300, bbox_inches='tight')
    print(f"✓ 保存完整loss曲线: {save_dir / 'lowdiff_loss_curve_full.png'}")
    plt.close()
    
    # 2. 绘制平滑后的曲线
    window_size = 100
    if len(losses) > window_size:
        smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        smoothed_steps = global_steps[window_size-1:]
        
        plt.figure(figsize=(12, 6))
        plt.plot(global_steps, losses, alpha=0.3, linewidth=0.5, label='原始Loss', color='#2E86AB')
        plt.plot(smoothed_steps, smoothed_losses, linewidth=2, color='#A23B72', label=f'移动平均 (窗口={window_size})')
        plt.xlabel('训练步数 (Global Steps)', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'GPT-2 训练Loss曲线{title_suffix} (平滑)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'lowdiff_loss_curve_smoothed.png', dpi=300, bbox_inches='tight')
        print(f"✓ 保存平滑loss曲线: {save_dir / 'lowdiff_loss_curve_smoothed.png'}")
        plt.close()
    
    # 3. 按Epoch分组绘制
    unique_epochs = sorted(set(epochs))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_epochs)))
    
    plt.figure(figsize=(14, 8))
    for idx, epoch in enumerate(unique_epochs):
        epoch_indices = [i for i, e in enumerate(epochs) if e == epoch]
        epoch_batches = [batches[i] for i in epoch_indices]
        epoch_losses = [losses[i] for i in epoch_indices]
        plt.plot(epoch_batches, epoch_losses, label=f'Epoch {epoch}', alpha=0.7, color=colors[idx])
    
    plt.xlabel('Batch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'GPT-2 训练Loss曲线{title_suffix} (按Epoch分组)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'lowdiff_loss_curve_by_epoch.png', dpi=300, bbox_inches='tight')
    print(f"✓ 保存分epoch的loss曲线: {save_dir / 'lowdiff_loss_curve_by_epoch.png'}")
    plt.close()
    
    # 4. 打印统计信息
    print("\n=== LowDiff训练统计 ===")
    print(f"总训练步数: {len(losses)}")
    print(f"Epoch数: {max(epochs) + 1}")
    print(f"初始Loss: {losses[0]:.4f}")
    print(f"最终Loss: {losses[-1]:.4f}")
    print(f"最低Loss: {min(losses):.4f}")
    print(f"平均Loss: {np.mean(losses):.4f}")
    print(f"Loss标准差: {np.std(losses):.4f}")
    
    return {
        'total_steps': len(losses),
        'epochs': max(epochs) + 1,
        'initial_loss': losses[0],
        'final_loss': losses[-1],
        'min_loss': min(losses),
        'mean_loss': np.mean(losses),
        'std_loss': np.std(losses)
    }

def main():
    # LowDiff日志文件路径
    log_file = Path(r"C:\Graduation project\LowDiff\gpt_lowdiff_20260213_022859.log")
    
    if not log_file.exists():
        print(f"❌ 错误: 找不到日志文件 {log_file}")
        return
    
    print(f"📖 正在读取LowDiff日志文件: {log_file}")
    epochs, batches, losses = parse_log_file(log_file)
    
    if not losses:
        print("❌ 错误: 未找到任何有效的loss数据")
        return
    
    print(f"✓ 成功解析 {len(losses)} 个loss值")
    
    # 保存到日志文件同目录
    save_dir = log_file.parent / "plots_lowdiff"
    print(f"\n📊 正在绘制LowDiff loss曲线...")
    stats = plot_loss_curves(epochs, batches, losses, save_dir, title_suffix=' - LowDiff策略')
    
    print(f"\n✅ 完成！所有图表已保存到: {save_dir}")

if __name__ == "__main__":
    main()

