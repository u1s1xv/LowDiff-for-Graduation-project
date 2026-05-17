#!/usr/bin/env python3
"""
Plot maximum checkpoint frequency for six strategies.

Usage examples:
  # replace values with your six y-values (order must match labels)
  python scripts/plot_max_ckpt_freq.py --values "1,0.8,0.5,0.3,0.6,0.4" --outfile resnet_max_ckpt_freq.png

  # or run without --values to use placeholder zeros and then edit the file
  python scripts/plot_max_ckpt_freq.py
"""
import argparse
import sys
import matplotlib.pyplot as plt


def parse_values(s: str):
    parts = [p.strip() for p in s.split(',') if p.strip()!='']
    if len(parts) != 6:
        raise ValueError('Expect exactly 6 comma-separated numeric values')
    try:
        return [float(x) for x in parts]
    except Exception as e:
        raise ValueError('All values must be numbers') from e


def main():
    labels = [
        'CheckFreq',
        'Gemini',
        'Naïve DC',
        'LowDiff',
        'LowDiff+(S)',
        'LowDiff+(P)'
    ]

    parser = argparse.ArgumentParser(description='Plot Resnet model maximum checkpoint frequency')
    parser.add_argument('--values', help='Comma-separated 6 y values (in the same order as labels)')
    parser.add_argument('--outfile', help='Output image path', default='resnet_max_ckpt_freq.png')
    parser.add_argument('--dpi', type=int, default=200, help='Output image DPI')
    args = parser.parse_args()

    if args.values:
        try:
            values = parse_values(args.values)
        except ValueError as e:
            print('Error parsing --values:', e)
            sys.exit(1)
    else:
        # Placeholder values: replace these with your own measurements
        values = [500,400, 1000, 500, 500, 500]

    fig, ax = plt.subplots(figsize=(10, 6))
    # use distinct colors for each bar so they are visually different
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
    bars = ax.bar(labels, values, color=colors)

    ax.set_ylim(0, 1000)
    ax.set_title('Resnet model maximum checkpoint frequency')
    ax.set_xlabel('Checkpoint strategy')
    ax.set_ylabel('Maximum checkpoint frequency')

    # add value labels above bars
    for bar, v in zip(bars, values):
        try:
            label = '' if v is None else f'{v:g}'
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, label, ha='center', va='bottom')
        except Exception:
            pass

    plt.tight_layout()
    plt.savefig(args.outfile, dpi=args.dpi)
    print(f'Saved plot to {args.outfile}')


if __name__ == '__main__':
    main()
