import matplotlib.pyplot as plt
import numpy as np
import os

# Set matplotlib backend to avoid display issues
plt.switch_backend('Agg')

# Checkpoint strategy labels
strategies = ['W/O CKPT', 'CheckFreq', 'Gemini', 'Naïve DC', 'LowDiff']

# Training time data (unit: minutes)
# Please fill in the corresponding minute values below
training_times_min = [
    20,  # W/O CKPT (fill in minutes)
    25,  # CheckFreq (fill in minutes)
    23,  # Gemini (fill in minutes)
    246,  # Naïve DC (fill in minutes)
    46,  # LowDiff (fill in minutes)
]

# Unit conversion: minutes to hours
training_times_hours = [minutes / 60 for minutes in training_times_min]

# Define different colors for each bar
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Plot bar chart
x_pos = np.arange(len(strategies))
bars = ax.bar(x_pos, training_times_hours, color=colors, edgecolor='black',
              linewidth=0.5, alpha=0.9, width=0.5)

# Set x-axis labels and ticks
ax.set_xticks(x_pos)
ax.set_xticklabels(strategies, fontsize=11)
ax.set_xlim(-0.6, len(strategies) - 0.4)

# Add time labels on top of each bar
for bar, hours in zip(bars, training_times_hours):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.03,
            f'{hours:.2f}h',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Set labels and title (all English)
ax.set_xlabel('Checkpoint Strategy', fontsize=12, fontweight='bold')
ax.set_ylabel('Training Time (h)', fontsize=12, fontweight='bold')
ax.set_title('Training Time Comparison of Different Checkpoint Strategies',
             fontsize=14, fontweight='bold', pad=15)

# Set y-axis: 0 to 2 with ticks every 0.2 (FIXED)
ax.set_ylim(0, 2)
ax.set_yticks(np.arange(0, 2.1, 0.2))
ax.set_yticklabels([f'{i:.1f}' for i in np.arange(0, 2.1, 0.2)], fontsize=10)

# Increase spacing between y-axis ticks
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))

# Add grid
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Adjust layout
plt.tight_layout()

# Create output directory if it doesn't exist
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save as PNG
png_path = os.path.join(output_dir, "training_time_comparison.png")
plt.savefig(png_path, dpi=300, bbox_inches='tight')
print(f"Chart saved to: {png_path}")

# Save as PDF
pdf_path = os.path.join(output_dir, "training_time_comparison.pdf")
plt.savefig(pdf_path, bbox_inches='tight')
print(f"Chart saved to: {pdf_path}")

# Save as SVG
svg_path = os.path.join(output_dir, "training_time_comparison.svg")
plt.savefig(svg_path, bbox_inches='tight')
print(f"Chart saved to: {svg_path}")

# Close figure to free memory
plt.close(fig)

print("Chart generation completed!")

