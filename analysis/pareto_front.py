import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import imageio.v2 as imageio
import random

file_dir = "analysis/outputs/frames"
os.makedirs(file_dir, exist_ok=True)
output_dir = "analysis/outputs/animations"
os.makedirs(output_dir, exist_ok=True)


def random_blue_shades(n):
    return [(random.uniform(0.3, 0.6), random.uniform(0.5, 0.8), 1.0) for _ in range(n)]

all_acc = []
all_params = []
all_colors = []

n_generations = 58

for gen in range(n_generations):
    acc, params = [], []
    
    colors = random_blue_shades(len(acc))

    all_acc.extend(acc)
    all_params.extend(params)
    all_colors.extend(colors)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.set_xlim(0.89, 0.945)
    ax.set_ylim(1e5, 1e8)

    ax.scatter(all_acc, all_params, alpha=0.5, color=all_colors)

    ax.plot(acc, params, color='gray', linewidth=1)
    ax.plot(0.937, 2450000, color='darkred', marker='*')

    best_idx = np.argmax(acc)
    ax.scatter(acc[best_idx], params[best_idx], color='darkgreen', s=100, marker='*')
    ax.text(acc[best_idx], params[best_idx], "Best", fontsize=9, color='darkred', ha='right')

    legend_elements = [
        Line2D([0], [0], marker='*', color='w', label='Point Transformers',
               markerfacecolor='darkred', markersize=12),
        Line2D([0], [0], color='gray', lw=2, label='Pareto Front'),
        Line2D([0], [0], marker='o', color='w', label='Variant',
               markerfacecolor='skyblue', markersize=8),
        Line2D([0], [0], marker="*", color='w', label='Best Variant',
               markerfacecolor='darkgreen', markersize=12)
    ]
    
    ax.legend(handles=legend_elements)
    ax.margins(2) 
    ax.set_yscale('log')
    ax.set_xlabel("Test Accuracy", fontsize=14)
    ax.set_ylabel("Parameters", fontsize=15)
    ax.set_title(f"Pareto Diagram Generation: {gen}")
    ax.set_frame_on(False)
    filename = f"{file_dir}/frame_{gen:02d}.png"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

frames = [imageio.imread(f"{file_dir}/frame_{i:02d}.png") for i in range(n_generations)]

output_path = os.path.join(output_dir, "pareto_evolution.gif")
imageio.mimsave(output_path, frames, duration=0.8)
