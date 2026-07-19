"""
fig_metric_distribution.py
Tự động quét các thư mục con trong results/, đọc kết quả results.json của từng đồ thị
và vẽ biểu đồ Violin + Boxplot biểu diễn phân phối kích thước phủ (cover_size) qua các seed.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Academic styling
try:
    plt.style.use(['science', 'ieee'])
except Exception:
    try:
        plt.style.use(['seaborn-v0_8-whitegrid'])
    except Exception:
        pass

COLORS = ['#4477AA', '#EE6677', '#228833', '#CCBB44']

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    
    if not os.path.exists(results_dir):
        print(f"Directory not found: {results_dir}")
        return

    # Quét tất cả thư mục con (tương ứng với các đồ thị đã chạy)
    graph_names = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    for graph_name in graph_names:
        json_path = os.path.join(results_dir, graph_name, "results.json")
        if not os.path.exists(json_path):
            continue
            
        print(f"Drawing distribution chart for: {graph_name}...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        metrics = data.get("metrics", {}).get(graph_name, {})
        if not metrics:
            continue
            
        conditions = list(metrics.keys())
        data_values = []
        labels = []
        
        # Ánh xạ tên hiển thị ngắn gọn
        short_names = {
            "AdaptiveEvolutionary": "EVO-RL\n(PPO)",
            "NoReinforcementLearning": "No-RL",
            "FixedOperatorStrategy": "Fixed",
            "RandomOperatorSelection": "Random"
        }
        
        for cond in conditions:
            seed_data = metrics[cond]
            values = [v["cover_size"] for k, v in seed_data.items() if k.isdigit()]
            data_values.append(values if values else [0.0])
            labels.append(short_names.get(cond, cond))
            
        fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
        
        # Vẽ violin plot
        parts = ax.violinplot(data_values, showmeans=False, showmedians=True)
        
        # Vẽ lồng box plot nhỏ ở giữa để hiển thị trung vị và phân vị
        ax.boxplot(data_values, widths=0.08, positions=np.arange(1, len(conditions) + 1), 
                   medianprops=dict(color="red", linestyle="-", lw=1.5),
                   boxprops=dict(color="black", lw=0.8),
                   whiskerprops=dict(color="black", lw=0.8),
                   capprops=dict(color="black", lw=0.8))
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(COLORS[i % len(COLORS)])
            pc.set_edgecolor('black')
            pc.set_linewidth(0.8)
            pc.set_alpha(0.6)
            
        # Customize lines của violin
        parts['cmedians'].set_edgecolor('red')
        parts['cmedians'].set_linewidth(1.5)
        parts['cmins'].set_edgecolor('black')
        parts['cmins'].set_linewidth(0.8)
        parts['cmaxes'].set_edgecolor('black')
        parts['cmaxes'].set_linewidth(0.8)
        
        ax.set_title(f"Distribution of s-Club Cover Sizes on: {graph_name}")
        ax.set_xlabel("Strategy")
        ax.set_ylabel("s-Club Cover Size")
        ax.set_xticks(np.arange(1, len(conditions) + 1))
        ax.set_xticklabels(labels)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        output_dir = os.path.join(results_dir, graph_name, "charts_output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "fig_metric_distribution.png")
        
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()