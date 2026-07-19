"""
fig_std_deviation.py
Tự động quét các thư mục con trong results/, đọc kết quả results.json của từng đồ thị
và vẽ biểu đồ cột biểu diễn độ lệch chuẩn (std) của kích thước phủ (cover_size).
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
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
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
            
        print(f"Drawing standard deviation chart for: {graph_name}...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        metrics = data.get("metrics", {}).get(graph_name, {})
        if not metrics:
            continue
            
        conditions = list(metrics.keys())
        std_devs = []
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
            std_devs.append(float(np.std(values)) if len(values) > 1 else 0.0)
            labels.append(short_names.get(cond, cond))
            
        fig, ax = plt.subplots(figsize=(5.0, 3.2), constrained_layout=True)
        bars = ax.bar(labels, std_devs, color=COLORS[:len(conditions)],
                      edgecolor='black', linewidth=0.8)
                      
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Std Dev of s-Club Cover Size")
        ax.set_title(f"Cover Size Standard Deviation on: {graph_name}")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Annotate giá trị lên đầu cột
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + (max(std_devs)*0.01 if max(std_devs) > 0 else 0.01),
                    f'{yval:.3f}', va='bottom', ha='center', fontsize=7, fontweight='bold')
                    
        output_dir = os.path.join(results_dir, graph_name, "charts_output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "fig_std_deviation.png")
        
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()