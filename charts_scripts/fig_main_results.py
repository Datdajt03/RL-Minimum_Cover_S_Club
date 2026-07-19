"""
fig_main_results.py
Tự động quét các thư mục con trong results/, đọc kết quả results.json của từng đồ thị
và vẽ biểu đồ cột so sánh kích thước phủ (cover_size) trung bình kèm độ lệch chuẩn (std).
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
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
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
            
        print(f"Drawing main results chart for: {graph_name}...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        metrics = data.get("metrics", {}).get(graph_name, {})
        if not metrics:
            continue
            
        conditions = list(metrics.keys())
        means = []
        stds = []
        labels = []
        
        # Ánh xạ tên hiển thị ngắn gọn
        short_names = {
            "AdaptiveEvolutionary": "EVO-RL (PPO)",
            "NoReinforcementLearning": "No-RL",
            "FixedOperatorStrategy": "Fixed",
            "RandomOperatorSelection": "Random"
        }
        
        for cond in conditions:
            seed_data = metrics[cond]
            # Lấy cover_size của các seed
            values = [v["cover_size"] for k, v in seed_data.items() if k.isdigit()]
            means.append(float(np.mean(values)) if values else 0.0)
            stds.append(float(np.std(values)) if len(values) > 1 else 0.0)
            labels.append(short_names.get(cond, cond))
            
        # Vẽ biểu đồ
        fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
        x = np.arange(len(conditions))
        bars = ax.bar(x, means, yerr=stds, color=COLORS[:len(conditions)],
                      capsize=5, edgecolor='black', linewidth=0.8, error_kw=dict(lw=1.0, capthick=1.0))
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Minimum s-Club Cover Size (mean ± std)")
        ax.set_title(f"Performance Comparison on Graph: {graph_name} ($s=2$)")
        
        # Annotate giá trị lên đầu cột
        for bar, m in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + (max(means)*0.02),
                    f'{m:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                    
        # Thiết lập giới hạn trục Y một cách hợp lý
        if means:
            ax.set_ylim(0, max(means) * 1.25)
            
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Lưu vào thư mục charts_output của đồ thị đó
        output_dir = os.path.join(results_dir, graph_name, "charts_output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "fig_main_results.png")
        
        fig.savefig(output_path, format='png')
        plt.close(fig)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()