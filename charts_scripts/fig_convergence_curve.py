import os
import json
import matplotlib.pyplot as plt
import numpy as np

def draw_convergence_curve(results_dir):
    """
    Quét các thư mục con kết quả, đọc kết quả results.json và vẽ biểu đồ đường cong hội tụ.
    Lưu biểu đồ vào thư mục con charts_output của từng đồ thị.
    """
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist.")
        return

    # Phong cách đồ thị khoa học
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 14
    })

    graph_names = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

    short_names = {
        "AdaptiveEvolutionary": "EVO-RL (Đề xuất)",
        "NoReinforcementLearning": "No-RL",
        "FixedOperatorStrategy": "Fixed",
        "RandomOperatorSelection": "Random"
    }

    colors = {
        "AdaptiveEvolutionary": "#1f77b4",  # Xanh lam đậm
        "NoReinforcementLearning": "#d62728",  # Đỏ
        "FixedOperatorStrategy": "#ff7f0e",  # Cam
        "RandomOperatorSelection": "#2ca02c"   # Xanh lá
    }

    markers = {
        "AdaptiveEvolutionary": "o",
        "NoReinforcementLearning": "s",
        "FixedOperatorStrategy": "^",
        "RandomOperatorSelection": "d"
    }

    for graph_name in graph_names:
        json_path = os.path.join(results_dir, graph_name, "results.json")
        if not os.path.exists(json_path):
            continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metrics = data.get('metrics', {}).get(graph_name, {})
            if not metrics:
                continue

            plt.figure(figsize=(8, 5))
            has_data = False

            for cond_name, cond_data in metrics.items():
                summary = cond_data.get('summary', {})
                hist = summary.get('history_cover_mean', [])
                
                if hist:
                    has_data = True
                    generations = np.arange(len(hist))
                    label = short_names.get(cond_name, cond_name)
                    color = colors.get(cond_name, None)
                    marker = markers.get(cond_name, None)
                    
                    # Vẽ đường cong, lấy markevery=20 để tránh dày đặc marker
                    plt.plot(generations, hist, label=label, color=color, 
                             marker=marker, markevery=max(1, len(hist)//10), 
                             linewidth=2.0, markersize=6)

            if not has_data:
                plt.close()
                continue

            plt.xlabel("Thế hệ (Generations)")
            plt.ylabel("Kích thước lớp phủ trung bình (Cover Size)")
            plt.title(f"Đường cong hội tụ kích thước phủ trung bình — Đồ thị {graph_name}")
            plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
            plt.tight_layout()

            # Tạo thư mục đầu ra
            out_dir = os.path.join(results_dir, graph_name, "charts_output")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "convergence_curve.png")
            
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"  [Convergence Curve] Saved chart to {out_path}")

        except Exception as e:
            print(f"  Error plotting convergence for {graph_name}: {e}")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.join(current_dir, "..", "results"))
    draw_convergence_curve(results_dir)

if __name__ == "__main__":
    main()
