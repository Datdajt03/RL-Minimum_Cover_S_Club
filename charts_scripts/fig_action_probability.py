import os
import json
import matplotlib.pyplot as plt
import numpy as np

def draw_action_probabilities(results_dir):
    """
    Quét các thư mục con kết quả, đọc kết quả results.json và vẽ biểu đồ xác suất chọn hành động di truyền.
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

            # Chúng ta chỉ vẽ cho chiến lược AdaptiveEvolutionary (EVO-RL)
            evo_data = metrics.get('AdaptiveEvolutionary', {})
            summary = evo_data.get('summary', {})
            prob_hist = summary.get('history_action_prob_mean', [])

            if not prob_hist:
                continue

            prob_hist = np.array(prob_hist)  # shape (generations, 3)
            generations = np.arange(len(prob_hist))

            plt.figure(figsize=(8, 5))
            
            # Vẽ 3 đường thẳng ứng với 3 toán tử
            plt.plot(generations, prob_hist[:, 0], label="Crossover ($a^{cross}$)", color="#1f77b4", linewidth=2.0)
            plt.plot(generations, prob_hist[:, 1], label="Repair Mut ($a^{repair}$)", color="#2ca02c", linewidth=2.0)
            plt.plot(generations, prob_hist[:, 2], label="Diversify Mut ($a^{div}$)", color="#d62728", linewidth=2.0)

            plt.xlabel("Thế hệ (Generations)")
            plt.ylabel("Xác suất lựa chọn toán tử")
            plt.ylim(0, 1.0)
            plt.title(f"Xác suất thích nghi toán tử di truyền EVO-RL — Đồ thị {graph_name}")
            plt.legend(loc='center right', frameon=True, facecolor='white', framealpha=0.9)
            plt.tight_layout()

            # Tạo thư mục đầu ra
            out_dir = os.path.join(results_dir, graph_name, "charts_output")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "action_selection_probability.png")
            
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"  [Action Probabilities] Saved chart to {out_path}")

        except Exception as e:
            print(f"  Error plotting action probabilities for {graph_name}: {e}")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.join(current_dir, "..", "results"))
    draw_action_probabilities(results_dir)

if __name__ == "__main__":
    main()
