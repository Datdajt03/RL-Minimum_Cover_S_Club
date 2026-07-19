"""
main.py — EVO-RL: Minimum s-Club Cover
Chạy thí nghiệm trên các tập đồ thị từ pbodulieu.

Khi khởi động sẽ hỏi:
  1. Chạy tất cả tập đỉnh
  2. Chọn từng tập đỉnh

Output:
  results.json          ← số liệu thô
  results_table.tex     ← bảng LaTeX dán vào bài báo
  results_summary.md    ← báo cáo Markdown
"""
import json
import os
import sys
import time
import numpy as np
import torch
import random

# Reconfigure stdout to use UTF-8 to prevent UnicodeEncodeError on Windows
if sys.stdout is not None:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from graph_loader import load_all_graphs, load_graph
from env import SClubEnvironment
from agent import (AdaptiveOperatorAgent, NoReinforcementLearningAgent,
                   FixedOperatorStrategyAgent, RandomOperatorSelectionAgent,
                   STATE_DIM, NUM_OPERATORS)
from train import train_agent, evaluate_agent
from experiment_config import ExperimentConfig

from experiment_config import ExperimentConfig

config = ExperimentConfig()

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'pbodulieu'))

# Tạo 20 hạt giống ngẫu nhiên thực sự khi khởi động
random.seed(int(time.time() * 1000) % 100000)
SEEDS    = [random.randint(1, 1000000) for _ in range(20)]
S_VALUE  = config.s_club_threshold

# Đồng bộ với experiment_config
HYPERPARAMETERS = {
    'learning_rate':    config.learning_rate,
    'batch_size':       config.batch_size,
    'max_epochs':       config.max_epochs,
    'hidden_dim':       config.hidden_dim,
    's_club_threshold': config.s_club_threshold,
    'pop_size':         config.pop_size,
    'num_seeds':        len(SEEDS),
    'discount':         config.discount,
    'clip_eps':         config.clip_eps,
    'entropy_coef':     config.entropy_coef,
    'grad_clip':        config.grad_clip,
    'alpha1':           10.0,
    'alpha2':           5.0,
    'alpha3':           5.0,
    'alpha4':           1.0,
    'penalty_lambda':   config.penalty_lambda,
    'max_generations':  config.max_generations,
    'stagnation_limit': config.stagnation_limit,
}

CONDITIONS = {
    'AdaptiveEvolutionary':      AdaptiveOperatorAgent,
    'NoReinforcementLearning':   NoReinforcementLearningAgent,
    'FixedOperatorStrategy':     FixedOperatorStrategyAgent,
    'RandomOperatorSelection':   RandomOperatorSelectionAgent,
}

SHORT_NAME = {
    'AdaptiveEvolutionary':    'EVO-RL (PPO)',
    'NoReinforcementLearning': 'No-RL',
    'FixedOperatorStrategy':   'Fixed',
    'RandomOperatorSelection': 'Random',
}

# ── Helpers ─────────────────────────────────────────────────────────────────

def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def ask_which_graphs(graphs: dict) -> dict:
    """Ask user which graph datasets to run."""
    names = list(graphs.keys())
    print("\n" + "="*55)
    print("  SELECT GRAPH DATASETS TO RUN")
    print("="*55)
    print("  [0] Run ALL datasets")
    for i, name in enumerate(names, 1):
        G = graphs[name]
        print(f"  [{i}] {name:30s} ({G.vcount():4d} nodes, {G.ecount():5d} edges)")
    print("  [q] Quit")
    print("="*55)

    while True:
        raw = input("\nEnter choice (e.g. 0 or 1,3,5): ").strip().lower()
        if raw == 'q':
            print("Quit.")
            sys.exit(0)
        if raw == '0':
            return graphs

        try:
            indices = [int(x.strip()) for x in raw.replace(',', ' ').split()]
            selected = {}
            for idx in indices:
                if 1 <= idx <= len(names):
                    selected[names[idx - 1]] = graphs[names[idx - 1]]
                else:
                    print(f"  Warning: invalid index {idx}, skipping.")
            if selected:
                print(f"\nSelected: {list(selected.keys())}")
                confirm = input("Confirm? (y/n): ").strip().lower()
                if confirm in ('y', 'yes', ''):
                    return selected
            else:
                print("No valid selection, please try again.")
        except ValueError:
            print("Invalid format. Enter 0 or numbers separated by commas.")


# ── Chạy thí nghiệm 1 đồ thị ────────────────────────────────────────────────

def run_graph(graph, graph_name, s):
    """Chạy tất cả conditions × seeds trên 1 đồ thị."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    config  = ExperimentConfig()

    print(f"\n--- {graph_name} ({graph.vcount()} nodes, "
          f"{graph.ecount()} edges, s={s}) ---")

    for cond_name, AgentClass in CONDITIONS.items():
        results[cond_name] = {}
        seed_covers = []

        for seed in SEEDS:
            set_all_seeds(seed)
            env = SClubEnvironment(graph=graph, s=s,
                                   pop_size=config.pop_size)
            agent = AgentClass(STATE_DIM, NUM_OPERATORS)
            # Optimizer chỉ dùng cho PPO agent thật
            optimizer = torch.optim.Adam(
                agent.policy_network.parameters(),
                lr=config.learning_rate)

            t0 = time.time()
            train_metrics = train_agent(agent, env, optimizer, device)
            eval_m = evaluate_agent(agent, env, device)
            elapsed = time.time() - t0

            cover = eval_m['cover_size']
            seed_covers.append(cover)
            results[cond_name][str(seed)] = {
                'cover_size':       cover,
                'cover_size_min':   eval_m['cover_size_min'],
                'solution_quality': eval_m['solution_quality'],
                'computation_time': round(elapsed, 2),
                'fully_covered':    eval_m['fully_covered'],
                'history_cover_sizes': train_metrics['cover_sizes'],
                'history_action_probs': train_metrics['action_probs'],
            }
            print(f"    {SHORT_NAME[cond_name]:15s} seed={seed}  "
                  f"cover={cover:.1f}  "
                  f"Q={eval_m['solution_quality']:.3f}  "
                  f"t={elapsed:.1f}s")

        # Tính trung bình lịch sử qua các seed
        history_covers_all = []
        history_probs_all = []
        for seed in SEEDS:
            if str(seed) in results[cond_name]:
                history_covers_all.append(results[cond_name][str(seed)]['history_cover_sizes'])
                history_probs_all.append(results[cond_name][str(seed)]['history_action_probs'])
        
        mean_history_covers = np.mean(history_covers_all, axis=0).tolist() if history_covers_all else []
        mean_history_probs = np.mean(history_probs_all, axis=0).tolist() if history_probs_all else []

        results[cond_name]['summary'] = {
            'cover_mean': round(float(np.mean(seed_covers)), 3),
            'cover_std':  round(float(np.std(seed_covers)),  3),
            'cover_min':  round(float(np.min(seed_covers)),  3),
            'history_cover_mean': mean_history_covers,
            'history_action_prob_mean': mean_history_probs,
        }
        print(f"    -> {SHORT_NAME[cond_name]:15s} mean={np.mean(seed_covers):.2f} "
              f"+-{np.std(seed_covers):.2f}")

    return results


# ── Sinh LaTeX ────────────────────────────────────────────────────────────────

# ── Sinh LaTeX ────────────────────────────────────────────────────────────────

def generate_combined_tex(all_results, graphs, filename='results_table.tex'):
    cnames = list(CONDITIONS.keys())
    lines  = []
    
    # Xác định số lượng lần chạy (seeds) thực tế đã thực hiện
    num_runs = 20  # Fallback mặc định
    if all_results:
        first_graph = list(all_results.keys())[0]
        if all_results[first_graph]:
            first_cond = list(all_results[first_graph].keys())[0]
            seeds_run = [s for s in all_results[first_graph][first_cond].keys() if s not in ('summary', '0.0')]
            if seeds_run:
                num_runs = len(seeds_run)
    
    # ── Bảng 1: Số club tối thiểu (mean \pm std) ──
    lines.append(r'% =============================================================================')
    lines.append(f'% BẢNG 1: Số club tối thiểu (mean \\pm std) trên {num_runs} lần chạy')
    lines.append(r'% =============================================================================')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    col = 'l|' + 'c' * len(cnames)
    lines.append(r'\caption{Kết quả số club tối thiểu (mean $\pm$ std) trên ' + str(num_runs) + r' lần chạy của bốn '
                 r'chiến lược trên các tập đồ thị benchmark với $s=2$. '
                 r'Giá trị nhỏ hơn là tốt hơn; in \textbf{đậm} là tốt nhất.}')
    lines.append(r'\label{tab:results_sclub}')
    lines.append(r'\begin{tabular}{' + col + r'}')
    lines.append(r'\toprule')

    header = r'\textbf{Tập đồ thị} & ' + ' & '.join(
        r'\textbf{' + SHORT_NAME[c] + '}' for c in cnames) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')

    for graph_name, gr in all_results.items():
        # Tìm mean nhỏ nhất (tốt nhất)
        means = {}
        for c in cnames:
            if c in gr and 'summary' in gr[c]:
                means[c] = gr[c]['summary']['cover_mean']
        best_mean = min(means.values()) if means else None

        row_cells = []
        for c in cnames:
            if c in gr and 'summary' in gr[c]:
                m  = gr[c]['summary']['cover_mean']
                sd = gr[c]['summary']['cover_std']
                cell = f'${m:.2f} \\pm {sd:.2f}$'
                if best_mean is not None and abs(m - best_mean) < 1e-6:
                    cell = r'\textbf{' + cell + '}'
            else:
                cell = r'\textemdash'
            row_cells.append(cell)

        G = graphs.get(graph_name)
        meta = f'({G.vcount()}v)' if G else ''
        lines.append(f'{graph_name} {meta} & ' + ' & '.join(row_cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    lines.append('\n')

    # ── Bảng 2: So sánh Best và Avg (Tương tự bài báo của Thầy) ──
    lines.append(r'% =============================================================================')
    lines.append(f'% BẢNG 2: So sánh kết quả tốt nhất (Best) và trung bình (Avg) trên {num_runs} lần chạy')
    lines.append(r'% =============================================================================')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    col_best_avg = 'l' + 'cc' * len(cnames)
    lines.append(r'\caption{Kết quả kích thước phủ tốt nhất (Best) và trung bình (Avg) trên ' + str(num_runs) + r' lần chạy của bốn chiến lược trên các tập đồ thị benchmark với $s=2$.}')
    lines.append(r'\label{tab:results_best_avg}')
    lines.append(r'\begin{tabular}{' + col_best_avg + r'}')
    lines.append(r'\toprule')
    
    # Header dòng 1
    sub_headers = []
    for c in cnames:
        sub_headers.append(rf'\multicolumn{{2}}{{c}}{{\textbf{{{SHORT_NAME[c]}}}}}')
    lines.append(r'\textbf{Tập đồ thị} & ' + ' & '.join(sub_headers) + r' \\')
    
    # Header dòng 2
    sub_sub = ['Best', 'Avg'] * len(cnames)
    lines.append(r' & ' + ' & '.join(rf'\textbf{{{s}}}' for s in sub_sub) + r' \\')
    lines.append(r'\midrule')

    for graph_name, gr in all_results.items():
        bests = {}
        for c in cnames:
            if c in gr:
                seed_vals = [gr[c][s]['cover_size'] for s in gr[c] if s not in ('summary', '0.0')]
                # Thử lấy từ các key là số hạt giống
                clean_vals = [v for v in seed_vals if isinstance(v, (int, float))]
                if clean_vals:
                    bests[c] = min(clean_vals)
                elif 'summary' in gr[c]:
                    bests[c] = gr[c]['summary']['cover_min']
        global_best_min = min(bests.values()) if bests else None

        row_cells = []
        for c in cnames:
            if c in gr and 'summary' in gr[c]:
                seed_vals = [gr[c][s]['cover_size'] for s in gr[c] if s not in ('summary', '0.0')]
                clean_vals = [v for v in seed_vals if isinstance(v, (int, float))]
                b = min(clean_vals) if clean_vals else gr[c]['summary']['cover_min']
                m = gr[c]['summary']['cover_mean']
                
                b_str = f'{b:.0f}'
                m_str = f'{m:.2f}'
                
                if global_best_min is not None and abs(b - global_best_min) < 1e-6:
                    b_str = r'\textbf{' + b_str + '}'
                row_cells.append(b_str)
                row_cells.append(m_str)
            else:
                row_cells.extend([r'\textemdash', r'\textemdash'])

        G = graphs.get(graph_name)
        meta = f'({G.vcount()}v)' if G else ''
        lines.append(f'{graph_name} {meta} & ' + ' & '.join(row_cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    lines.append('\n')

    # ── Bảng 3: So sánh Thời gian chạy trung bình ──
    lines.append(r'% =============================================================================')
    lines.append(f'% BẢNG 3: So sánh thời gian tính toán trung bình (giây) trên {num_runs} lần chạy')
    lines.append(r'% =============================================================================')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    col_time = 'l' + 'c' * len(cnames)
    lines.append(r'\caption{So sánh thời gian tính toán trung bình (giây) trên ' + str(num_runs) + r' lần chạy của bốn chiến lược trên các tập đồ thị benchmark với $s=2$.}')
    lines.append(r'\label{tab:results_time}')
    lines.append(r'\begin{tabular}{' + col_time + r'}')
    lines.append(r'\toprule')
    header_time = r'\textbf{Tập đồ thị} & ' + ' & '.join(
        rf'\textbf{{{SHORT_NAME[c]} (s)}}' for c in cnames) + r' \\'
    lines.append(header_time)
    lines.append(r'\midrule')

    for graph_name, gr in all_results.items():
        row_cells = []
        for c in cnames:
            if c in gr:
                times = [gr[c][s]['computation_time'] for s in gr[c] if s not in ('summary', '0.0')]
                clean_times = [t for t in times if isinstance(t, (int, float))]
                if clean_times:
                    avg_time = np.mean(clean_times)
                    row_cells.append(f'{avg_time:.2f}')
                else:
                    row_cells.append(r'\textemdash')
            else:
                row_cells.append(r'\textemdash')

        G = graphs.get(graph_name)
        meta = f'({G.vcount()}v)' if G else ''
        lines.append(f'{graph_name} {meta} & ' + ' & '.join(row_cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    tex_str = '\n'.join(lines)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(tex_str)
    print(f'Saved: {filename}')


def generate_detailed_tex(all_results, graphs, filename='results_table_detailed.tex'):
    cnames = list(CONDITIONS.keys())
    lines = []
    
    # Xác định số lượng lần chạy (seeds) thực tế đã thực hiện
    num_runs = 20  # Fallback mặc định
    if all_results:
        first_graph = list(all_results.keys())[0]
        if all_results[first_graph]:
            first_cond = list(all_results[first_graph].keys())[0]
            seeds_run = [s for s in all_results[first_graph][first_cond].keys() if s not in ('summary', '0.0')]
            if seeds_run:
                num_runs = len(seeds_run)
    
    for graph_name, gr in all_results.items():
        lines.append(r'% =============================================================================')
        lines.append(f'% BẢNG CHI TIẾT {num_runs} LẦN CHẠY: {graph_name}')
        lines.append(r'% =============================================================================')
        lines.append(r'\begin{table}[htbp]')
        lines.append(r'\centering')
        lines.append(f'\\caption{{Kết quả chi tiết kích thước phủ trên {num_runs} lần chạy của bốn chiến lược trên đồ thị {graph_name} ($s=2$).}}')
        lines.append(f'\\label{{tab:detailed_{graph_name}}}')
        
        # 5 cột: Hạt giống, EVO-RL (PPO), No-RL, Fixed, Random
        lines.append(r'\begin{tabular}{c|cccc}')
        lines.append(r'\toprule')
        
        header = r'\textbf{Hạt giống} & ' + ' & '.join(
            r'\textbf{' + SHORT_NAME[c] + '}' for c in cnames) + r' \\'
        lines.append(header)
        lines.append(r'\midrule')
        
        first_cond = cnames[0]
        if first_cond in gr:
            seeds = [s for s in gr[first_cond] if s not in ('summary', '0.0')]
            try:
                sorted_seeds = sorted(seeds, key=lambda x: int(x))
            except ValueError:
                sorted_seeds = sorted(seeds)
        else:
            sorted_seeds = []
            
        for idx, seed_str in enumerate(sorted_seeds, 1):
            row_cells = []
            for c in cnames:
                if c in gr and seed_str in gr[c]:
                    val = gr[c][seed_str]['cover_size']
                    row_cells.append(f'{val:.0f}')
                else:
                    row_cells.append(r'\textemdash')
            lines.append(f'Lần {idx} ({seed_str}) & ' + ' & '.join(row_cells) + r' \\')
            
        lines.append(r'\midrule')
        
        # Thêm dòng Best và Avg ở cuối bảng chi tiết
        best_cells = []
        avg_cells = []
        for c in cnames:
            if c in gr and 'summary' in gr[c]:
                seed_vals = [gr[c][s]['cover_size'] for s in gr[c] if s not in ('summary', '0.0')]
                clean_vals = [v for v in seed_vals if isinstance(v, (int, float))]
                best_val = min(clean_vals) if clean_vals else gr[c]['summary']['cover_min']
                avg_val = gr[c]['summary']['cover_mean']
                best_cells.append(f'{best_val:.0f}')
                avg_cells.append(f'{avg_val:.2f}')
            else:
                best_cells.append(r'\textemdash')
                avg_cells.append(r'\textemdash')
                
        lines.append(r'\textbf{Best} & ' + ' & '.join(best_cells) + r' \\')
        lines.append(r'\textbf{Avg} & ' + ' & '.join(avg_cells) + r' \\')
        
        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{table}')
        lines.append('\n')
        
    tex_str = '\n'.join(lines)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(tex_str)
    print(f'Saved: {filename}')


def generate_latex_charts(all_results, graphs, filename='results_chart.tex'):
    cnames = list(CONDITIONS.keys())
    lines = []
    
    short_names = {
        "AdaptiveEvolutionary": "EVO-RL",
        "NoReinforcementLearning": "No-RL",
        "FixedOperatorStrategy": "Fixed",
        "RandomOperatorSelection": "Random"
    }

    # Màu sắc cố định cho các đường cong hội tụ
    cond_colors = {
        "AdaptiveEvolutionary": "blue",
        "NoReinforcementLearning": "red",
        "FixedOperatorStrategy": "orange",
        "RandomOperatorSelection": "olive"
    }

    def sample_coords(data_list, step=10):
        if not data_list:
            return ""
        coords = []
        for idx, val in enumerate(data_list):
            if idx % step == 0 or idx == len(data_list) - 1:
                coords.append(f"({idx}, {val:.3f})")
        return " ".join(coords)

    for graph_name, gr in all_results.items():
        means = {}
        stds = {}
        for c in cnames:
            if c in gr and 'summary' in gr[c]:
                means[c] = gr[c]['summary']['cover_mean']
                stds[c] = gr[c]['summary']['cover_std']
        
        if not means:
            continue
            
        ymax = max(means.values()) * 1.3
        
        # 1. BIỂU ĐỒ CỘT SO SÁNH KÍCH THƯỚC PHỦ TRUNG BÌNH
        lines.append(f'% =============================================================================')
        lines.append(f'% BIỂU ĐỒ 1: SO SÁNH KÍCH THƯỚC PHỦ TRÊN ĐỒ THỊ {graph_name}')
        lines.append(f'% =============================================================================')
        lines.append(r'\begin{figure}[htbp]')
        lines.append(r'\centering')
        lines.append(r'\begin{tikzpicture}')
        lines.append(r'\begin{axis}[')
        lines.append(r'    ybar,')
        lines.append(r'    width=0.9\columnwidth,')
        lines.append(r'    height=5.8cm,')
        lines.append(r'    bar width=16pt,')
        lines.append(r'    ylabel={Kích thước lớp phủ trung bình},')
        lines.append(r'    xlabel={Chiến lược lựa chọn toán tử},')
        lines.append(r'    symbolic x coords={' + ', '.join(short_names.values()) + r'},')
        lines.append(r'    xtick=data,')
        lines.append(r'    nodes near coords,')
        lines.append(r'    nodes near coords style={font=\footnotesize, anchor=south, /pgf/number format/fixed, /pgf/number format/precision=2},')
        lines.append(r'    ymin=0,')
        lines.append(f'    ymax={ymax:.2f},')
        lines.append(r'    ymajorgrids=true,')
        lines.append(r'    grid style={dashed, gray!30},')
        lines.append(r'    enlarge x limits=0.25,')
        lines.append(r'    xticklabel style={font=\footnotesize},')
        lines.append(r'    yticklabel style={font=\footnotesize},')
        lines.append(r']')
        
        lines.append(r'\addplot+[')
        lines.append(r'    fill={rgb,255:red,68;green,114;blue,196},')
        lines.append(r'    draw=black,')
        lines.append(r'    error bars/.cd,')
        lines.append(r'    y dir=both,')
        lines.append(r'    y explicit')
        lines.append(r'] coordinates {')
        
        for c in cnames:
            short_name = short_names[c]
            mean_val = means.get(c, 0.0)
            std_val = stds.get(c, 0.0)
            lines.append(f'    ({short_name}, {mean_val:.2f}) +- (0.0, {std_val:.3f})')
            
        lines.append(r'};')
        lines.append(r'\end{axis}')
        lines.append(r'\end{tikzpicture}')
        lines.append(r'\caption{So sánh kích thước lớp phủ trung bình ($s=' + str(S_VALUE) + r'$) trên đồ thị ' + graph_name + r'.}')
        lines.append(r'\label{fig:chart_cover_' + graph_name + r'}')
        lines.append(r'\end{figure}')
        lines.append('\n')

        # 2. BIỂU ĐỒ ĐƯỜNG CONG HỘI TỤ (CONVERGENCE CURVE)
        has_history_cover = any('history_cover_mean' in gr[c]['summary'] for c in cnames if c in gr and 'summary' in gr[c])
        if has_history_cover:
            lines.append(f'% =============================================================================')
            lines.append(f'% BIỂU ĐỒ 2: ĐƯỜNG CONG HỘI TỤ QUA CÁC THẾ HỆ TRÊN ĐỒ THỊ {graph_name}')
            lines.append(f'% =============================================================================')
            lines.append(r'\begin{figure}[htbp]')
            lines.append(r'\centering')
            lines.append(r'\begin{tikzpicture}')
            lines.append(r'\begin{axis}[')
            lines.append(r'    width=0.9\columnwidth,')
            lines.append(r'    height=5.8cm,')
            lines.append(r'    xlabel={Thế hệ},')
            lines.append(r'    ylabel={Kích thước lớp phủ},')
            lines.append(r'    grid=both,')
            lines.append(r'    grid style={dashed, gray!30},')
            lines.append(r'    legend style={at={(0.5,-0.25)}, anchor=north, legend columns=4, font=\tiny},')
            lines.append(r'    xticklabel style={font=\footnotesize},')
            lines.append(r'    yticklabel style={font=\footnotesize},')
            lines.append(r']')

            for c in cnames:
                if c in gr and 'summary' in gr[c] and 'history_cover_mean' in gr[c]['summary']:
                    hist = gr[c]['summary']['history_cover_mean']
                    coords_str = sample_coords(hist, step=10)
                    color = cond_colors[c]
                    lines.append(r'\addplot[' + color + r', thick, mark=none] coordinates {' + coords_str + r'};')
                    lines.append(r'\addlegendentry{' + short_names[c] + r'}')

            lines.append(r'\end{axis}')
            lines.append(r'\end{tikzpicture}')
            lines.append(r'\caption{Đường cong hội tụ kích thước lớp phủ trung bình qua các thế hệ trên đồ thị ' + graph_name + r'.}')
            lines.append(r'\label{fig:chart_conv_' + graph_name + r'}')
            lines.append(r'\end{figure}')
            lines.append('\n')

        # 3. BIỂU ĐỒ XÁC SUẤT CHỌN HÀNH ĐỘNG (ACTION SELECTION PROBABILITY) CỦA EVO-RL
        evo_name = "AdaptiveEvolutionary"
        if evo_name in gr and 'summary' in gr[evo_name] and 'history_action_prob_mean' in gr[evo_name]['summary']:
            prob_hist = gr[evo_name]['summary']['history_action_prob_mean']
            if prob_hist:
                lines.append(f'% =============================================================================')
                lines.append(f'% BIỂU ĐỒ 3: XÁC SUẤT CHỌN TOÁN TỬ CỦA EVO-RL TRÊN ĐỒ THỊ {graph_name}')
                lines.append(f'% =============================================================================')
                lines.append(r'\begin{figure}[htbp]')
                lines.append(r'\centering')
                lines.append(r'\begin{tikzpicture}')
                lines.append(r'\begin{axis}[')
                lines.append(r'    width=0.9\columnwidth,')
                lines.append(r'    height=5.8cm,')
                lines.append(r'    xlabel={Thế hệ},')
                lines.append(r'    ylabel={Xác suất chọn toán tử},')
                lines.append(r'    ymin=0, ymax=1.0,')
                lines.append(r'    grid=both,')
                lines.append(r'    grid style={dashed, gray!30},')
                lines.append(r'    legend style={at={(0.5,-0.25)}, anchor=north, legend columns=3, font=\tiny},')
                lines.append(r'    xticklabel style={font=\footnotesize},')
                lines.append(r'    yticklabel style={font=\footnotesize},')
                lines.append(r']')

                # Tách thành 3 mảng xác suất cho 3 toán tử
                crossover_probs = [step_probs[0] for step_probs in prob_hist]
                repair_probs = [step_probs[1] for step_probs in prob_hist]
                diversify_probs = [step_probs[2] for step_probs in prob_hist]

                lines.append(r'\addplot[blue, thick, mark=none] coordinates {' + sample_coords(crossover_probs, step=10) + r'};')
                lines.append(r'\addlegendentry{Crossover ($a^{\text{cross}}$)}')

                lines.append(r'\addplot[green!70!black, thick, mark=none] coordinates {' + sample_coords(repair_probs, step=10) + r'};')
                lines.append(r'\addlegendentry{Repair Mut ($a^{\text{repair}}$)}')

                lines.append(r'\addplot[red, thick, mark=none] coordinates {' + sample_coords(diversify_probs, step=10) + r'};')
                lines.append(r'\addlegendentry{Diversify Mut ($a^{\text{div}}$)}')

                lines.append(r'\end{axis}')
                lines.append(r'\end{tikzpicture}')
                lines.append(r'\caption{Sự thay đổi xác suất chọn các toán tử di truyền thích nghi của tác nhân EVO-RL qua các thế hệ trên đồ thị ' + graph_name + r'.}')
                lines.append(r'\label{fig:chart_probs_' + graph_name + r'}')
                lines.append(r'\end{figure}')
                lines.append('\n')

    tex_str = '\n'.join(lines)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(tex_str)
    print(f'Saved: {filename}')


# ── Sinh Markdown ─────────────────────────────────────────────────────────────

def generate_md(all_results, graphs, filename='results_summary.md', seeds_list=None):
    if seeds_list is None:
        seeds_list = SEEDS
    cnames = list(CONDITIONS.keys())
    lines  = []
    lines.append('# Kết Quả Thực Nghiệm — EVO-RL: Minimum s-Club Cover\n')

    lines.append('## Cấu hình thực nghiệm\n')
    lines.append(f'| Tham số | Giá trị |')
    lines.append(f'|---------|---------|')
    for k, v in HYPERPARAMETERS.items():
        lines.append(f'| `{k}` | {v} |')
    lines.append('')

    lines.append('## Mô tả tập đồ thị\n')
    lines.append('| Tập đồ thị | Số đỉnh | Số cạnh | Định dạng |')
    lines.append('|------------|---------|---------|-----------|')
    for gname, G in graphs.items():
        ext = '.gml' if gname in ('karate', 'dolphins', 'football') else '.clq'
        lines.append(f'| {gname} | {G.vcount()} | {G.ecount()} | `{ext}` |')
    lines.append('')

    lines.append(f'## Kết quả với s = {S_VALUE}\n')
    lines.append('> Số club **nhỏ hơn = tốt hơn**. Giá trị in **đậm** = tốt nhất trong hàng.\n')

    # Header bảng
    h_cells = ['**Tập đồ thị**'] + [f'**{SHORT_NAME[c]}**' for c in cnames]
    s_cells = ['---'] + ['---:'] * len(cnames)
    lines.append('| ' + ' | '.join(h_cells) + ' |')
    lines.append('| ' + ' | '.join(s_cells) + ' |')

    for graph_name, gr in all_results.items():
        means = {c: gr[c]['summary']['cover_mean']
                 for c in cnames if c in gr and 'summary' in gr[c]}
        best_mean = min(means.values()) if means else None

        row = [f'`{graph_name}`']
        for c in cnames:
            if c in gr and 'summary' in gr[c]:
                m  = gr[c]['summary']['cover_mean']
                sd = gr[c]['summary']['cover_std']
                cell = f'{m:.2f} ± {sd:.2f}'
                if best_mean is not None and abs(m - best_mean) < 1e-6:
                    cell = f'**{cell}**'
                row.append(cell)
            else:
                row.append('—')
        lines.append('| ' + ' | '.join(row) + ' |')

    lines.append('\n## Kết quả chi tiết theo seed\n')
    for graph_name, gr in all_results.items():
        lines.append(f'### {graph_name}\n')
        
        # Tạo bảng động dựa trên số seed thực tế trong seeds_list
        seed_headers = [f'Seed {s}' for s in seeds_list]
        lines.append('| Chiến lược | ' + ' | '.join(seed_headers) + ' | Mean ± Std |')
        lines.append('|------------|' + '|'.join(['---'] * len(seeds_list)) + '|------------|')
        
        for c in cnames:
            if c not in gr:
                continue
            
            seed_vals = []
            for s in seeds_list:
                val = gr[c].get(str(s), {}).get('cover_size', '—')
                if isinstance(val, (int, float)):
                    seed_vals.append(f'{val:.2f}')
                else:
                    seed_vals.append(str(val))
            
            if 'summary' in gr[c]:
                m  = gr[c]['summary']['cover_mean']
                sd = gr[c]['summary']['cover_std']
                summary = f'{m:.2f} ± {sd:.2f}'
            else:
                summary = '—'
            
            lines.append(f'| {SHORT_NAME[c]} | ' + ' | '.join(seed_vals) + f' | {summary} |')
        lines.append('')

    lines.append('\n## Bảng so sánh Best và Avg (Tương tự bài báo của Thầy)\n')
    h_cells = ['**Tập đồ thị**']
    for c in cnames:
        h_cells.append(f'**{SHORT_NAME[c]} Best**')
        h_cells.append(f'**{SHORT_NAME[c]} Avg**')
    s_cells = ['---'] + ['---:'] * (len(cnames) * 2)
    lines.append('| ' + ' | '.join(h_cells) + ' |')
    lines.append('| ' + ' | '.join(s_cells) + ' |')
    for graph_name, gr in all_results.items():
        row = [f'`{graph_name}`']
        bests = {}
        for c in cnames:
            if c in gr:
                seed_vals = [gr[c][s]['cover_size'] for s in gr[c] if s not in ('summary', '0.0')]
                clean_vals = [v for v in seed_vals if isinstance(v, (int, float))]
                if clean_vals:
                    bests[c] = min(clean_vals)
                elif 'summary' in gr[c]:
                    bests[c] = gr[c]['summary']['cover_min']
        global_best_min = min(bests.values()) if bests else None
        
        for c in cnames:
            if c in gr and 'summary' in gr[c]:
                seed_vals = [gr[c][s]['cover_size'] for s in gr[c] if s not in ('summary', '0.0')]
                clean_vals = [v for v in seed_vals if isinstance(v, (int, float))]
                b = min(clean_vals) if clean_vals else gr[c]['summary']['cover_min']
                m = gr[c]['summary']['cover_mean']
                b_str = f'{b:.0f}'
                if global_best_min is not None and abs(b - global_best_min) < 1e-6:
                    b_str = f'**{b_str}**'
                row.append(b_str)
                row.append(f'{m:.2f}')
            else:
                row.extend(['—', '—'])
        lines.append('| ' + ' | '.join(row) + ' |')
        
    lines.append('\n## Bảng so sánh Thời gian chạy trung bình (giây)\n')
    h_cells = ['**Tập đồ thị**'] + [f'**{SHORT_NAME[c]} (giây)**' for c in cnames]
    s_cells = ['---'] + ['---:'] * len(cnames)
    lines.append('| ' + ' | '.join(h_cells) + ' |')
    lines.append('| ' + ' | '.join(s_cells) + ' |')
    for graph_name, gr in all_results.items():
        row = [f'`{graph_name}`']
        for c in cnames:
            if c in gr:
                times = [gr[c][s]['computation_time'] for s in gr[c] if s not in ('summary', '0.0')]
                clean_times = [t for t in times if isinstance(t, (int, float))]
                if clean_times:
                    row.append(f'{np.mean(clean_times):.2f}s')
                else:
                    row.append('—')
            else:
                row.append('—')
        lines.append('| ' + ' | '.join(row) + ' |')
    lines.append('')

    lines.append(r'## Ghi chú')
    lines.append('- **EVO-RL (PPO)**: Agent học tăng cường PPO chọn toán tử thích nghi')
    lines.append('- **No-RL**: Xác suất toán tử cố định đều nhau')
    lines.append('- **Fixed**: Lịch trình toán tử cố định xoay vòng 0→1→2')
    lines.append('- **Random**: Chọn toán tử ngẫu nhiên hoàn toàn')
    lines.append(f'\n_Sinh tự động bởi `main.py` — s={S_VALUE}, seeds={seeds_list}_\n')

    md_str = '\n'.join(lines)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(md_str)
    print(f'Saved: {filename}')


def get_next_run_index():
    idx = 1
    while True:
        if (os.path.exists(f'results_{idx}.json') or 
            os.path.exists(f'results_table_{idx}.tex') or 
            os.path.exists(f'results_summary_{idx}.md')):
            idx += 1
        else:
            return idx


def main():
    print("\n" + "="*55)
    print("  EVO-RL: Minimum s-Club Cover Experiment")
    print("  RL-based Adaptive Operator Selection for EA")
    print("="*55)

    # Check data directory
    if not os.path.isdir(DATA_DIR):
        print(f"\nERROR: Data directory not found:\n  {DATA_DIR}")
        sys.exit(1)

    # Load all graphs
    print(f"\nLoading graphs from: {DATA_DIR}")
    all_graphs = load_all_graphs(DATA_DIR)
    if not all_graphs:
        print("ERROR: No graph files found (.gml or .clq)!")
        sys.exit(1)

    # Parse command line args for automated execution
    selected_graphs = {}
    auto_mode = False
    
    if "--run" in sys.argv:
        auto_mode = True
        idx_arg = sys.argv.index("--run") + 1
        if idx_arg < len(sys.argv):
            val = sys.argv[idx_arg]
            names = list(all_graphs.keys())
            if val == "0":
                selected_graphs = all_graphs
            else:
                parts = [x.strip() for x in val.replace(',', ' ').split()]
                for part in parts:
                    if part.isdigit():
                        idx = int(part)
                        if 1 <= idx <= len(names):
                            selected_graphs[names[idx - 1]] = all_graphs[names[idx - 1]]
                    elif part in all_graphs:
                        selected_graphs[part] = all_graphs[part]
        if not selected_graphs:
            print("WARNING: No valid graph selection found. Running ALL graphs.")
            selected_graphs = all_graphs
    else:
        # Ask user which graphs to run (Interactive Mode)
        selected_graphs = ask_which_graphs(all_graphs)

    print(f"\nWill run: {len(selected_graphs)} graph(s) x "
          f"{len(CONDITIONS)} methods x {len(SEEDS)} seeds")
    print(f"s = {S_VALUE}, max_epochs = {HYPERPARAMETERS['max_epochs']}, "
          f"max_generations = {HYPERPARAMETERS['max_generations']}, "
          f"stagnation_limit = {HYPERPARAMETERS['stagnation_limit']}")
    
    if not auto_mode:
        input("\nPress Enter to start (Ctrl+C to cancel)...")

    # Lấy index đánh số thứ tự trước khi chạy
    run_idx = get_next_run_index()

    # Chạy thực nghiệm
    t_total_start = time.time()
    
    # Khởi tạo all_results trống
    all_results = {}
    for graph_name in selected_graphs.keys():
        all_results[graph_name] = {}
        for cond_name in CONDITIONS.keys():
            all_results[graph_name][cond_name] = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'='*55}")
        print(f"  === RUNNING SEED {seed} ({seed_idx + 1}/{len(SEEDS)}) ===")
        print(f"{'='*55}")
        
        for graph_name, G in selected_graphs.items():
            print(f"\n--- {graph_name} (s={S_VALUE}, seed={seed}) ---")
            for cond_name, AgentClass in CONDITIONS.items():
                set_all_seeds(seed)
                env = SClubEnvironment(graph=G, s=S_VALUE, pop_size=config.pop_size)
                agent = AgentClass(STATE_DIM, NUM_OPERATORS)
                
                optimizer = torch.optim.Adam(
                    agent.policy_network.parameters(),
                    lr=config.learning_rate)

                t0 = time.time()
                train_agent(agent, env, optimizer, device)
                eval_m = evaluate_agent(agent, env, device)
                elapsed = time.time() - t0

                cover = eval_m['cover_size']
                all_results[graph_name][cond_name][str(seed)] = {
                    'cover_size':       cover,
                    'cover_size_min':   eval_m['cover_size_min'],
                    'solution_quality': eval_m['solution_quality'],
                    'computation_time': round(elapsed, 2),
                    'fully_covered':    eval_m['fully_covered'],
                    'history_cover_sizes': train_metrics['cover_sizes'],
                    'history_action_probs': train_metrics['action_probs'],
                }
                print(f"    {SHORT_NAME[cond_name]:15s} cover={cover:.1f}  "
                      f"Q={eval_m['solution_quality']:.3f}  "
                      f"t={elapsed:.1f}s")
                      
        # --- Sau khi hoàn thành seed hiện tại cho TẤT CẢ đồ thị ---
        current_seeds_run = SEEDS[:seed_idx + 1]
        
        # Tạo thư mục kết quả gốc
        results_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))
        os.makedirs(results_root, exist_ok=True)
        
        for graph_name in selected_graphs.keys():
            # Tạo thư mục con tương ứng với đồ thị
            graph_dir = os.path.join(results_root, graph_name)
            os.makedirs(graph_dir, exist_ok=True)
            
            # Lấy kết quả của riêng đồ thị này
            graph_results = {graph_name: all_results[graph_name]}
            
            for cond_name in CONDITIONS.keys():
                covers = []
                history_covers_all = []
                history_probs_all = []
                for s in current_seeds_run:
                    if str(s) in graph_results[graph_name][cond_name]:
                        covers.append(graph_results[graph_name][cond_name][str(s)]['cover_size'])
                        history_covers_all.append(graph_results[graph_name][cond_name][str(s)]['history_cover_sizes'])
                        history_probs_all.append(graph_results[graph_name][cond_name][str(s)]['history_action_probs'])
                if covers:
                    mean_history_covers = np.mean(history_covers_all, axis=0).tolist() if history_covers_all else []
                    mean_history_probs = np.mean(history_probs_all, axis=0).tolist() if history_probs_all else []
                    
                    graph_results[graph_name][cond_name]['summary'] = {
                        'cover_mean': round(float(np.mean(covers)), 3),
                        'cover_std':  round(float(np.std(covers)),  3),
                        'cover_min':  round(float(np.min(covers)),  3),
                        'history_cover_mean': mean_history_covers,
                        'history_action_prob_mean': mean_history_probs,
                    }
                    
            # 1. Ghi file kết quả tích lũy JSON tăng dần cho đồ thị này (lưu dự phòng)
            json_fn  = os.path.join(graph_dir, f'results_{run_idx}.json')
            json_def = os.path.join(graph_dir, 'results.json')

            output = {
                'hyperparameters': HYPERPARAMETERS,
                's_value':         S_VALUE,
                'seeds':           current_seeds_run,
                'graphs_run':      [graph_name],
                'metrics':         graph_results,
            }

            with open(json_fn, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            with open(json_def, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            # 2. Ghi file kết quả riêng biệt cho RIÊNG seed này (độc lập)
            formatted_seed_results = {graph_name: {}}
            for cond_name in CONDITIONS.keys():
                metrics = all_results[graph_name][cond_name][str(seed)]
                formatted_seed_results[graph_name][cond_name] = {
                    str(seed): metrics,
                    'summary': {
                        'cover_mean': metrics['cover_size'],
                        'cover_std': 0.0,
                        'cover_min': metrics['cover_size'],
                        'history_cover_mean': metrics['history_cover_sizes'],
                        'history_action_prob_mean': metrics['history_action_probs'],
                    }
                }
                
            json_seed_fn = os.path.join(graph_dir, f'results_run_{run_idx}_seed_{seed}.json')
            md_seed_fn   = os.path.join(graph_dir, f'results_summary_run_{run_idx}_seed_{seed}.md')
            
            output_seed = {
                'hyperparameters': HYPERPARAMETERS,
                's_value':         S_VALUE,
                'seed':            seed,
                'graphs_run':      [graph_name],
                'metrics':         formatted_seed_results,
            }
            
            with open(json_seed_fn, 'w', encoding='utf-8') as f:
                json.dump(output_seed, f, indent=2, ensure_ascii=False)
                
            single_graph_dict = {graph_name: selected_graphs[graph_name]}
            generate_md(formatted_seed_results, single_graph_dict, md_seed_fn, [seed])
            
        print(f"  [Seed {seed}] Da backup ket qua JSON vao results/<graph_name>/")

    # --- HẠT GIỐNG ĐÃ CHẠY XONG HOÀN TOÀN ---
    # Sinh báo cáo LaTeX và Markdown tổng kết cuối cùng 1 lần duy nhất cho mỗi đồ thị
    print(f"\n{'='*55}")
    print("  Generating final LaTeX tables and Markdown reports...")
    print(f"{'='*55}")
    
    results_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))
    for graph_name in selected_graphs.keys():
        graph_dir = os.path.join(results_root, graph_name)
        graph_results = {graph_name: all_results[graph_name]}
        
        # Tính toán summary của cả 20 seeds
        for cond_name in CONDITIONS.keys():
            covers = []
            history_covers_all = []
            history_probs_all = []
            for s in SEEDS:
                if str(s) in graph_results[graph_name][cond_name]:
                    covers.append(graph_results[graph_name][cond_name][str(s)]['cover_size'])
                    history_covers_all.append(graph_results[graph_name][cond_name][str(s)]['history_cover_sizes'])
                    history_probs_all.append(graph_results[graph_name][cond_name][str(s)]['history_action_probs'])
            if covers:
                mean_history_covers = np.mean(history_covers_all, axis=0).tolist() if history_covers_all else []
                mean_history_probs = np.mean(history_probs_all, axis=0).tolist() if history_probs_all else []
                
                graph_results[graph_name][cond_name]['summary'] = {
                    'cover_mean': round(float(np.mean(covers)), 3),
                    'cover_std':  round(float(np.std(covers)),  3),
                    'cover_min':  round(float(np.min(covers)),  3),
                    'history_cover_mean': mean_history_covers,
                    'history_action_prob_mean': mean_history_probs,
                }
                
        # Khởi tạo đường dẫn file LaTeX và Markdown
        tex_fn  = os.path.join(graph_dir, f'results_table_{run_idx}.tex')
        tex_def = os.path.join(graph_dir, 'results_table.tex')
        
        tex_det_fn  = os.path.join(graph_dir, f'results_table_detailed_{run_idx}.tex')
        tex_det_def = os.path.join(graph_dir, 'results_table_detailed.tex')
        
        md_fn   = os.path.join(graph_dir, f'results_summary_{run_idx}.md')
        md_def  = os.path.join(graph_dir, 'results_summary.md')
        
        single_graph_dict = {graph_name: selected_graphs[graph_name]}
        
        # 1. Sinh LaTeX Table tổng hợp (3 bảng)
        generate_combined_tex(graph_results, single_graph_dict, tex_fn)
        generate_combined_tex(graph_results, single_graph_dict, tex_def)
        
        # 2. Sinh LaTeX Table Chi tiết 20 lần chạy
        generate_detailed_tex(graph_results, single_graph_dict, tex_det_fn)
        generate_detailed_tex(graph_results, single_graph_dict, tex_det_def)
        
        # 3. Sinh báo cáo Markdown tổng hợp
        generate_md(graph_results, single_graph_dict, md_fn, SEEDS)
        generate_md(graph_results, single_graph_dict, md_def, SEEDS)

        # 3.5. Sinh mã nguồn biểu đồ LaTeX PGFPlots TikZ
        chart_tex_fn = os.path.join(graph_dir, f'results_chart_{run_idx}.tex')
        chart_tex_def = os.path.join(graph_dir, 'results_chart.tex')
        generate_latex_charts(graph_results, single_graph_dict, chart_tex_fn)
        generate_latex_charts(graph_results, single_graph_dict, chart_tex_def)

    # 4. Tự động vẽ biểu đồ cho các đồ thị đã chạy
    try:
        print("\n" + "-"*55)
        print("  Auto-generating comparison charts...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        import generate_charts
        generate_charts.main()
    except Exception as e:
        print(f"  Warning: Could not generate charts: {e}")

    total_time = time.time() - t_total_start
    print(f"\n{'='*55}")
    print(f"  Finished! Total time: {total_time/60:.1f} minutes")
    print(f"{'='*55}")
    print(f"All files saved in results/ subdirectories with run index {run_idx}.")


if __name__ == '__main__':
    main()