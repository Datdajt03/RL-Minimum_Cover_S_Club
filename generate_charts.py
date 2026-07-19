"""
generate_charts.py
File điều phối chính để chạy cả 5 script vẽ biểu đồ:
1. fig_main_results.py
2. fig_metric_distribution.py
3. fig_std_deviation.py
4. fig_convergence_curve.py
5. fig_action_probability.py
"""
import os
import sys

def main():
    print("="*60)
    print("  Generating Charts for EVO-RL s-Club Cover Experiment...")
    print("="*60)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(current_dir, "charts_scripts")
    
    if not os.path.exists(scripts_dir):
        print(f"ERROR: Charts script directory not found at: {scripts_dir}")
        return
        
    sys.path.append(scripts_dir)
    
    try:
        import fig_main_results
        print("\n[1/5] Running fig_main_results.py...")
        fig_main_results.main()
    except Exception as e:
        print(f"Error running fig_main_results: {e}")
        
    try:
        import fig_metric_distribution
        print("\n[2/5] Running fig_metric_distribution.py...")
        fig_metric_distribution.main()
    except Exception as e:
        print(f"Error running fig_metric_distribution: {e}")
        
    try:
        import fig_std_deviation
        print("\n[3/5] Running fig_std_deviation.py...")
        fig_std_deviation.main()
    except Exception as e:
        print(f"Error running fig_std_deviation: {e}")

    try:
        import fig_convergence_curve
        print("\n[4/5] Running fig_convergence_curve.py...")
        fig_convergence_curve.main()
    except Exception as e:
        print(f"Error running fig_convergence_curve: {e}")

    try:
        import fig_action_probability
        print("\n[5/5] Running fig_action_probability.py...")
        fig_action_probability.main()
    except Exception as e:
        print(f"Error running fig_action_probability: {e}")
        
    print("\n" + "="*60)
    print("  All 5 charts generated successfully!")
    print("  Check the 'charts_output' directory inside each graph folder under 'results/'.")
    print("="*60)

if __name__ == "__main__":
    main()
