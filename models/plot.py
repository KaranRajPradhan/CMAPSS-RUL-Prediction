import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_engine_rul_comparison(csv_file, engine_id):
    results_df = pd.read_csv(csv_file)

    if engine_id == -1:
        engine_ids = results_df['EngineID'].unique()
        plot_filename = "rul_comparison_all_engines.png"
    else:
        engine_ids = [engine_id]
        plot_filename = f"engine_{engine_id}_rul_comparison.png"

    plt.figure(figsize=(15, 8))

    for current_engine_id in engine_ids:
        engine_results = results_df[results_df['EngineID'] == current_engine_id]

        if len(engine_results) == 0:
            continue

        plt.plot(engine_results.index, engine_results['Actual_RUL'], 
                 color='blue', marker='o', linestyle='-', markersize=5, linewidth=2)
        plt.plot(engine_results.index, engine_results['Predicted_RUL'],
                 color='orange', marker='x', linestyle='--', markersize=5, linewidth=2)

    plt.title('Predicted vs Actual RUL for Engine')
    plt.xlabel('Cycle')
    plt.ylabel('RUL')
    plt.legend(['RUL (Actual)', 'RUL (Predicted)'], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    os.makedirs('rul_comparison_plots', exist_ok=True)
    plot_path = os.path.join('rul_comparison_plots', plot_filename)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    print(f"Plot saved as '{plot_path}'")

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <csv_file> <engine_id>")
        print("Use -1 to plot all engines")
        sys.exit(1)

    try:
        csv_file = sys.argv[1]
        engine_id = int(sys.argv[2])
        plot_engine_rul_comparison(csv_file, engine_id)
    except ValueError:
        print("Engine ID must be an integer.")
        print("Use -1 to plot all engines")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()