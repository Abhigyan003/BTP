import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g. SMD, MSL, SWaT)')
    parser.add_argument('--use_causal', action='store_true', help='Visualize Causal Enhanced results')
    args = parser.parse_args()

    # Dynamic File Paths based on Causal Flag
    suffix = "_causal" if args.use_causal else ""
    csv_path = f"results/csv/{args.dataset}_benchmark{suffix}.csv"
    output_img = f"results/image/{args.dataset}_benchmark{suffix}.png"
    
    mode_label = "Physics-Aware (Causal)" if args.use_causal else "Standard Shape-Based"

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        print(f"Run: python -m scripts.benchmark --dataset {args.dataset}" + (" --use_causal" if args.use_causal else ""))
        return

    df = pd.read_csv(csv_path)
    print(f"Visualizing {mode_label} results for {args.dataset}...")

    # Prepare Data
    df_melt = df.melt(id_vars=['Machine'], 
                      value_vars=['Raw_F1', 'PA_F1'], 
                      var_name='Metric', 
                      value_name='Score')
    
    df_melt['Metric'] = df_melt['Metric'].replace({'Raw_F1': 'Strict F1', 'PA_F1': 'Paper F1 (PA)'})

    # Setup Plot
    sns.set_theme(style="whitegrid")
    width = max(14, len(df) * 0.4)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, 10), sharex=True)

    # ==========================================
    # PLOT 1: ACCURACY
    # ==========================================
    sns.barplot(x='Machine', y='Score', hue='Metric', data=df_melt, ax=ax1, palette="viridis")
    
    avg_pa = df['PA_F1'].mean()
    ax1.axhline(avg_pa, color='red', linestyle='--', linewidth=2, label=f'Avg PA F1: {avg_pa:.2f}')
    
    ax1.set_title(f"Accuracy: {mode_label} - {args.dataset}", fontsize=14, fontweight='bold')
    ax1.set_ylabel("F1 Score", fontsize=12)
    ax1.set_ylim(0, 1.15)
    ax1.legend(loc='upper right', frameon=True)

    # ==========================================
    # PLOT 2: EFFICIENCY
    # ==========================================
    sns.lineplot(x='Machine', y='InitTime', data=df, ax=ax2, marker='o', color='#d62728', linewidth=2)
    ax2.fill_between(df['Machine'], df['InitTime'], color='#d62728', alpha=0.1)
    
    ax2.set_title(f"Efficiency: Model Initialization Time (Avg: {df['InitTime'].mean():.3f}s)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Time (Seconds)", fontsize=12)
    ax2.set_xlabel("Target Entity ID", fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_img, dpi=300)
    print(f"Success! Plot saved to {output_img}")

if __name__ == "__main__":
    main()