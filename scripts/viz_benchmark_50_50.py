import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # 1. Load Data
    csv_path = "results/csv/SMD_benchmark_50_50.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        print("Please run 'run_full_benchmark.py' first!")
        return

    df = pd.read_csv(csv_path)

    # 2. Prepare Data for F1 Plot (Melting)
    # This transforms columns 'Raw_F1' and 'PA_F1' into rows for easier plotting
    df_melt = df.melt(id_vars=['Machine'], 
                      value_vars=['Raw_F1', 'PA_F1'], 
                      var_name='Metric', 
                      value_name='Score')
    
    # Rename for cleaner legend
    df_melt['Metric'] = df_melt['Metric'].replace({'Raw_F1': 'Strict F1 (Raw)', 'PA_F1': 'Paper F1 (PA)'})

    # 3. Setup Canvas
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # ==========================================
    # PLOT 1: ACCURACY (Raw vs PA)
    # ==========================================
    # Grouped Bar Chart
    sns.barplot(x='Machine', y='Score', hue='Metric', data=df_melt, ax=ax1, palette="viridis")
    
    # Add a line for Average PA F1
    avg_pa = df['PA_F1'].mean()
    ax1.axhline(avg_pa, color='red', linestyle='--', linewidth=2, label=f'Avg PA F1: {avg_pa:.2f}')
    
    # Formatting
    ax1.set_title("Accuracy: Strict vs. Point-Adjusted (Target Domain)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("F1 Score", fontsize=12)
    ax1.set_ylim(0, 1.1) # Leave room for legend
    ax1.legend(loc='upper right', frameon=True)
    
    # Add text to explain the "Gap"
    ax1.text(0, 1.02, "The gap between bars represents anomaly segments detected correctly but partially.", 
             transform=ax1.transAxes, fontsize=10, color='gray', style='italic')

    # ==========================================
    # PLOT 2: INITIALIZATION TIME
    # ==========================================
    # Line chart with area fill
    sns.lineplot(x='Machine', y='InitTime', data=df, ax=ax2, marker='o', color='#d62728', linewidth=2)
    ax2.fill_between(df['Machine'], df['InitTime'], color='#d62728', alpha=0.1)
    
    # Formatting
    ax2.set_title(f"Efficiency: Model Initialization Time (Avg: {df['InitTime'].mean():.3f}s)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Time (Seconds)", fontsize=12)
    ax2.set_xlabel("Target Machine ID", fontsize=12)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Layout Adjustment
    plt.tight_layout()
    
    # Save
    save_name = "results/image/smd_50_50_results.png"
    plt.savefig(save_name, dpi=300)
    print(f"Comparison plot saved to {save_name}")
    print("Open this image to see the performance gap and efficiency stability.")

if __name__ == "__main__":
    main()