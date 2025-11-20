import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    csv_path = "results/csv/target_comparison.csv"
    if not os.path.exists(csv_path):
        print("Error: CSV not found. Run run_target_comparison.py first.")
        return

    df = pd.read_csv(csv_path)
    
    # Melt Data for Seaborn
    df_f1 = df.melt(id_vars=['Machine'], value_vars=['Omni_F1', 'Scratch_F1'], var_name='Method', value_name='F1 Score')
    df_time = df.melt(id_vars=['Machine'], value_vars=['Omni_Time', 'Scratch_Time'], var_name='Method', value_name='Time (s)')
    
    # Rename labels for clarity
    df_f1['Method'] = df_f1['Method'].replace({'Omni_F1': 'OmniTransfer', 'Scratch_F1': 'Train from Scratch'})
    df_time['Method'] = df_time['Method'].replace({'Omni_Time': 'OmniTransfer', 'Scratch_Time': 'Train from Scratch'})

    # Setup Plot
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # ==========================================
    # PLOT 1: EFFICIENCY (Time)
    # ==========================================
    sns.barplot(x='Machine', y='Time (s)', hue='Method', data=df_time, ax=ax1, palette=["#2ecc71", "#e74c3c"])
    
    ax1.set_yscale("log") # Crucial because Scratch is ~100x slower
    ax1.set_title("Efficiency Comparison: Initialization Time (Log Scale)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Time (Seconds) - Log Scale", fontsize=12)
    ax1.legend(loc='upper right')
    
    # Add text annotation for average speedup
    avg_omni = df['Omni_Time'].mean()
    avg_scratch = df['Scratch_Time'].mean()
    speedup = avg_scratch / avg_omni
    ax1.text(0.02, 0.9, f"Avg Speedup: {speedup:.1f}x Faster", transform=ax1.transAxes, 
             fontsize=12, fontweight='bold', color='green', bbox=dict(facecolor='white', alpha=0.8))

    # ==========================================
    # PLOT 2: ACCURACY (F1 Score)
    # ==========================================
    sns.barplot(x='Machine', y='F1 Score', hue='Method', data=df_f1, ax=ax2, palette=["#2ecc71", "#e74c3c"])
    
    ax2.set_title("Accuracy Comparison: Point-Adjusted F1 Score", fontsize=14, fontweight='bold')
    ax2.set_ylabel("F1 Score", fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='lower right')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig("results/image/target_comparison_plot.png", dpi=300)
    print("Saved comparison plot to target_comparison_plot.png")

if __name__ == "__main__":
    main()