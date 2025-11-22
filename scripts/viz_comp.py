import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g. SMD, MSL, SWaT)')
    parser.add_argument('--use_causal', action='store_true', help='Visualize Causal Enhanced results')
    parser.add_argument('--use_entropy', action='store_true', help='Visualize Entropic results')
    args = parser.parse_args()

    # Dynamic File Paths
    tag = "_entropy" if args.use_entropy else "_periodic"
    if args.use_causal: tag += "_causal"
    
    csv_path = f"results/csv/{args.dataset}_comparison{tag}.csv"
    output_img = f"results/image/{args.dataset}_comparison{tag}.png"
    
    omni_label = "OmniTransfer"
    if args.use_entropy: omni_label += " (Entropy"
    else: omni_label += " (Periodic"
    
    if args.use_causal: omni_label += " + Causal)"
    else: omni_label += ")"

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        print(f"Run: python -m scripts.compare --dataset {args.dataset}" + (" --use_entropy" if args.use_entropy else "") + (" --use_causal" if args.use_causal else ""))
        return

    df = pd.read_csv(csv_path)
    print(f"Comparing {omni_label} vs Scratch for {args.dataset}...")
    
    # Melt Data
    df_f1 = df.melt(id_vars=['Machine'], value_vars=['Omni_F1', 'Scratch_F1'], 
                    var_name='Method', value_name='F1 Score')
    df_time = df.melt(id_vars=['Machine'], value_vars=['Omni_Time', 'Scratch_Time'], 
                      var_name='Method', value_name='Time (s)')
    
    # Rename for Legend
    mapper = {'Omni_F1': omni_label, 'Scratch_F1': 'Train from Scratch',
              'Omni_Time': omni_label, 'Scratch_Time': 'Train from Scratch'}
    
    df_f1['Method'] = df_f1['Method'].replace(mapper)
    df_time['Method'] = df_time['Method'].replace(mapper)

    # Plot Setup
    sns.set_theme(style="whitegrid")
    width = max(15, len(df) * 0.45)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, 12))

    # 1. Efficiency (Log Scale)
    sns.barplot(x='Machine', y='Time (s)', hue='Method', data=df_time, ax=ax1, palette=["#2ecc71", "#e74c3c"])
    ax1.set_yscale("log") 
    ax1.set_title(f"Efficiency Comparison: {args.dataset} (Log Scale)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Time (s) - Log Scale")
    ax1.legend(loc='upper right')
    
    # Speedup Text
    avg_omni = df['Omni_Time'].mean()
    avg_scratch = df['Scratch_Time'].mean()
    if avg_omni > 0:
        speedup = avg_scratch / avg_omni
        ax1.text(0.02, 0.9, f"Avg Speedup: {speedup:.1f}x Faster", transform=ax1.transAxes, 
                 fontsize=12, fontweight='bold', color='green', bbox=dict(facecolor='white', alpha=0.8))

    # 2. Accuracy
    sns.barplot(x='Machine', y='F1 Score', hue='Method', data=df_f1, ax=ax2, palette=["#2ecc71", "#e74c3c"])
    ax2.set_title(f"Accuracy Comparison: {omni_label} vs Scratch", fontsize=14, fontweight='bold')
    ax2.set_ylabel("F1 Score (PA)")
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='lower right')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_img, dpi=300)
    print(f"Success! Plot saved to {output_img}")

if __name__ == "__main__":
    main()