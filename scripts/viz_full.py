import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Visualize Full Comparison Results')
    parser.add_argument('--datasets', type=str, nargs='+', required=True, 
                       help='Dataset names (e.g., SMD MSL SWaT)')
    args = parser.parse_args()
    
    datasets_str = '_'.join(args.datasets)
    csv_path = f"results/csv/{datasets_str}_full_comparison.csv"
    overall_summary_path = f"results/csv/{datasets_str}_overall_summary.csv"
    per_dataset_summary_path = f"results/csv/{datasets_str}_per_dataset_summary.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        print(f"Run: python -m scripts.full_comparison --datasets {' '.join(args.datasets)}")
        return
    
    # Load data
    df = pd.read_csv(csv_path)
    overall_summary = pd.read_csv(overall_summary_path)
    per_dataset_summary = pd.read_csv(per_dataset_summary_path, index_col=[0, 1])
    
    # Sort configs for consistent ordering
    config_order = [
        'TranAD_Scratch',
        'Omni_Periodic', 
        'Omni_Entropy',
        'Omni_Causal',
        'Omni_Entropy_Causal'
    ]
    
    overall_summary['Config'] = pd.Categorical(overall_summary['Config'], categories=config_order, ordered=True)
    overall_summary = overall_summary.sort_values('Config')
    
    # Setup style
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(18, 12))
    
    # ==========================================
    # PLOT 1: F1 SCORE COMPARISON (BAR)
    # ==========================================
    ax1 = plt.subplot(2, 3, 1)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax1.bar(range(len(overall_summary)), overall_summary['Avg_F1'], color=colors, alpha=0.8, edgecolor='black')
    
    ax1.set_xticks(range(len(overall_summary)))
    ax1.set_xticklabels([c.replace('_', '\n') for c in overall_summary['Config']], fontsize=9)
    ax1.set_ylabel('Average F1 Score (PA)', fontsize=11, fontweight='bold')
    datasets_title = ' + '.join(args.datasets)
    ax1.set_title(f'{datasets_title}: F1 Score Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, overall_summary['Avg_F1'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # ==========================================
    # PLOT 2: TRAINING TIME COMPARISON (BAR)
    # ==========================================
    ax2 = plt.subplot(2, 3, 2)
    bars = ax2.bar(range(len(overall_summary)), overall_summary['Avg_Time'], color=colors, alpha=0.8, edgecolor='black')
    
    ax2.set_xticks(range(len(overall_summary)))
    ax2.set_xticklabels([c.replace('_', '\n') for c in overall_summary['Config']], fontsize=9)
    ax2.set_ylabel('Average Training Time (s)', fontsize=11, fontweight='bold')
    ax2.set_title(f'{datasets_title}: Training Speed Comparison', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, overall_summary['Avg_Time']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(overall_summary['Avg_Time'])*0.02, 
                f'{val:.2f}s', ha='center', fontsize=10, fontweight='bold')
    
    # ==========================================
    # PLOT 3: F1 vs TIME SCATTER
    # ==========================================
    ax3 = plt.subplot(2, 3, 3)
    for i, (config, color) in enumerate(zip(config_order, colors)):
        mask = overall_summary['Config'] == config
        ax3.scatter(overall_summary[mask]['Avg_Time'], overall_summary[mask]['Avg_F1'], 
                   s=300, color=color, alpha=0.7, edgecolors='black', linewidth=2,
                   label=config.replace('_', ' '))
    
    ax3.set_xlabel('Training Time (s)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax3.set_title('Accuracy vs Speed Trade-off', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=8, loc='best', framealpha=0.9)
    ax3.grid(alpha=0.3)
    
    # ==========================================
    # PLOT 4: PER-DATASET COMPARISON TABLE
    # ==========================================
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    
    # Create pivot table: Dataset vs Config (from main dataframe)
    pivot_dataset = df.groupby(['Dataset', 'Config'])['F1'].mean().unstack('Config')
    pivot_dataset = pivot_dataset[config_order]  # Reorder columns
    
    # Create table
    table_data = []
    table_data.append(['Dataset'] + [c.replace('_', '\n') for c in config_order])
    
    for dataset in pivot_dataset.index:
        row = [dataset]
        for config in config_order:
            val = pivot_dataset.loc[dataset, config]
            row.append(f'{val:.3f}')
        table_data.append(row)
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15] + [0.17]*len(config_order))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(len(config_order) + 1):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white')
    
    # Color code cells by performance
    for i in range(1, len(table_data)):
        table[(i, 0)].set_facecolor('#ecf0f1')
        table[(i, 0)].set_text_props(weight='bold')
        
        for j in range(1, len(config_order) + 1):
            val = float(table_data[i][j])
            if val >= 0.9:
                color = '#2ecc71'  # Green
            elif val >= 0.8:
                color = '#f39c12'  # Orange
            else:
                color = '#e74c3c'  # Red
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_text_props(color='white', weight='bold')
    
    ax4.set_title('Per-Dataset F1 Scores', fontsize=13, fontweight='bold', pad=20)
    
    # ==========================================
    # PLOT 5: F1 DISTRIBUTION (VIOLIN)
    # ==========================================
    ax5 = plt.subplot(2, 3, 5)
    df['Config'] = pd.Categorical(df['Config'], categories=config_order, ordered=True)
    
    violin = ax5.violinplot([df[df['Config'] == c]['F1'].values for c in config_order],
                            positions=range(len(config_order)), widths=0.7,
                            showmeans=True, showmedians=True)
    
    for i, pc in enumerate(violin['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax5.set_xticks(range(len(config_order)))
    ax5.set_xticklabels([c.replace('_', '\n') for c in config_order], fontsize=9)
    ax5.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax5.set_title('F1 Score Distribution', fontsize=13, fontweight='bold')
    ax5.set_ylim(0, 1.1)
    ax5.grid(axis='y', alpha=0.3)
    
    # ==========================================
    # PLOT 6: SPEEDUP COMPARISON
    # ==========================================
    ax6 = plt.subplot(2, 3, 6)
    baseline_time = overall_summary[overall_summary['Config'] == 'TranAD_Scratch']['Avg_Time'].values[0]
    speedups = baseline_time / overall_summary['Avg_Time']
    
    bars = ax6.bar(range(len(overall_summary)), speedups, color=colors, alpha=0.8, edgecolor='black')
    ax6.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Baseline (1x)')
    
    ax6.set_xticks(range(len(overall_summary)))
    ax6.set_xticklabels([c.replace('_', '\n') for c in overall_summary['Config']], fontsize=9)
    ax6.set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
    ax6.set_title(f'Training Speedup vs TranAD Scratch', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, speedups):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val:.1f}x', ha='center', fontsize=10, fontweight='bold')
    
    # ==========================================
    # FINAL LAYOUT & SAVE
    # ==========================================
    plt.suptitle(f'OmniTransfer Full Comparison: {datasets_title}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    datasets_str = '_'.join(args.datasets)
    output_path = f"results/image/{datasets_str}_full_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: {output_path}")
    
    # ==========================================
    # PRINT KEY INSIGHTS
    # ==========================================
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")
    
    best_f1_config = overall_summary.loc[overall_summary['Avg_F1'].idxmax(), 'Config']
    best_f1 = overall_summary['Avg_F1'].max()
    
    fastest_config = overall_summary.loc[overall_summary['Avg_Time'].idxmin(), 'Config']
    fastest_time = overall_summary['Avg_Time'].min()
    
    print(f"Best F1 Score: {best_f1_config} ({best_f1:.4f})")
    print(f"Fastest Training: {fastest_config} ({fastest_time:.2f}s)")
    print(f"Baseline Time: {baseline_time:.2f}s")
    print(f"Max Speedup: {speedups.max():.1f}x")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
