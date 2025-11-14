"""
Create Dataset-Specific Analysis Charts
Use runs CSV data for analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_parser import RAGDataParser
from visualization_utils import RAGVisualizer

def create_dataset_performance_analysis(parser, visualizer, output_dir):
    all_data = parser.load_all_runs()
    run_labels = parser.get_run_labels()
    datasets = ['HotpotQA', 'MS MARCO', 'Natural Questions', 'TriviaQA']
    dataset_keys = ['hotpotqa', 'ms_marco', 'natural_questions', 'triviaqa']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dataset-Specific Performance Analysis (CSV Real Data)', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    for i, (dataset_name, dataset_key) in enumerate(zip(datasets, dataset_keys)):
        ax = axes[i]
        exp_names = []
        accuracies = []
        faithfulness = []
        recalls = []
        for run_name, data in all_data.items():
            if dataset_key in data['datasets']:
                exp_names.append(run_labels.get(run_name, run_name).replace(' ', '\n'))
                m = data['datasets'][dataset_key]
                accuracies.append(m['accuracy'] * 100)
                faithfulness.append(m['faithfulness'])
                recalls.append(m['context_recall'] * 100)
        x = np.arange(len(exp_names))
        width = 0.25
        bars1 = ax.bar(x - width, accuracies, width, label='Accuracy (%)', color=visualizer.color_schemes['datasets'][i], alpha=0.8)
        bars2 = ax.bar(x, faithfulness, width, label='Faithfulness', color=visualizer.color_schemes['datasets'][i], alpha=0.6)
        bars3 = ax.bar(x + width, [r/100 for r in recalls], width, label='Context Recall', color=visualizer.color_schemes['datasets'][i], alpha=0.4)
        for bar, value in zip(bars1, accuracies):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 1, f'{value:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_title(f'{dataset_name}', fontweight='bold', fontsize=12)
        ax.set_ylabel('Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(max(accuracies), max(recalls)) + 5)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'dataset_specific_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Dataset-specific analysis saved to: {output_path}")
    return fig

def create_best_worst_comparison(parser, visualizer, output_dir):
    all_data = parser.load_all_runs()
    run_labels = parser.get_run_labels()
    items = [(run_labels.get(k, k), v['overall']['overall_accuracy'] * 100, k) for k, v in all_data.items()]
    best_name, best_acc, best_key = max(items, key=lambda x: x[1])
    worst_name, worst_acc, worst_key = min(items, key=lambda x: x[1])
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Best vs Worst System Comparison\n{best_name} vs {worst_name}', fontsize=16, fontweight='bold')
    metrics = ['Overall Accuracy', 'Faithfulness', 'Context Recall']
    best_data = all_data[best_key]['overall']
    worst_data = all_data[worst_key]['overall']
    best_values = [best_data['overall_accuracy'] * 100, best_data['overall_faithfulness'], best_data['overall_context_recall'] * 100]
    worst_values = [worst_data['overall_accuracy'] * 100, worst_data['overall_faithfulness'], worst_data['overall_context_recall'] * 100]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, best_values, width, label=f'Best: {best_name}', 
                   color='green', alpha=0.7)
    bars2 = ax1.bar(x + width/2, worst_values, width, label=f'Worst: {worst_name}', 
                   color='red', alpha=0.7)
    
    # Add value labels
    for bars, values in [(bars1, best_values), (bars2, worst_values)]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if 'Chunks' in metrics[list(bars).index(bar)]:
                label = f'{value:.1f}'
            elif 'F1' in metrics[list(bars).index(bar)]:
                label = f'{value:.2f}'
            else:
                label = f'{value:.1f}%'
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_title('Overall Performance Metrics', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Dataset-specific accuracy comparison
    datasets = ['HotpotQA', 'MS MARCO', 'Natural Questions', 'TriviaQA']
    dataset_keys = ['hotpotqa', 'ms_marco', 'natural_questions', 'triviaqa']
    
    best_dataset_acc = [all_data[best_key]['datasets'][dk]['accuracy'] * 100 for dk in dataset_keys]
    worst_dataset_acc = [all_data[worst_key]['datasets'][dk]['accuracy'] * 100 for dk in dataset_keys]
    
    x2 = np.arange(len(datasets))
    bars3 = ax2.bar(x2 - width/2, best_dataset_acc, width, label=f'Best: {best_name}', 
                   color='green', alpha=0.7)
    bars4 = ax2.bar(x2 + width/2, worst_dataset_acc, width, label=f'Worst: {worst_name}', 
                   color='red', alpha=0.7)
    
    # Add value labels
    for bars, values in [(bars3, best_dataset_acc), (bars4, worst_dataset_acc)]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_title('Dataset-Specific Accuracy', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(datasets, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance gap analysis
    gaps = [best - worst for best, worst in zip(best_dataset_acc, worst_dataset_acc)]
    
    colors_gap = ['green' if gap > 0 else 'red' for gap in gaps]
    bars5 = ax3.bar(datasets, gaps, color=colors_gap, alpha=0.7)
    
    for bar, gap in zip(bars5, gaps):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1),
               f'{gap:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
               fontsize=10, fontweight='bold')
    
    ax3.set_title('Performance Gap (Best - Worst)', fontweight='bold')
    ax3.set_ylabel('Accuracy Difference (%)')
    ax3.set_xticklabels(datasets, rotation=45, ha='right')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary insights
    ax4.axis('off')
    
    avg_gap = np.mean(gaps)
    max_gap = max(gaps)
    min_gap = min(gaps)
    
    insights_text = f"""
    Performance Analysis Summary:
    
    Best System: {best_name}
    • Overall Accuracy: {best_acc:.1f}%
    • Strongest Dataset: {datasets[best_dataset_acc.index(max(best_dataset_acc))]} ({max(best_dataset_acc):.1f}%)
    • Weakest Dataset: {datasets[best_dataset_acc.index(min(best_dataset_acc))]} ({min(best_dataset_acc):.1f}%)
    
    Worst System: {worst_name}
    • Overall Accuracy: {worst_acc:.1f}%
    • Strongest Dataset: {datasets[worst_dataset_acc.index(max(worst_dataset_acc))]} ({max(worst_dataset_acc):.1f}%)
    • Weakest Dataset: {datasets[worst_dataset_acc.index(min(worst_dataset_acc))]} ({min(worst_dataset_acc):.1f}%)
    
    Performance Gaps:
    • Average Gap: {avg_gap:.1f}%
    • Largest Gap: {max_gap:.1f}% ({datasets[gaps.index(max_gap)]})
    • Smallest Gap: {min_gap:.1f}% ({datasets[gaps.index(min_gap)]})
    
    Key Insight:
    {best_name} consistently outperforms 
    {worst_name} across all datasets, with the 
    largest improvement on {datasets[gaps.index(max_gap)]}.
    """
    
    ax4.text(0.1, 0.9, insights_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F9FA', edgecolor='#DEE2E6'))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'best_worst_system_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Best vs worst comparison saved to: {output_path}")
    
    return fig

def create_system_ranking_analysis(parser, visualizer, output_dir):
    all_data = parser.load_all_runs()
    run_labels = parser.get_run_labels()
    systems_data = {}
    for run_name, data in all_data.items():
        o = data['overall']
        systems_data[run_labels.get(run_name, run_name)] = {
            'accuracy': o['overall_accuracy'] * 100,
            'faithfulness': o['overall_faithfulness'],
            'retrieval_rate': o['overall_context_recall'] * 100
        }
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('System Ranking Analysis (CSV Real Data)', fontsize=16, fontweight='bold')
    metrics = ['accuracy', 'faithfulness', 'retrieval_rate']
    titles = ['Overall Accuracy (%)', 'Faithfulness', 'Context Recall (%)']
    axes = [ax1, ax2, ax3]
    for metric, title, ax in zip(metrics, titles, axes):
        sorted_systems = sorted(systems_data.items(), key=lambda x: x[1][metric], reverse=True)
        names = [s[0] for s in sorted_systems]
        values = [s[1][metric] for s in sorted_systems]
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
        bars = ax.barh(names, values, color=colors, alpha=0.8, edgecolor='black')
        for bar, value in zip(bars, values):
            w = bar.get_width()
            label = f'{value:.2f}' if metric == 'faithfulness' else f'{value:.1f}%'
            ax.text(w + w*0.01, bar.get_y() + bar.get_height()/2, label, ha='left', va='center', fontsize=10, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Performance')
        ax.grid(True, alpha=0.3)
        for j, bar in enumerate(bars):
            rank = j + 1
            ax.text(0.02, bar.get_y() + bar.get_height()/2, f'#{rank}', ha='left', va='center', fontsize=12, fontweight='bold', color='white', bbox=dict(boxstyle="circle,pad=0.1", facecolor='black', alpha=0.7))
    ax4.axis('off')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'system_ranking_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"System ranking analysis saved to: {output_path}")
    return fig

def main():
    """Main function to generate dataset analysis charts"""
    
    # Initialize parser and visualizer
    parser = RAGDataParser()
    visualizer = RAGVisualizer()
    output_dir = "/home/pushihao/RAG/Reports/docs/pics"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating dataset-specific analysis charts...")
    
    # 1. Dataset performance analysis
    print("\n1. Creating dataset-specific performance analysis...")
    fig1 = create_dataset_performance_analysis(parser, visualizer, output_dir)
    if fig1:
        plt.close(fig1)
    
    # 2. Best vs worst comparison
    print("\n2. Creating best vs worst system comparison...")
    fig2 = create_best_worst_comparison(parser, visualizer, output_dir)
    if fig2:
        plt.close(fig2)
    
    # 3. System ranking analysis
    print("\n3. Creating system ranking analysis...")
    fig3 = create_system_ranking_analysis(parser, visualizer, output_dir)
    if fig3:
        plt.close(fig3)
    
    print("\nAll dataset analysis charts created successfully!")

if __name__ == "__main__":
    main()