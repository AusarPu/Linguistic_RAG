"""
Create Dataset-Specific Analysis Charts
Generates detailed analysis charts for individual datasets using real experimental data
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import sys

# Add the current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_parser import RAGDataParser
from visualization_utils import RAGVisualizer

def create_dataset_performance_analysis(parser, visualizer, output_dir):
    """Create detailed dataset performance analysis"""
    
    all_data = parser.load_all_experiments()
    
    # Define experiment labels
    experiment_labels = {
        'final_result_2_preprocess_think': 'Complete System (Original)',
        'final_result_8_preprocess_think_usefulness_v2': 'Complete System (Optimized)',
        'final_result_3_preprocess_think_no_rewriter': 'No Query Rewriter',
        'final_result_4_preprocess_think_no_usefulness': 'No Usefulness Judge',
        'final_result_5_preprocess_think_no_dense_chunks': 'No Dense Chunk Retrieval',
        'final_result_6_preprocess_think_no_dense_keywords': 'No Dense Keyword Retrieval',
        'final_result_7_preprocess_think_no_dense_questions': 'No Dense Question Retrieval'
    }
    
    datasets = ['HotpotQA', 'MS MARCO', 'Natural Questions', 'TriviaQA']
    dataset_keys = ['hotpotqa', 'ms_marco', 'natural_questions', 'triviaqa']
    
    # Create a 2x2 subplot for each dataset
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dataset-Specific Performance Analysis\n(Real Experimental Results)', 
                fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, (dataset_name, dataset_key) in enumerate(zip(datasets, dataset_keys)):
        ax = axes[i]
        
        # Collect data for this dataset
        exp_names = []
        accuracies = []
        f1_scores = []
        retrieval_rates = []
        
        for exp_key, exp_label in experiment_labels.items():
            if exp_key in all_data and 'error' not in all_data[exp_key]:
                summary = all_data[exp_key]['summary']
                if 'error' not in summary and dataset_key in summary['datasets']:
                    dataset_data = summary['datasets'][dataset_key]
                    exp_names.append(exp_label.replace(' ', '\n'))  # Break long labels
                    accuracies.append(dataset_data['accuracy'] * 100)
                    f1_scores.append(dataset_data['f1_score'])
                    retrieval_rates.append(dataset_data['retrieval_success_rate'] * 100)
        
        # Create grouped bar chart
        x = np.arange(len(exp_names))
        width = 0.25
        
        bars1 = ax.bar(x - width, accuracies, width, label='Accuracy (%)', 
                      color=visualizer.color_schemes['datasets'][i], alpha=0.8)
        bars2 = ax.bar(x, f1_scores, width, label='F1 Score', 
                      color=visualizer.color_schemes['datasets'][i], alpha=0.6)
        bars3 = ax.bar(x + width, [r/100 for r in retrieval_rates], width, label='Retrieval Rate', 
                      color=visualizer.color_schemes['datasets'][i], alpha=0.4)
        
        # Add value labels on bars
        for bar, value in zip(bars1, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_title(f'{dataset_name}', fontweight='bold', fontsize=12)
        ax.set_ylabel('Performance Score')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(max(accuracies), max([f*100 for f in f1_scores]), max(retrieval_rates)) + 5)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'dataset_specific_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Dataset-specific analysis saved to: {output_path}")
    
    return fig

def create_best_worst_comparison(parser, visualizer, output_dir):
    """Create comparison between best and worst performing systems"""
    
    all_data = parser.load_all_experiments()
    
    # Find best and worst systems based on overall accuracy
    system_accuracies = {}
    
    experiment_labels = {
        'final_result_2_preprocess_think': 'Complete System (Original)',
        'final_result_8_preprocess_think_usefulness_v2': 'Complete System (Optimized)',
        'final_result_3_preprocess_think_no_rewriter': 'No Query Rewriter',
        'final_result_4_preprocess_think_no_usefulness': 'No Usefulness Judge',
        'final_result_5_preprocess_think_no_dense_chunks': 'No Dense Chunk Retrieval',
        'final_result_6_preprocess_think_no_dense_keywords': 'No Dense Keyword Retrieval',
        'final_result_7_preprocess_think_no_dense_questions': 'No Dense Question Retrieval'
    }
    
    for exp_key, exp_label in experiment_labels.items():
        if exp_key in all_data and 'error' not in all_data[exp_key]:
            summary = all_data[exp_key]['summary']
            if 'error' not in summary:
                accuracy = summary['overall']['overall_accuracy'] * 100
                system_accuracies[exp_label] = (accuracy, exp_key)
    
    # Find best and worst
    best_system = max(system_accuracies.items(), key=lambda x: x[1][0])
    worst_system = min(system_accuracies.items(), key=lambda x: x[1][0])
    
    best_name, (best_acc, best_key) = best_system
    worst_name, (worst_acc, worst_key) = worst_system
    
    # Create comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Best vs Worst System Comparison\n{best_name} vs {worst_name}', 
                fontsize=16, fontweight='bold')
    
    # 1. Overall metrics comparison
    metrics = ['Overall Accuracy', 'F1 Score', 'Retrieval Success Rate', 'Avg Chunks/Query']
    
    best_data = all_data[best_key]['summary']['overall']
    worst_data = all_data[worst_key]['summary']['overall']
    
    best_values = [
        best_data['overall_accuracy'] * 100,
        best_data['overall_f1_score'],
        best_data['overall_retrieval_success_rate'] * 100,
        best_data['avg_chunks_per_question']
    ]
    
    worst_values = [
        worst_data['overall_accuracy'] * 100,
        worst_data['overall_f1_score'],
        worst_data['overall_retrieval_success_rate'] * 100,
        worst_data['avg_chunks_per_question']
    ]
    
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
    
    best_dataset_acc = [all_data[best_key]['summary']['datasets'][dk]['accuracy'] * 100 for dk in dataset_keys]
    worst_dataset_acc = [all_data[worst_key]['summary']['datasets'][dk]['accuracy'] * 100 for dk in dataset_keys]
    
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
    """Create system ranking analysis across different metrics"""
    
    all_data = parser.load_all_experiments()
    
    experiment_labels = {
        'final_result_2_preprocess_think': 'Complete (Orig)',
        'final_result_8_preprocess_think_usefulness_v2': 'Complete (Opt)',
        'final_result_3_preprocess_think_no_rewriter': 'No Rewriter',
        'final_result_4_preprocess_think_no_usefulness': 'No Usefulness',
        'final_result_5_preprocess_think_no_dense_chunks': 'No Dense Chunk',
        'final_result_6_preprocess_think_no_dense_keywords': 'No Dense Keyword',
        'final_result_7_preprocess_think_no_dense_questions': 'No Dense Question'
    }
    
    # Collect all metrics
    systems_data = {}
    
    for exp_key, exp_label in experiment_labels.items():
        if exp_key in all_data and 'error' not in all_data[exp_key]:
            summary = all_data[exp_key]['summary']
            if 'error' not in summary:
                overall = summary['overall']
                systems_data[exp_label] = {
                    'accuracy': overall['overall_accuracy'] * 100,
                    'f1_score': overall['overall_f1_score'],
                    'retrieval_rate': overall['overall_retrieval_success_rate'] * 100,
                    'avg_chunks': overall['avg_chunks_per_question']
                }
    
    # Create ranking visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('System Ranking Analysis Across Different Metrics\n(Real Experimental Results)', 
                fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'f1_score', 'retrieval_rate', 'avg_chunks']
    metric_titles = ['Overall Accuracy (%)', 'F1 Score', 'Retrieval Success Rate (%)', 'Avg Chunks per Query']
    axes = [ax1, ax2, ax3, ax4]
    
    for i, (metric, title, ax) in enumerate(zip(metrics, metric_titles, axes)):
        # Sort systems by this metric
        if metric == 'avg_chunks':
            # For chunks, lower is better (more efficient)
            sorted_systems = sorted(systems_data.items(), key=lambda x: x[1][metric])
        else:
            # For other metrics, higher is better
            sorted_systems = sorted(systems_data.items(), key=lambda x: x[1][metric], reverse=True)
        
        system_names = [s[0] for s in sorted_systems]
        values = [s[1][metric] for s in sorted_systems]
        
        # Create horizontal bar chart
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(system_names)))
        if metric == 'avg_chunks':
            colors = colors[::-1]  # Reverse colors for chunks (lower is better)
        
        bars = ax.barh(system_names, values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, values):
            width = bar.get_width()
            if metric == 'f1_score':
                label = f'{value:.2f}'
            elif metric == 'avg_chunks':
                label = f'{value:.1f}'
            else:
                label = f'{value:.1f}%'
            ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                   label, ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Performance Score')
        ax.grid(True, alpha=0.3)
        
        # Add ranking numbers
        for j, (bar, name) in enumerate(zip(bars, system_names)):
            rank = j + 1
            ax.text(0.02, bar.get_y() + bar.get_height()/2,
                   f'#{rank}', ha='left', va='center', 
                   fontsize=12, fontweight='bold', color='white',
                   bbox=dict(boxstyle="circle,pad=0.1", facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the figure
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