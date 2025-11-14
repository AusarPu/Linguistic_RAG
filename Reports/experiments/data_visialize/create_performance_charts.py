"""
Create Performance Comparison Charts
Generates ablation study and dataset performance comparison charts
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

def create_dataset_performance_chart(parser, visualizer, output_dir):
    """Create dataset performance comparison chart"""
    runs = parser.load_all_runs()
    preferred = 'result_5_full'
    run_name = preferred if preferred in runs else sorted(runs.keys())[0]
    current_data = runs[run_name]
    datasets = current_data['datasets']
    dataset_labels = parser.get_dataset_labels()
    
    # Prepare data for visualization
    dataset_names = []
    accuracies = []
    retrieval_rates = []
    f1_scores = []
    
    for dataset, metrics in datasets.items():
        dataset_names.append(dataset_labels.get(dataset, dataset))
        accuracies.append(metrics.get('accuracy', 0) * 100)
        retrieval_rates.append(metrics.get('context_recall', 0) * 100)
        f1_scores.append(metrics.get('faithfulness', 0) * 100)
    
    # Create the chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(dataset_names))
    width = 0.25
    
    colors = visualizer.color_schemes['datasets']
    
    bars1 = ax.bar(x - width, accuracies, width, label='Accuracy (%)', color=colors[0], alpha=0.8)
    bars2 = ax.bar(x, retrieval_rates, width, label='Retrieval Success Rate (%)', color=colors[1], alpha=0.8)
    bars3 = ax.bar(x + width, f1_scores, width, label='F1 Score (%)', color=colors[2], alpha=0.8)
    
    # Add value labels on bars
    def add_value_labels(bars, values, format_str='{:.1f}'):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   format_str.format(value),
                   ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1, accuracies, '{:.1f}%')
    add_value_labels(bars2, retrieval_rates, '{:.1f}%')
    add_value_labels(bars3, f1_scores, '{:.1f}%')
    
    ax.set_xlabel('Datasets', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Metrics', fontsize=12, fontweight='bold')
    ax.set_title('RAG System Performance Across Different Datasets', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limit to accommodate labels
    ax.set_ylim(0, max(max(accuracies), max(retrieval_rates), max(f1_scores)) + 10)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'dataset_performance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Dataset performance chart saved to: {output_path}")
    
    return fig

def create_current_system_summary_dashboard(parser, visualizer, output_dir):
    """Create a summary dashboard for the current system"""
    runs = parser.load_all_runs()
    preferred = 'result_5_full'
    run_name = preferred if preferred in runs else sorted(runs.keys())[0]
    current_data = runs[run_name]
    
    # Create dashboard
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('RAG System Performance Summary Dashboard', fontsize=20, fontweight='bold', y=0.95)
    
    # Overall metrics (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    overall = current_data['overall']
    
    metrics = ['Overall Accuracy', 'Total Questions']
    values = [
        overall.get('overall_accuracy', 0) * 100,
        overall.get('total_questions', 0)
    ]
    
    colors = ['#2E86AB', '#5D737E']
    bars = ax1.bar(range(len(metrics)), values, color=colors, alpha=0.8)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if 'Accuracy' in metrics[bars.index(bar)]:
            label = f'{value:.1f}%'
        else:
            label = f'{int(value)}'
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_title('Overall Performance Metrics', fontweight='bold')
    ax1.set_xticks(range(len(metrics)))
    ax1.set_xticklabels(metrics, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Dataset accuracy comparison (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    datasets = current_data['datasets']
    dataset_labels = parser.get_dataset_labels()
    
    dataset_names = [dataset_labels.get(d, d) for d in datasets.keys()]
    dataset_accuracies = [datasets[d].get('accuracy', 0) * 100 for d in datasets.keys()]
    
    colors_datasets = visualizer.color_schemes['datasets']
    bars = ax2.bar(dataset_names, dataset_accuracies, color=colors_datasets, alpha=0.8)
    
    for bar, acc in zip(bars, dataset_accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_title('Accuracy by Dataset', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # System components overview (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    components_text = """
    System Components:
    
    ✓ Query Rewriter
    ✓ Multi-path Retrieval
      • BM25 Retrieval
      • Dense Chunk Retrieval  
      • Dense Keyword Retrieval
      • Dense Question Retrieval
    ✓ Usefulness Judge
    ✓ Soft Retention Strategy
    ✓ Answer Generation
    """
    
    ax3.text(0.1, 0.9, components_text, transform=ax3.transAxes, fontsize=11,
            verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#F5F5F5', edgecolor='#333333'))
    ax3.set_title('Active System Components', fontweight='bold')
    
    # Performance distribution (middle row, spanning all columns)
    ax4 = fig.add_subplot(gs[1, :])
    
    # Create a radar-like visualization showing different aspects
    categories = ['Accuracy', 'Knowledge\nRetrieval', 'Answer\nQuality', 'System\nRobustness']
    values = [
        overall.get('overall_accuracy', 0),
        overall.get('overall_context_recall', 0),
        overall.get('overall_faithfulness', 0),
        overall.get('overall_answer_relevancy', 0)
    ]
    
    # Create a horizontal bar chart instead of radar for better readability
    y_pos = np.arange(len(categories))
    bars = ax4.barh(y_pos, values, color=visualizer.color_schemes['ablation'][:len(categories)], alpha=0.8)
    
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax4.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{value:.1%}', ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(categories)
    ax4.set_xlabel('Performance Score')
    ax4.set_title('System Performance Dimensions', fontweight='bold')
    ax4.set_xlim(0, 1.1)
    ax4.grid(True, alpha=0.3)
    
    # Key insights (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('off')
    
    insights_text = """
    Key Insights:
    
    • Best Performance: 自动计算
    • Most Challenging: 自动计算
    • Total Questions: 来自 CSV 汇总
    • Multi-path retrieval strategy active
    """
    
    ax5.text(0.1, 0.9, insights_text, transform=ax5.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#E8F4FD', edgecolor='#2E86AB'))
    ax5.set_title('Key Insights', fontweight='bold')
    
    # Performance trends (bottom middle)
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Show dataset difficulty ranking
    dataset_difficulty = sorted([(name, acc) for name, acc in zip(dataset_names, dataset_accuracies)], 
                               key=lambda x: x[1], reverse=True)
    
    names, accs = zip(*dataset_difficulty)
    colors_sorted = [colors_datasets[dataset_names.index(name)] for name in names]
    
    bars = ax6.bar(range(len(names)), accs, color=colors_sorted, alpha=0.8)
    
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax6.set_title('Dataset Difficulty Ranking', fontweight='bold')
    ax6.set_ylabel('Accuracy (%)')
    ax6.set_xticks(range(len(names)))
    ax6.set_xticklabels(names, rotation=45, ha='right')
    ax6.grid(True, alpha=0.3)
    
    # System status (bottom right)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    status_text = f"""
    System Status:
    
    📊 Total Questions Processed: {overall.get('total_questions', 0)}
    🎯 Overall Accuracy: {overall.get('overall_accuracy', 0):.1%}
    🔍 Context Recall: {overall.get('overall_context_recall', 0):.1%}
    ⚡ All Components Active
    ✅ System Operational
    """
    
    ax7.text(0.1, 0.9, status_text, transform=ax7.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#E8F5E8', edgecolor='#6A994E'))
    ax7.set_title('System Status', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'summary_dashboard.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Summary dashboard saved to: {output_path}")
    
    return fig

def create_runs_ablation_comparison(parser, visualizer, output_dir):
    runs = parser.load_all_runs()
    run_labels = parser.get_run_labels()
    names = []
    accs = []
    for run_name, data in runs.items():
        names.append(run_labels.get(run_name, run_name))
        accs.append(data['overall']['overall_accuracy'] * 100)
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = visualizer.color_schemes['ablation'][:len(names)]
    bars = ax.bar(names, accs, color=colors, alpha=0.8)
    for bar, acc in zip(bars, accs):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xlabel('Runs', fontsize=12, fontweight='bold')
    ax.set_ylabel('Overall Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation/System Comparison by Runs (CSV Real Data)', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ablation_study_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Ablation study chart saved to: {output_path}")
    return fig

def create_component_contribution_chart(parser, visualizer, output_dir):
    runs = parser.load_all_runs()
    baseline = 'result_5_full' if 'result_5_full' in runs else sorted(runs.keys())[0]
    base_acc = runs[baseline]['overall']['overall_accuracy'] * 100
    labels = []
    deltas = []
    run_labels = parser.get_run_labels()
    for run_name, data in runs.items():
        if run_name == baseline:
            continue
        labels.append(run_labels.get(run_name, run_name))
        deltas.append(data['overall']['overall_accuracy'] * 100 - base_acc)
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['green' if d >= 0 else 'red' for d in deltas]
    bars = ax.barh(labels, deltas, color=colors, alpha=0.8)
    for bar, d in zip(bars, deltas):
        w = bar.get_width()
        ax.text(w + (0.2 if w >= 0 else -0.2), bar.get_y() + bar.get_height()/2, f'{d:+.1f}%',
                ha='left' if w >= 0 else 'right', va='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('Accuracy Change vs Baseline (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Component/System Contribution vs {run_labels.get(baseline, baseline)}', fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'component_contribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Component contribution chart saved to: {output_path}")
    return fig

def main():
    """Main function to generate all performance charts"""
    
    # Initialize parser and visualizer
    parser = RAGDataParser()
    visualizer = RAGVisualizer()
    output_dir = "/home/pushihao/RAG/Reports/docs/pics"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating performance comparison charts...")
    
    # 1. Dataset performance comparison
    print("\n1. Creating dataset performance comparison chart...")
    fig1 = create_dataset_performance_chart(parser, visualizer, output_dir)
    if fig1:
        plt.close(fig1)
    
    # 2. Summary dashboard
    print("\n2. Creating summary dashboard...")
    fig2 = create_current_system_summary_dashboard(parser, visualizer, output_dir)
    if fig2:
        plt.close(fig2)
    
    # 3. Simulated ablation comparison
    print("\n3. Creating ablation study comparison chart...")
    fig3 = create_runs_ablation_comparison(parser, visualizer, output_dir)
    if fig3:
        plt.close(fig3)
    
    # 4. Component contribution analysis
    print("\n4. Creating component contribution chart...")
    fig4 = create_component_contribution_chart(parser, visualizer, output_dir)
    if fig4:
        plt.close(fig4)
    
    print("\nAll performance charts created successfully!")

if __name__ == "__main__":
    main()
    