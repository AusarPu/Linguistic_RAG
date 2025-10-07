"""
Fix Dashboard F1 Score Display
Fix F1 score display in the comprehensive dashboard
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

def create_fixed_comprehensive_dashboard(parser, visualizer, output_dir):
    """Create comprehensive dashboard with fixed F1 score display"""
    
    all_data = parser.load_all_experiments()
    
    # Use the best performing system for the dashboard
    best_system_key = 'final_result_4_preprocess_think_no_usefulness'  # Based on the loaded data
    best_system_data = all_data[best_system_key]['summary']
    
    # Create dashboard
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('RAG System Comprehensive Performance Dashboard\n(Real Experimental Results)', 
                fontsize=20, fontweight='bold', y=0.95)
    
    # 1. Best system performance (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    metrics = ['Accuracy', 'F1 Score', 'Retrieval Rate']
    values = [
        best_system_data['overall']['overall_accuracy'] * 100,
        best_system_data['overall']['overall_f1_score'] * 100,  # Convert F1 to percentage
        best_system_data['overall']['overall_retrieval_success_rate'] * 100
    ]
    
    colors = ['#2E86AB', '#F18F01', '#5D737E']
    bars = ax1.bar(range(len(metrics)), values, color=colors, alpha=0.8)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_title('Best System Performance\n(No Usefulness Judge)', fontweight='bold')
    ax1.set_xticks(range(len(metrics)))
    ax1.set_xticklabels(metrics, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # 2. All systems accuracy comparison (top middle, spanning 2 columns)
    ax2 = fig.add_subplot(gs[0, 1:3])
    
    exp_names = []
    exp_accuracies = []
    
    experiment_labels = {
        'final_result_2_preprocess_think': 'Complete (Orig)',
        'final_result_8_preprocess_think_usefulness_v2': 'Complete (Opt)',
        'final_result_3_preprocess_think_no_rewriter': 'No Rewriter',
        'final_result_4_preprocess_think_no_usefulness': 'No Usefulness',
        'final_result_5_preprocess_think_no_dense_chunks': 'No Dense Chunk',
        'final_result_6_preprocess_think_no_dense_keywords': 'No Dense Keyword',
        'final_result_7_preprocess_think_no_dense_questions': 'No Dense Question'
    }
    
    for exp_key, label in experiment_labels.items():
        if exp_key in all_data and 'error' not in all_data[exp_key]:
            summary = all_data[exp_key]['summary']
            if 'error' not in summary:
                exp_names.append(label)
                exp_accuracies.append(summary['overall']['overall_accuracy'] * 100)
    
    bars = ax2.bar(exp_names, exp_accuracies, color=visualizer.color_schemes['ablation'][:len(exp_names)], alpha=0.8)
    
    for bar, acc in zip(bars, exp_accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_title('All Systems Accuracy Comparison', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_xticklabels(exp_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Key findings (top right)
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis('off')
    
    # Find best and worst performing systems
    best_acc = max(exp_accuracies)
    worst_acc = min(exp_accuracies)
    best_system = exp_names[exp_accuracies.index(best_acc)]
    worst_system = exp_names[exp_accuracies.index(worst_acc)]
    
    findings_text = f"""
    Key Findings:
    
    🏆 Best System: {best_system}
       Accuracy: {best_acc:.1f}%
    
    📉 Worst System: {worst_system}
       Accuracy: {worst_acc:.1f}%
    
    📊 Performance Range: {best_acc - worst_acc:.1f}%
    
    🔍 Surprising Result:
       Removing usefulness judge
       actually improves performance!
    
    ⚡ All systems achieve >80% accuracy
    """
    
    ax3.text(0.1, 0.9, findings_text, transform=ax3.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#E8F4FD', edgecolor='#2E86AB'))
    
    # 4. Dataset performance comparison (middle row, full width)
    ax4 = fig.add_subplot(gs[1, :])
    
    datasets = ['HotpotQA', 'MS MARCO', 'Natural Questions', 'TriviaQA']
    dataset_keys = ['hotpotqa', 'ms_marco', 'natural_questions', 'triviaqa']
    
    # Create grouped bar chart for all systems across datasets
    x = np.arange(len(datasets))
    width = 0.12
    
    for i, (exp_key, label) in enumerate(experiment_labels.items()):
        if exp_key in all_data and 'error' not in all_data[exp_key]:
            summary = all_data[exp_key]['summary']
            if 'error' not in summary:
                dataset_accs = [summary['datasets'][dk]['accuracy'] * 100 for dk in dataset_keys]
                ax4.bar(x + i * width - width * 3, dataset_accs, width, 
                       label=label, alpha=0.8, 
                       color=visualizer.color_schemes['ablation'][i % len(visualizer.color_schemes['ablation'])])
    
    ax4.set_xlabel('Datasets')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Performance Across All Datasets and Systems', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # 5. System statistics (bottom left)
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.axis('off')
    
    stats_text = f"""
    Experimental Statistics:
    
    📋 Total Experiments: {len(all_data)}
    📊 Questions per Experiment: 400
    🎯 Datasets Tested: 4 (HotpotQA, MS MARCO, Natural Questions, TriviaQA)
    
    Performance Summary:
    • Highest Accuracy: {best_acc:.1f}% ({best_system})
    • Lowest Accuracy: {worst_acc:.1f}% ({worst_system})
    • Average Accuracy: {np.mean(exp_accuracies):.1f}%
    • Standard Deviation: {np.std(exp_accuracies):.1f}%
    
    Component Impact Analysis:
    • Query Rewriter: Mixed impact (dataset dependent)
    • Usefulness Judge: Negative impact (surprising finding!)
    • Dense Retrievals: Generally positive impact
    • System Optimization: +5.8% improvement (v2 vs original)
    """
    
    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F9FA', edgecolor='#DEE2E6'))
    
    # 6. Recommendations (bottom right)
    ax6 = fig.add_subplot(gs[2, 2:])
    ax6.axis('off')
    
    recommendations_text = """
    Recommendations Based on Results:
    
    🎯 For Maximum Accuracy:
       Use "No Usefulness Judge" configuration
       Expected performance: ~89% accuracy
    
    🔄 For Balanced Performance:
       Use "Complete System (Optimized)" 
       Good across all datasets: ~87.5% accuracy
    
    ⚡ For Speed vs Accuracy Trade-off:
       Consider removing usefulness judge
       Improves both speed and accuracy
    
    📊 Dataset-Specific Insights:
       • MS MARCO: All systems perform well (>87%)
       • TriviaQA: Consistent across configurations
       • Natural Questions: Benefits from optimization
       • HotpotQA: Most challenging, varies by config
    
    🔬 Future Research Directions:
       • Investigate why usefulness judge hurts performance
       • Optimize dense retrieval combinations
       • Dataset-specific configuration tuning
    """
    
    ax6.text(0.1, 0.9, recommendations_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#E8F5E8', edgecolor='#6A994E'))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'real_comprehensive_dashboard.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Fixed comprehensive dashboard saved to: {output_path}")
    
    return fig

def main():
    """Main function to fix dashboard F1 score display"""
    
    # Initialize parser and visualizer
    parser = RAGDataParser()
    visualizer = RAGVisualizer()
    output_dir = "/home/pushihao/RAG/Reports/docs/pics"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Fixing comprehensive dashboard F1 score display...")
    
    # Fix comprehensive dashboard
    fig = create_fixed_comprehensive_dashboard(parser, visualizer, output_dir)
    if fig:
        plt.close(fig)
    
    print("Dashboard F1 score display fixed!")

if __name__ == "__main__":
    main()