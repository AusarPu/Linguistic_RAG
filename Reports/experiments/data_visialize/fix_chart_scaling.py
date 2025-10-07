"""
Fix Chart Scaling Issues
Regenerate charts with proper scaling for F1 scores and other metrics
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

def create_fixed_ablation_comparison(parser, visualizer, output_dir):
    """Create ablation study comparison with fixed scaling"""
    
    # Load all experimental data
    all_data = parser.load_all_experiments()
    
    # Define experiment mapping and their descriptions
    experiment_mapping = {
        'final_result_2_preprocess_think': 'Complete System\n(Original)',
        'final_result_8_preprocess_think_usefulness_v2': 'Complete System\n(Optimized)',
        'final_result_3_preprocess_think_no_rewriter': 'No Query\nRewriter',
        'final_result_4_preprocess_think_no_usefulness': 'No Usefulness\nJudge',
        'final_result_5_preprocess_think_no_dense_chunks': 'No Dense Chunk\nRetrieval',
        'final_result_6_preprocess_think_no_dense_keywords': 'No Dense Keyword\nRetrieval',
        'final_result_7_preprocess_think_no_dense_questions': 'No Dense Question\nRetrieval'
    }
    
    # Extract data for visualization
    experiment_names = []
    accuracies = []
    f1_scores = []
    retrieval_rates = []
    
    for exp_key in experiment_mapping.keys():
        if exp_key in all_data and 'error' not in all_data[exp_key]:
            summary = all_data[exp_key].get('summary', {})
            if 'error' not in summary:
                overall = summary.get('overall', {})
                experiment_names.append(experiment_mapping[exp_key])
                accuracies.append(overall.get('overall_accuracy', 0) * 100)
                f1_scores.append(overall.get('overall_f1_score', 0) * 100)  # Convert to percentage
                retrieval_rates.append(overall.get('overall_retrieval_success_rate', 0) * 100)
    
    # Create the chart with better scaling
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Chart 1: All metrics comparison with proper scaling
    x = np.arange(len(experiment_names))
    width = 0.25
    colors = visualizer.color_schemes['ablation'][:3]
    
    bars1 = ax1.bar(x - width, accuracies, width, label='Accuracy (%)', 
                   color=colors[0], alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x, f1_scores, width, label='F1 Score (%)', 
                   color=colors[1], alpha=0.8, edgecolor='black', linewidth=1)
    bars3 = ax1.bar(x + width, retrieval_rates, width, label='Retrieval Success Rate (%)', 
                   color=colors[2], alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bars, values, format_str in [(bars1, accuracies, '{:.1f}%'), 
                                     (bars2, f1_scores, '{:.1f}%'), 
                                     (bars3, retrieval_rates, '{:.1f}%')]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   format_str.format(value), ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
    
    # Highlight the best performing systems
    best_acc_idx = accuracies.index(max(accuracies))
    bars1[best_acc_idx].set_edgecolor('red')
    bars1[best_acc_idx].set_linewidth(3)
    
    ax1.set_xlabel('System Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Performance Score (%)', fontsize=12, fontweight='bold')
    ax1.set_title('RAG System Ablation Study: Performance Comparison\n(All Metrics Scaled to Percentages for Better Visibility)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(experiment_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)  # Set consistent scale
    
    # Chart 2: Focus on accuracy differences with zoomed scale
    bars4 = ax2.bar(experiment_names, accuracies, color=colors[0], alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar, acc in zip(bars4, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
               f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Highlight best system
    bars4[best_acc_idx].set_edgecolor('red')
    bars4[best_acc_idx].set_linewidth(3)
    
    ax2.set_xlabel('System Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Overall Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Detailed Accuracy Comparison (Zoomed Scale)', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(experiment_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Set y-axis to focus on the actual range of values
    min_acc = min(accuracies)
    max_acc = max(accuracies)
    margin = (max_acc - min_acc) * 0.1
    ax2.set_ylim(min_acc - margin, max_acc + margin)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'real_ablation_study_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Fixed ablation study chart saved to: {output_path}")
    
    return fig

def create_fixed_dataset_analysis(parser, visualizer, output_dir):
    """Create dataset analysis with fixed F1 score scaling"""
    
    all_data = parser.load_all_experiments()
    
    # Define experiment labels
    experiment_labels = {
        'final_result_2_preprocess_think': 'Complete (Orig)',
        'final_result_8_preprocess_think_usefulness_v2': 'Complete (Opt)',
        'final_result_3_preprocess_think_no_rewriter': 'No Rewriter',
        'final_result_4_preprocess_think_no_usefulness': 'No Usefulness',
        'final_result_5_preprocess_think_no_dense_chunks': 'No Dense Chunk',
        'final_result_6_preprocess_think_no_dense_keywords': 'No Dense Keyword',
        'final_result_7_preprocess_think_no_dense_questions': 'No Dense Question'
    }
    
    datasets = ['HotpotQA', 'MS MARCO', 'Natural Questions', 'TriviaQA']
    dataset_keys = ['hotpotqa', 'ms_marco', 'natural_questions', 'triviaqa']
    
    # Create a 2x2 subplot for each dataset
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Dataset-Specific Performance Analysis\n(Fixed Scaling for Better Visibility)', 
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
                    exp_names.append(exp_label)
                    accuracies.append(dataset_data['accuracy'] * 100)
                    f1_scores.append(dataset_data['f1_score'] * 100)  # Convert to percentage
                    retrieval_rates.append(dataset_data['retrieval_success_rate'] * 100)
        
        # Create grouped bar chart with consistent scaling
        x = np.arange(len(exp_names))
        width = 0.25
        
        bars1 = ax.bar(x - width, accuracies, width, label='Accuracy (%)', 
                      color=visualizer.color_schemes['datasets'][i], alpha=0.8)
        bars2 = ax.bar(x, f1_scores, width, label='F1 Score (%)', 
                      color=visualizer.color_schemes['datasets'][i], alpha=0.6)
        bars3 = ax.bar(x + width, retrieval_rates, width, label='Retrieval Rate (%)', 
                      color=visualizer.color_schemes['datasets'][i], alpha=0.4)
        
        # Add value labels on bars (only for accuracy to avoid clutter)
        for bar, value in zip(bars1, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_title(f'{dataset_name}', fontweight='bold', fontsize=12)
        ax.set_ylabel('Performance Score (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)  # Consistent scale for all subplots
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'dataset_specific_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Fixed dataset analysis saved to: {output_path}")
    
    return fig

def create_fixed_version_comparison(parser, visualizer, output_dir):
    """Create version comparison with better metric scaling"""
    
    all_data = parser.load_all_experiments()
    
    # Compare final_result_2 (original) vs final_result_8 (optimized)
    original_key = 'final_result_2_preprocess_think'
    optimized_key = 'final_result_8_preprocess_think_usefulness_v2'
    
    if original_key not in all_data or optimized_key not in all_data:
        print("Error: Required experiment data not found for version comparison")
        return None
    
    original_data = all_data[original_key]['summary']
    optimized_data = all_data[optimized_key]['summary']
    
    # Create comparison dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('System Version Comparison: Original vs Optimized\n(Fixed Scaling for Better Visibility)', 
                fontsize=16, fontweight='bold')
    
    # 1. Overall metrics comparison - all converted to percentages
    metrics = ['Accuracy', 'F1 Score', 'Retrieval Rate']
    original_values = [
        original_data['overall']['overall_accuracy'] * 100,
        original_data['overall']['overall_f1_score'] * 100,  # Convert to percentage
        original_data['overall']['overall_retrieval_success_rate'] * 100
    ]
    optimized_values = [
        optimized_data['overall']['overall_accuracy'] * 100,
        optimized_data['overall']['overall_f1_score'] * 100,  # Convert to percentage
        optimized_data['overall']['overall_retrieval_success_rate'] * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, original_values, width, label='Original System', 
                   color='#3498DB', alpha=0.8)
    bars2 = ax1.bar(x + width/2, optimized_values, width, label='Optimized System', 
                   color='#E74C3C', alpha=0.8)
    
    # Add value labels
    for bars, values in [(bars1, original_values), (bars2, optimized_values)]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_title('Overall Performance Metrics (%)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # 2. Dataset-specific accuracy comparison
    datasets = ['HotpotQA', 'MS MARCO', 'Natural Questions', 'TriviaQA']
    dataset_keys = ['hotpotqa', 'ms_marco', 'natural_questions', 'triviaqa']
    
    original_dataset_acc = [original_data['datasets'][key]['accuracy'] * 100 for key in dataset_keys]
    optimized_dataset_acc = [optimized_data['datasets'][key]['accuracy'] * 100 for key in dataset_keys]
    
    x2 = np.arange(len(datasets))
    bars3 = ax2.bar(x2 - width/2, original_dataset_acc, width, label='Original System', 
                   color='#3498DB', alpha=0.8)
    bars4 = ax2.bar(x2 + width/2, optimized_dataset_acc, width, label='Optimized System', 
                   color='#E74C3C', alpha=0.8)
    
    # Add value labels
    for bars, values in [(bars3, original_dataset_acc), (bars4, optimized_dataset_acc)]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_title('Accuracy by Dataset (%)', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(datasets, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Improvement analysis
    improvements = [opt - orig for opt, orig in zip(optimized_values, original_values)]
    improvement_labels = ['Accuracy\nImprovement', 'F1 Score\nImprovement', 'Retrieval Rate\nImprovement']
    
    colors_improvement = ['green' if imp > 0 else 'red' for imp in improvements]
    bars5 = ax3.bar(improvement_labels, improvements, color=colors_improvement, alpha=0.7)
    
    for bar, imp in zip(bars5, improvements):
        height = bar.get_height()
        if abs(height) > 0.01:
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.2 if height > 0 else -0.5),
                   f'+{imp:.1f}%' if imp > 0 else f'{imp:.1f}%', 
                   ha='center', va='bottom' if height > 0 else 'top', 
                   fontsize=11, fontweight='bold')
    
    ax3.set_title('Performance Improvements (%)', fontweight='bold')
    ax3.set_ylabel('Improvement (percentage points)')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax4.axis('off')
    
    summary_text = f"""
    System Comparison Summary:
    
    Original System (final_result_2):
    • Overall Accuracy: {original_data['overall']['overall_accuracy']:.1%}
    • F1 Score: {original_data['overall']['overall_f1_score']:.1%}
    • Retrieval Success: {original_data['overall']['overall_retrieval_success_rate']:.1%}
    • Avg Chunks/Query: {original_data['overall']['avg_chunks_per_question']:.1f}
    
    Optimized System (final_result_8):
    • Overall Accuracy: {optimized_data['overall']['overall_accuracy']:.1%}
    • F1 Score: {optimized_data['overall']['overall_f1_score']:.1%}
    • Retrieval Success: {optimized_data['overall']['overall_retrieval_success_rate']:.1%}
    • Avg Chunks/Query: {optimized_data['overall']['avg_chunks_per_question']:.1f}
    
    Key Improvements:
    • Accuracy: +{(optimized_data['overall']['overall_accuracy'] - original_data['overall']['overall_accuracy']) * 100:.1f} percentage points
    • F1 Score: +{(optimized_data['overall']['overall_f1_score'] - original_data['overall']['overall_f1_score']) * 100:.1f} percentage points
    • Perfect Retrieval Success Rate Achieved
    
    Note: Soft retention strategy and improved 
    usefulness detection prompts implemented
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F9FA', edgecolor='#DEE2E6'))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'version_comparison_dashboard.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Fixed version comparison dashboard saved to: {output_path}")
    
    return fig

def main():
    """Main function to regenerate charts with fixed scaling"""
    
    # Initialize parser and visualizer
    parser = RAGDataParser()
    visualizer = RAGVisualizer()
    output_dir = "/home/pushihao/RAG/Reports/docs/pics"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Regenerating charts with fixed scaling...")
    
    # 1. Fixed ablation comparison
    print("\n1. Creating fixed ablation study comparison...")
    fig1 = create_fixed_ablation_comparison(parser, visualizer, output_dir)
    if fig1:
        plt.close(fig1)
    
    # 2. Fixed dataset analysis
    print("\n2. Creating fixed dataset analysis...")
    fig2 = create_fixed_dataset_analysis(parser, visualizer, output_dir)
    if fig2:
        plt.close(fig2)
    
    # 3. Fixed version comparison
    print("\n3. Creating fixed version comparison...")
    fig3 = create_fixed_version_comparison(parser, visualizer, output_dir)
    if fig3:
        plt.close(fig3)
    
    print("\nAll charts regenerated with fixed scaling!")

if __name__ == "__main__":
    main()