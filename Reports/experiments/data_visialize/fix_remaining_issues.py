"""
Fix Remaining Chart Issues
Fix F1 score display and arrow overlapping issues
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# Add the current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_parser import RAGDataParser
from visualization_utils import RAGVisualizer

def create_fixed_best_worst_comparison(parser, visualizer, output_dir):
    """Create best vs worst comparison with fixed F1 score display"""
    
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
    
    # 1. Overall metrics comparison - ALL converted to percentages
    metrics = ['Overall Accuracy', 'F1 Score', 'Retrieval Success Rate', 'Avg Chunks/Query']
    
    best_data = all_data[best_key]['summary']['overall']
    worst_data = all_data[worst_key]['summary']['overall']
    
    best_values = [
        best_data['overall_accuracy'] * 100,
        best_data['overall_f1_score'] * 100,  # Convert F1 to percentage
        best_data['overall_retrieval_success_rate'] * 100,
        best_data['avg_chunks_per_question']  # Keep as is for chunks
    ]
    
    worst_values = [
        worst_data['overall_accuracy'] * 100,
        worst_data['overall_f1_score'] * 100,  # Convert F1 to percentage
        worst_data['overall_retrieval_success_rate'] * 100,
        worst_data['avg_chunks_per_question']  # Keep as is for chunks
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, best_values, width, label=f'Best: {best_name}', 
                   color='green', alpha=0.7)
    bars2 = ax1.bar(x + width/2, worst_values, width, label=f'Worst: {worst_name}', 
                   color='red', alpha=0.7)
    
    # Add value labels with proper formatting
    for bars, values in [(bars1, best_values), (bars2, worst_values)]:
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            if 'Chunks' in metrics[i]:
                label = f'{value:.1f}'
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
    • F1 Score: {best_data['overall_f1_score']*100:.1f}%
    • Strongest Dataset: {datasets[best_dataset_acc.index(max(best_dataset_acc))]} ({max(best_dataset_acc):.1f}%)
    • Weakest Dataset: {datasets[best_dataset_acc.index(min(best_dataset_acc))]} ({min(best_dataset_acc):.1f}%)
    
    Worst System: {worst_name}
    • Overall Accuracy: {worst_acc:.1f}%
    • F1 Score: {worst_data['overall_f1_score']*100:.1f}%
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
    print(f"Fixed best vs worst comparison saved to: {output_path}")
    
    return fig

def create_fixed_rag_architecture_diagram(output_dir="/home/pushihao/RAG/Reports/docs/pics"):
    """Create RAG system architecture diagram with better arrow positioning"""
    
    # Create figure with high DPI for publication quality
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define color scheme
    colors = {
        'input': '#E3F2FD',      # Light blue
        'processing': '#E8F5E8',  # Light green
        'retrieval': '#FFF3E0',   # Light orange
        'judgment': '#FCE4EC',    # Light pink
        'generation': '#F3E5F5',  # Light purple
        'output': '#E0F2F1',      # Light teal
        'border': '#37474F'       # Dark gray
    }
    
    # Component definitions with positions and properties
    components = {
        # Input Layer
        'User Query': {
            'pos': (1, 10.5, 2, 0.8),
            'color': colors['input'],
            'type': 'input'
        },
        
        # Query Processing Layer
        'Query Rewriter': {
            'pos': (1, 9, 2, 0.8),
            'color': colors['processing'],
            'type': 'processing'
        },
        
        # Multi-path Retrieval Layer
        'Multi-path\nRetrieval System': {
            'pos': (5, 9, 3, 0.8),
            'color': colors['retrieval'],
            'type': 'retrieval'
        },
        
        # Retrieval Components
        'BM25\nRetrieval': {
            'pos': (2, 7, 1.5, 0.8),
            'color': colors['retrieval'],
            'type': 'retrieval'
        },
        'Dense Chunk\nRetrieval': {
            'pos': (4, 7, 1.5, 0.8),
            'color': colors['retrieval'],
            'type': 'retrieval'
        },
        'Dense Keyword\nRetrieval': {
            'pos': (6, 7, 1.5, 0.8),
            'color': colors['retrieval'],
            'type': 'retrieval'
        },
        'Dense Question\nRetrieval': {
            'pos': (8, 7, 1.5, 0.8),
            'color': colors['retrieval'],
            'type': 'retrieval'
        },
        
        # Knowledge Base
        'Knowledge Base\n(Indexed Documents)': {
            'pos': (10.5, 6.5, 1.5, 2.5),
            'color': colors['processing'],
            'type': 'storage'
        },
        
        # Judgment Layer
        'Usefulness\nJudge': {
            'pos': (4.5, 5, 2, 0.8),
            'color': colors['judgment'],
            'type': 'judgment'
        },
        
        # Retention Strategy
        'Soft Retention\nStrategy': {
            'pos': (4.5, 3.5, 2, 0.8),
            'color': colors['judgment'],
            'type': 'judgment'
        },
        
        # Generation Layer
        'Answer\nGeneration': {
            'pos': (8, 3.5, 2, 0.8),
            'color': colors['generation'],
            'type': 'generation'
        },
        
        # Output Layer
        'Final Answer': {
            'pos': (8, 1.5, 2, 0.8),
            'color': colors['output'],
            'type': 'output'
        }
    }
    
    # Draw components
    component_centers = {}
    for name, props in components.items():
        x, y, w, h = props['pos']
        
        # Create rounded rectangle
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05",
            facecolor=props['color'],
            edgecolor=colors['border'],
            linewidth=2
        )
        ax.add_patch(rect)
        
        # Add component text
        ax.text(x + w/2, y + h/2, name, 
               ha='center', va='center',
               fontsize=11, fontweight='bold',
               wrap=True)
        
        # Store center for arrow connections
        component_centers[name] = (x + w/2, y + h/2)
    
    # Define connections (arrows) with better positioning to avoid text overlap
    connections = [
        # Main flow - use edge points instead of centers
        (('User Query', 'bottom'), ('Query Rewriter', 'top')),
        (('Query Rewriter', 'right'), ('Multi-path\nRetrieval System', 'left')),
        
        # Retrieval paths - connect to bottom of multi-path system
        (('Multi-path\nRetrieval System', 'bottom-left'), ('BM25\nRetrieval', 'top')),
        (('Multi-path\nRetrieval System', 'bottom'), ('Dense Chunk\nRetrieval', 'top')),
        (('Multi-path\nRetrieval System', 'bottom'), ('Dense Keyword\nRetrieval', 'top')),
        (('Multi-path\nRetrieval System', 'bottom-right'), ('Dense Question\nRetrieval', 'top')),
        
        # Knowledge base connections - connect from right side
        (('Knowledge Base\n(Indexed Documents)', 'left'), ('BM25\nRetrieval', 'right')),
        (('Knowledge Base\n(Indexed Documents)', 'left'), ('Dense Chunk\nRetrieval', 'right')),
        (('Knowledge Base\n(Indexed Documents)', 'left'), ('Dense Keyword\nRetrieval', 'right')),
        (('Knowledge Base\n(Indexed Documents)', 'left'), ('Dense Question\nRetrieval', 'right')),
        
        # Judgment flow - connect from bottom of retrieval components
        (('BM25\nRetrieval', 'bottom'), ('Usefulness\nJudge', 'top-left')),
        (('Dense Chunk\nRetrieval', 'bottom'), ('Usefulness\nJudge', 'top')),
        (('Dense Keyword\nRetrieval', 'bottom'), ('Usefulness\nJudge', 'top')),
        (('Dense Question\nRetrieval', 'bottom'), ('Usefulness\nJudge', 'top-right')),
        
        # Retention and generation
        (('Usefulness\nJudge', 'bottom'), ('Soft Retention\nStrategy', 'top')),
        (('Soft Retention\nStrategy', 'right'), ('Answer\nGeneration', 'left')),
        (('Answer\nGeneration', 'bottom'), ('Final Answer', 'top'))
    ]
    
    # Helper function to get edge points
    def get_edge_point(component_name, edge):
        x, y, w, h = components[component_name]['pos']
        if edge == 'top':
            return (x + w/2, y + h)
        elif edge == 'bottom':
            return (x + w/2, y)
        elif edge == 'left':
            return (x, y + h/2)
        elif edge == 'right':
            return (x + w, y + h/2)
        elif edge == 'top-left':
            return (x + w/4, y + h)
        elif edge == 'top-right':
            return (x + 3*w/4, y + h)
        elif edge == 'bottom-left':
            return (x + w/4, y)
        elif edge == 'bottom-right':
            return (x + 3*w/4, y)
        else:
            return component_centers[component_name]
    
    # Draw arrows with better positioning
    for (start_comp, start_edge), (end_comp, end_edge) in connections:
        start_pos = get_edge_point(start_comp, start_edge)
        end_pos = get_edge_point(end_comp, end_edge)
        
        ax.annotate('', xy=end_pos, xytext=start_pos,
                   arrowprops=dict(arrowstyle='->', lw=2, color=colors['border']))
    
    # Add title
    ax.text(6, 11.5, 'RAG System Architecture', 
           ha='center', va='center',
           fontsize=20, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=colors['input'], label='Input Layer'),
        mpatches.Patch(color=colors['processing'], label='Processing Layer'),
        mpatches.Patch(color=colors['retrieval'], label='Retrieval Layer'),
        mpatches.Patch(color=colors['judgment'], label='Judgment Layer'),
        mpatches.Patch(color=colors['generation'], label='Generation Layer'),
        mpatches.Patch(color=colors['output'], label='Output Layer')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Add annotations for key features
    ax.text(0.5, 5.5, 'Key Features:\n• Multi-path retrieval\n• Usefulness judgment\n• Soft retention strategy', 
           ha='left', va='top',
           fontsize=10, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=colors['border']))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'rag_system_architecture.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Fixed architecture diagram saved to: {output_path}")
    
    return fig

def create_fixed_experiment_design_overview(output_dir="/home/pushihao/RAG/Reports/docs/pics"):
    """Create experiment design overview with better arrow positioning"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define experiment hierarchy with better positioning
    experiments = {
        'Complete System\n(Baseline)': {
            'pos': (4, 8.5, 2, 0.8),
            'color': '#2E86AB',
            'level': 0
        },
        'No Query\nRewriter': {
            'pos': (1, 6.5, 1.5, 0.8),
            'color': '#A23B72',
            'level': 1
        },
        'No Usefulness\nJudge': {
            'pos': (3, 6.5, 1.5, 0.8),
            'color': '#F18F01',
            'level': 1
        },
        'No Dense Chunk\nRetrieval': {
            'pos': (5, 6.5, 1.5, 0.8),
            'color': '#C73E1D',
            'level': 1
        },
        'No Dense Keyword\nRetrieval': {
            'pos': (7, 6.5, 1.5, 0.8),
            'color': '#5D737E',
            'level': 1
        },
        'No Dense Question\nRetrieval': {
            'pos': (4, 4.5, 1.5, 0.8),
            'color': '#8B5A3C',
            'level': 1
        }
    }
    
    # Draw experiment boxes
    centers = {}
    for name, props in experiments.items():
        x, y, w, h = props['pos']
        
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05",
            facecolor=props['color'],
            edgecolor='white',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(rect)
        
        ax.text(x + w/2, y + h/2, name, 
               ha='center', va='center',
               fontsize=10, fontweight='bold',
               color='white')
        
        centers[name] = (x + w/2, y + h/2)
    
    # Draw connections from baseline to ablations with curved arrows to avoid text
    baseline_center = centers['Complete System\n(Baseline)']
    baseline_x, baseline_y = baseline_center
    
    for name, props in experiments.items():
        if props['level'] == 1:
            target_center = centers[name]
            target_x, target_y = target_center
            
            # Use curved arrows to avoid overlapping text
            if target_x < baseline_x:  # Left side
                # Curve left then down
                ax.annotate('', xy=(target_x + 0.75, target_y + 0.4), 
                           xytext=(baseline_x - 0.75, baseline_y - 0.4),
                           arrowprops=dict(arrowstyle='->', lw=2, color='#333333',
                                         connectionstyle="arc3,rad=-0.3"))
            elif target_x > baseline_x:  # Right side
                # Curve right then down
                ax.annotate('', xy=(target_x - 0.75, target_y + 0.4), 
                           xytext=(baseline_x + 0.75, baseline_y - 0.4),
                           arrowprops=dict(arrowstyle='->', lw=2, color='#333333',
                                         connectionstyle="arc3,rad=0.3"))
            else:  # Center (bottom one)
                # Straight down
                ax.annotate('', xy=(target_x, target_y + 0.4), 
                           xytext=(baseline_x, baseline_y - 0.4),
                           arrowprops=dict(arrowstyle='->', lw=2, color='#333333'))
    
    # Add title
    ax.text(5, 9.5, 'Ablation Study Design', 
           ha='center', va='center',
           fontsize=18, fontweight='bold')
    
    # Add description
    ax.text(5, 2.5, 'Each ablation study removes one key component\nto measure its contribution to overall performance', 
           ha='center', va='center',
           fontsize=12,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='#F5F5F5', edgecolor='#333333'))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'experiment_design_overview.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Fixed experiment design overview saved to: {output_path}")
    
    return fig

def main():
    """Main function to fix remaining chart issues"""
    
    # Initialize parser and visualizer
    parser = RAGDataParser()
    visualizer = RAGVisualizer()
    output_dir = "/home/pushihao/RAG/Reports/docs/pics"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Fixing remaining chart issues...")
    
    # 1. Fix best vs worst comparison (F1 score issue)
    print("\n1. Fixing best vs worst comparison F1 score display...")
    fig1 = create_fixed_best_worst_comparison(parser, visualizer, output_dir)
    if fig1:
        plt.close(fig1)
    
    # 2. Fix RAG architecture diagram (arrow overlap issue)
    print("\n2. Fixing RAG architecture diagram arrow positioning...")
    fig2 = create_fixed_rag_architecture_diagram(output_dir)
    if fig2:
        plt.close(fig2)
    
    # 3. Fix experiment design overview (arrow overlap issue)
    print("\n3. Fixing experiment design overview arrow positioning...")
    fig3 = create_fixed_experiment_design_overview(output_dir)
    if fig3:
        plt.close(fig3)
    
    print("\nAll remaining chart issues fixed!")

if __name__ == "__main__":
    main()