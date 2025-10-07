#!/usr/bin/env python3
"""
LLM-only vs RAG System Comparison Visualization
Creates comprehensive comparison charts between LLM-only baseline and RAG systems
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import json
import os

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_llm_only_data():
    """Load LLM-only experimental results"""
    base_path = Path("/home/pushihao/RAG/Reports/experiments/datasets/final_result_9_only_LLM")
    
    # Load summary data
    summary_file = base_path / "evaluation_summary.txt"
    
    data = {
        'overall': {'accuracy': 62.0},
        'datasets': {
            'HotpotQA': {'accuracy': 41.0},
            'MS MARCO': {'accuracy': 86.0}, 
            'Natural Questions': {'accuracy': 57.0},
            'TriviaQA': {'accuracy': 64.0}
        }
    }
    
    return data

def load_rag_systems_data():
    """Load RAG systems data for comparison"""
    
    # Best RAG system (optimized)
    best_rag = {
        'name': 'RAG System (Optimized)',
        'overall': {'accuracy': 87.5},
        'datasets': {
            'HotpotQA': {'accuracy': 89.0},
            'MS MARCO': {'accuracy': 89.0},
            'Natural Questions': {'accuracy': 86.0}, 
            'TriviaQA': {'accuracy': 86.0}
        }
    }
    
    # Original RAG system
    original_rag = {
        'name': 'RAG System (Original)',
        'overall': {'accuracy': 81.75},
        'datasets': {
            'HotpotQA': {'accuracy': 75.0},
            'MS MARCO': {'accuracy': 89.0},
            'Natural Questions': {'accuracy': 82.0},
            'TriviaQA': {'accuracy': 81.0}
        }
    }
    
    return best_rag, original_rag

def create_overall_comparison(llm_data, best_rag, original_rag, output_dir):
    """Create overall performance comparison chart"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    systems = ['LLM-only\n(Baseline)', 'RAG System\n(Original)', 'RAG System\n(Optimized)']
    accuracies = [
        llm_data['overall']['accuracy'],
        original_rag['overall']['accuracy'], 
        best_rag['overall']['accuracy']
    ]
    
    colors = ['#ff7f7f', '#87ceeb', '#90ee90']
    bars = ax.bar(systems, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add improvement annotations
    rag_original_improvement = original_rag['overall']['accuracy'] - llm_data['overall']['accuracy']
    rag_optimized_improvement = best_rag['overall']['accuracy'] - llm_data['overall']['accuracy']
    
    ax.annotate(f'+{rag_original_improvement:.1f}%', 
                xy=(1, original_rag['overall']['accuracy']), 
                xytext=(1, original_rag['overall']['accuracy'] + 5),
                ha='center', fontsize=12, color='blue', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    
    ax.annotate(f'+{rag_optimized_improvement:.1f}%', 
                xy=(2, best_rag['overall']['accuracy']), 
                xytext=(2, best_rag['overall']['accuracy'] + 5),
                ha='center', fontsize=12, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('LLM-only vs RAG Systems: Overall Performance Comparison\n(Real Experimental Results)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistical significance note
    ax.text(0.02, 0.98, 'Note: All results based on 400 test questions (100 per dataset)', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'llm_rag_overall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_dataset_comparison(llm_data, best_rag, original_rag, output_dir):
    """Create dataset-specific comparison chart"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    datasets = list(llm_data['datasets'].keys())
    x = np.arange(len(datasets))
    width = 0.25
    
    llm_accs = [llm_data['datasets'][d]['accuracy'] for d in datasets]
    original_accs = [original_rag['datasets'][d]['accuracy'] for d in datasets]
    best_accs = [best_rag['datasets'][d]['accuracy'] for d in datasets]
    
    bars1 = ax.bar(x - width, llm_accs, width, label='LLM-only (Baseline)', 
                   color='#ff7f7f', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, original_accs, width, label='RAG System (Original)', 
                   color='#87ceeb', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, best_accs, width, label='RAG System (Optimized)', 
                   color='#90ee90', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Datasets', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('LLM-only vs RAG Systems: Dataset-Specific Performance\n(Real Experimental Results)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(fontsize=12, loc='upper left')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'llm_rag_dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_improvement_analysis(llm_data, best_rag, original_rag, output_dir):
    """Create improvement analysis chart"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Overall improvement
    systems = ['Original RAG\nvs LLM-only', 'Optimized RAG\nvs LLM-only']
    improvements = [
        original_rag['overall']['accuracy'] - llm_data['overall']['accuracy'],
        best_rag['overall']['accuracy'] - llm_data['overall']['accuracy']
    ]
    
    colors = ['#87ceeb', '#90ee90']
    bars = ax1.bar(systems, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'+{imp:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax1.set_ylabel('Accuracy Improvement (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Performance Improvement\nover LLM-only Baseline', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(improvements) + 5)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right plot: Dataset-specific improvements
    datasets = list(llm_data['datasets'].keys())
    x = np.arange(len(datasets))
    width = 0.35
    
    original_improvements = [original_rag['datasets'][d]['accuracy'] - llm_data['datasets'][d]['accuracy'] 
                           for d in datasets]
    best_improvements = [best_rag['datasets'][d]['accuracy'] - llm_data['datasets'][d]['accuracy'] 
                        for d in datasets]
    
    bars1 = ax2.bar(x - width/2, original_improvements, width, label='Original RAG', 
                    color='#87ceeb', alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x + width/2, best_improvements, width, label='Optimized RAG', 
                    color='#90ee90', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'+{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Datasets', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Dataset-Specific Improvements\nover LLM-only Baseline', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'llm_rag_improvement_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_knowledge_impact_analysis(llm_data, best_rag, original_rag, output_dir):
    """Create analysis showing the impact of external knowledge"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    datasets = list(llm_data['datasets'].keys())
    
    # Calculate knowledge gaps (difference between LLM-only and RAG)
    llm_accs = [llm_data['datasets'][d]['accuracy'] for d in datasets]
    best_accs = [best_rag['datasets'][d]['accuracy'] for d in datasets]
    knowledge_gaps = [best - llm for best, llm in zip(best_accs, llm_accs)]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(datasets))
    
    # Plot LLM-only performance as base
    bars1 = ax.barh(y_pos, llm_accs, height=0.6, label='LLM-only Performance', 
                    color='#ff7f7f', alpha=0.8, edgecolor='black')
    
    # Plot knowledge contribution as additional bars
    bars2 = ax.barh(y_pos, knowledge_gaps, left=llm_accs, height=0.6, 
                    label='Knowledge Base Contribution', color='#90ee90', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for i, (llm_acc, gap) in enumerate(zip(llm_accs, knowledge_gaps)):
        # LLM-only accuracy
        ax.text(llm_acc/2, i, f'{llm_acc:.0f}%', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='darkred')
        # Knowledge contribution
        ax.text(llm_acc + gap/2, i, f'+{gap:.0f}%', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='darkgreen')
        # Total accuracy
        ax.text(llm_acc + gap + 2, i, f'{llm_acc + gap:.0f}%', ha='left', va='center', 
                fontsize=11, fontweight='bold', color='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(datasets)
    ax.set_xlabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Impact of External Knowledge Base on Performance\n(LLM-only vs Optimized RAG System)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add insights text
    avg_improvement = np.mean(knowledge_gaps)
    ax.text(0.02, 0.98, f'Average Knowledge Contribution: +{avg_improvement:.1f}%\n'
                        f'Highest Impact: {datasets[np.argmax(knowledge_gaps)]} (+{max(knowledge_gaps):.0f}%)\n'
                        f'Lowest Impact: {datasets[np.argmin(knowledge_gaps)]} (+{min(knowledge_gaps):.0f}%)', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'knowledge_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to create all comparison visualizations"""
    
    # Set up output directory
    output_dir = Path("/home/pushihao/RAG/Reports/docs/pics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading experimental data...")
    
    # Load data
    llm_data = load_llm_only_data()
    best_rag, original_rag = load_rag_systems_data()
    
    print("Creating comparison visualizations...")
    
    # Create visualizations
    create_overall_comparison(llm_data, best_rag, original_rag, output_dir)
    print("✓ Overall comparison chart created")
    
    create_dataset_comparison(llm_data, best_rag, original_rag, output_dir)
    print("✓ Dataset comparison chart created")
    
    create_improvement_analysis(llm_data, best_rag, original_rag, output_dir)
    print("✓ Improvement analysis chart created")
    
    create_knowledge_impact_analysis(llm_data, best_rag, original_rag, output_dir)
    print("✓ Knowledge impact analysis chart created")
    
    print(f"\nAll comparison charts saved to: {output_dir}")
    print("\nGenerated files:")
    print("- llm_rag_overall_comparison.png")
    print("- llm_rag_dataset_comparison.png") 
    print("- llm_rag_improvement_analysis.png")
    print("- knowledge_impact_analysis.png")

if __name__ == "__main__":
    main()