"""
Create Advanced Visualization Charts
Reads real metrics from runs CSV
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

def create_performance_heatmap(parser, visualizer, output_dir):
    runs = parser.load_all_runs()
    datasets = ['HotpotQA', 'MS MARCO', 'Natural Questions', 'TriviaQA']
    dataset_keys = ['hotpotqa', 'ms_marco', 'natural_questions', 'triviaqa']
    run_labels = parser.get_run_labels()
    matrix = []
    row_names = []
    for run_name, data in runs.items():
        row = []
        for dk in dataset_keys:
            if dk in data['datasets']:
                row.append(data['datasets'][dk]['accuracy'] * 100)
            else:
                row.append(0)
        matrix.append(row)
        row_names.append(run_labels.get(run_name, run_name))
    df = pd.DataFrame(matrix, index=row_names, columns=datasets)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlBu_r', center=df.values.mean(), square=False, ax=ax,
                cbar_kws={'shrink': 0.8, 'label': 'Accuracy (%)'})
    ax.set_title('Performance Matrix: Runs × Datasets (Accuracy %)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Datasets', fontsize=12, fontweight='bold')
    ax.set_ylabel('Runs', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'performance_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Performance heatmap saved to: {output_path}")
    return fig

def create_comprehensive_radar_chart(parser, visualizer, output_dir):
    runs = parser.load_all_runs()
    run_labels = parser.get_run_labels()
    keys = list(runs.keys())
    a_key = keys[0]
    b_key = 'result_6_bm25' if 'result_6_bm25' in runs else keys[-1]
    dimensions = ['Accuracy', 'Context Recall', 'Context Precision', 'Faithfulness', 'Answer Relevancy']
    data = {}
    for key in [a_key, b_key]:
        o = runs[key]['overall']
        vals = [
            float(o['overall_accuracy']),
            float(o['overall_context_recall']),
            float(o['overall_context_precision']),
            float(o['overall_faithfulness']),
            float(o['overall_answer_relevancy'])
        ]
        data[run_labels.get(key, key)] = vals
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]
    colors = visualizer.color_schemes['comparison']
    for i, (name, values) in enumerate(data.items()):
        v = values + values[:1]
        ax.plot(angles, v, 'o-', linewidth=2, label=name, color=colors[i % len(colors)])
        ax.fill(angles, v, alpha=0.25, color=colors[i % len(colors)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions)
    ax.set_ylim(0, 1)
    ax.set_title('System Metrics Radar (Real CSV)', fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'comprehensive_radar_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Comprehensive radar chart saved to: {output_path}")
    return fig

def create_retrieval_analysis_chart(parser, visualizer, output_dir):
    runs = parser.load_all_runs()
    preferred = 'result_5_full' if 'result_5_full' in runs else list(runs.keys())[0]
    data = runs[preferred]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Retrieval Analysis (Real CSV)', fontsize=16, fontweight='bold')
    ds_labels = parser.get_dataset_labels()
    names = []
    accs = []
    recalls = []
    precs = []
    for dk, lab in ds_labels.items():
        if dk in data['datasets']:
            names.append(lab)
            accs.append(data['datasets'][dk]['accuracy'] * 100)
            recalls.append(data['datasets'][dk]['context_recall'] * 100)
            precs.append(data['datasets'][dk]['context_precision'] * 100)
    x = np.arange(len(names))
    ax1.bar(names, recalls, color=visualizer.color_schemes['ablation'][:len(names)])
    ax1.set_title('Context Recall by Dataset (%)')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    ax2.bar(names, precs, color=visualizer.color_schemes['ablation'][:len(names)])
    ax2.set_title('Context Precision by Dataset (%)')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax3.plot(names, accs, 'o-', linewidth=2, color='#2E86AB')
    ax3.set_title('Accuracy by Dataset (%)')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)
    ax4.scatter(recalls, accs, c=precs, cmap='RdYlGn')
    ax4.set_xlabel('Context Recall (%)')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Accuracy vs Recall (colored by Precision)')
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'retrieval_analysis_dashboard.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Retrieval analysis dashboard saved to: {output_path}")
    return fig

def create_system_comparison_matrix(parser, visualizer, output_dir):
    runs = parser.load_all_runs()
    run_labels = parser.get_run_labels()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    col_labels = ['Run', 'Accuracy (%)', 'Context Recall (%)', 'Context Precision (%)']
    rows = []
    for run_name, data in runs.items():
        o = data['overall']
        rows.append([
            run_labels.get(run_name, run_name),
            f"{o['overall_accuracy']*100:.1f}",
            f"{o['overall_context_recall']*100:.1f}",
            f"{o['overall_context_precision']*100:.1f}"
        ])
    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.set_title('System Comparison Matrix (Real CSV)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'system_comparison_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"System comparison matrix saved to: {output_path}")
    return fig

def main():
    """Main function to generate all advanced charts"""
    
    # Initialize parser and visualizer
    parser = RAGDataParser()
    visualizer = RAGVisualizer()
    output_dir = "/home/pushihao/RAG/Reports/docs/pics"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating advanced visualization charts...")
    
    # 1. Performance heatmap
    print("\n1. Creating performance heatmap...")
    fig1 = create_performance_heatmap(parser, visualizer, output_dir)
    if fig1:
        plt.close(fig1)
    
    # 2. Comprehensive radar chart
    print("\n2. Creating comprehensive radar chart...")
    fig2 = create_comprehensive_radar_chart(parser, visualizer, output_dir)
    if fig2:
        plt.close(fig2)
    
    # 3. Retrieval analysis dashboard
    print("\n3. Creating retrieval analysis dashboard...")
    fig3 = create_retrieval_analysis_chart(parser, visualizer, output_dir)
    if fig3:
        plt.close(fig3)
    
    # 4. System comparison matrix
    print("\n4. Creating system comparison matrix...")
    fig4 = create_system_comparison_matrix(parser, visualizer, output_dir)
    if fig4:
        plt.close(fig4)
    
    print("\nAll advanced charts created successfully!")

if __name__ == "__main__":
    main()
import os
import numpy as np
import matplotlib.pyplot as plt
from visualization_utils import RAGVisualizer

def plot_multipath_fusion_cost(output_dir: str, output_name: str) -> str:
    datasets = ["HotpotQA", "TriviaQA"]
    r2_acc = [0.75 * 100, 0.77 * 100]
    r4_acc = [0.69 * 100, 0.74 * 100]
    r2_rec = [0.88 * 100, 0.79 * 100]
    r4_rec = [0.90 * 100, 0.82 * 100]

    viz = RAGVisualizer(output_dir)
    c1, c2 = viz.color_schemes["comparison"][0], viz.color_schemes["comparison"][1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Cost of Naive Multi-path Fusion (R4)", fontsize=16, fontweight="bold")
    x = np.arange(len(datasets))
    width = 0.35

    b1 = ax1.bar(x - width/2, r2_acc, width, label="R2: P1 + P2", color=c1, alpha=0.85)
    b2 = ax1.bar(x + width/2, r4_acc, width, label="R4: P1 + P2 + P3", color=c2, alpha=0.85)
    ax1.set_title("Accuracy (%)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, axis="y")
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom")

    b3 = ax2.bar(x - width/2, r2_rec, width, label="R2: P1 + P2", color=c1, alpha=0.85)
    b4 = ax2.bar(x + width/2, r4_rec, width, label="R4: P1 + P2 + P3", color=c2, alpha=0.85)
    ax2.set_title("Context Recall (%)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis="y")
    for bars in [b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path