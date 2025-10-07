"""
Create RAG System Architecture Diagram
Generates a comprehensive architecture diagram showing all system components
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import os

def create_rag_architecture_diagram(output_dir="/home/pushihao/RAG/Reports/docs/pics"):
    """Create comprehensive RAG system architecture diagram"""
    
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
            'pos': (10.5, 7, 1.5, 2),
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
    
    # Define connections (arrows)
    connections = [
        # Main flow
        ('User Query', 'Query Rewriter'),
        ('Query Rewriter', 'Multi-path\nRetrieval System'),
        
        # Retrieval paths
        ('Multi-path\nRetrieval System', 'BM25\nRetrieval'),
        ('Multi-path\nRetrieval System', 'Dense Chunk\nRetrieval'),
        ('Multi-path\nRetrieval System', 'Dense Keyword\nRetrieval'),
        ('Multi-path\nRetrieval System', 'Dense Question\nRetrieval'),
        
        # Knowledge base connections
        ('Knowledge Base\n(Indexed Documents)', 'BM25\nRetrieval'),
        ('Knowledge Base\n(Indexed Documents)', 'Dense Chunk\nRetrieval'),
        ('Knowledge Base\n(Indexed Documents)', 'Dense Keyword\nRetrieval'),
        ('Knowledge Base\n(Indexed Documents)', 'Dense Question\nRetrieval'),
        
        # Judgment flow
        ('BM25\nRetrieval', 'Usefulness\nJudge'),
        ('Dense Chunk\nRetrieval', 'Usefulness\nJudge'),
        ('Dense Keyword\nRetrieval', 'Usefulness\nJudge'),
        ('Dense Question\nRetrieval', 'Usefulness\nJudge'),
        
        # Retention and generation
        ('Usefulness\nJudge', 'Soft Retention\nStrategy'),
        ('Soft Retention\nStrategy', 'Answer\nGeneration'),
        ('Answer\nGeneration', 'Final Answer')
    ]
    
    # Draw arrows
    for start, end in connections:
        start_pos = component_centers[start]
        end_pos = component_centers[end]
        
        # Calculate arrow positions to avoid overlapping with boxes
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        length = np.sqrt(dx**2 + dy**2)
        
        if length > 0:
            # Normalize direction
            dx_norm = dx / length
            dy_norm = dy / length
            
            # Offset from box edges
            offset = 0.4
            start_arrow = (start_pos[0] + dx_norm * offset, start_pos[1] + dy_norm * offset)
            end_arrow = (end_pos[0] - dx_norm * offset, end_pos[1] - dy_norm * offset)
            
            ax.annotate('', xy=end_arrow, xytext=start_arrow,
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
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'rag_system_architecture.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Architecture diagram saved to: {output_path}")
    
    return fig

def create_experiment_design_overview(output_dir="/home/pushihao/RAG/Reports/docs/pics"):
    """Create experiment design overview diagram"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define experiment hierarchy
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
    
    # Draw connections from baseline to ablations
    baseline_center = centers['Complete System\n(Baseline)']
    for name, props in experiments.items():
        if props['level'] == 1:
            target_center = centers[name]
            ax.annotate('', xy=target_center, xytext=baseline_center,
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
    print(f"Experiment design overview saved to: {output_path}")
    
    return fig

if __name__ == "__main__":
    # Create output directory
    output_dir = "/home/pushihao/RAG/Reports/docs/pics"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate architecture diagram
    print("Creating RAG system architecture diagram...")
    fig1 = create_rag_architecture_diagram(output_dir)
    plt.close(fig1)
    
    # Generate experiment design overview
    print("Creating experiment design overview...")
    fig2 = create_experiment_design_overview(output_dir)
    plt.close(fig2)
    
    print("Architecture diagrams created successfully!")