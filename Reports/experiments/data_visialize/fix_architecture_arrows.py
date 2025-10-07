"""
Fix RAG Architecture Diagram Arrow Overlapping
Fix the arrow overlapping issue with Knowledge Base component
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import os

def create_fixed_rag_architecture_diagram(output_dir="/home/pushihao/RAG/Reports/docs/pics"):
    """Create RAG system architecture diagram with completely fixed arrow positioning"""
    
    # Create figure with high DPI for publication quality
    fig, ax = plt.subplots(figsize=(18, 12))  # Increased width to give more space
    ax.set_xlim(0, 14)  # Expanded x-axis
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
    
    # Component definitions with better positioning to avoid overlaps
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
        
        # Retrieval Components - spread out more
        'BM25\nRetrieval': {
            'pos': (1.5, 7, 1.8, 0.8),
            'color': colors['retrieval'],
            'type': 'retrieval'
        },
        'Dense Chunk\nRetrieval': {
            'pos': (3.8, 7, 1.8, 0.8),
            'color': colors['retrieval'],
            'type': 'retrieval'
        },
        'Dense Keyword\nRetrieval': {
            'pos': (6.1, 7, 1.8, 0.8),
            'color': colors['retrieval'],
            'type': 'retrieval'
        },
        'Dense Question\nRetrieval': {
            'pos': (8.4, 7, 1.8, 0.8),
            'color': colors['retrieval'],
            'type': 'retrieval'
        },
        
        # Knowledge Base - moved further right and made taller
        'Knowledge Base\n(Indexed Documents)': {
            'pos': (11.5, 6, 2, 3),  # Moved right and made taller
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
    
    # Helper function to get specific edge points
    def get_connection_point(component_name, side, offset=0):
        x, y, w, h = components[component_name]['pos']
        if side == 'top':
            return (x + w/2 + offset, y + h)
        elif side == 'bottom':
            return (x + w/2 + offset, y)
        elif side == 'left':
            return (x, y + h/2 + offset)
        elif side == 'right':
            return (x + w, y + h/2 + offset)
        elif side == 'top-left':
            return (x + w/4, y + h)
        elif side == 'top-right':
            return (x + 3*w/4, y + h)
        elif side == 'bottom-left':
            return (x + w/4, y)
        elif side == 'bottom-right':
            return (x + 3*w/4, y)
        else:
            return component_centers[component_name]
    
    # Define connections with very specific positioning to avoid overlaps
    connections = [
        # Main flow
        (get_connection_point('User Query', 'bottom'), get_connection_point('Query Rewriter', 'top')),
        (get_connection_point('Query Rewriter', 'right'), get_connection_point('Multi-path\nRetrieval System', 'left')),
        
        # Multi-path to retrieval components - use specific points
        (get_connection_point('Multi-path\nRetrieval System', 'bottom', -1), get_connection_point('BM25\nRetrieval', 'top')),
        (get_connection_point('Multi-path\nRetrieval System', 'bottom', -0.3), get_connection_point('Dense Chunk\nRetrieval', 'top')),
        (get_connection_point('Multi-path\nRetrieval System', 'bottom', 0.3), get_connection_point('Dense Keyword\nRetrieval', 'top')),
        (get_connection_point('Multi-path\nRetrieval System', 'bottom', 1), get_connection_point('Dense Question\nRetrieval', 'top')),
        
        # Knowledge base connections - use curved paths to avoid text overlap
        # These will be drawn separately with curved arrows
        
        # Retrieval to usefulness judge
        (get_connection_point('BM25\nRetrieval', 'bottom'), get_connection_point('Usefulness\nJudge', 'top-left')),
        (get_connection_point('Dense Chunk\nRetrieval', 'bottom'), get_connection_point('Usefulness\nJudge', 'top', -0.3)),
        (get_connection_point('Dense Keyword\nRetrieval', 'bottom'), get_connection_point('Usefulness\nJudge', 'top', 0.3)),
        (get_connection_point('Dense Question\nRetrieval', 'bottom'), get_connection_point('Usefulness\nJudge', 'top-right')),
        
        # Flow continuation
        (get_connection_point('Usefulness\nJudge', 'bottom'), get_connection_point('Soft Retention\nStrategy', 'top')),
        (get_connection_point('Soft Retention\nStrategy', 'right'), get_connection_point('Answer\nGeneration', 'left')),
        (get_connection_point('Answer\nGeneration', 'bottom'), get_connection_point('Final Answer', 'top'))
    ]
    
    # Draw regular arrows
    for start_pos, end_pos in connections:
        ax.annotate('', xy=end_pos, xytext=start_pos,
                   arrowprops=dict(arrowstyle='->', lw=2, color=colors['border']))
    
    # Draw curved arrows from Knowledge Base to avoid text overlap
    kb_center = get_connection_point('Knowledge Base\n(Indexed Documents)', 'left')
    
    # Define target points on the right side of retrieval components
    retrieval_components = ['BM25\nRetrieval', 'Dense Chunk\nRetrieval', 'Dense Keyword\nRetrieval', 'Dense Question\nRetrieval']
    
    for i, comp in enumerate(retrieval_components):
        target_point = get_connection_point(comp, 'right')
        
        # Create curved connection to avoid overlapping with component text
        # Use different curve directions for different components
        if i < 2:  # Top two components
            curve_style = "arc3,rad=-0.3"
        else:  # Bottom two components
            curve_style = "arc3,rad=0.3"
        
        ax.annotate('', xy=target_point, xytext=kb_center,
                   arrowprops=dict(arrowstyle='->', lw=2, color=colors['border'],
                                 connectionstyle=curve_style))
    
    # Add title
    ax.text(7, 11.5, 'RAG System Architecture', 
           ha='center', va='center',
           fontsize=20, fontweight='bold')
    
    # Add legend - positioned to not interfere with components
    legend_elements = [
        mpatches.Patch(color=colors['input'], label='Input Layer'),
        mpatches.Patch(color=colors['processing'], label='Processing Layer'),
        mpatches.Patch(color=colors['retrieval'], label='Retrieval Layer'),
        mpatches.Patch(color=colors['judgment'], label='Judgment Layer'),
        mpatches.Patch(color=colors['generation'], label='Generation Layer'),
        mpatches.Patch(color=colors['output'], label='Output Layer')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 0.95))
    
    # Add annotations for key features - positioned to not interfere
    ax.text(0.5, 4.5, 'Key Features:\n• Multi-path retrieval\n• Usefulness judgment\n• Soft retention strategy', 
           ha='left', va='top',
           fontsize=10, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=colors['border']))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'rag_system_architecture.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Fixed RAG architecture diagram saved to: {output_path}")
    
    return fig

def main():
    """Main function to fix architecture diagram arrows"""
    
    output_dir = "/home/pushihao/RAG/Reports/docs/pics"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Fixing RAG architecture diagram arrow overlapping...")
    
    # Fix architecture diagram
    fig = create_fixed_rag_architecture_diagram(output_dir)
    if fig:
        plt.close(fig)
    
    print("RAG architecture diagram arrow overlapping fixed!")

if __name__ == "__main__":
    main()