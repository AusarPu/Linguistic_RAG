"""
Visualization Utilities for RAG System Evaluation Results
Common plotting functions and configurations for generating charts
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')


class RAGVisualizer:
    """Visualization utilities for RAG evaluation results"""
    
    def __init__(self, output_dir: str = "/home/pushihao/RAG/Reports/docs/pics"):
        self.output_dir = output_dir
        self.setup_style()
        
    def setup_style(self):
        """Setup matplotlib and seaborn styles"""
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure matplotlib for better appearance
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.format': 'png',
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })
        
        # Color palette for consistency
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'info': '#5D737E',
            'light': '#F5F5F5',
            'dark': '#2C3E50'
        }
        
        # Color schemes for different chart types
        self.color_schemes = {
            'ablation': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#5D737E', '#8B5A3C', '#6A994E'],
            'datasets': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
            'comparison': ['#3498DB', '#E74C3C'],
            'heatmap': 'RdYlBu_r'
        }
    
    def save_figure(self, filename: str, fig: Optional[plt.Figure] = None):
        """Save figure with consistent settings"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        filepath = os.path.join(self.output_dir, filename)
        
        if fig is None:
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        else:
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        
        print(f"Saved: {filepath}")
    
    def create_grouped_bar_chart(self, data: pd.DataFrame, x_col: str, y_cols: List[str], 
                                title: str, xlabel: str, ylabel: str, 
                                labels: Optional[Dict[str, str]] = None,
                                colors: Optional[List[str]] = None,
                                figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Create a grouped bar chart"""
        fig, ax = plt.subplots(figsize=figsize)
        
        if colors is None:
            colors = self.color_schemes['ablation'][:len(y_cols)]
        
        x = np.arange(len(data))
        width = 0.8 / len(y_cols)
        
        for i, col in enumerate(y_cols):
            label = labels.get(col, col) if labels else col
            ax.bar(x + i * width - width * (len(y_cols) - 1) / 2, 
                   data[col], width, label=label, color=colors[i % len(colors)])
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(data[x_col], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_heatmap(self, data: pd.DataFrame, title: str, 
                      xlabel: str, ylabel: str,
                      figsize: Tuple[int, int] = (10, 8),
                      annot: bool = True, fmt: str = '.2f') -> plt.Figure:
        """Create a heatmap"""
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(data, annot=annot, fmt=fmt, cmap=self.color_schemes['heatmap'],
                   center=0.5, square=True, ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        plt.tight_layout()
        return fig
    
    def create_radar_chart(self, data: Dict[str, List[float]], labels: List[str],
                          title: str, figsize: Tuple[int, int] = (10, 10)) -> plt.Figure:
        """Create a radar chart"""
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = self.color_schemes['comparison']
        
        for i, (name, values) in enumerate(data.items()):
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def create_dashboard(self, data: Dict[str, Any], title: str,
                        figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """Create a comprehensive dashboard"""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.95)
        
        return fig
    
    def create_line_chart(self, data: pd.DataFrame, x_col: str, y_cols: List[str],
                         title: str, xlabel: str, ylabel: str,
                         labels: Optional[Dict[str, str]] = None,
                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Create a line chart"""
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = self.color_schemes['ablation']
        
        for i, col in enumerate(y_cols):
            label = labels.get(col, col) if labels else col
            ax.plot(data[x_col], data[col], marker='o', linewidth=2, 
                   label=label, color=colors[i % len(colors)])
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold', pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_box_plot(self, data: List[List[float]], labels: List[str],
                       title: str, xlabel: str, ylabel: str,
                       figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Create a box plot"""
        fig, ax = plt.subplots(figsize=figsize)
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        colors = self.color_schemes['datasets']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_stacked_bar_chart(self, data: pd.DataFrame, x_col: str, y_cols: List[str],
                                title: str, xlabel: str, ylabel: str,
                                labels: Optional[Dict[str, str]] = None,
                                figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Create a stacked bar chart"""
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = self.color_schemes['ablation'][:len(y_cols)]
        
        bottom = np.zeros(len(data))
        
        for i, col in enumerate(y_cols):
            label = labels.get(col, col) if labels else col
            ax.bar(data[x_col], data[col], bottom=bottom, 
                   label=label, color=colors[i])
            bottom += data[col]
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold', pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def add_value_labels(self, ax, bars, format_str: str = '{:.1%}'):
        """Add value labels on top of bars"""
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       format_str.format(height),
                       ha='center', va='bottom', fontsize=9)
    
    def create_comparison_bars(self, data: Dict[str, float], title: str,
                              xlabel: str, ylabel: str,
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Create simple comparison bar chart"""
        fig, ax = plt.subplots(figsize=figsize)
        
        names = list(data.keys())
        values = list(data.values())
        colors = self.color_schemes['comparison'][:len(names)]
        
        bars = ax.bar(names, values, color=colors)
        self.add_value_labels(ax, bars)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def create_metric_summary_box(self, ax, title: str, metrics: Dict[str, float],
                                 position: Tuple[float, float, float, float]):
        """Create a summary box with key metrics"""
        x, y, width, height = position
        
        # Create fancy box
        box = FancyBboxPatch((x, y), width, height,
                           boxstyle="round,pad=0.02",
                           facecolor=self.colors['light'],
                           edgecolor=self.colors['primary'],
                           linewidth=2)
        ax.add_patch(box)
        
        # Add title
        ax.text(x + width/2, y + height - 0.05, title,
               ha='center', va='top', fontsize=12, fontweight='bold')
        
        # Add metrics
        y_pos = y + height - 0.15
        for metric, value in metrics.items():
            if isinstance(value, float):
                text = f"{metric}: {value:.1%}" if value <= 1 else f"{metric}: {value:.2f}"
            else:
                text = f"{metric}: {value}"
            ax.text(x + 0.02, y_pos, text, ha='left', va='top', fontsize=10)
            y_pos -= 0.08


def create_architecture_diagram():
    """Create RAG system architecture diagram"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define components and their positions
    components = {
        'Query Input': (1, 8.5, 1.5, 0.8),
        'Query Rewriter': (1, 7, 1.5, 0.8),
        'Multi-path Retrieval': (4, 7, 2, 0.8),
        'BM25 Retrieval': (3, 5.5, 1.2, 0.6),
        'Dense Chunk': (4.5, 5.5, 1.2, 0.6),
        'Dense Keywords': (6, 5.5, 1.2, 0.6),
        'Dense Questions': (7.5, 5.5, 1.2, 0.6),
        'Usefulness Judge': (4, 3.5, 2, 0.8),
        'Soft Retention': (4, 2, 2, 0.8),
        'Answer Generation': (7.5, 2, 1.5, 0.8),
        'Final Answer': (7.5, 0.5, 1.5, 0.8)
    }
    
    # Colors for different component types
    colors = {
        'Query Input': '#E8F4FD',
        'Query Rewriter': '#B3E5FC',
        'Multi-path Retrieval': '#81C784',
        'BM25 Retrieval': '#A5D6A7',
        'Dense Chunk': '#A5D6A7',
        'Dense Keywords': '#A5D6A7',
        'Dense Questions': '#A5D6A7',
        'Usefulness Judge': '#FFB74D',
        'Soft Retention': '#FFCC02',
        'Answer Generation': '#F48FB1',
        'Final Answer': '#E1BEE7'
    }
    
    # Draw components
    for name, (x, y, w, h) in components.items():
        rect = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.05",
                            facecolor=colors.get(name, '#F5F5F5'),
                            edgecolor='#333333',
                            linewidth=1.5)
        ax.add_patch(rect)
        
        # Add text
        ax.text(x + w/2, y + h/2, name, ha='center', va='center',
               fontsize=10, fontweight='bold', wrap=True)
    
    # Draw arrows to show flow
    arrows = [
        ((1.75, 8.5), (1.75, 7.8)),  # Query Input -> Query Rewriter
        ((2.5, 7.4), (4, 7.4)),      # Query Rewriter -> Multi-path
        ((5, 7), (3.6, 6.1)),        # Multi-path -> BM25
        ((5, 7), (5.1, 6.1)),        # Multi-path -> Dense Chunk
        ((5, 7), (6.6, 6.1)),        # Multi-path -> Dense Keywords
        ((5, 7), (8.1, 6.1)),        # Multi-path -> Dense Questions
        ((3.6, 5.5), (4.5, 4.3)),    # BM25 -> Usefulness
        ((5.1, 5.5), (5, 4.3)),      # Dense Chunk -> Usefulness
        ((6.6, 5.5), (5.5, 4.3)),    # Dense Keywords -> Usefulness
        ((8.1, 5.5), (5.8, 4.3)),    # Dense Questions -> Usefulness
        ((5, 3.5), (5, 2.8)),        # Usefulness -> Soft Retention
        ((6, 2.4), (7.5, 2.4)),      # Soft Retention -> Generation
        ((8.25, 2), (8.25, 1.3))     # Generation -> Final Answer
    ]
    
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='#333333'))
    
    ax.set_title('RAG System Architecture', fontsize=18, fontweight='bold', pad=20)
    
    return fig


if __name__ == "__main__":
    # Test visualization utilities
    visualizer = RAGVisualizer()
    
    # Test architecture diagram
    fig = create_architecture_diagram()
    visualizer.save_figure('test_architecture.png', fig)
    plt.close(fig)
    
    print("Visualization utilities created successfully!")