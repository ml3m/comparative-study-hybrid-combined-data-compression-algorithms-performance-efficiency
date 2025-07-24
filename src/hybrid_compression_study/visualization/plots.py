"""
Visualization components for compression analysis.

This module provides comprehensive plotting and visualization tools
for analyzing compression algorithm performance and results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..benchmarks.suite import BenchmarkResult, BenchmarkSummary


# Set up plotting style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


class CompressionVisualizer:
    """
    Main visualization class for compression analysis.
    """
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 12)
        
        # Set style
        if style in plt.style.available:
            plt.style.use(style)
    
    def plot_compression_ratios(self, results: List[BenchmarkResult], 
                              save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot compression ratios comparison across algorithms.
        
        Args:
            results: List of benchmark results
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        # Prepare data
        df = pd.DataFrame([
            {
                'Algorithm': r.algorithm_name,
                'Dataset': r.dataset_name,
                'File': r.file_name,
                'Compression_Ratio': r.compression_ratio,
                'Original_Size_KB': r.original_size / 1024,
                'Success': r.success
            }
            for r in results if r.success
        ])
        
        if df.empty:
            raise ValueError("No successful results to plot")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Box plot of compression ratios by algorithm
        sns.boxplot(data=df, x='Algorithm', y='Compression_Ratio', ax=ax1)
        ax1.set_title('Compression Ratio Distribution by Algorithm', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Algorithm', fontsize=12)
        ax1.set_ylabel('Compression Ratio', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot of compression ratio vs file size
        for i, algo in enumerate(df['Algorithm'].unique()):
            algo_data = df[df['Algorithm'] == algo]
            ax2.scatter(algo_data['Original_Size_KB'], algo_data['Compression_Ratio'], 
                       label=algo, alpha=0.7, s=60, color=self.colors[i % len(self.colors)])
        
        ax2.set_title('Compression Ratio vs File Size', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Original File Size (KB)', fontsize=12)
        ax2.set_ylabel('Compression Ratio', fontsize=12)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_performance_comparison(self, results: List[BenchmarkResult],
                                  save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot performance comparison (speed vs compression).
        
        Args:
            results: List of benchmark results
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        # Prepare data
        df = pd.DataFrame([
            {
                'Algorithm': r.algorithm_name,
                'Compression_Ratio': r.compression_ratio,
                'Total_Time': r.total_time,
                'Throughput_MBps': r.throughput_mbps,
                'Dataset': r.dataset_name,
                'File_Size_MB': r.original_size / (1024 * 1024)
            }
            for r in results if r.success and r.total_time > 0
        ])
        
        if df.empty:
            raise ValueError("No successful results with timing data to plot")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Compression Ratio vs Processing Time
        for i, algo in enumerate(df['Algorithm'].unique()):
            algo_data = df[df['Algorithm'] == algo]
            ax1.scatter(algo_data['Total_Time'], algo_data['Compression_Ratio'],
                       label=algo, alpha=0.7, s=60, color=self.colors[i % len(self.colors)])
        
        ax1.set_xlabel('Total Processing Time (seconds)', fontsize=12)
        ax1.set_ylabel('Compression Ratio', fontsize=12)
        ax1.set_title('Compression Ratio vs Processing Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Throughput by Algorithm
        sns.boxplot(data=df, x='Algorithm', y='Throughput_MBps', ax=ax2)
        ax2.set_title('Throughput Distribution by Algorithm', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Algorithm', fontsize=12)
        ax2.set_ylabel('Throughput (MB/s)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance efficiency (ratio/time)
        df['Efficiency'] = df['Compression_Ratio'] / df['Total_Time']
        sns.barplot(data=df.groupby('Algorithm')['Efficiency'].mean().reset_index(), 
                   x='Algorithm', y='Efficiency', ax=ax3)
        ax3.set_title('Compression Efficiency (Ratio/Time)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Algorithm', fontsize=12)
        ax3.set_ylabel('Efficiency (Ratio/Second)', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Processing time vs file size
        for i, algo in enumerate(df['Algorithm'].unique()):
            algo_data = df[df['Algorithm'] == algo]
            ax4.scatter(algo_data['File_Size_MB'], algo_data['Total_Time'],
                       label=algo, alpha=0.7, s=60, color=self.colors[i % len(self.colors)])
        
        ax4.set_xlabel('File Size (MB)', fontsize=12)
        ax4.set_ylabel('Processing Time (seconds)', fontsize=12)
        ax4.set_title('Processing Time vs File Size', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_algorithm_heatmap(self, results: List[BenchmarkResult],
                              metric: str = 'compression_ratio',
                              save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create a heatmap showing algorithm performance across datasets.
        
        Args:
            results: List of benchmark results
            metric: Metric to visualize ('compression_ratio', 'total_time', 'throughput_mbps')
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        # Prepare data
        metric_map = {
            'compression_ratio': 'Compression Ratio',
            'total_time': 'Total Time (s)',
            'throughput_mbps': 'Throughput (MB/s)'
        }
        
        if metric not in metric_map:
            raise ValueError(f"Unknown metric: {metric}. Available: {list(metric_map.keys())}")
        
        df = pd.DataFrame([
            {
                'Algorithm': r.algorithm_name,
                'Dataset': f"{r.dataset_name}:{r.file_name}",
                metric: getattr(r, metric)
            }
            for r in results if r.success
        ])
        
        if df.empty:
            raise ValueError("No successful results to plot")
        
        # Create pivot table
        pivot_table = df.pivot_table(values=metric, index='Algorithm', columns='Dataset', aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=(max(12, len(pivot_table.columns)), max(8, len(pivot_table.index))))
        
        # Create heatmap
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, cbar_kws={'label': metric_map[metric]})
        
        ax.set_title(f'Algorithm Performance Heatmap: {metric_map[metric]}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Dataset:File', fontsize=12)
        ax.set_ylabel('Algorithm', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self, results: List[BenchmarkResult]) -> go.Figure:
        """
        Create an interactive Plotly dashboard.
        
        Args:
            results: List of benchmark results
            
        Returns:
            Plotly Figure object
        """
        # Prepare data
        df = pd.DataFrame([
            {
                'Algorithm': r.algorithm_name,
                'Dataset': r.dataset_name,
                'File': r.file_name,
                'Compression_Ratio': r.compression_ratio,
                'Total_Time': r.total_time,
                'Original_Size_MB': r.original_size / (1024 * 1024),
                'Compressed_Size_MB': r.compressed_size / (1024 * 1024),
                'Throughput': r.throughput_mbps,
                'Space_Saved_%': r.compression_percentage
            }
            for r in results if r.success
        ])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Compression Ratio vs Time', 'Throughput by Algorithm', 
                          'File Size Impact', 'Space Savings Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Scatter plot of compression ratio vs time
        for algo in df['Algorithm'].unique():
            algo_data = df[df['Algorithm'] == algo]
            fig.add_trace(
                go.Scatter(
                    x=algo_data['Total_Time'],
                    y=algo_data['Compression_Ratio'],
                    mode='markers',
                    name=algo,
                    text=algo_data['File'],
                    hovertemplate='<b>%{text}</b><br>Time: %{x:.3f}s<br>Ratio: %{y:.2f}x<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Plot 2: Box plot of throughput
        for algo in df['Algorithm'].unique():
            algo_data = df[df['Algorithm'] == algo]
            fig.add_trace(
                go.Box(
                    y=algo_data['Throughput'],
                    name=algo,
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Plot 3: File size impact
        fig.add_trace(
            go.Scatter(
                x=df['Original_Size_MB'],
                y=df['Compression_Ratio'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df['Total_Time'],
                    colorscale='Viridis',
                    colorbar=dict(title="Time (s)", x=1.02),
                    showscale=True
                ),
                text=df['Algorithm'],
                hovertemplate='<b>%{text}</b><br>Size: %{x:.2f}MB<br>Ratio: %{y:.2f}x<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Plot 4: Space savings histogram
        fig.add_trace(
            go.Histogram(
                x=df['Space_Saved_%'],
                nbinsx=20,
                name='Space Saved',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Compression Algorithm Performance Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Total Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Compression Ratio", row=1, col=1)
        
        fig.update_xaxes(title_text="Algorithm", row=1, col=2)
        fig.update_yaxes(title_text="Throughput (MB/s)", row=1, col=2)
        
        fig.update_xaxes(title_text="Original File Size (MB)", row=2, col=1, type="log")
        fig.update_yaxes(title_text="Compression Ratio", row=2, col=1)
        
        fig.update_xaxes(title_text="Space Saved (%)", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        return fig
    
    def plot_pipeline_breakdown(self, pipeline_results: Dict[str, Any],
                               save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot pipeline stage breakdown.
        
        Args:
            pipeline_results: Pipeline results with stage information
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        if 'stage_results' not in pipeline_results:
            raise ValueError("Pipeline results must contain stage_results")
        
        stages = pipeline_results['stage_results']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Extract stage data
        stage_names = [stage['stage_name'] for stage in stages]
        input_sizes = [stage['input_size'] for stage in stages]
        output_sizes = [stage['output_size'] for stage in stages]
        compression_ratios = [stage['input_size'] / stage['output_size'] if stage['output_size'] > 0 else 0 
                            for stage in stages]
        
        # Plot 1: Size reduction through pipeline
        x = np.arange(len(stage_names))
        width = 0.35
        
        ax1.bar(x - width/2, input_sizes, width, label='Input Size', alpha=0.8)
        ax1.bar(x + width/2, output_sizes, width, label='Output Size', alpha=0.8)
        
        ax1.set_xlabel('Pipeline Stages', fontsize=12)
        ax1.set_ylabel('Size (bytes)', fontsize=12)
        ax1.set_title('Data Size Through Pipeline Stages', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(stage_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Compression ratio per stage
        bars = ax2.bar(stage_names, compression_ratios, alpha=0.8, color=self.colors[:len(stage_names)])
        
        ax2.set_xlabel('Pipeline Stages', fontsize=12)
        ax2.set_ylabel('Compression Ratio', fontsize=12)
        ax2.set_title('Compression Ratio by Stage', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, ratio in zip(bars, compression_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{ratio:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_summary_report(summary: BenchmarkSummary, output_dir: Path) -> None:
    """
    Create a comprehensive visual report from benchmark summary.
    
    Args:
        summary: BenchmarkSummary object
        output_dir: Directory to save report files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = CompressionVisualizer()
    
    # Generate all plots
    plots = [
        ('compression_ratios.png', visualizer.plot_compression_ratios),
        ('performance_comparison.png', visualizer.plot_performance_comparison),
        ('algorithm_heatmap.png', lambda r, p: visualizer.plot_algorithm_heatmap(r, 'compression_ratio', p))
    ]
    
    print("📊 Generating visualization report...")
    
    for filename, plot_func in plots:
        try:
            filepath = output_dir / filename
            fig = plot_func(summary.results, filepath)
            plt.close(fig)  # Free memory
            print(f"   ✅ Created: {filepath}")
        except Exception as e:
            print(f"   ❌ Failed to create {filename}: {str(e)}")
    
    # Create interactive dashboard
    try:
        dashboard = visualizer.create_interactive_dashboard(summary.results)
        dashboard_path = output_dir / 'interactive_dashboard.html'
        dashboard.write_html(str(dashboard_path))
        print(f"   ✅ Created interactive dashboard: {dashboard_path}")
    except Exception as e:
        print(f"   ❌ Failed to create interactive dashboard: {str(e)}")
    
    print(f"📈 Report saved to: {output_dir}") 