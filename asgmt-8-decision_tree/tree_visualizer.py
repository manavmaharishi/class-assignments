"""
Visualization utilities for Decision Tree CART implementation
Provides tree plotting and analysis visualization functions

Author: Student
Date: November 2025
Course: Data Analytics and Visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from typing import Optional, Tuple, Dict, List
from decision_tree_cart import DecisionTreeCART, Node


class TreeVisualizer:
    """
    Visualization class for Decision Trees
    """
    
    def __init__(self, tree: DecisionTreeCART):
        """
        Initialize the visualizer.
        
        Args:
            tree: Trained DecisionTreeCART instance
        """
        self.tree = tree
        self.node_positions = {}
        
    def plot_tree(self, 
                  figsize: Tuple[int, int] = (15, 10),
                  node_size: float = 1500,
                  font_size: int = 10,
                  save_path: Optional[str] = None) -> None:
        """
        Plot the decision tree structure.
        
        Args:
            figsize: Figure size (width, height)
            node_size: Size of the nodes
            font_size: Font size for text
            save_path: Path to save the figure (optional)
        """
        if self.tree.root is None:
            raise ValueError("Tree has not been trained yet.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate positions for all nodes
        self._calculate_positions(self.tree.root, 0, 0, 1.0)
        
        # Draw nodes and edges
        self._draw_tree(ax, self.tree.root, node_size, font_size)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.title(f'Decision Tree Visualization\n'
                 f'Task: {self.tree.task_type.title()}, '
                 f'Depth: {self.tree.get_depth()}, '
                 f'Leaves: {self.tree.get_n_leaves()}',
                 fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _calculate_positions(self, node: Node, depth: int, x: float, width: float) -> None:
        """
        Calculate positions for tree nodes recursively.
        
        Args:
            node: Current node
            depth: Current depth
            x: X position
            width: Available width for subtree
        """
        if node is None:
            return
        
        # Calculate y position based on depth
        max_depth = self.tree.get_depth()
        y = 1.0 - (depth / max_depth) if max_depth > 0 else 0.5
        
        self.node_positions[id(node)] = (x, y)
        
        if node.value is None:  # Internal node
            # Recursively position children
            child_width = width / 2
            if node.left:
                self._calculate_positions(node.left, depth + 1, x - child_width/2, child_width)
            if node.right:
                self._calculate_positions(node.right, depth + 1, x + child_width/2, child_width)
    
    def _draw_tree(self, ax, node: Node, node_size: float, font_size: int) -> None:
        """
        Draw the tree recursively.
        
        Args:
            ax: Matplotlib axis
            node: Current node
            node_size: Size of nodes
            font_size: Font size
        """
        if node is None:
            return
        
        node_id = id(node)
        x, y = self.node_positions[node_id]
        
        # Choose colors based on node type
        if node.value is not None:  # Leaf node
            color = 'lightgreen'
            if self.tree.task_type == 'classification':
                text = f'Class: {node.value}\\nSamples: {node.samples}\\nGini: {node.impurity:.3f}'
            else:
                text = f'Value: {node.value:.3f}\\nSamples: {node.samples}\\nMSE: {node.impurity:.3f}'
        else:  # Internal node
            color = 'lightblue'
            feature_name = self.tree.feature_names[node.feature]
            text = f'{feature_name} ≤ {node.threshold:.3f}\\nSamples: {node.samples}\\nImpurity: {node.impurity:.3f}'
        
        # Draw node
        circle = plt.Circle((x, y), 0.05, color=color, ec='black', linewidth=2)
        ax.add_patch(circle)
        
        # Add text
        ax.text(x, y, text, ha='center', va='center', fontsize=font_size-2,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
        # Draw edges to children
        if node.left:
            left_id = id(node.left)
            left_x, left_y = self.node_positions[left_id]
            ax.plot([x, left_x], [y, left_y], 'k-', linewidth=2)
            
            # Add "True" label
            mid_x, mid_y = (x + left_x) / 2, (y + left_y) / 2
            ax.text(mid_x - 0.02, mid_y, 'True', fontsize=font_size-3, 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            
            self._draw_tree(ax, node.left, node_size, font_size)
        
        if node.right:
            right_id = id(node.right)
            right_x, right_y = self.node_positions[right_id]
            ax.plot([x, right_x], [y, right_y], 'k-', linewidth=2)
            
            # Add "False" label
            mid_x, mid_y = (x + right_x) / 2, (y + right_y) / 2
            ax.text(mid_x + 0.02, mid_y, 'False', fontsize=font_size-3,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            
            self._draw_tree(ax, node.right, node_size, font_size)
    
    def plot_feature_importance(self, 
                              figsize: Tuple[int, int] = (10, 6),
                              save_path: Optional[str] = None) -> None:
        """
        Plot feature importance.
        
        Args:
            figsize: Figure size
            save_path: Path to save the figure (optional)
        """
        importance = self.tree.feature_importance()
        
        if not importance:
            print("No feature importance available.")
            return
        
        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        features, scores = zip(*sorted_features)
        
        plt.figure(figsize=figsize)
        bars = plt.bar(range(len(features)), scores, color='skyblue', alpha=0.7, edgecolor='navy')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Features', fontsize=12, fontweight='bold')
        plt.ylabel('Importance Score', fontsize=12, fontweight='bold')
        plt.title('Feature Importance in Decision Tree', fontsize=14, fontweight='bold')
        plt.xticks(range(len(features)), features, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_tree_statistics(self, 
                           figsize: Tuple[int, int] = (12, 8),
                           save_path: Optional[str] = None) -> None:
        """
        Plot various tree statistics.
        
        Args:
            figsize: Figure size
            save_path: Path to save the figure (optional)
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Tree depth distribution
        depths = []
        samples_per_depth = []
        self._collect_depth_stats(self.tree.root, 0, depths, samples_per_depth)
        
        depth_counts = {}
        for d in depths:
            depth_counts[d] = depth_counts.get(d, 0) + 1
        
        ax1.bar(depth_counts.keys(), depth_counts.values(), color='lightcoral', alpha=0.7)
        ax1.set_xlabel('Depth Level')
        ax1.set_ylabel('Number of Nodes')
        ax1.set_title('Nodes per Depth Level')
        ax1.grid(axis='y', alpha=0.3)
        
        # Samples per depth
        depth_samples = {}
        for d, s in zip(depths, samples_per_depth):
            if d not in depth_samples:
                depth_samples[d] = []
            depth_samples[d].append(s)
        
        avg_samples = {d: np.mean(samples) for d, samples in depth_samples.items()}
        ax2.bar(avg_samples.keys(), avg_samples.values(), color='lightgreen', alpha=0.7)
        ax2.set_xlabel('Depth Level')
        ax2.set_ylabel('Average Samples')
        ax2.set_title('Average Samples per Depth Level')
        ax2.grid(axis='y', alpha=0.3)
        
        # Feature usage count
        feature_usage = {}
        self._collect_feature_usage(self.tree.root, feature_usage)
        
        if feature_usage:
            features = list(feature_usage.keys())
            usage_counts = list(feature_usage.values())
            
            ax3.barh(range(len(features)), usage_counts, color='lightsalmon', alpha=0.7)
            ax3.set_yticks(range(len(features)))
            ax3.set_yticklabels([self.tree.feature_names[f] for f in features])
            ax3.set_xlabel('Usage Count')
            ax3.set_title('Feature Usage in Splits')
            ax3.grid(axis='x', alpha=0.3)
        
        # Tree summary statistics
        stats_text = f"""Tree Statistics:
        
Task Type: {self.tree.task_type.title()}
Maximum Depth: {self.tree.get_depth()}
Number of Leaves: {self.tree.get_n_leaves()}
Total Nodes: {len(depths)}

Parameters:
Max Depth: {self.tree.max_depth}
Min Samples Split: {self.tree.min_samples_split}
Min Samples Leaf: {self.tree.min_samples_leaf}
Min Impurity Decrease: {self.tree.min_impurity_decrease}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _collect_depth_stats(self, node: Node, depth: int, depths: List[int], samples: List[int]) -> None:
        """
        Collect depth and sample statistics recursively.
        
        Args:
            node: Current node
            depth: Current depth
            depths: List to store depths
            samples: List to store sample counts
        """
        if node is None:
            return
        
        depths.append(depth)
        samples.append(node.samples)
        
        if node.left:
            self._collect_depth_stats(node.left, depth + 1, depths, samples)
        if node.right:
            self._collect_depth_stats(node.right, depth + 1, depths, samples)
    
    def _collect_feature_usage(self, node: Node, usage_dict: Dict[int, int]) -> None:
        """
        Collect feature usage statistics recursively.
        
        Args:
            node: Current node
            usage_dict: Dictionary to store feature usage counts
        """
        if node is None or node.value is not None:
            return
        
        usage_dict[node.feature] = usage_dict.get(node.feature, 0) + 1
        
        self._collect_feature_usage(node.left, usage_dict)
        self._collect_feature_usage(node.right, usage_dict)


def plot_decision_boundary_2d(tree: DecisionTreeCART, 
                              X: np.ndarray, 
                              y: np.ndarray,
                              feature_names: List[str] = None,
                              figsize: Tuple[int, int] = (10, 8),
                              resolution: int = 100,
                              save_path: Optional[str] = None) -> None:
    """
    Plot decision boundary for 2D classification problems.
    
    Args:
        tree: Trained DecisionTreeCART instance
        X: Feature matrix (should be 2D)
        y: Target values
        feature_names: Names of the two features
        figsize: Figure size
        resolution: Resolution of the decision boundary
        save_path: Path to save the figure (optional)
    """
    if X.shape[1] != 2:
        raise ValueError("This function only works with 2D data (2 features).")
    
    if tree.task_type != 'classification':
        raise ValueError("Decision boundary plotting is only available for classification.")
    
    # Create a mesh
    h = (X[:, 0].max() - X[:, 0].min()) / resolution
    xx, yy = np.meshgrid(np.arange(X[:, 0].min() - h, X[:, 0].max() + h, h),
                        np.arange(X[:, 1].min() - h, X[:, 1].max() + h, h))
    
    # Make predictions on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = tree.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=figsize)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    
    # Plot data points
    unique_classes = np.unique(y)
    colors = plt.cm.RdYlBu(np.linspace(0, 1, len(unique_classes)))
    
    for i, cls in enumerate(unique_classes):
        mask = y == cls
        plt.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], label=f'Class {cls}', 
                   s=50, alpha=0.8, edgecolor='black')
    
    plt.xlabel(feature_names[0] if feature_names else 'Feature 1', fontsize=12, fontweight='bold')
    plt.ylabel(feature_names[1] if feature_names else 'Feature 2', fontsize=12, fontweight='bold')
    plt.title('Decision Tree Decision Boundary', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()