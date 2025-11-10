"""
Visual Tree Demonstration Script
Shows the actual decision tree structure and visualizations

Run this to see your trees visually!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decision_tree_cart import DecisionTreeCART
from example_datasets import DatasetExamples
from tree_visualizer import TreeVisualizer, plot_decision_boundary_2d

def show_tree_structure():
    """Display a simple tree structure as text"""
    print("🌳 DECISION TREE STRUCTURE VISUALIZATION")
    print("="*50)
    
    # Create a simple dataset for clear visualization
    X_iris, y_iris, _ = DatasetExamples.load_dataset('iris')
    
    # Use only first 2 features for clearer visualization
    X_simple = X_iris.iloc[:, :2]  # Sepal length and width
    
    # Train a shallow tree for clear visualization
    tree = DecisionTreeCART(
        task_type='classification', 
        max_depth=3, 
        min_samples_split=10,
        random_state=42
    )
    tree.fit(X_simple, y_iris)
    
    print(f"Dataset: Iris (simplified - 2 features)")
    print(f"Features used: {list(X_simple.columns)}")
    print(f"Tree depth: {tree.get_depth()}")
    print(f"Number of leaves: {tree.get_n_leaves()}")
    
    # Print tree structure in text format
    def print_tree_structure(node, depth=0, feature_names=None):
        """Print tree structure as text"""
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(X_simple.columns))]
        
        indent = "  " * depth
        
        if node.value is not None:  # Leaf node
            print(f"{indent}├─ LEAF: Class {node.value} (samples: {node.samples}, gini: {node.impurity:.3f})")
        else:  # Internal node
            feature_name = feature_names[node.feature]
            print(f"{indent}├─ {feature_name} <= {node.threshold:.3f} (samples: {node.samples}, gini: {node.impurity:.3f})")
            
            if node.left:
                print(f"{indent}│  └─ TRUE:")
                print_tree_structure(node.left, depth + 2, feature_names)
            
            if node.right:
                print(f"{indent}│  └─ FALSE:")
                print_tree_structure(node.right, depth + 2, feature_names)
    
    print("\n📊 TREE STRUCTURE (Text Format):")
    print("-" * 40)
    print_tree_structure(tree.root, feature_names=list(X_simple.columns))
    
    return tree, X_simple, y_iris

def create_visual_tree():
    """Create visual tree plots"""
    print("\n\n🎨 CREATING VISUAL TREE PLOTS...")
    print("="*40)
    
    tree, X_simple, y_iris = show_tree_structure()
    
    try:
        # Create visualizer
        viz = TreeVisualizer(tree)
        
        print("1. Creating tree structure diagram...")
        viz.plot_tree(figsize=(14, 8), font_size=8)
        
        print("2. Creating feature importance plot...")
        viz.plot_feature_importance(figsize=(8, 5))
        
        print("3. Creating decision boundary plot...")
        plot_decision_boundary_2d(
            tree=tree,
            X=X_simple.values,
            y=y_iris.values,
            feature_names=list(X_simple.columns),
            figsize=(10, 6)
        )
        
        print("4. Creating tree statistics...")
        viz.plot_tree_statistics(figsize=(12, 8))
        
        print("✅ All visualizations created!")
        
    except Exception as e:
        print(f"⚠️  Could not create plots (may need display): {e}")
        print("💡 Try running in Jupyter notebook for best visualization!")
    
    return tree

def show_prediction_path():
    """Show how a prediction travels through the tree"""
    print("\n\n🎯 PREDICTION PATH DEMONSTRATION")
    print("="*40)
    
    tree, X_simple, y_iris = show_tree_structure()
    
    # Take a sample for prediction
    sample = X_simple.iloc[0]  # First sample
    actual_class = y_iris.iloc[0]
    
    print(f"Sample features: {dict(sample)}")
    print(f"Actual class: {actual_class}")
    
    # Manual prediction to show path
    def trace_prediction(node, sample_values, depth=0, path="Root"):
        """Trace the prediction path through the tree"""
        indent = "  " * depth
        
        if node.value is not None:  # Leaf node
            print(f"{indent}└─ 🏁 PREDICTION: Class {node.value}")
            return node.value
        else:
            feature_name = list(X_simple.columns)[node.feature]
            feature_value = sample_values[node.feature]
            threshold = node.threshold
            
            if feature_value <= threshold:
                print(f"{indent}├─ {feature_name} = {feature_value:.3f} <= {threshold:.3f} → Go LEFT")
                return trace_prediction(node.left, sample_values, depth + 1, path + " → Left")
            else:
                print(f"{indent}├─ {feature_name} = {feature_value:.3f} > {threshold:.3f} → Go RIGHT")
                return trace_prediction(node.right, sample_values, depth + 1, path + " → Right")
    
    print(f"\n📍 TRACING PREDICTION PATH:")
    predicted = trace_prediction(tree.root, sample.values)
    
    print(f"\n✅ Final prediction: Class {predicted}")
    print(f"🎯 Correct prediction: {'YES' if predicted == actual_class else 'NO'}")

def interactive_tree_explorer():
    """Create an interactive tree exploration"""
    print("\n\n🔍 INTERACTIVE TREE EXPLORER")
    print("="*35)
    
    # Create different tree sizes for comparison
    X_iris, y_iris, _ = DatasetExamples.load_dataset('iris')
    
    depths = [2, 3, 5]
    trees = {}
    
    for depth in depths:
        tree = DecisionTreeCART(
            task_type='classification',
            max_depth=depth,
            random_state=42
        )
        tree.fit(X_iris, y_iris)
        trees[depth] = tree
        
        print(f"\nTree with max_depth={depth}:")
        print(f"  • Actual depth: {tree.get_depth()}")
        print(f"  • Number of leaves: {tree.get_n_leaves()}")
        
        # Test accuracy
        predictions = tree.predict(X_iris)
        accuracy = np.mean(predictions == y_iris.values)
        print(f"  • Training accuracy: {accuracy:.3f}")
        
        # Show feature importance
        importance = tree.feature_importance()
        top_feature = max(importance.items(), key=lambda x: x[1])
        print(f"  • Most important feature: {top_feature[0]} ({top_feature[1]:.3f})")
    
    return trees

def main():
    """Run all tree visualization demonstrations"""
    print("🌳 COMPLETE TREE VISUALIZATION DEMO")
    print("="*50)
    
    # 1. Show tree structure as text
    tree, X_simple, y_iris = show_tree_structure()
    
    # 2. Create visual plots
    create_visual_tree()
    
    # 3. Show prediction path
    show_prediction_path()
    
    # 4. Interactive exploration
    trees = interactive_tree_explorer()
    
    print("\n" + "="*50)
    print("🎉 TREE VISUALIZATION COMPLETE!")
    print("\n💡 To see the visual plots:")
    print("   • Run this script in an environment with display support")
    print("   • Or better yet, open 'decision_tree_cart_demo.ipynb' in Jupyter")
    print("   • The notebook has interactive plots and full visualizations!")
    
    return tree, trees

if __name__ == "__main__":
    main()