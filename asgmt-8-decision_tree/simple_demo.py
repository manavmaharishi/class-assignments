"""
Simple demonstration of CART Decision Tree implementation
Quick showcase of key features

Run this script to see your CART implementation in action!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decision_tree_cart import DecisionTreeCART
from example_datasets import DatasetExamples
from evaluation_metrics import TreeEvaluator
from tree_visualizer import TreeVisualizer

def main():
    print("🌳 DECISION TREE CART DEMONSTRATION")
    print("="*50)
    
    # 1. Classification Example
    print("\n📊 CLASSIFICATION EXAMPLE: IRIS DATASET")
    print("-" * 40)
    
    # Load and split iris data
    X_iris, y_iris, _ = DatasetExamples.load_dataset('iris')
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)
    
    print(f"Dataset: {X_iris.shape[0]} samples, {X_iris.shape[1]} features")
    print(f"Classes: {sorted(y_iris.unique())}")
    
    # Train classifier
    tree_clf = DecisionTreeCART(task_type='classification', max_depth=4, random_state=42)
    tree_clf.fit(X_train, y_train)
    
    # Evaluate
    evaluator_clf = TreeEvaluator(tree_clf)
    results_clf = evaluator_clf.evaluate(X_test, y_test, show_plots=False)
    
    print(f"✅ Classification Results:")
    print(f"   • Accuracy: {results_clf['accuracy']:.3f}")
    print(f"   • Tree Depth: {results_clf['tree_depth']}")
    print(f"   • Number of Leaves: {results_clf['n_leaves']}")
    
    # Show feature importance
    importance = tree_clf.feature_importance()
    print(f"   • Most Important Feature: {max(importance.items(), key=lambda x: x[1])[0]}")
    
    # 2. Regression Example
    print("\n🏠 REGRESSION EXAMPLE: HOUSING PRICES")
    print("-" * 40)
    
    # Load housing data
    X_house, y_house, _ = DatasetExamples.load_dataset('housing_prices', n_samples=500)
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_house, y_house, test_size=0.3, random_state=42)
    
    print(f"Dataset: {X_house.shape[0]} samples, {X_house.shape[1]} features")
    print(f"Price range: ${y_house.min():,.0f} - ${y_house.max():,.0f}")
    
    # Train regressor
    tree_reg = DecisionTreeCART(task_type='regression', max_depth=6, min_samples_split=5, random_state=42)
    tree_reg.fit(X_train_h, y_train_h)
    
    # Evaluate
    evaluator_reg = TreeEvaluator(tree_reg)
    results_reg = evaluator_reg.evaluate(X_test_h, y_test_h, show_plots=False)
    
    print(f"✅ Regression Results:")
    print(f"   • R² Score: {results_reg['r2']:.3f}")
    print(f"   • RMSE: ${results_reg['rmse']:,.0f}")
    print(f"   • Tree Depth: {results_reg['tree_depth']}")
    print(f"   • Number of Leaves: {results_reg['n_leaves']}")
    
    # Show most important features
    importance_reg = tree_reg.feature_importance()
    top_features = sorted(importance_reg.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"   • Top Features:")
    for i, (feat, score) in enumerate(top_features):
        print(f"     {i+1}. {feat}: {score:.3f}")
    
    # 3. Make some predictions
    print("\n🎯 PREDICTION EXAMPLES")
    print("-" * 25)
    
    # Classification predictions
    sample_iris = X_test.iloc[:3]
    pred_iris = tree_clf.predict(sample_iris)
    true_iris = y_test.iloc[:3]
    
    print("Classification Predictions:")
    for i in range(3):
        print(f"   Sample {i+1}: Predicted={pred_iris[i]}, Actual={true_iris.iloc[i]}")
    
    # Regression predictions  
    sample_house = X_test_h.iloc[:3]
    pred_house = tree_reg.predict(sample_house)
    true_house = y_test_h.iloc[:3]
    
    print("\nRegression Predictions:")
    for i in range(3):
        print(f"   House {i+1}: Predicted=${pred_house[i]:,.0f}, Actual=${true_house.iloc[i]:,.0f}")
    
    # 4. Quick visualization
    print("\n📈 CREATING VISUALIZATIONS...")
    print("-" * 30)
    
    try:
        # Feature importance plot
        viz_clf = TreeVisualizer(tree_clf)
        print("• Plotting classification feature importance...")
        viz_clf.plot_feature_importance(figsize=(8, 4))
        
        viz_reg = TreeVisualizer(tree_reg)
        print("• Plotting regression feature importance...")
        viz_reg.plot_feature_importance(figsize=(10, 4))
        
        print("✅ Visualizations created successfully!")
        
    except Exception as e:
        print(f"⚠️  Visualization skipped (display may not be available): {e}")
    
    print("\n" + "="*50)
    print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("📚 Your CART implementation is working perfectly!")
    print("\n💡 Next steps:")
    print("   1. Open the Jupyter notebook: decision_tree_cart_demo.ipynb")
    print("   2. Run all cells to see the full interactive demonstration")
    print("   3. Experiment with different parameters and datasets")
    
    return tree_clf, tree_reg

if __name__ == "__main__":
    classifier, regressor = main()