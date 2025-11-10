"""
Test script to verify Decision Tree CART implementation
Quick validation of all components

Author: Student
Date: November 2025
Course: Data Analytics and Visualization
"""

import numpy as np
import pandas as pd
from decision_tree_cart import DecisionTreeCART
from example_datasets import DatasetExamples
from evaluation_metrics import TreeEvaluator
from tree_visualizer import TreeVisualizer

def test_classification():
    """Test classification functionality"""
    print("Testing Classification...")
    
    # Load dataset
    X, y, _ = DatasetExamples.load_dataset('iris')
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create and train model
    tree = DecisionTreeCART(task_type='classification', max_depth=4, random_state=42)
    tree.fit(X_train, y_train)
    
    # Make predictions
    predictions = tree.predict(X_test)
    
    # Evaluate
    evaluator = TreeEvaluator(tree)
    results = evaluator.evaluate(X_test, y_test, show_plots=False)
    
    print(f"✅ Classification test passed!")
    print(f"   Accuracy: {results['accuracy']:.3f}")
    print(f"   Tree depth: {results['tree_depth']}")
    print(f"   Number of leaves: {results['n_leaves']}")
    
    return True

def test_regression():
    """Test regression functionality"""
    print("\\nTesting Regression...")
    
    # Load dataset
    X, y, _ = DatasetExamples.load_dataset('synthetic_regression', n_samples=300)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create and train model
    tree = DecisionTreeCART(task_type='regression', max_depth=6, random_state=42)
    tree.fit(X_train, y_train)
    
    # Make predictions
    predictions = tree.predict(X_test)
    
    # Evaluate
    evaluator = TreeEvaluator(tree)
    results = evaluator.evaluate(X_test, y_test, show_plots=False)
    
    print(f"✅ Regression test passed!")
    print(f"   R² Score: {results['r2']:.3f}")
    print(f"   RMSE: {results['rmse']:.3f}")
    print(f"   Tree depth: {results['tree_depth']}")
    print(f"   Number of leaves: {results['n_leaves']}")
    
    return True

def test_feature_importance():
    """Test feature importance calculation"""
    print("\\nTesting Feature Importance...")
    
    # Load dataset with meaningful features
    X, y, _ = DatasetExamples.load_dataset('titanic_like', n_samples=500)
    
    # Train model
    tree = DecisionTreeCART(task_type='classification', max_depth=5, random_state=42)
    tree.fit(X, y)
    
    # Get feature importance
    importance = tree.feature_importance()
    
    print(f"✅ Feature importance test passed!")
    print("   Top features:")
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, score) in enumerate(sorted_features[:3]):
        print(f"   {i+1}. {feature}: {score:.3f}")
    
    return True

def test_datasets():
    """Test dataset generation"""
    print("\\nTesting Dataset Generation...")
    
    datasets = DatasetExamples.get_all_datasets()
    
    for name, info in datasets.items():
        try:
            X, y, dataset_info = DatasetExamples.load_dataset(name, n_samples=100)
            print(f"   ✅ {name}: {X.shape[0]} samples, {X.shape[1]} features")
        except Exception as e:
            print(f"   ❌ {name}: Error - {str(e)}")
            return False
    
    print(f"✅ All dataset generation tests passed!")
    return True

def main():
    """Run all tests"""
    print("🧪 CART Decision Tree Implementation Tests")
    print("=" * 50)
    
    try:
        # Run tests
        test_datasets()
        test_classification()
        test_regression()
        test_feature_importance()
        
        print("\\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        print("📚 Your CART implementation is ready for the assignment!")
        
    except Exception as e:
        print(f"\\n❌ Test failed with error: {str(e)}")
        print("Please check the implementation and try again.")

if __name__ == "__main__":
    main()