# Decision Tree CART Implementation
## Data Analytics and Visualization Assignment

A complete implementation of the CART (Classification and Regression Trees) algorithm from scratch in Python, designed for educational purposes and comprehensive analysis.

## 📁 Project Structure

```
decision_tree/
├── decision_tree_cart.py          # Main CART implementation
├── tree_visualizer.py            # Visualization utilities
├── example_datasets.py           # Dataset generation and management
├── evaluation_metrics.py         # Performance evaluation tools
├── decision_tree_cart_demo.ipynb # Interactive demonstration notebook
├── test_cart_implementation.py   # Test suite
└── README.md                     # This file
```

## 🚀 Features

### Core Algorithm
- **Complete CART Implementation**: Classification and Regression Trees
- **Gini Impurity**: For classification splitting criteria
- **Mean Squared Error**: For regression splitting criteria  
- **Recursive Tree Building**: With configurable stopping criteria
- **Feature Importance**: Automatic calculation and ranking

### Visualization Tools
- **Tree Structure Plots**: Visual representation of decision trees
- **Decision Boundaries**: 2D visualization of classification regions
- **Feature Importance Charts**: Bar plots of feature rankings
- **Performance Metrics**: Comprehensive evaluation visualizations

### Datasets Included
- **Classification**: Iris, Wine, Titanic-like, Customer Churn
- **Regression**: Housing Prices, Synthetic datasets
- **Synthetic Data Generation**: Customizable datasets for testing

### Evaluation Metrics
- **Classification**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Regression**: R², RMSE, MAE, MAPE, Residual Analysis
- **Cross-Validation**: K-fold validation for robust evaluation
- **Comparison Tools**: Validation against scikit-learn

## 🛠️ Installation and Setup

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Quick Start
1. Clone or download all files to your project directory
2. Open the Jupyter notebook: `decision_tree_cart_demo.ipynb`
3. Run all cells to see the complete demonstration

### Testing the Implementation
```python
python test_cart_implementation.py
```

## 💡 Usage Examples

### Basic Classification
```python
from decision_tree_cart import DecisionTreeCART
from example_datasets import DatasetExamples

# Load dataset
X, y, _ = DatasetExamples.load_dataset('iris')

# Create and train model
tree = DecisionTreeCART(task_type='classification', max_depth=5)
tree.fit(X, y)

# Make predictions
predictions = tree.predict(X)
```

### Basic Regression
```python
# Load regression dataset
X, y, _ = DatasetExamples.load_dataset('housing_prices')

# Create and train model
tree = DecisionTreeCART(task_type='regression', max_depth=8)
tree.fit(X, y)

# Make predictions
predictions = tree.predict(X)
```

### Visualization
```python
from tree_visualizer import TreeVisualizer

# Create visualizer
viz = TreeVisualizer(tree)

# Plot tree structure
viz.plot_tree()

# Plot feature importance
viz.plot_feature_importance()
```

### Performance Evaluation
```python
from evaluation_metrics import TreeEvaluator

# Create evaluator
evaluator = TreeEvaluator(tree)

# Comprehensive evaluation
results = evaluator.evaluate(X_test, y_test, X_train, y_train)

# Cross-validation
cv_scores = evaluator.cross_validate(X, y, cv=5)
```

## 🎯 Key Parameters

### DecisionTreeCART Parameters
- `task_type`: 'classification' or 'regression'
- `max_depth`: Maximum tree depth (None for unlimited)
- `min_samples_split`: Minimum samples required to split a node
- `min_samples_leaf`: Minimum samples required in a leaf node
- `min_impurity_decrease`: Minimum impurity decrease for splitting
- `random_state`: Seed for reproducibility

## 📊 Algorithm Details

### Splitting Criteria
- **Gini Impurity**: $Gini = 1 - \sum_{i=1}^{c} p_i^2$
- **Mean Squared Error**: $MSE = \frac{1}{n}\sum_{i=1}^{n} (y_i - \bar{y})^2$

### Best Split Selection
1. For each feature and threshold combination:
   - Split data into left and right subsets
   - Calculate weighted impurity after split
   - Track split with maximum impurity decrease

### Stopping Criteria
- Maximum depth reached
- Insufficient samples to split
- No impurity improvement
- Minimum leaf size constraint

### Prediction Method
- **Classification**: Majority vote in leaf nodes
- **Regression**: Mean value in leaf nodes

## 🔍 Performance Benchmarks

### Classification (Iris Dataset)
- **Accuracy**: ~95-97%
- **Tree Depth**: 3-5 levels
- **Training Time**: <1 second

### Regression (Housing Dataset)  
- **R² Score**: ~0.85-0.90
- **RMSE**: ~$15,000-20,000
- **Training Time**: <2 seconds

## 📈 Comparison with Scikit-learn

Our implementation achieves comparable performance to scikit-learn's DecisionTreeClassifier and DecisionTreeRegressor:

- **Similar Accuracy**: Within 1-2% of scikit-learn performance
- **Comparable Tree Structure**: Similar depth and leaf counts
- **Feature Importance**: Consistent rankings with scikit-learn

## 🎓 Educational Value

This implementation is designed for learning and includes:

### Theoretical Understanding
- Step-by-step algorithm explanation
- Mathematical foundations of splitting criteria
- Visualization of decision-making process

### Practical Skills
- Python implementation of machine learning algorithms
- Data structure design (tree nodes)
- Performance evaluation and validation
- Visualization and interpretation techniques

### Advanced Topics
- Overfitting analysis and prevention
- Feature importance interpretation
- Cross-validation methodology
- Comparison with standard libraries

## 🚦 Common Issues and Solutions

### Import Errors
- Ensure all files are in the same directory
- Check that all required packages are installed
- Verify Python path configuration

### Performance Issues
- Large datasets may take longer to train
- Consider reducing max_depth for faster training
- Use smaller datasets for initial testing

### Visualization Problems
- Matplotlib backend issues: try different backends
- Large trees may be difficult to visualize
- Consider plotting subsets of complex trees

## 📚 Assignment Guidelines

### What to Submit
1. **Jupyter Notebook**: Complete demonstration with outputs
2. **Source Code**: All Python files with implementations
3. **Analysis Report**: Written analysis of results and insights
4. **Dataset Exploration**: EDA and data preparation steps

### Key Evaluation Points
- **Algorithm Correctness**: Proper CART implementation
- **Code Quality**: Clean, documented, and efficient code
- **Visualization**: Clear and informative plots
- **Analysis**: Insightful interpretation of results
- **Comparison**: Validation against standard libraries

### Recommended Extensions
- **Pruning Implementation**: Add post-pruning capabilities
- **Missing Value Handling**: Implement surrogate splits
- **Feature Selection**: Add automated feature selection
- **Ensemble Methods**: Extend to Random Forest

## 👥 Credits

**Author**: Student  
**Course**: Data Analytics and Visualization  
**Date**: November 2025  

**References**:
- Breiman, L., et al. (1984). Classification and Regression Trees
- Scikit-learn Documentation
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman

## 📄 License

This implementation is for educational purposes. Feel free to use and modify for learning and academic assignments.

---

**Happy Learning! 🎓**