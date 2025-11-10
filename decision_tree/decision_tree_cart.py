"""
Decision Tree Implementation using CART Algorithm
Classification and Regression Trees (CART) for Data Analytics

Author: Student
Date: November 2025
Course: Data Analytics and Visualization
"""

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional, Any


class Node:
    """
    Represents a node in the decision tree.
    """
    def __init__(self, 
                 feature: Optional[int] = None, 
                 threshold: Optional[float] = None,
                 left: Optional['Node'] = None, 
                 right: Optional['Node'] = None,
                 value: Optional[Any] = None,
                 samples: int = 0,
                 impurity: float = 0.0):
        """
        Initialize a tree node.
        
        Args:
            feature: Index of feature used for splitting
            threshold: Threshold value for the split
            left: Left child node
            right: Right child node
            value: Prediction value for leaf nodes
            samples: Number of samples in this node
            impurity: Gini impurity or MSE of this node
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.samples = samples
        self.impurity = impurity


class DecisionTreeCART:
    """
    Decision Tree implementation using CART algorithm.
    Supports both classification and regression.
    """
    
    def __init__(self, 
                 task_type: str = 'classification',
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 min_impurity_decrease: float = 0.0,
                 random_state: Optional[int] = None):
        """
        Initialize the Decision Tree.
        
        Args:
            task_type: 'classification' or 'regression'
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required in a leaf
            min_impurity_decrease: Minimum impurity decrease for split
            random_state: Random seed for reproducibility
        """
        self.task_type = task_type
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root = None
        self.feature_names = None
        self.classes = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """
        Calculate Gini impurity for classification.
        
        Args:
            y: Target values
            
        Returns:
            Gini impurity score
        """
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)
    
    def _mse(self, y: np.ndarray) -> float:
        """
        Calculate Mean Squared Error for regression.
        
        Args:
            y: Target values
            
        Returns:
            MSE score
        """
        if len(y) == 0:
            return 0.0
        
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)
    
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """
        Calculate impurity based on task type.
        
        Args:
            y: Target values
            
        Returns:
            Impurity score
        """
        if self.task_type == 'classification':
            return self._gini_impurity(y)
        else:
            return self._mse(y)
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find the best feature and threshold for splitting.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Tuple of (best_feature, best_threshold, best_impurity_decrease)
        """
        best_feature = None
        best_threshold = None
        best_impurity_decrease = 0.0
        current_impurity = self._calculate_impurity(y)
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # Try splits at midpoints between unique values
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                # Split the data
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                # Check if split creates valid children
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                
                # Calculate weighted impurity after split
                n_left, n_right = len(left_y), len(right_y)
                n_total = len(y)
                
                left_impurity = self._calculate_impurity(left_y)
                right_impurity = self._calculate_impurity(right_y)
                
                weighted_impurity = (n_left / n_total) * left_impurity + (n_right / n_total) * right_impurity
                impurity_decrease = current_impurity - weighted_impurity
                
                if impurity_decrease > best_impurity_decrease:
                    best_impurity_decrease = impurity_decrease
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_impurity_decrease
    
    def _create_leaf(self, y: np.ndarray) -> Any:
        """
        Create a leaf node value.
        
        Args:
            y: Target values
            
        Returns:
            Prediction value for the leaf
        """
        if self.task_type == 'classification':
            # Return the most common class
            counter = Counter(y)
            return counter.most_common(1)[0][0]
        else:
            # Return the mean for regression
            return np.mean(y)
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively build the decision tree.
        
        Args:
            X: Feature matrix
            y: Target values
            depth: Current depth in the tree
            
        Returns:
            Root node of the (sub)tree
        """
        n_samples = len(y)
        current_impurity = self._calculate_impurity(y)
        
        # Create leaf node if stopping criteria are met
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            return Node(value=self._create_leaf(y), 
                       samples=n_samples, 
                       impurity=current_impurity)
        
        # Find the best split
        best_feature, best_threshold, best_impurity_decrease = self._find_best_split(X, y)
        
        # Create leaf if no good split found
        if best_feature is None or best_impurity_decrease < self.min_impurity_decrease:
            return Node(value=self._create_leaf(y), 
                       samples=n_samples, 
                       impurity=current_impurity)
        
        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature=best_feature,
                   threshold=best_threshold,
                   left=left_child,
                   right=right_child,
                   samples=n_samples,
                   impurity=current_impurity)
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'DecisionTreeCART':
        """
        Train the decision tree.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Self for method chaining
        """
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Store classes for classification
        if self.task_type == 'classification':
            self.classes = np.unique(y)
        
        # Build the tree
        self.root = self._build_tree(X, y)
        
        return self
    
    def _predict_sample(self, x: np.ndarray, node: Node) -> Any:
        """
        Predict a single sample by traversing the tree.
        
        Args:
            x: Single sample features
            node: Current node
            
        Returns:
            Prediction value
        """
        # If leaf node, return the value
        if node.value is not None:
            return node.value
        
        # Traverse left or right based on feature threshold
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if self.root is None:
            raise ValueError("Tree has not been trained. Call fit() first.")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = []
        for x in X:
            pred = self._predict_sample(x, self.root)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities (for classification only).
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of class probabilities
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks.")
        
        if self.root is None:
            raise ValueError("Tree has not been trained. Call fit() first.")
        
        # For simplicity, return one-hot encoded predictions
        # In a full implementation, you'd traverse to leaves and calculate actual probabilities
        predictions = self.predict(X)
        n_classes = len(self.classes)
        probabilities = np.zeros((len(X), n_classes))
        
        for i, pred in enumerate(predictions):
            class_idx = np.where(self.classes == pred)[0][0]
            probabilities[i, class_idx] = 1.0
        
        return probabilities
    
    def get_depth(self, node: Optional[Node] = None) -> int:
        """
        Get the depth of the tree.
        
        Args:
            node: Starting node (default: root)
            
        Returns:
            Maximum depth of the tree
        """
        if node is None:
            node = self.root
        
        if node is None or node.value is not None:
            return 0
        
        return 1 + max(self.get_depth(node.left), self.get_depth(node.right))
    
    def get_n_leaves(self, node: Optional[Node] = None) -> int:
        """
        Get the number of leaves in the tree.
        
        Args:
            node: Starting node (default: root)
            
        Returns:
            Number of leaf nodes
        """
        if node is None:
            node = self.root
        
        if node is None:
            return 0
        
        if node.value is not None:
            return 1
        
        return self.get_n_leaves(node.left) + self.get_n_leaves(node.right)
    
    def feature_importance(self) -> dict:
        """
        Calculate feature importance based on impurity decrease.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.root is None:
            raise ValueError("Tree has not been trained. Call fit() first.")
        
        importance_scores = np.zeros(len(self.feature_names))
        
        def calculate_importance(node):
            if node is None or node.value is not None:
                return
            
            # Calculate importance as weighted impurity decrease
            left_samples = node.left.samples if node.left else 0
            right_samples = node.right.samples if node.right else 0
            total_samples = left_samples + right_samples
            
            if total_samples > 0:
                left_impurity = node.left.impurity if node.left else 0
                right_impurity = node.right.impurity if node.right else 0
                
                weighted_impurity = (left_samples / total_samples) * left_impurity + \
                                  (right_samples / total_samples) * right_impurity
                
                importance_decrease = node.impurity - weighted_impurity
                importance_scores[node.feature] += importance_decrease * total_samples
            
            calculate_importance(node.left)
            calculate_importance(node.right)
        
        calculate_importance(self.root)
        
        # Normalize importance scores
        total_importance = np.sum(importance_scores)
        if total_importance > 0:
            importance_scores /= total_importance
        
        return dict(zip(self.feature_names, importance_scores))