"""
Evaluation metrics and analysis tools for Decision Tree CART implementation
Provides comprehensive evaluation capabilities for both classification and regression

Author: Student
Date: November 2025
Course: Data Analytics and Visualization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from collections import Counter
from decision_tree_cart import DecisionTreeCART


class ClassificationEvaluator:
    """
    Evaluation metrics for classification tasks.
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, classes: Optional[List] = None):
        """
        Initialize the evaluator.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            classes: List of class labels (optional)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = classes if classes is not None else np.unique(np.concatenate([y_true, y_pred]))
        self.n_classes = len(self.classes)
    
    def accuracy(self) -> float:
        """Calculate accuracy score."""
        return np.mean(self.y_true == self.y_pred)
    
    def confusion_matrix(self) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Returns:
            Confusion matrix as numpy array
        """
        cm = np.zeros((self.n_classes, self.n_classes), dtype=int)
        
        for i, true_class in enumerate(self.classes):
            for j, pred_class in enumerate(self.classes):
                cm[i, j] = np.sum((self.y_true == true_class) & (self.y_pred == pred_class))
        
        return cm
    
    def precision_recall_f1(self, average: str = 'macro') -> Dict[str, float]:
        """
        Calculate precision, recall, and F1-score.
        
        Args:
            average: Averaging method ('macro', 'micro', 'weighted')
            
        Returns:
            Dictionary with precision, recall, and F1-score
        """
        cm = self.confusion_matrix()
        
        # Per-class metrics
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        
        for i in range(self.n_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            
            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precision_per_class.append(precision)
            
            # Recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recall_per_class.append(recall)
            
            # F1-score
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_per_class.append(f1)
        
        # Calculate averages
        if average == 'macro':
            precision = np.mean(precision_per_class)
            recall = np.mean(recall_per_class)
            f1 = np.mean(f1_per_class)
        elif average == 'micro':
            # Micro-average
            tp_total = np.sum([cm[i, i] for i in range(self.n_classes)])
            fp_total = np.sum(cm) - tp_total - np.sum([np.sum(cm[i, :]) - cm[i, i] for i in range(self.n_classes)])
            fn_total = np.sum(cm) - tp_total - np.sum([np.sum(cm[:, i]) - cm[i, i] for i in range(self.n_classes)])
            
            precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
            recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif average == 'weighted':
            # Weighted by support
            class_counts = [np.sum(cm[i, :]) for i in range(self.n_classes)]
            total_count = np.sum(class_counts)
            
            weights = [count / total_count for count in class_counts]
            precision = np.average(precision_per_class, weights=weights)
            recall = np.average(recall_per_class, weights=weights)
            f1 = np.average(f1_per_class, weights=weights)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class
        }
    
    def classification_report(self) -> str:
        """
        Generate a detailed classification report.
        
        Returns:
            Formatted classification report string
        """
        cm = self.confusion_matrix()
        metrics = self.precision_recall_f1()
        
        # Header
        report = "Classification Report\\n"
        report += "=" * 60 + "\\n"
        report += f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}\\n"
        report += "-" * 60 + "\\n"
        
        # Per-class metrics
        for i, cls in enumerate(self.classes):
            support = np.sum(cm[i, :])
            report += f"{cls:<15} {metrics['precision_per_class'][i]:<12.3f} "
            report += f"{metrics['recall_per_class'][i]:<12.3f} "
            report += f"{metrics['f1_per_class'][i]:<12.3f} {support:<12}\\n"
        
        report += "-" * 60 + "\\n"
        
        # Overall metrics
        total_support = np.sum(cm)
        report += f"{'Accuracy':<15} {'':<12} {'':<12} {self.accuracy():<12.3f} {total_support:<12}\\n"
        report += f"{'Macro Avg':<15} {metrics['precision']:<12.3f} "
        report += f"{metrics['recall']:<12.3f} {metrics['f1_score']:<12.3f} {total_support:<12}\\n"
        
        # Weighted average
        weighted_metrics = self.precision_recall_f1('weighted')
        report += f"{'Weighted Avg':<15} {weighted_metrics['precision']:<12.3f} "
        report += f"{weighted_metrics['recall']:<12.3f} {weighted_metrics['f1_score']:<12.3f} {total_support:<12}\\n"
        
        return report
    
    def plot_confusion_matrix(self, 
                            figsize: Tuple[int, int] = (8, 6),
                            normalize: bool = False,
                            save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            figsize: Figure size
            normalize: Whether to normalize the matrix
            save_path: Path to save the figure (optional)
        """
        cm = self.confusion_matrix()
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=figsize)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.colorbar()
        
        # Add labels
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontweight='bold')
        
        plt.ylabel('True Label', fontweight='bold')
        plt.xlabel('Predicted Label', fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class RegressionEvaluator:
    """
    Evaluation metrics for regression tasks.
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Initialize the evaluator.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.residuals = y_true - y_pred
    
    def mean_squared_error(self) -> float:
        """Calculate Mean Squared Error."""
        return np.mean(self.residuals ** 2)
    
    def root_mean_squared_error(self) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(self.mean_squared_error())
    
    def mean_absolute_error(self) -> float:
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(self.residuals))
    
    def r_squared(self) -> float:
        """Calculate R-squared (coefficient of determination)."""
        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    def mean_absolute_percentage_error(self) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = self.y_true != 0
        if not np.any(mask):
            return np.inf
        
        return np.mean(np.abs(self.residuals[mask] / self.y_true[mask])) * 100
    
    def regression_report(self) -> str:
        """
        Generate a detailed regression report.
        
        Returns:
            Formatted regression report string
        """
        report = "Regression Report\\n"
        report += "=" * 40 + "\\n"
        report += f"Mean Squared Error (MSE):        {self.mean_squared_error():.6f}\\n"
        report += f"Root Mean Squared Error (RMSE):  {self.root_mean_squared_error():.6f}\\n"
        report += f"Mean Absolute Error (MAE):       {self.mean_absolute_error():.6f}\\n"
        report += f"R-squared (R²):                  {self.r_squared():.6f}\\n"
        report += f"Mean Absolute Percentage Error:  {self.mean_absolute_percentage_error():.2f}%\\n"
        report += "\\n"
        report += f"Residual Statistics:\\n"
        report += f"Mean:                            {np.mean(self.residuals):.6f}\\n"
        report += f"Standard Deviation:              {np.std(self.residuals):.6f}\\n"
        report += f"Min:                             {np.min(self.residuals):.6f}\\n"
        report += f"Max:                             {np.max(self.residuals):.6f}\\n"
        
        return report
    
    def plot_residuals(self, 
                      figsize: Tuple[int, int] = (12, 8),
                      save_path: Optional[str] = None) -> None:
        """
        Plot residual analysis.
        
        Args:
            figsize: Figure size
            save_path: Path to save the figure (optional)
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Predicted vs Actual
        min_val = min(np.min(self.y_true), np.min(self.y_pred))
        max_val = max(np.max(self.y_true), np.max(self.y_pred))
        
        ax1.scatter(self.y_true, self.y_pred, alpha=0.6, color='blue')
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Predicted vs Actual')
        ax1.grid(True, alpha=0.3)
        
        # Add R² to the plot
        ax1.text(0.05, 0.95, f'R² = {self.r_squared():.3f}', 
                transform=ax1.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Residuals vs Predicted
        ax2.scatter(self.y_pred, self.residuals, alpha=0.6, color='green')
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals vs Predicted')
        ax2.grid(True, alpha=0.3)
        
        # Histogram of residuals
        ax3.hist(self.residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Residuals')
        ax3.grid(True, alpha=0.3)
        
        # Q-Q plot (approximation using manual implementation)
        sorted_residuals = np.sort(self.residuals)
        n = len(sorted_residuals)
        theoretical_quantiles = np.array([np.percentile(np.random.normal(0, 1, 10000), 
                                                       100 * (i + 0.5) / n) for i in range(n)])
        
        ax4.scatter(theoretical_quantiles, sorted_residuals, alpha=0.6, color='purple')
        
        # Add reference line
        slope, intercept = np.polyfit(theoretical_quantiles, sorted_residuals, 1)
        ax4.plot(theoretical_quantiles, slope * theoretical_quantiles + intercept, 'r--', linewidth=2)
        
        ax4.set_xlabel('Theoretical Quantiles')
        ax4.set_ylabel('Sample Quantiles')
        ax4.set_title('Q-Q Plot (Normal Distribution)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class TreeEvaluator:
    """
    Comprehensive evaluator for Decision Tree models.
    """
    
    def __init__(self, tree: DecisionTreeCART):
        """
        Initialize the evaluator.
        
        Args:
            tree: Trained DecisionTreeCART instance
        """
        self.tree = tree
    
    def evaluate(self, 
                X_test: Union[np.ndarray, pd.DataFrame], 
                y_test: Union[np.ndarray, pd.Series],
                X_train: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                y_train: Optional[Union[np.ndarray, pd.Series]] = None,
                show_plots: bool = True) -> Dict[str, any]:
        """
        Comprehensive evaluation of the decision tree.
        
        Args:
            X_test: Test features
            y_test: Test targets
            X_train: Training features (optional, for comparison)
            y_train: Training targets (optional, for comparison)
            show_plots: Whether to display plots
            
        Returns:
            Dictionary with evaluation results
        """
        # Make predictions
        y_pred = self.tree.predict(X_test)
        results = {}
        
        if self.tree.task_type == 'classification':
            evaluator = ClassificationEvaluator(y_test, y_pred)
            
            # Basic metrics
            results['accuracy'] = evaluator.accuracy()
            metrics = evaluator.precision_recall_f1()
            results.update(metrics)
            
            # Confusion matrix
            results['confusion_matrix'] = evaluator.confusion_matrix()
            
            # Classification report
            results['report'] = evaluator.classification_report()
            print(results['report'])
            
            if show_plots:
                evaluator.plot_confusion_matrix()
            
            # Training accuracy if available
            if X_train is not None and y_train is not None:
                y_train_pred = self.tree.predict(X_train)
                train_evaluator = ClassificationEvaluator(y_train, y_train_pred)
                results['train_accuracy'] = train_evaluator.accuracy()
                print(f"\\nTraining Accuracy: {results['train_accuracy']:.4f}")
                print(f"Test Accuracy: {results['accuracy']:.4f}")
                print(f"Overfitting Gap: {results['train_accuracy'] - results['accuracy']:.4f}")
        
        else:  # Regression
            evaluator = RegressionEvaluator(y_test, y_pred)
            
            # Basic metrics
            results['mse'] = evaluator.mean_squared_error()
            results['rmse'] = evaluator.root_mean_squared_error()
            results['mae'] = evaluator.mean_absolute_error()
            results['r2'] = evaluator.r_squared()
            results['mape'] = evaluator.mean_absolute_percentage_error()
            
            # Regression report
            results['report'] = evaluator.regression_report()
            print(results['report'])
            
            if show_plots:
                evaluator.plot_residuals()
            
            # Training metrics if available
            if X_train is not None and y_train is not None:
                y_train_pred = self.tree.predict(X_train)
                train_evaluator = RegressionEvaluator(y_train, y_train_pred)
                results['train_r2'] = train_evaluator.r_squared()
                results['train_rmse'] = train_evaluator.root_mean_squared_error()
                print(f"\\nTraining R²: {results['train_r2']:.4f}")
                print(f"Test R²: {results['r2']:.4f}")
                print(f"Training RMSE: {results['train_rmse']:.4f}")
                print(f"Test RMSE: {results['rmse']:.4f}")
        
        # Tree structure information
        results['tree_depth'] = self.tree.get_depth()
        results['n_leaves'] = self.tree.get_n_leaves()
        results['feature_importance'] = self.tree.feature_importance()
        
        print(f"\\nTree Structure:")
        print(f"Depth: {results['tree_depth']}")
        print(f"Number of Leaves: {results['n_leaves']}")
        
        return results
    
    def cross_validate(self, 
                      X: Union[np.ndarray, pd.DataFrame],
                      y: Union[np.ndarray, pd.Series],
                      cv: int = 5,
                      random_state: int = 42) -> Dict[str, List[float]]:
        """
        Perform cross-validation evaluation.
        
        Args:
            X: Features
            y: Targets
            cv: Number of cross-validation folds
            random_state: Random seed
            
        Returns:
            Dictionary with cross-validation scores
        """
        np.random.seed(random_state)
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        fold_size = n_samples // cv
        scores = {}
        
        if self.tree.task_type == 'classification':
            scores['accuracy'] = []
            scores['precision'] = []
            scores['recall'] = []
            scores['f1_score'] = []
        else:
            scores['mse'] = []
            scores['rmse'] = []
            scores['mae'] = []
            scores['r2'] = []
        
        for fold in range(cv):
            # Split data
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < cv - 1 else n_samples
            
            test_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            X_train_fold = X[train_indices]
            y_train_fold = y[train_indices]
            X_test_fold = X[test_indices]
            y_test_fold = y[test_indices]
            
            # Create and train new tree for this fold
            fold_tree = DecisionTreeCART(
                task_type=self.tree.task_type,
                max_depth=self.tree.max_depth,
                min_samples_split=self.tree.min_samples_split,
                min_samples_leaf=self.tree.min_samples_leaf,
                min_impurity_decrease=self.tree.min_impurity_decrease,
                random_state=random_state + fold
            )
            
            fold_tree.fit(X_train_fold, y_train_fold)
            y_pred_fold = fold_tree.predict(X_test_fold)
            
            # Calculate metrics
            if self.tree.task_type == 'classification':
                evaluator = ClassificationEvaluator(y_test_fold, y_pred_fold)
                scores['accuracy'].append(evaluator.accuracy())
                metrics = evaluator.precision_recall_f1()
                scores['precision'].append(metrics['precision'])
                scores['recall'].append(metrics['recall'])
                scores['f1_score'].append(metrics['f1_score'])
            else:
                evaluator = RegressionEvaluator(y_test_fold, y_pred_fold)
                scores['mse'].append(evaluator.mean_squared_error())
                scores['rmse'].append(evaluator.root_mean_squared_error())
                scores['mae'].append(evaluator.mean_absolute_error())
                scores['r2'].append(evaluator.r_squared())
        
        # Print cross-validation results
        print(f"Cross-Validation Results ({cv}-fold):")
        print("=" * 40)
        
        for metric, score_list in scores.items():
            mean_score = np.mean(score_list)
            std_score = np.std(score_list)
            print(f"{metric.upper()}: {mean_score:.4f} (+/- {std_score:.4f})")
        
        return scores