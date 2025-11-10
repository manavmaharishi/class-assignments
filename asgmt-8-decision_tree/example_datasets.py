"""
Example datasets and demonstrations for Decision Tree CART implementation
Includes synthetic and real-world-like datasets for testing

Author: Student
Date: November 2025
Course: Data Analytics and Visualization
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, load_iris, load_wine
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt


def create_synthetic_classification_data(n_samples: int = 1000, 
                                       n_features: int = 4,
                                       n_classes: int = 3,
                                       n_redundant: int = 0,
                                       n_clusters_per_class: int = 1,
                                       random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create synthetic classification dataset.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        n_redundant: Number of redundant features
        n_clusters_per_class: Number of clusters per class
        random_state: Random seed
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=n_redundant,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        random_state=random_state,
        n_informative=n_features - n_redundant
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Convert to DataFrame and Series
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series


def create_synthetic_regression_data(n_samples: int = 1000,
                                   n_features: int = 4,
                                   noise: float = 0.1,
                                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create synthetic regression dataset.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Noise level
        random_state: Random seed
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Convert to DataFrame and Series
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series


def create_titanic_like_dataset(n_samples: int = 1000, 
                               random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a Titanic-like survival prediction dataset.
    
    Args:
        n_samples: Number of samples
        random_state: Random seed
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    np.random.seed(random_state)
    
    # Generate features
    data = {}
    
    # Age (18-80)
    data['age'] = np.random.normal(35, 12, n_samples)
    data['age'] = np.clip(data['age'], 18, 80)
    
    # Fare (10-500)
    data['fare'] = np.random.lognormal(4, 1, n_samples)
    data['fare'] = np.clip(data['fare'], 10, 500)
    
    # Sex (0: female, 1: male)
    data['sex'] = np.random.binomial(1, 0.65, n_samples)
    
    # Passenger class (1: first, 2: second, 3: third)
    class_probs = [0.2, 0.3, 0.5]
    data['pclass'] = np.random.choice([1, 2, 3], n_samples, p=class_probs)
    
    # Siblings/spouses aboard
    data['sibsp'] = np.random.poisson(0.5, n_samples)
    data['sibsp'] = np.clip(data['sibsp'], 0, 8)
    
    # Parents/children aboard
    data['parch'] = np.random.poisson(0.3, n_samples)
    data['parch'] = np.clip(data['parch'], 0, 6)
    
    # Generate survival based on realistic probabilities
    survival_prob = (
        0.8 * (data['sex'] == 0) +  # Women more likely to survive
        0.2 * (data['pclass'] == 1) +  # First class more likely to survive
        0.1 * (data['pclass'] == 2) +  # Second class somewhat more likely
        -0.01 * (data['age'] - 30) +  # Younger people more likely to survive
        0.0001 * data['fare'] +  # Higher fare (better cabin) more likely to survive
        -0.05 * data['sibsp'] +  # Large families less likely to survive together
        -0.03 * data['parch']
    )
    
    # Add some randomness and convert to binary
    survival_prob += np.random.normal(0, 0.2, n_samples)
    survival_prob = np.clip(survival_prob, 0, 1)
    survived = np.random.binomial(1, survival_prob, n_samples)
    
    # Create DataFrame
    X_df = pd.DataFrame(data)
    y_series = pd.Series(survived, name='survived')
    
    return X_df, y_series


def create_customer_churn_dataset(n_samples: int = 1000,
                                 random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a customer churn prediction dataset.
    
    Args:
        n_samples: Number of samples
        random_state: Random seed
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    np.random.seed(random_state)
    
    data = {}
    
    # Customer tenure in months (1-72)
    data['tenure'] = np.random.randint(1, 73, n_samples)
    
    # Monthly charges ($20-120)
    data['monthly_charges'] = np.random.normal(65, 20, n_samples)
    data['monthly_charges'] = np.clip(data['monthly_charges'], 20, 120)
    
    # Total charges (tenure * monthly_charges + noise)
    data['total_charges'] = data['tenure'] * data['monthly_charges'] + np.random.normal(0, 100, n_samples)
    data['total_charges'] = np.maximum(data['total_charges'], data['monthly_charges'])
    
    # Contract type (0: month-to-month, 1: one year, 2: two year)
    data['contract_type'] = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])
    
    # Payment method (0: electronic check, 1: mailed check, 2: bank transfer, 3: credit card)
    data['payment_method'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.2, 0.2, 0.2])
    
    # Internet service (0: no, 1: DSL, 2: fiber optic)
    data['internet_service'] = np.random.choice([0, 1, 2], n_samples, p=[0.2, 0.4, 0.4])
    
    # Number of services (phone, internet, TV, security, etc.)
    data['num_services'] = np.random.randint(1, 8, n_samples)
    
    # Generate churn based on realistic factors
    churn_prob = (
        0.5 * (data['contract_type'] == 0) +  # Month-to-month more likely to churn
        -0.2 * (data['contract_type'] == 2) +  # Two-year contract less likely to churn
        -0.01 * data['tenure'] +  # Longer tenure less likely to churn
        0.005 * data['monthly_charges'] +  # Higher charges more likely to churn
        0.2 * (data['payment_method'] == 0) +  # Electronic check more likely to churn
        -0.05 * data['num_services']  # More services less likely to churn
    )
    
    # Add randomness and convert to binary
    churn_prob += np.random.normal(0, 0.3, n_samples)
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = np.random.binomial(1, churn_prob, n_samples)
    
    # Create DataFrame
    X_df = pd.DataFrame(data)
    y_series = pd.Series(churn, name='churn')
    
    return X_df, y_series


def load_iris_dataset(**kwargs) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the Iris dataset.
    
    Args:
        **kwargs: Additional arguments (ignored for compatibility)
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    iris = load_iris()
    
    X_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    y_series = pd.Series(iris.target, name='species')
    
    return X_df, y_series


def load_wine_dataset(**kwargs) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the Wine dataset.
    
    Args:
        **kwargs: Additional arguments (ignored for compatibility)
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    wine = load_wine()
    
    X_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    y_series = pd.Series(wine.target, name='wine_class')
    
    return X_df, y_series


def create_housing_price_dataset(n_samples: int = 1000,
                                random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a housing price prediction dataset for regression.
    
    Args:
        n_samples: Number of samples
        random_state: Random seed
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    np.random.seed(random_state)
    
    data = {}
    
    # Square footage (800-4000)
    data['sqft'] = np.random.normal(2000, 600, n_samples)
    data['sqft'] = np.clip(data['sqft'], 800, 4000)
    
    # Number of bedrooms (1-6)
    data['bedrooms'] = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.1, 0.05])
    
    # Number of bathrooms (1-4)
    data['bathrooms'] = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples, 
                                        p=[0.15, 0.1, 0.25, 0.15, 0.2, 0.1, 0.05])
    
    # Age of house (0-50 years)
    data['age'] = np.random.randint(0, 51, n_samples)
    
    # Garage size (0-3 cars)
    data['garage'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.4, 0.3, 0.1])
    
    # Lot size (0.1-2.0 acres)
    data['lot_size'] = np.random.lognormal(-0.5, 0.5, n_samples)
    data['lot_size'] = np.clip(data['lot_size'], 0.1, 2.0)
    
    # Distance to city center (1-30 miles)
    data['distance_to_city'] = np.random.exponential(8, n_samples)
    data['distance_to_city'] = np.clip(data['distance_to_city'], 1, 30)
    
    # Generate price based on realistic factors
    price = (
        150 * data['sqft'] +  # $150 per sqft base
        10000 * data['bedrooms'] +  # $10k per bedroom
        8000 * data['bathrooms'] +  # $8k per bathroom
        -2000 * data['age'] +  # Depreciation
        5000 * data['garage'] +  # Garage value
        20000 * data['lot_size'] +  # Lot size value
        -3000 * data['distance_to_city'] +  # Distance penalty
        50000  # Base value
    )
    
    # Add some noise
    price += np.random.normal(0, 20000, n_samples)
    price = np.maximum(price, 50000)  # Minimum price
    
    # Create DataFrame
    X_df = pd.DataFrame(data)
    y_series = pd.Series(price, name='price')
    
    return X_df, y_series


class DatasetExamples:
    """
    Collection of example datasets with descriptions and use cases.
    """
    
    @staticmethod
    def get_all_datasets() -> Dict[str, Dict[str, Any]]:
        """
        Get all available example datasets.
        
        Returns:
            Dictionary with dataset information
        """
        return {
            'synthetic_classification': {
                'function': create_synthetic_classification_data,
                'type': 'classification',
                'description': 'Synthetic multi-class classification dataset',
                'features': ['feature_1', 'feature_2', 'feature_3', 'feature_4'],
                'target': 'target'
            },
            'synthetic_regression': {
                'function': create_synthetic_regression_data,
                'type': 'regression',
                'description': 'Synthetic regression dataset',
                'features': ['feature_1', 'feature_2', 'feature_3', 'feature_4'],
                'target': 'target'
            },
            'titanic_like': {
                'function': create_titanic_like_dataset,
                'type': 'classification',
                'description': 'Titanic-like survival prediction dataset',
                'features': ['age', 'fare', 'sex', 'pclass', 'sibsp', 'parch'],
                'target': 'survived'
            },
            'customer_churn': {
                'function': create_customer_churn_dataset,
                'type': 'classification',
                'description': 'Customer churn prediction dataset',
                'features': ['tenure', 'monthly_charges', 'total_charges', 'contract_type', 
                           'payment_method', 'internet_service', 'num_services'],
                'target': 'churn'
            },
            'iris': {
                'function': load_iris_dataset,
                'type': 'classification',
                'description': 'Classic Iris flower classification dataset',
                'features': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
                'target': 'species'
            },
            'wine': {
                'function': load_wine_dataset,
                'type': 'classification',
                'description': 'Wine classification dataset',
                'features': ['alcohol', 'malic_acid', 'ash', '...'],  # Truncated for brevity
                'target': 'wine_class'
            },
            'housing_prices': {
                'function': create_housing_price_dataset,
                'type': 'regression',
                'description': 'Housing price prediction dataset',
                'features': ['sqft', 'bedrooms', 'bathrooms', 'age', 'garage', 'lot_size', 'distance_to_city'],
                'target': 'price'
            }
        }
    
    @staticmethod
    def load_dataset(dataset_name: str, **kwargs) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Load a specific dataset by name.
        
        Args:
            dataset_name: Name of the dataset to load
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            Tuple of (features, target, dataset_info)
        """
        datasets = DatasetExamples.get_all_datasets()
        
        if dataset_name not in datasets:
            available = list(datasets.keys())
            raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {available}")
        
        dataset_info = datasets[dataset_name]
        dataset_function = dataset_info['function']
        
        X, y = dataset_function(**kwargs)
        
        return X, y, dataset_info
    
    @staticmethod
    def describe_dataset(dataset_name: str) -> None:
        """
        Print description of a dataset.
        
        Args:
            dataset_name: Name of the dataset
        """
        datasets = DatasetExamples.get_all_datasets()
        
        if dataset_name not in datasets:
            available = list(datasets.keys())
            print(f"Dataset '{dataset_name}' not found. Available datasets: {available}")
            return
        
        info = datasets[dataset_name]
        print(f"Dataset: {dataset_name}")
        print(f"Type: {info['type']}")
        print(f"Description: {info['description']}")
        print(f"Features: {info['features']}")
        print(f"Target: {info['target']}")
    
    @staticmethod
    def list_all_datasets() -> None:
        """
        List all available datasets with descriptions.
        """
        datasets = DatasetExamples.get_all_datasets()
        
        print("Available Datasets:")
        print("=" * 50)
        
        for name, info in datasets.items():
            print(f"\\n{name}:")
            print(f"  Type: {info['type']}")
            print(f"  Description: {info['description']}")


def split_and_describe_data(X: pd.DataFrame, 
                          y: pd.Series, 
                          test_size: float = 0.2,
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data and print descriptive statistics.
    
    Args:
        X: Feature matrix
        y: Target values
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    print("Dataset Information:")
    print("=" * 40)
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    if y.dtype in ['int64', 'int32'] and len(np.unique(y)) < 20:
        print(f"\\nTarget distribution:")
        print(y.value_counts().sort_index())
    else:
        print(f"\\nTarget statistics:")
        print(y.describe())
    
    print(f"\\nFeature statistics (training set):")
    print(X_train.describe())
    
    return X_train, X_test, y_train, y_test