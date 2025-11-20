import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def manual_pca(X, n_components=2):
    """
    Manual PCA implementation showing all mathematical steps
    
    Steps:
    1. Center the data (subtract mean)
    2. Calculate covariance matrix
    3. Find eigenvalues and eigenvectors
    4. Sort by eigenvalues (descending)
    5. Select top n_components
    6. Transform data to new space
    """
    print("=== MANUAL PCA STEP-BY-STEP ===")
    
    print("Step 1: Centering the data...")
    X_centered = X - np.mean(X, axis=0)
    print(f"Original data shape: {X.shape}")
    print(f"Mean of original data: {np.mean(X, axis=0)}")
    print(f"Mean of centered data: {np.mean(X_centered, axis=0)}")
    
    #   covariance matrix
    print("\nStep 2: Computing covariance matrix...")
    # Covariance matrix: C = (1/n-1) * X_T @ X
    n_samples = X_centered.shape[0]
    covariance_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
    print(f"Covariance matrix shape: {covariance_matrix.shape}")
    print(f"Covariance matrix:\n{covariance_matrix}")
    
    # eigenvalues and eigenvectors
    print("\nStep 3: Computing eigenvalues and eigenvectors...")
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors shape: {eigenvectors.shape}")
    
    # sorting by eigenvalues (descending)
    print("\nStep 4: Sorting by eigenvalues (descending)...")
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[sorted_indices]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    print(f"Sorted eigenvalues: {eigenvalues_sorted}")
    
    # selecting top n_components
    print(f"\nStep 5: Selecting top {n_components} components...")
    selected_eigenvalues = eigenvalues_sorted[:n_components]
    selected_eigenvectors = eigenvectors_sorted[:, :n_components]
    print(f"Selected eigenvalues: {selected_eigenvalues}")
    print(f"Selected eigenvectors shape: {selected_eigenvectors.shape}")
    
    # transforming data to new space
    print(f"\nStep 6: Transforming data to {n_components}D space...")
    X_transformed = X_centered @ selected_eigenvectors
    
    # Calculate explained variance ratio
    total_variance = np.sum(eigenvalues_sorted)
    explained_variance_ratio = selected_eigenvalues / total_variance
    
    print(f"Transformed data shape: {X_transformed.shape}")
    print(f"Explained variance ratio: {explained_variance_ratio}")
    print(f"Total variance explained: {explained_variance_ratio.sum():.3f} ({explained_variance_ratio.sum()*100:.1f}%)")
    
    return X_transformed, selected_eigenvectors, explained_variance_ratio, selected_eigenvalues

def standardize_features(X):
    """Manual standardization: (X - mean) / std"""
    print("=== MANUAL STANDARDIZATION ===")
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)  # ddof=1 for sample standard deviation
    X_standardized = (X - mean) / std
    
    print(f"Original mean: {mean}")
    print(f"Original std: {std}")
    print(f"Standardized mean: {np.mean(X_standardized, axis=0)}")
    print(f"Standardized std: {np.std(X_standardized, axis=0, ddof=1)}")
    
    return X_standardized

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Manual standardization
X_scaled = standardize_features(X)

# Apply manual PCA to reduce to 2 principal components
X_pca, components, explained_variance_ratio, eigenvalues = manual_pca(X_scaled, n_components=2)

# Print detailed insights from manual PCA
print("\n" + "="*50)
print("=== MANUAL PCA RESULTS SUMMARY ===")
print("="*50)
print(f"Original dataset shape: {X.shape}")
print(f"Reduced dataset shape: {X_pca.shape}")
print(f"\nExplained variance ratio: {explained_variance_ratio}")
print(f"Total variance explained: {explained_variance_ratio.sum():.3f} ({explained_variance_ratio.sum()*100:.1f}%)")
print(f"\nPrincipal Component 1 explains: {explained_variance_ratio[0]*100:.1f}% of variance")
print(f"Principal Component 2 explains: {explained_variance_ratio[1]*100:.1f}% of variance")

print(f"\nOriginal features: {iris.feature_names}")
print(f"PCA Components (how much each original feature contributes):")
for i, component in enumerate(components.T):
    print(f"  PC{i+1}: {component}")
    
print(f"\nEigenvalues: {eigenvalues}")
print(f"Component loadings (eigenvectors):")
for i in range(len(iris.feature_names)):
    print(f"  {iris.feature_names[i]:20s}: PC1={components[i,0]:6.3f}, PC2={components[i,1]:6.3f}")

# Visualize the transformed dataset
plt.figure(figsize=(8, 6))
for target, color, label in zip([0, 1, 2], ['r', 'g', 'b'], iris.target_names):
    plt.scatter(X_pca[y == target, 0], X_pca[y == target, 1],
                color=color, label=label, edgecolor='k', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

