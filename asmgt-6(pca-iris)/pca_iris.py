import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA







# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Print key insights from PCA
print("=== PCA INSIGHTS ===")
print(f"Original dataset shape: {X.shape}")
print(f"Reduced dataset shape: {X_pca.shape}")
print(f"\nExplained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f} ({pca.explained_variance_ratio_.sum()*100:.1f}%)")
print(f"\nPrincipal Component 1 explains: {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
print(f"Principal Component 2 explains: {pca.explained_variance_ratio_[1]*100:.1f}% of variance")

print(f"\nOriginal features: {iris.feature_names}")
print(f"PCA Components (how much each original feature contributes):")
for i, component in enumerate(pca.components_):
    print(f"  PC{i+1}: {component}")

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

