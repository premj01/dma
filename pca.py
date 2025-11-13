# ------------------------------------------------------------
# Perform PCA (Principal Component Analysis) on Random Dataset
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1️⃣ Generate a random dataset
np.random.seed(42)

# Let's simulate 3 classes around different centers (like in real data)
n_samples_per_class = 60

# Randomly generated 5-dimensional data
class1 = np.random.normal(loc=[2, 2, 2, 2, 2], scale=0.5, size=(n_samples_per_class, 5))
class2 = np.random.normal(loc=[5, 5, 5, 5, 5], scale=0.6, size=(n_samples_per_class, 5))
class3 = np.random.normal(loc=[8, 2, 6, 3, 7], scale=0.7, size=(n_samples_per_class, 5))

# Combine into a single dataset
X = np.vstack((class1, class2, class3))
y = np.array([0]*n_samples_per_class + [1]*n_samples_per_class + [2]*n_samples_per_class)

# Create a DataFrame
df = pd.DataFrame(X, columns=['Feature1','Feature2','Feature3','Feature4','Feature5'])
df['Class'] = y

print("✅ Random Dataset Created (first 5 rows):")
print(df.head())

# 2️⃣ Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('Class', axis=1))

# 3️⃣ Apply PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# 4️⃣ Explained variance ratio (how much info each PC keeps)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# 5️⃣ Plot variance per component
plt.figure(figsize=(7,5))
plt.bar(range(1, 6), explained_variance, alpha=0.7, label='Individual Variance')
plt.step(range(1, 6), cumulative_variance, where='mid', color='red', label='Cumulative Variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Each Principal Component')
plt.legend()
plt.grid(True)
plt.show()

# 6️⃣ Visualize data in first 2 principal components
plt.figure(figsize=(7,5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=40)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - Projection onto First Two Components')
plt.grid(True)
plt.show()

# 7️⃣ Create a PCA loadings table (how each feature contributes to components)
loadings = pd.DataFrame(pca.components_, columns=df.columns[:-1])
loadings.index = [f'PC{i+1}' for i in range(pca.n_components_)]
print("\n🔹 PCA Component Loadings:")
print(loadings)

# 8️⃣ Interpretation summary
print("\n📊 PCA Interpretation Summary:")
for i, var in enumerate(explained_variance, 1):
    print(f" - PC{i} explains {var*100:.2f}% of total variance.")

print("""
✅ PCA Summary:
- PCA reduces high-dimensional data into a smaller number of components that retain most information.
- PC1 captures the direction of maximum variance.
- PC2 captures the next most significant variance, orthogonal to PC1.
- Usually, the first few components (like PC1 + PC2) explain most of the variance.
""")
