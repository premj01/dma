# 📘 Perform PCA (Principal Component Analysis)

# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Load dataset
iris = load_iris()
X = iris.data
y = iris.target
features = iris.feature_names

# Step 3: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA (reduce to 2 components for easy visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 5: Create a DataFrame for visualization
df_pca = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
df_pca["Target"] = y

# Step 6: Explained variance (importance of components)
print("Explained Variance Ratio (importance of each component):")
print(pca.explained_variance_ratio_)

# Step 7: Visualize the two PCA components
plt.figure(figsize=(6, 4))
plt.scatter(df_pca["PC1"], df_pca["PC2"], c=y, cmap="rainbow")
plt.title("PCA - Dimensionality Reduction (2 Components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


# 📘 Principal Component Analysis (PCA) on Retail Dataset

# Step 1: Import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 2: Load your dataset
data = pd.read_csv("retail.csv")  # 👈 Make sure retail.csv is in the same folder
print("📊 Retail Dataset Preview:")
print(data.head())

# Step 3: Separate features (X) and labels (y)
X = data.drop("Category", axis=1)
y = data["Category"]

# Step 4: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply PCA (reduce to 2 principal components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 6: Create a DataFrame with PCA results
pca_df = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
pca_df["Category"] = y

# Step 7: Print explained variance
print("\nExplained Variance Ratio (importance of each component):")
print(pca.explained_variance_ratio_)

# Step 8: Visualize PCA results
plt.figure(figsize=(6, 4))
for label in pca_df["Category"].unique():
    plt.scatter(
        pca_df[pca_df["Category"] == label]["PC1"],
        pca_df[pca_df["Category"] == label]["PC2"],
        label=label,
    )
plt.title("PCA - Retail Dataset (2 Components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()
