# --------------------------------------------------------
# Perform PCA (Principal Component Analysis)
# on Transformed_Sales_Data.csv
# --------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1️⃣ Load the dataset
file_path = "Transformed_Sales_Data.csv"  # <- put your file path here
df = pd.read_csv("C:/Users/shreyash/Downloads/Transformed_Sales_Data.csv")

print("✅ Dataset Loaded Successfully\n")
print(df.head())

# 2️⃣ Select only numeric columns for PCA
numeric_cols = df.select_dtypes(include=[np.number]).columns
X = df[numeric_cols].dropna()

print("\nNumeric columns used for PCA:\n", list(numeric_cols))

# 3️⃣ Standardize the features (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Apply PCA
pca = PCA(n_components=len(numeric_cols))
X_pca = pca.fit_transform(X_scaled)

# 5️⃣ Explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# 6️⃣ Plot explained variance
plt.figure(figsize=(7,5))
plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.7, label='Individual Variance')
plt.step(range(1, len(explained_variance)+1), cumulative_variance, where='mid', color='red', label='Cumulative Variance')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Components')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7️⃣ Visualize PCA projection (first two PCs)
plt.figure(figsize=(7,5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], color='green', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - Data Projected to First Two Components')
plt.grid(True)
plt.show()

# 8️⃣ Create a PCA loadings DataFrame (feature contributions)
loadings = pd.DataFrame(
    pca.components_,
    columns=numeric_cols,
    index=[f'PC{i+1}' for i in range(len(numeric_cols))]
)

print("\n🔹 PCA Loadings (Feature Contributions):")
print(loadings.round(4))

# 9️⃣ Display explained variance summary
print("\n📊 Explained Variance by Component:")
for i, var in enumerate(explained_variance, 1):
    print(f"PC{i}: {var*100:.2f}% of total variance")

# 🔟 Interpretation
print("""
✅ Interpretation:
- PCA reduces high-dimensional data to fewer components that retain most information.
- PC1 explains the largest variance (most informative direction).
- PC2 explains the next major variance, orthogonal to PC1.
- You can keep the first few PCs (e.g., first 2 or 3) that together explain over 95% variance
  for simpler visualization or model building.
""")
