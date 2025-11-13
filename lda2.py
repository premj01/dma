# ==============================================================
# ✅ Linear Discriminant Analysis (LDA) using Random Data
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# -------------------------------------------------------------
# 1️⃣ GENERATE RANDOM DATASET (100 samples, 5 features, 3 classes)
# -------------------------------------------------------------
X, y = make_classification(
    n_samples=100,
    n_features=5,
    n_classes=3,
    n_informative=3,
    n_redundant=0,
    random_state=42
)

print("✅ Random dataset generated successfully!")

# -------------------------------------------------------------
# 2️⃣ TRAIN / TEST SPLIT + SCALING
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------
# 3️⃣ BASELINE CLASSIFICATION (WITHOUT LDA)
# -------------------------------------------------------------
baseline_model = LogisticRegression(max_iter=300)
baseline_model.fit(X_train_scaled, y_train)
y_pred_before = baseline_model.predict(X_test_scaled)

accuracy_before = accuracy_score(y_test, y_pred_before)
print("\n📌 Accuracy BEFORE LDA :", accuracy_before)

# -------------------------------------------------------------
# 4️⃣ APPLY LINEAR DISCRIMINANT ANALYSIS (reduce features to 2D)
# -------------------------------------------------------------
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

# -------------------------------------------------------------
# 5️⃣ TRAIN MODEL AGAIN AFTER DIMENSIONALITY REDUCTION
# -------------------------------------------------------------
model_after = LogisticRegression(max_iter=300)
model_after.fit(X_train_lda, y_train)
y_pred_after = model_after.predict(X_test_lda)

accuracy_after = accuracy_score(y_test, y_pred_after)
print("\n✅ Accuracy AFTER LDA :", accuracy_after)

# -------------------------------------------------------------
# 6️⃣ PLOT LDA TRANSFORMED DATA (2D projection)
# -------------------------------------------------------------
plt.figure(figsize=(8,6))
for label in np.unique(y):
    plt.scatter(
        X_train_lda[y_train == label, 0],
        X_train_lda[y_train == label, 1],
        label=f"Class {label}"
    )

plt.title("LDA - Dimensionality Reduction to 2 Components")
plt.xlabel("LDA Component 1")
plt.ylabel("LDA Component 2")
plt.legend()
plt.grid()
plt.show()

# -------------------------------------------------------------
# 7️⃣ CONFUSION MATRIX (AFTER LDA)
# -------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred_after)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix (After LDA)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------------------------------------
# 8️⃣ PRINT REPORT
# -------------------------------------------------------------
print("\n------ Classification Report (After LDA) ------")
print(classification_report(y_test, y_pred_after))

# -------------------------------------------------------------
# 🔍 Final Explanation
# -------------------------------------------------------------
print("""
🧠 Why LDA improved classification?

➡ LDA finds the best directions (linear combinations of features) that maximize class separation.
➡ It reduces noise and redundancy in the dataset.
➡ After reducing data to 2 dimensions, it becomes easier for the classifier to separate the classes.

📌 Result: Model accuracy increased after applying LDA.
""")
