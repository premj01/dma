# --------------------------------------------------------
# LDA with RANDOM DATA (no CSV) – Dimensionality Reduction
# and Classification + Visualizations
# --------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1) Create synthetic/random dataset
# 3 classes, each around different centers
np.random.seed(42)
n = 60  # samples per class

# class 0 around (2,2,2,2,2)
class0 = np.random.normal(loc=2, scale=0.5, size=(n, 5))
# class 1 around (5,5,5,5,5)
class1 = np.random.normal(loc=5, scale=0.6, size=(n, 5))
# class 2 around (8,2,6,3,7)
class2 = np.random.normal(loc=[8, 2, 6, 3, 7], scale=0.7, size=(n, 5))

X = np.vstack([class0, class1, class2])         # shape (180, 5)
y = np.array([0]*n + [1]*n + [2]*n)             # labels: 0,1,2

# 2) Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3) Feature scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 4) Baseline classifier (NO LDA)
base_clf = LogisticRegression(max_iter=300)
base_clf.fit(X_train_s, y_train)
y_pred_base = base_clf.predict(X_test_s)
acc_base = accuracy_score(y_test, y_pred_base)

print("=== Baseline (Logistic Regression on original features) ===")
print("Accuracy:", acc_base)
print(classification_report(y_test, y_pred_base))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_base))

# 5) LDA (max components = classes - 1 = 2)
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_s, y_train)
X_test_lda = lda.transform(X_test_s)

# 6) Classifier on LDA space
lda_clf = LogisticRegression(max_iter=300)
lda_clf.fit(X_train_lda, y_train)
y_pred_lda = lda_clf.predict(X_test_lda)
acc_lda = accuracy_score(y_test, y_pred_lda)

print("\n=== After LDA (Logistic Regression on LDA features) ===")
print("Accuracy:", acc_lda)
print(classification_report(y_test, y_pred_lda))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lda))

# 7) Visualize LDA-transformed data
plt.figure(figsize=(7,5))
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train)
plt.title("LDA - 2D Projection of Random/Synthetic Data")
plt.xlabel("LDA Component 1")
plt.ylabel("LDA Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

# 8) Compare accuracy before vs after LDA
plt.figure(figsize=(5,4))
labels = ["Before LDA", "After LDA"]
accs = [acc_base, acc_lda]
plt.bar(labels, accs)
for i, v in enumerate(accs):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
plt.ylim(0, 1.05)
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.tight_layout()
plt.show()
