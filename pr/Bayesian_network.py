# 📘 Simple Bayesian Network using Naive Bayes

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Step 2: Split data (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Step 3: Train Bayesian model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# 📘 Simple Bayesian Network Classification (Naive Bayes)

# Step 1: Import libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load real-world dataset (Iris)
iris = load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame for clarity
df = pd.DataFrame(X, columns=iris.feature_names)
df["Target"] = iris.target_names[y]
print("📊 Sample Data:")
print(df.head())


# Step 3: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 4: Create and train Bayesian model (Naive Bayes)
model = GaussianNB()
model.fit(X_train, y_train)

# Step 5: Predict class labels
y_pred = model.predict(X_test)

# Step 6: Evaluate model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%\n")
print("Confusion Matrix:\n", cm)
print(
    "\nClassification Report:\n",
    classification_report(y_test, y_pred, target_names=iris.target_names),
)

# Step 7: Visualize confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt="d",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
)
plt.title("Confusion Matrix - Naive Bayes Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
