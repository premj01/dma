# 📘 Decision Tree and Random Forest Classification

# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 2: Load labelled dataset (Iris)
iris = load_iris()
X, y = iris.data, iris.target


# Step 3: Split into training and testing data (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 4: Train Decision Tree model (using Gini Impurity)
dt_model = DecisionTreeClassifier(criterion="gini", random_state=42)
dt_model.fit(X_train, y_train)

# Step 5: Train Random Forest model (ensemble of Decision Trees)
rf_model = RandomForestClassifier(n_estimators=100, criterion="gini", random_state=42)
rf_model.fit(X_train, y_train)

# Step 6: Predict and evaluate both models
y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Step 7: Evaluate accuracy
acc_dt = accuracy_score(y_test, y_pred_dt)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("✅ Decision Tree Accuracy:", round(acc_dt * 100, 2), "%")
print("✅ Random Forest Accuracy:", round(acc_rf * 100, 2), "%")

# Step 8: Display confusion matrices
cm1 = confusion_matrix(y_test, y_pred_dt)
print("\n📊 Decision Tree Confusion Matrix:")
print(cm1)

cm2 = confusion_matrix(y_test, y_pred_rf)
print("\n📊 Random Forest Confusion Matrix:")
print(cm2)

# Step 9: Detailed classification report
print("\n📄 Classification Report (Decision Tree):")
print(classification_report(y_test, y_pred_dt, target_names=iris.target_names))

print("\n📄 Classification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf, target_names=iris.target_names))

# fig ,axes = plt.subplot(1,2)

plt.imshow(cm1, cmap="Blues")
plt.title("Title")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

plt.imshow(cm2, cmap="Greens")
plt.title("Title")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()


# 📘 Decision Tree vs Random Forest using CSV Dataset

# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load your dataset (replace filename with your actual file)
# Example: "retail_customers.csv" with columns like Age, Income, Spending, Category
data = pd.read_csv("CM_retail.csv")
print("📊 Dataset Preview:")
print(data.head())

# Step 3: Separate features and target (label)
# Assuming last column is the label (change if needed)
X = data.iloc[:, :-1]  # features
y = data.iloc[:, -1]  # target

# Encode labels if they are categorical
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# Step 4: Split dataset into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 5: Train Decision Tree model (Gini Impurity)
dt_model = DecisionTreeClassifier(criterion="gini", random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Step 6: Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, criterion="gini", random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Step 7: Evaluate both models
acc_dt = accuracy_score(y_test, y_pred_dt)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("\n✅ Model Accuracies:")
print(f"Decision Tree Accuracy: {acc_dt*100:.2f}%")
print(f"Random Forest Accuracy: {acc_rf*100:.2f}%")

print("\n📊 Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

print("\n📊 Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\n📄 Classification Report (Decision Tree):")
print(classification_report(y_test, y_pred_dt))

print("\n📄 Classification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))
