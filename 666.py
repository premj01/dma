# 🚀 Minimal Example: Decision Tree vs Random Forest (Random Data)
# Author: Shreyash Dongare

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Generate random labeled data (features + target)
np.random.seed(42)
X = np.random.randint(18, 60, (100, 4))  # age-like features
y = np.random.randint(0, 2, 100)         # binary labels (0 = No, 1 = Yes)
df = pd.DataFrame(X, columns=["Age", "Income", "Score", "Experience"])
df["Purchase"] = y

# 2️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(df.drop("Purchase", axis=1), y, test_size=0.3, random_state=42)

# 3️⃣ Train models
dt = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, criterion="gini", random_state=42).fit(X_train, y_train)

# 4️⃣ Evaluate
for name, model in [("Decision Tree", dt), ("Random Forest", rf)]:
    pred = model.predict(X_test)
    print(f"\n🌳 {name} Accuracy:", round(accuracy_score(y_test, pred), 3))
    print(classification_report(y_test, pred))

# 5️⃣ Visualize Decision Tree
plt.figure(figsize=(8,5))
plot_tree(dt, feature_names=X_train.columns, class_names=["No", "Yes"], filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

# 6️⃣ Feature Importance (Random Forest)
imp = rf.feature_importances_
plt.bar(X_train.columns, imp, color='teal')
plt.title("Feature Importance - Random Forest")
plt.show()

# 7️⃣ Gini & Information Gain Example
def gini(p): return 1 - sum([i**2 for i in p])
parent, left, right = gini([0.5,0.5]), gini([0.7,0.3]), gini([0.4,0.6])
info_gain = parent - (0.6*left + 0.4*right)
print(f"\n🧮 Gini(70/30): {round(left,3)}, Info Gain Example: {round(info_gain,3)}")
