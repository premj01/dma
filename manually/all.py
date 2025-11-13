##### EXP 2

import pandas as pd
import numpy as np
import re

# 1. Load the Excel file created from ETL step
df = pd.read_excel('Combined_Data.xlsx')
# 2. Display initial infoprint("Initial Dataset Info:")
print(df.info())
# 3. Handle Missing Values
df['Region'] = df['Region'].fillna('Unknown')
df['Sales'] = df['Sales'].fillna(df['Sales'].mean())
# 4. Remove Duplicates
df.drop_duplicates(inplace=True)
# 5. Format Text Columns
df['Region'] = df['Region'].str.title()
df['Customer_Name'] = df['Customer_Name'].str.title()
# 6. Convert Date Column
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
# 7. Clean Product Names
df['Product'] = df['Product'].str.strip().str.capitalize()
# 8. Create a New Feature – Sales Category
df['Sales_Category'] = pd.cut(df['Sales'],
bins=[0, 30000, 50000, 70000],
labels=['Low', 'Medium', 'High'])
# 9. Display Cleaned Data
print("\nCleaned & Preprocessed Data:")print(df.head())
# 10. Save Final Cleaned Dataset
df.to_excel('Final_Cleaned_Data.xlsx', index=False)
print("\n Final Cleaned Data saved as 'Final_Cleaned_Data.xlsx'")

##### EXP 3

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
viral = np.random.multivariate_normal([5, 2, 1, 3], np.eye(4), 50)
bacterial = np.random.multivariate_normal([10, 5, 6, 7], np.eye(4), 50)
X = np.vstack((viral, bacterial))
y = np.array([0]*50 + [1]*50) # 0=viral, 1=bacterial
classes = np.unique(y)
n_features = X.shape[1]

mean_vectors = []
for c in classes:
    mean_vectors.append(np.mean(X[y==c], axis=0))

S_w = np.zeros((n_features, n_features))
for c, mv in zip(classes, mean_vectors):
    class_scatter = np.zeros((n_features, n_features))
    for row in X[y==c]:
        row, mv = row.reshape(n_features,1), mv.reshape(n_features,1)
        class_scatter += (row - mv).dot((row - mv).T)
S_w += class_scatter


overall_mean = np.mean(X, axis=0).reshape(n_features,1)
S_b = np.zeros((n_features, n_features))
for i, mean_vec in enumerate(mean_vectors):
    n = X[y==i].shape[0]
    mean_vec = mean_vec.reshape(n_features,1)
    S_b += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_w).dot(S_b))# Sort eigenvectors by decreasing eigenvalues
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

W = eig_pairs[0][1].reshape(n_features, 1)
X_lda = X.dot(W)
plt.figure(figsize=(8,5))
plt.scatter(X_lda[y==0], np.zeros_like(X_lda[y==0]), color='blue', label='Viral')
plt.scatter(X_lda[y==1], np.zeros_like(X_lda[y==1]), color='red', label='Bacterial')
plt.xlabel('LDA Component 1')
plt.title('LDA from Scratch: Viral vs Bacterial')
plt.legend()


####### EXP 4

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
viral = np.random.multivariate_normal([5, 2, 1, 3], np.eye(4), 50)
bacterial = np.random.multivariate_normal([10, 5, 6, 7], np.eye(4), 50)
X = np.vstack((viral, bacterial))
y = np.array([0]*50 + [1]*50) # 0=viral, 1=bacterial

X_meaned = X - np.mean(X, axis=0)
cov_mat = np.cov(X_meaned, rowvar=False)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

W = np.hstack((eig_pairs[0][1].reshape(-1,1), eig_pairs[1][1].reshape(-1,1)))
X_pca = X_meaned.dot(W)

plt.figure(figsize=(8,5))
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='blue', label='Viral')
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='red', label='Bacterial')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA from Scratch: Viral vs Bacterial')
plt.legend()
plt.show()


##### EXP 5

from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt

# Dataset 1: Retail transactions
transactions = [
    ['Milk', 'Bread', 'Eggs'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread'],
    ['Milk', 'Eggs'],
    ['Bread', 'Eggs', 'Butter'],
    ['Milk', 'Bread', 'Eggs', 'Butter']
]
def get_support(itemset, transactions):
    return sum(1 for t in transactions if set(itemset).issubset(set(t))) / len(transactions)

# Apriori algorithm
def apriori(transactions, min_support=0.3):
    items = set(i for t in transactions for i in t)
    frequent = []
    for i in range(1, len(items)+1):
        for combo in combinations(items, i):
            support = get_support(combo, transactions)
            if support >= min_support:
                frequent.append((combo, support))
    return frequent

def generate_rules(frequent):
    rules = []
    for itemset, support in frequent:
        if len(itemset) >= 2:
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    consequent = tuple(set(itemset) - set(antecedent))
                    conf = get_support(itemset, transactions) / get_support(antecedent, transactions)
                    lift = conf / get_support(consequent, transactions)
                    rules.append((antecedent, consequent, conf, lift))
    return rules
frequent_itemsets = apriori(transactions)
rules = generate_rules(frequent_itemsets) 
for r in rules:
    print(f"Rule: {r[0]} → {r[1]}, Confidence: {r[2]:.2f}, Lift: {r[3]:.2f}")

def draw_graph(rules, title):
    G = nx.DiGraph()
    for a, c, conf, lift in rules:
        G.add_edge(', '.join(a), ', '.join(c), label=f'C:{conf:.2f}, L:{lift:.2f}')
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title(title)
    plt.show()
frequent_itemsets = apriori(transactions)
rules = generate_rules(frequent_itemsets)
draw_graph(rules, "Retail Basket Association Rules")


######## EXP 6

import math
# Format: [Study Hours, Attendance %, Grade, Passed]
data = [
    [5, 90, 'A', 'Yes'],
    [2, 69, 'C', 'Yes'],
    [4, 80, 'B', 'Yes'],
    [1, 50, 'D', 'No'],
    [3, 75, 'B', 'Yes'],
    [2, 55, 'C', 'No']
]
# Gini Impurity Function
def gini(groups):
    total = sum(len(g) for g in groups)
    score = 0
    for group in groups:
        size = len(group)
        if size == 0: continue
        yes = sum(1 for r in group if r[-1] == 'Yes')
        no = size - yes
        p_yes = yes / size
        p_no = no / size
        score += (1 - p_yes**2 - p_no**2) * (size / total)
    return score
# Entropy Function for Information Gain
def entropy(group):
    size = len(group)
    if size == 0: return 0
    yes = sum(1 for r in group if r[-1] == 'Yes')
    no = size - yes
    p_yes = yes / size if yes else 0
    p_no = no / size if no else 0
    return -sum(p * math.log2(p) for p in [p_yes, p_no] if p > 0)
# Split by Grade
grade_A = [r for r in data if r[2] == 'A']
grade_B = [r for r in data if r[2] == 'B']
grade_C = [r for r in data if r[2] == 'C']
grade_D = [r for r in data if r[2] == 'D']

# Gini Impurity
gini_score = gini([grade_A, grade_B, grade_C, grade_D])
print("Gini Impurity for Grade Split:", round(gini_score, 3))

# Information Gain
total_entropy = entropy(data)
split_entropy = (
    len(grade_A)/len(data)*entropy(grade_A) +
    len(grade_B)/len(data)*entropy(grade_B) +
    len(grade_C)/len(data)*entropy(grade_C) +
    len(grade_D)/len(data)*entropy(grade_D)
)
info_gain = total_entropy - split_entropy
print("Information Gain for Grade Split:", round(info_gain, 3))
# Simple Decision Rule Based on Grade
def predict(row):
    return 'Yes' if row[2] in ['A', 'B'] else 'No'

# Accuracy Calculation
correct = sum(1 for r in data if predict(r) == r[-1])
accuracy = correct / len(data)
print("Model Accuracy:", round(accuracy, 2))


###### EXP 7

from itertools import combinations
import math, collections
import networkx as nx
import matplotlib.pyplot as plt

# ----------------------- Data (same as yours) -----------------------
transactions = [
    ['Milk', 'Bread', 'Eggs'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread'],
    ['Milk', 'Eggs'],
    ['Bread', 'Eggs', 'Butter'],
    ['Milk', 'Bread', 'Eggs', 'Butter']
]

# ----------------------- Helpers -----------------------
def get_support(itemset, transactions):
    iset = set(itemset)
    return sum(1 for t in transactions if iset.issubset(set(t))) / len(transactions)

# ----------------------- FP-Tree Node -----------------------
class FPNode:
    __slots__ = ("item", "count", "parent", "children", "link")
    def __init__(self, item, parent=None):
        self.item = item
        self.count = 1
        self.parent = parent
        self.children = {}   # item -> FPNode
        self.link = None     # next node with same item

# ----------------------- Build FP-Tree -----------------------
def build_fptree(transactions, min_support):
    n = len(transactions)
    min_count = math.ceil(min_support * n)

    # 1) count items and keep only frequent ones
    cnt = collections.Counter()
    for t in transactions:
        cnt.update(set(t))  # avoid dup items inside a txn
    freq_items = {i:c for i,c in cnt.items() if c >= min_count}
    if not freq_items: 
        return None, {}, min_count

    # global order: by freq desc, then lexicographic
    order = sorted(freq_items.items(), key=lambda x:(-x[1], str(x[0])))
    rank = {i:r for r,(i,_) in enumerate(order)}

    # header table: item -> [total_count, first_node]
    header = {i:[c, None] for i,c in freq_items.items()}

    root = FPNode(None)

    def add_transaction(t):
        # filter & sort by global order
        items = [i for i in t if i in freq_items]
        items.sort(key=lambda x: rank[x])
        cur = root
        for it in items:
            if it not in cur.children:
                child = FPNode(it, cur)
                cur.children[it] = child
                # link into header chain
                head = header[it][1]
                if head is None: header[it][1] = child
                else:
                    while head.link: head = head.link
                    head.link = child
                cur = child
            else:
                cur.children[it].count += 1
                cur = cur.children[it]

    for t in transactions:
        add_transaction(t)

    return root, header, min_count

# ----------------------- Mine FP-Tree -----------------------
def ascend_path(node):
    path = []
    while node.parent and node.parent.item is not None:
        node = node.parent
        path.append(node.item)
    return path

def conditional_base(item, header):
    base = []  # list of (path, count)
    node = header[item][1]
    while node:
        path = ascend_path(node)
        if path:
            base.append((path, node.count))
        node = node.link
    return base

def fpgrowth(transactions, min_support=0.3):
    root, header, min_count = build_fptree(transactions, min_support)
    if header == {}: return []

    n = len(transactions)
    freq_counts = {}  # frozenset(items) -> absolute count

    # items in increasing frequency order (suffix growth)
    items_inc = sorted(header.items(), key=lambda kv:(kv[1][0], str(kv[0])))

    def mine(header, suffix):
        # header: item -> [total_count, first_node]
        items = sorted(header.items(), key=lambda kv:(kv[1][0], str(kv[0])))
        for item, (tot, head) in items:
            new_itemset = tuple(sorted(suffix + [item]))
            freq_counts[frozenset(new_itemset)] = tot

            # build conditional pattern base
            base = conditional_base(item, header)
            cond_txns = []
            for path, c in base:
                for _ in range(c):
                    cond_txns.append(path)
            if not cond_txns: 
                continue

            # build conditional tree with SAME absolute min_count
            cond_root, cond_header, _ = build_fptree(cond_txns, min_support=min_count/len(cond_txns))
            # ensure absolute threshold
            # prune header entries below min_count
            if cond_header:
                pruned = {i:v for i,v in cond_header.items() if v[0] >= min_count}
                if pruned:
                    mine(pruned, list(new_itemset))

    mine(header, [])
    # convert absolute counts -> relative supports
    return [(tuple(sorted(list(iset))), cnt/n) for iset, cnt in freq_counts.items()]

# ----------------------- Rules (confidence & lift) -----------------------
def generate_rules(frequent, transactions, min_conf=0.0):
    # build quick support dict
    supp = {tuple(sorted(fs)): s for fs, s in frequent}
    rules = []
    for itemset, s_ab in frequent:
        if len(itemset) < 2: 
            continue
        for r in range(1, len(itemset)):
            for A in combinations(itemset, r):
                A = tuple(sorted(A))
                B = tuple(sorted(set(itemset)-set(A)))
                s_a = supp.get(A, get_support(A, transactions))
                s_b = supp.get(B, get_support(B, transactions))
                if s_a == 0 or s_b == 0: 
                    continue
                conf = s_ab / s_a
                if conf >= min_conf:
                    lift = conf / s_b
                    rules.append((A, B, conf, lift))
    return rules

# ----------------------- Graph (reuse your style) -----------------------
def draw_graph(rules, title):
    G = nx.DiGraph()
    for a, c, conf, lift in rules:
        G.add_edge(', '.join(a), ', '.join(c), label=f'C:{conf:.2f}, L:{lift:.2f}')
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'), font_color='red')
    plt.title(title); plt.show()

# ----------------------- Run (like your Apriori flow) -----------------------
frequent_itemsets = fpgrowth(transactions, min_support=0.3)
rules = generate_rules(frequent_itemsets, transactions)

print("Frequent itemsets:")
for fs, s in sorted(frequent_itemsets, key=lambda x:(-x[1], x[0])):
    print(f"{fs} -> support={s:.2f}")

print("\nRules:")
for a, b, c, l in rules:
    print(f"Rule: {a} → {b}, Confidence: {c:.2f}, Lift: {l:.2f}")

draw_graph(rules, "Retail Basket Association Rules (FP-Growth)")




####### EXP 8

# Format: [Age, Gender, ChestPainType, HeartDisease]
data = [
    [63, 'Male', 'Typical', 'Yes'],
    [37, 'Female', 'Non-anginal', 'No'],
    [56, 'Male', 'Atypical', 'Yes'],
    [41, 'Female', 'Atypical', 'No'],
    [67, 'Male', 'Asymptomatic', 'Yes'],
    [62, 'Female', 'Typical', 'No']
]
def prior(label):
    return sum(1 for r in data if r[-1] == label) / len(data)

def cond_prob(index, value, label):
    count = sum(1 for r in data if r[index] == value and r[-1] == label)
    total = sum(1 for r in data if r[-1] == label)
    return count / total if total else 0

def predict(evidence):
    labels = ['Yes', 'No']
    probs = {}
    for label in labels:
        p = prior(label)
        for i, val in enumerate(evidence):
            p *= cond_prob(i, val, label)
        probs[label] = p
    return max(probs, key=probs.get)

# Evaluate accuracy and confusion matrix
y_true = [r[-1] for r in data]
y_pred = [predict(r[:-1]) for r in data]

tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'Yes' and yp == 'Yes')
tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'No' and yp == 'No')
fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'No' and yp == 'Yes')
fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'Yes' and yp == 'No')

accuracy = (tp + tn) / len(data)

print("Accuracy:", round(accuracy, 2))
print("Confusion Matrix:")
print(f"TP: {tp}, FN: {fn}")
print(f"FP: {fp}, TN: {tn}")


###### EXP 9

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error
import numpy as np
# Sample dataset (Monthly Sales)
data = {
'Month': pd.date_range(start='2020-01', periods=24, freq='M'),
'Sales':
[200,220,250,270,300,320,350,370,400,420,450,470,490,510,530,550,580,600,620,640,670,690,710,730]
}
df = pd.DataFrame(data)
df.set_index('Month', inplace=True)
# Decomposition
result = seasonal_decompose(df['Sales'], model='additive', period=12)
result.plot()
plt.show()
# Moving Average Forecast
df['MA_3'] = df['Sales'].rolling(window=3).mean()

# Forecast next value
forecast = df['MA_3'].iloc[-1]
print("Next Month Forecast:", forecast)
# Error metric (example)
actual = df['Sales'].iloc[-1]
error = mean_absolute_error([actual], [forecast])
print("MAE:", error)

##### EXP 10


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Create a dummy dataset array using numpy
np.random.seed(0)
num_days = 365
dates = pd.date_range(start='2020-01-01', periods=num_days, freq='D')

# Construct a synthetic daily new cases series with an increasing trend and noise
trend = np.linspace(50, 5000, num_days)
noise = np.random.normal(loc=0, scale=200, size=num_days)
daily_new = np.maximum(0, (trend + noise)).astype(int)

# Cumulative confirmed cases
confirmed = np.cumsum(daily_new)

# Numpy array representation (day index, confirmed)
dummy_array = np.column_stack((np.arange(num_days), confirmed))

# Convert to DataFrame expected by the rest of the notebook
data = pd.DataFrame({'Date': dates, 'Confirmed': confirmed})
data.set_index('Date', inplace=True)

# Visualize Total Confirmed Cases
plt.figure(figsize=(10, 6))
data.plot(title='Total COVID-19 Confirmed Cases in India Over Time (Dummy Data)', figsize=(10,6))
plt.xlabel('Date')
plt.ylabel('Total Confirmed Cases')
plt.grid(True)
plt.show()

# Stationarity test
result = adfuller(data['Confirmed'])
print(f'ADF Statistic: {result[0]:.10f}')
print(f'p-value: {result[1]:.10f}')

# Differencing if non-stationary
data_diff = data['Confirmed'].diff().dropna()

# Fit ARIMA model
model = ARIMA(data_diff, order=(2,1,2))
model_fit = model.fit()
# print(model_fit.summary()) # Removed model summary from output

# Forecast
forecast_steps = 10
forecast = model_fit.forecast(steps=forecast_steps)

# Create a date index for the forecast
last_date = data.index[-1]
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
forecast.index = forecast_index

# Plot Actual Daily New Cases and Forecast
plt.figure(figsize=(12, 7))
plt.plot(data_diff.tail(100), label='Actual Daily New Cases (last 100 days)') # Plot last 100 actual values for better visualization
plt.plot(forecast, label='Forecasted Daily New Cases', color='red')
plt.title('ARIMA Forecast of Daily New COVID-19 Cases in India (Dummy Data)')
plt.xlabel('Date')
plt.ylabel('Daily New Cases')
plt.legend()
plt.grid(True)
plt.show()

try:
    predictions = model_fit.predict(start=len(data_diff)-forecast_steps, end=len(data_diff)-1)
    rmse = np.sqrt(mean_squared_error(data_diff[-forecast_steps:], predictions))
    print(f"RMSE: {rmse:.10f}")
except:
    print("Could not calculate RMSE for the last", forecast_steps, "steps.")