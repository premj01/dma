# 🚀 Association Rule Mining using Apriori Algorithm on Retail Dataset
# Author: Shreyash Dongare

import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt

# 🧾 Step 1: Load the retail dataset
file_path = "C:/Users/shreyash/Downloads/archive (4)/retail_sales_dataset.csv"  # ← make sure your file is in the same folder
data = pd.read_csv(file_path)

print("✅ Dataset Loaded Successfully!\n")
print(data.head())

# 🧺 Step 2: Group items by Transaction ID
# We'll assume your dataset has "Transaction ID" and "Product Category" columns
transactions = data.groupby("Transaction ID")["Product Category"].apply(list).tolist()

# 🧱 Step 3: Convert transactions into a binary matrix
items = sorted({i for t in transactions for i in t})
df = pd.DataFrame([{i: i in t for i in items} for t in transactions]).astype(int)
n = len(df)

# 🔍 Step 4: Generate association rules manually
rules = []
min_support = 0.1  # Minimum support threshold (you can adjust)

for a, b in combinations(items, 2):
    both = (df[a] & df[b]).sum()
    s = both / n
    if s >= min_support:
        c1, c2 = s / (df[a].sum() / n), s / (df[b].sum() / n)
        rules += [
            {'Rule': f'{a} → {b}', 'Support': s, 'Confidence': c1, 'Lift': c1 / (df[b].sum() / n)},
            {'Rule': f'{b} → {a}', 'Support': s, 'Confidence': c2, 'Lift': c2 / (df[a].sum() / n)}
        ]

# ✅ Step 5: Display results
if rules:
    rules_df = pd.DataFrame(rules).round(3).sort_values(by='Lift', ascending=False).reset_index(drop=True)
    print("\n=== Association Rules using Apriori ===\n")
    print(rules_df)

    # 📈 Step 6: Plot top 5 rules by Lift
    top_rules = rules_df.head(5)
    plt.barh(top_rules['Rule'], top_rules['Lift'], color='lightgreen')
    plt.xlabel('Lift Value')
    plt.ylabel('Rule')
    plt.title('Top 5 Association Rules by Lift')
    plt.gca().invert_yaxis()
    plt.show()

    # 💡 Step 7: Interpretation
    print("\n=== Interpretation ===")
    print("➡ Lift > 1 → Positive association (items bought together often).")
    print("➡ Higher Confidence → Stronger prediction of buying the consequent item.")
else:
    print("⚠️ No association rules found. Try lowering min_support value.")
