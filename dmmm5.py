# 🚀 Association Rule Mining using Apriori Algorithm (Simple + Visual)
# Author: Shreyash Dongare

import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt

# 🧺 Step 1: Create sample retail dataset
dataset = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter', 'Jam'],
    ['Milk', 'Bread', 'Eggs'],
    ['Milk', 'Eggs', 'Butter'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread', 'Butter', 'Eggs'],
    ['Butter', 'Jam'],
    ['Milk', 'Bread'],
    ['Bread', 'Butter'],
    ['Milk', 'Eggs']
]

# 🧾 Step 2: Convert to binary matrix
items = sorted({i for basket in dataset for i in basket})
df = pd.DataFrame([{i: i in basket for i in items} for basket in dataset]).astype(int)
n = len(df)

# 🔍 Step 3: Generate association rules
rules = []
for a, b in combinations(items, 2):
    both = (df[a] & df[b]).sum()
    s = both / n
    if s >= 0.2:
        c1, c2 = s / (df[a].sum() / n), s / (df[b].sum() / n)
        rules += [
            {'Rule': f'{a}→{b}', 'Support': s, 'Confidence': c1, 'Lift': c1 / (df[b].sum() / n)},
            {'Rule': f'{b}→{a}', 'Support': s, 'Confidence': c2, 'Lift': c2 / (df[a].sum() / n)}
        ]

rules_df = pd.DataFrame(rules).round(3).sort_values(by='Lift', ascending=False)

# 📊 Step 4: Display results
print("=== Apriori Association Rules ===\n")
print(rules_df)

# 📈 Step 5: Visualize top 5 rules by Lift
top_rules = rules_df.head(5)
plt.barh(top_rules['Rule'], top_rules['Lift'], color='skyblue')
plt.xlabel('Lift Value')
plt.ylabel('Rule')
plt.title('Top 5 Association Rules by Lift')
plt.gca().invert_yaxis()
plt.show()

# 💡 Step 6: Interpretation
print("\n➡ Lift > 1 → Positive association (items bought together).")
print("➡ Higher Confidence → Stronger relationship between items.")
