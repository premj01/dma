# 📘 Simple Apriori Algorithm for Association Rule Mining

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# Step 2: Create a simple retail dataset (transactions)
transactions = [
    ["milk", "bread", "butter"],
    ["bread", "butter"],
    ["milk", "bread"],
    ["milk", "bread", "butter", "jam"],
    ["bread", "jam"],
    ["milk", "bread", "butter"],
]

# Step 3: Convert transactions into one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Step 4: Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
print("🧺 Frequent Itemsets:")
print(frequent_itemsets)


# Step 5: Generate association rules (support, confidence, lift)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print("\n🔗 Association Rules:")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

# Step 6: Interpret top rules
print("\n🧠 Interpretation:")
for i, row in rules.iterrows():
    print(
        f"If a customer buys {list(row['antecedents'])}, "
        f"they are likely to buy {list(row['consequents'])} "
        f"(confidence = {row['confidence']:.2f}, lift = {row['lift']:.2f})"
    )
