# 📘 Simple FP-Growth Algorithm for Market Basket Analysis

# Step 1: Import required libraries

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Step 2: Create simple dataset (transactions)
# Each list = items purchased together by one customer
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

# Step 4: Apply FP-Growth algorithm to find frequent itemsets
frequent_itemsets = fpgrowth(df, min_support=0.4, use_colnames=True)
print("🧺 Frequent Itemsets:")
print(frequent_itemsets)

# Step 5: Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
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




# ✅ FP-Growth: Frequent Itemsets + Association Rules (simple & short)

!pip -q install mlxtend

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# -----------------------------
# 1) Sample transactions (replace with your data if needed)
# Each list is a customer basket (products bought together)
transactions = [
    ['milk','bread','eggs'],
    ['milk','bread'],
    ['bread','butter'],
    ['milk','eggs','cookies'],
    ['bread','eggs'],
    ['milk','bread','butter'],
    ['bread','cookies'],
    ['milk','bread','eggs','butter'],
    ['milk','cookies'],
    ['bread','butter','jam'],
]

# --- (Optional) Load from CSV ---
# If you have a CSV where each row is a basket like: "milk,bread,eggs"
# df_raw = pd.read_csv('baskets.csv', header=None)
# transactions = df_raw[0].apply(lambda x: [i.strip() for i in str(x).split(',')]).tolist()

# -----------------------------
# 2) One-hot encode the baskets
te = TransactionEncoder().fit(transactions)
df = pd.DataFrame(te.transform(transactions), columns=te.columns_)

# -----------------------------
# 3) FP-Growth: find frequent itemsets
# Adjust min_support (e.g., 0.2 to 0.5) to control how frequent is “frequent”
frequent_itemsets = fpgrowth(df, min_support=0.3, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)
print("🧺 Frequent Itemsets:")
print(frequent_itemsets)

# -----------------------------
# 4) Generate association rules
# metric='confidence' is common; you can also use 'lift' or 'leverage'
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)
rules = rules.sort_values(['lift','confidence'], ascending=False)
print("\n🔗 Association Rules (sorted by lift, confidence):")
print(rules[['antecedents','consequents','support','confidence','lift']])

# -----------------------------
# 5) Simple, human-readable interpretations
def nice(s): return ', '.join(list(s))
print("\n🧠 Easy Interpretations:")
for _, r in rules.head(5).iterrows():
    print(f"If a basket has [{nice(r['antecedents'])}], "
          f"then it often also has [{nice(r['consequents'])}] "
          f"(conf={r['confidence']:.2f}, lift={r['lift']:.2f})")
