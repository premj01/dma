import pandas as pd
import json

# -------------------------------
# 1️⃣ Extract
# -------------------------------
# Read the raw data
df_raw = pd.read_csv("a_raw_sales_data_with_nulls.csv")

print("🔹 Extracted Raw Data:")
print(df_raw.head())
print(f"\nTotal rows before cleaning: {len(df_raw)}")

# -------------------------------
# 2️⃣ Transform
# -------------------------------

# Convert Date to datetime
df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")

# Handle missing values
# Remove rows with missing critical fields
df_clean = df_raw.dropna(subset=["Region", "Product", "Discount", "Date"])

# Fill missing text fields if needed
df_clean["SalesCategory"] = df_clean["SalesCategory"].fillna("Uncategorized")

# Calculate TotalSales and NetSales
df_clean["TotalSales"] = df_clean["Quantity"] * df_clean["UnitPrice"]
df_clean["NetSales"] = df_clean["TotalSales"] - df_clean["Discount"]


# Parse JSON-like CustomerInfo
def parse_json_safe(text):
    try:
        return json.loads(text)
    except:
        return {}


customer_info_df = df_clean["CustomerInfo"].apply(parse_json_safe).apply(pd.Series)

# Merge parsed info back
df_transformed = pd.concat(
    [df_clean.drop(columns=["CustomerInfo"]), customer_info_df], axis=1
)

# Add derived columns (Year, Month)
df_transformed["Year"] = df_transformed["Date"].dt.year
df_transformed["Month"] = df_transformed["Date"].dt.month_name()


# Basic sentiment tagging from feedback
def get_sentiment(feedback):
    feedback = str(feedback).lower()
    if any(
        word in feedback
        for word in ["great", "excellent", "good", "love", "works", "perfect"]
    ):
        return "Positive"
    elif any(
        word in feedback for word in ["late", "bad", "poor", "expensive", "not worth"]
    ):
        return "Negative"
    else:
        return "Neutral"


df_transformed["Sentiment"] = df_transformed["CustomerFeedback"].apply(get_sentiment)

print("\n🔹 Transformed & Cleaned Data:")
print(df_transformed.head())
print(f"\nTotal rows after cleaning: {len(df_transformed)}")

# -------------------------------
# 3️⃣ Load
# -------------------------------

# Save cleaned data
output_file = "transformed_sales_data.csv"
df_transformed.to_csv(output_file, index=False)

print(f"\n✅ ETL completed successfully! Cleaned data saved as '{output_file}'")
