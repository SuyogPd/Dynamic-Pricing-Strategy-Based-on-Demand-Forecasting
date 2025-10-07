import numpy as np
import pandas as pd

# Load train data with cost & price
train = pd.read_csv("train_with_price.csv", parse_dates=["date"])

# Example: Last known price & cost per item
item_cost = train.groupby("item_id")["cost"].first().to_dict()
item_price = train.groupby("item_id")["price"].mean().to_dict()

# -------------------------------
# Example Forecasted Sales
# (Replace this with your ML predictions)
# Format: DataFrame with [date, store_id, item_id, forecasted_sales]
# -------------------------------
forecast_dates = pd.date_range("2018-04-01", periods=7)  # Example: next 7 days
unique_store_items = train[["store_id", "item_id"]].drop_duplicates()

forecast_data = []
for date in forecast_dates:
    for _, row in unique_store_items.iterrows():
        forecast_data.append({
            "date": date,
            "store_id": row["store_id"],
            "item_id": row["item_id"],
            "forecasted_sales": np.random.randint(50, 200)  # Replace with model prediction
        })

forecast_df = pd.DataFrame(forecast_data)

# -------------------------------
# Dynamic Pricing Optimization
# -------------------------------
elasticity = -0.7  # price elasticity of demand (approx, can be tuned)

pricing_results = []

for idx, row in forecast_df.iterrows():
    store = row["store_id"]
    item = row["item_id"]
    date = row["date"]
    demand = row["forecasted_sales"]
    
    cost = item_cost[item]
    base_price = item_price[item]
    
    best_price = base_price
    best_profit = -1
    
    # Try prices ±20% of base price
    for p in np.arange(0.8 * base_price, 1.2 * base_price, 1):
        price_factor = (p - base_price) / base_price
        adjusted_demand = demand * (1 + elasticity * price_factor)
        adjusted_demand = max(0, int(adjusted_demand))  # no negative demand
        
        profit = (p - cost) * adjusted_demand
        
        if profit > best_profit:
            best_profit = profit
            best_price = int(p)
    
    pricing_results.append({
        "date": date,
        "store_id": store,
        "item_id": item,
        "base_price": int(base_price),
        "optimal_price": best_price,
        "forecasted_sales": demand,
        "expected_profit": int(best_profit)
    })

# Convert to DataFrame
dynamic_pricing_df = pd.DataFrame(pricing_results)

# Save for further use
dynamic_pricing_df.to_csv("dynamic_pricing_results.csv", index=False)

print(dynamic_pricing_df.head(10))




results = dynamic_pricing_df.copy()

# Profit at base price
results["profit_base"] = (results["base_price"] - 
                          results["base_price"].map(item_cost)) * results["forecasted_sales"]

# Profit at optimal price (already in expected_profit column)
results["profit_opt"] = results["expected_profit"]

# Profit gain %
results["profit_gain_pct"] = ((results["profit_opt"] - results["profit_base"]) / 
                              results["profit_base"].replace(0, np.nan)) * 100

# Summary
avg_gain = results["profit_gain_pct"].mean()
hit_rate = (results["profit_opt"] > results["profit_base"]).mean() * 100
total_gain = results["profit_opt"].sum() - results["profit_base"].sum()

print(f"✅ Average Profit Gain: {avg_gain:.2f}%")
print(f"✅ Hit Rate: {hit_rate:.2f}%")
print(f"✅ Total Profit Gain: {total_gain:.2f}")




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dynamic pricing results
results = pd.read_csv("dynamic_pricing_results.csv", parse_dates=["date"])

# Load item costs (needed for base profit)
train = pd.read_csv("train_with_price.csv")
item_cost = train.groupby("item_id")["cost"].first().to_dict()

# -------------------------------
# Step 1: Profit at base price
# -------------------------------
results["profit_base"] = (results["base_price"] - results["item_id"].map(item_cost)) * results["forecasted_sales"]

# Already have expected_profit as profit at optimal price
results["profit_opt"] = results["expected_profit"]

# -------------------------------
# Step 2: Profit gain percentage (handle zero base profit)
# -------------------------------
results = results[results["profit_base"] > 0]  # only consider rows with base profit > 0
results["profit_gain_pct"] = ((results["profit_opt"] - results["profit_base"]) / results["profit_base"]) * 100

# -------------------------------
# Step 3: Aggregate Metrics
# -------------------------------
avg_gain = results["profit_gain_pct"].mean()
hit_rate = (results["profit_opt"] > results["profit_base"]).mean() * 100
total_gain = results["profit_opt"].sum() - results["profit_base"].sum()

print(f"✅ Average Profit Gain: {avg_gain:.2f}%")
print(f"✅ Hit Rate: {hit_rate:.2f}%")
print(f"✅ Total Profit Gain: {total_gain:.2f}")

# -------------------------------
# Step 4: Visualize Base vs Optimal Profit
# -------------------------------
# Pick a sample store-item for plotting
sample_store = 1
sample_item = 1

sample_df = results[(results["store_id"] == sample_store) & 
                    (results["item_id"] == sample_item)].copy()

plt.figure(figsize=(10,6))
plt.plot(sample_df["date"], sample_df["profit_base"], label="Profit at Base Price", marker="o")
plt.plot(sample_df["date"], sample_df["profit_opt"], label="Profit at Optimal Price", marker="s")
plt.xlabel("Date")
plt.ylabel("Profit")
plt.title(f"Profit Comparison (Store {sample_store}, Item {sample_item})")
plt.legend()
plt.grid(True)
plt.show()
