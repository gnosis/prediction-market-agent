import pandas as pd

# Load market dataset
df = pd.read_csv("data/open_omen_markets_2026-03-03.csv")

print("\n=== Columns ===")
print(df.columns.tolist())

df["collateralVolume"] = pd.to_numeric(df["collateralVolume"], errors="coerce")


print("\n=== Markets per Category ===")
print(df["category"].value_counts())

print("\n=== Average Liquidity by Category ===")
print(
    df.groupby("category")["collateralVolume"]
    .mean()
    .sort_values(ascending=False)
)

print("\n=== Average YES Probability by Category ===")
print(
    df.groupby("category")["p_yes"]
    .mean()
    .sort_values(ascending=False)
)

print("\n=== Combined Summary ===")

summary = df.groupby("category").agg({
    "collateralVolume": "mean",
    "p_yes": "mean",
    "market_id": "count"
}).rename(columns={"market_id": "market_count"})

print(summary.sort_values("collateralVolume", ascending=False))
