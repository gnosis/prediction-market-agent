import pandas as pd

df = pd.read_csv("omen_resolved_snapshots.csv")

# Clean
df = df.dropna(subset=["prob_at_close", "resolution"])

# Bucket probabilities
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0]

df["prob_bucket"] = pd.cut(df["prob_at_close"], bins=bins, include_lowest=True)

summary = (
    df.groupby("prob_bucket", observed=False)
    .agg(
        n=("market_id", "count"),
        avg_pred=("prob_at_close", "mean"),
        actual_yes_rate=("resolution", "mean")
    )
    .reset_index()
)

summary["mispricing"] = summary["actual_yes_rate"] - summary["avg_pred"]

print(summary)

print("overall YES rate")
print(df["resolution"].mean())
print(df["resolution"].value_counts())

print("probability range")
print(df["prob_at_close"].describe())

print("inspecting first few rows")
print(df[["question", "prob_at_close", "resolution"]].head(15))


print("check if high prob_at_close really corresponds to YES more often")
print(df.sort_values("prob_at_close", ascending=False)[["question", "prob_at_close", "resolution"]].head(20))

