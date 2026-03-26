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

print("\noverall YES rate")
print(df["resolution"].mean())
print(df["resolution"].value_counts())

print("probability range")
print(df["prob_at_close"].describe())

print("inspecting first few rows")
print(df[["question", "prob_at_close", "resolution"]].head(15))


print("check if high prob_at_close really corresponds to YES more often")
print(df.sort_values("prob_at_close", ascending=False)[["question", "prob_at_close", "resolution"]].head(20))


print("\nYES rate by category")
cat_summary = (
    df.groupby("category")
      .agg(
          n=("market_id", "count"),
          yes_rate=("resolution", "mean"),
          avg_prob=("prob_at_close", "mean"),
      )
      .sort_values("n", ascending=False)
)
print(cat_summary.head(15))



print("\nDuration-level base rates")
df["duration_bucket"] = pd.cut(
    df["duration_hours"],
    bins=[0, 24, 72, 168, 336, 10000],
    include_lowest=True
)

print("\nYES rate by duration bucket")
duration_summary = (
    df.groupby("duration_bucket", observed=False)
      .agg(
          n=("market_id", "count"),
          yes_rate=("resolution", "mean"),
          avg_prob=("prob_at_close", "mean"),
      )
      .reset_index()
)
print(duration_summary)


print("\n Volumn / liquidity effect")
df["volume_bucket"] = pd.qcut(df["volume"], q=4, duplicates="drop")

print("\nYES rate by volume bucket")
volume_summary = (
    df.groupby("volume_bucket", observed=False)
      .agg(
          n=("market_id", "count"),
          yes_rate=("resolution", "mean"),
          avg_prob=("prob_at_close", "mean"),
      )
      .reset_index()
)
print(volume_summary)

