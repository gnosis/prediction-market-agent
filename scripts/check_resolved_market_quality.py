import pandas as pd

df = pd.read_csv("resolved_omen_markets.csv")

print("\n=== COLUMN COMPLETENESS ===")
print("rows:", len(df))
print("non-null p_yes:", df["p_yes"].notna().sum())
print("non-null outcome_yes:", df["outcome_yes"].notna().sum())
print("non-null both:", df[["p_yes", "outcome_yes"]].dropna().shape[0])

print("\n=== SAMPLE ROWS WITH p_yes ===")
print(df[df["p_yes"].notna()][["title", "p_yes", "outcome_yes", "usdVolume"]].head(10))
