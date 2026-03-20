import pandas as pd

df = pd.read_csv("analyzed_markets.csv")

# edge = model - market
df["edge"] = df["ai_prob"] - df["market_prob"]

# absolute edge
df["abs_edge"] = df["edge"].abs()

print("\n=== EDGE SUMMARY ===")
print(df["edge"].describe())

# where model thinks YES vs NO
print("\n=== DIRECTION ===")
print((df["edge"] > 0).value_counts())

# large edge opportunities
print("\n=== HIGH EDGE MARKETS ===")
print(df[df["abs_edge"] > 0.05][["title", "market_prob", "ai_prob", "edge"]].head(10))




bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]

df["prob_bin"] = pd.cut(df["market_prob"], bins=bins, labels=labels)

bin_analysis = df.groupby("prob_bin").agg(
    avg_edge=("edge", "mean"),
    avg_abs_edge=("abs_edge", "mean"),
    count=("edge", "size")
)

print("\n=== EDGE BY MARKET PROBABILITY ===")
print(bin_analysis)


print("\n=== VERDICT COUNTS ===")
print(df["verdict"].value_counts())




#if avg_edge is negative, model is less confident than market --> 
# "the model tends to be more septical in high confidence markets"

#if avg_edge is positive, model thinks market is underestimating YES
# "model identifies potential underevaluation in low-probability events

#if edge ~ 0, model aggrees with market
