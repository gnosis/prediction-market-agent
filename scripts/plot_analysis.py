import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("omen_resolved_snapshots.csv")

# Clean data
df = df.dropna(subset=["prob_at_close", "resolution"])
df = df[(df["prob_at_close"] >= 0) & (df["prob_at_close"] <= 1)]

# =========================
# 1. CALIBRATION CURVE
# =========================

bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0]

df["prob_bucket"] = pd.cut(df["prob_at_close"], bins=bins, include_lowest=True)

summary = (
    df.groupby("prob_bucket", observed=False)
    .agg(
        avg_pred=("prob_at_close", "mean"),
        actual_yes_rate=("resolution", "mean"),
        n=("market_id", "count")
    )
    .reset_index()
)

# Remove tiny buckets for a cleaner plot
summary = summary[summary["n"] >= 10]

figsize=(6.5, 4.5)

plt.plot(
    summary["avg_pred"],
    summary["actual_yes_rate"],
    marker="o",
    markersize=7,
    linewidth=2,
    color="#6637b8",
    label="Observed"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=2,
    color="#23143d",
    label="Perfect Calibration"
)

# Optional annotation to reinforce the gap
plt.annotate(
    "~14% overall YES rate",
    xy=(0.52, 0.14),
    xytext=(0.62, 0.24),
    arrowprops=dict(arrowstyle="->", color="#23143d"),
    fontsize=10
)

plt.xlabel("Predicted Probability")
plt.ylabel("Actual YES Rate")
plt.title("Market Calibration Curve")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(frameon=False)
plt.grid(False)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.savefig("calibration_curve.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: calibration_curve.png")

# =========================
# 2. LIQUIDITY CHART
# =========================

df["volume_bucket"] = pd.qcut(df["volume"], q=4, duplicates="drop")

volume_summary = (
    df.groupby("volume_bucket", observed=False)
    .agg(
        yes_rate=("resolution", "mean"),
        n=("market_id", "count")
    )
    .reset_index()
)

figsize=(6.5, 4.5)

x_pos = range(len(volume_summary))
labels = ["Low", "Med-Low", "Med-High", "High"]

bars = plt.bar(
    x_pos,
    volume_summary["yes_rate"],
    color="#6637b8"
)

# Add value labels above bars
for i, v in enumerate(volume_summary["yes_rate"]):
    plt.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=10)

plt.xlabel("Volume Bucket")
plt.ylabel("YES Rate")
plt.title("Outcome Frequency by Market Liquidity")
plt.xticks(x_pos, labels, rotation=0)
plt.grid(False)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.savefig("liquidity_vs_outcome.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: liquidity_vs_outcome.png")
