import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt


def pick_latest_csv(pattern: str = "data/open_omen_markets_*.csv") -> str:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching {pattern}. Run export script first.")
    return files[-1]


def main(csv_path: str | None = None) -> None:
    if csv_path is None:
        csv_path = pick_latest_csv()
        print(f"Using latest file: {csv_path}")

    df = pd.read_csv(csv_path).dropna(subset=["p_yes"])
    df["p_yes"] = df["p_yes"].astype(float)

    print("N markets:", len(df))
    print("\nSummary stats:")
    print(df["p_yes"].describe())

    near_50 = ((df["p_yes"] >= 0.45) & (df["p_yes"] <= 0.55)).mean()
    near_60 = ((df["p_yes"] >= 0.40) & (df["p_yes"] <= 0.60)).mean()
    extreme = ((df["p_yes"] <= 0.10) | (df["p_yes"] >= 0.90)).mean()

    print("\nShare near 0.5 (0.45–0.55):", round(near_50, 3))
    print("Share near 0.5-ish (0.40–0.60):", round(near_60, 3))
    print("Share extreme (<=0.10 or >=0.90):", round(extreme, 3))

    plt.hist(df["p_yes"], bins=15)
    plt.title("Distribution of Market Probabilities (p_yes)")
    plt.xlabel("Probability YES")
    plt.ylabel("Count")
    plt.tight_layout()

    out_plot = "outputs/p_yes_histogram.png"
    plt.savefig(out_plot)
    print(f"\nSaved histogram to {out_plot}")


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) == 2 else None
    main(csv_arg)
