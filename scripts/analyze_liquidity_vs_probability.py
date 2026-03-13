import sys
import glob
import os
import pandas as pd


def get_latest_csv():
    files = glob.glob("data/open_omen_markets_*.csv")
    if not files:
        raise FileNotFoundError("No open_omen_markets CSV found in /data")
    return max(files, key=os.path.getctime)


def main(csv_path=None):

    if csv_path is None:
        csv_path = get_latest_csv()
        print(f"Using latest file: {csv_path}")

    df = pd.read_csv(csv_path)

    # Keep only rows with probability
    df = df.dropna(subset=["p_yes"])
    df["p_yes"] = df["p_yes"].astype(float)

    # Use usdVolume as liquidity proxy
    df["usdVolume"] = pd.to_numeric(df["usdVolume"], errors="coerce")
    df = df.dropna(subset=["usdVolume"])

    print("\nMarkets usable:", len(df))

    if len(df) < 6:
        print("Not enough markets to segment.")
        return

    # Create tercile liquidity bins
    df["liq_bin"] = pd.qcut(df["usdVolume"], q=3, labels=["Low", "Medium", "High"])

    print("\nCounts by liquidity bin:")
    print(df["liq_bin"].value_counts())

    print("\nProbability summary by liquidity bin:")
    print(df.groupby("liq_bin")["p_yes"].describe())

    print("\nShare near 0.5 (0.45–0.55) by liquidity:")
    for name, group in df.groupby("liq_bin"):
        share = ((group["p_yes"] >= 0.45) & (group["p_yes"] <= 0.55)).mean()
        print(f"{name}: {round(share, 3)}")

    print("\nStd deviation of probabilities by liquidity:")
    for name, group in df.groupby("liq_bin"):
        print(f"{name}: {round(group['p_yes'].std(), 3)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
