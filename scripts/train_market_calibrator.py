import joblib
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss


# -------------------------
# 1. Load data
# -------------------------
df = pd.read_csv("omen_resolved_snapshots.csv")

df = df.dropna(subset=["prob_at_close", "resolution"])
df = df[(df["prob_at_close"] >= 0) & (df["prob_at_close"] <= 1)]

# Sort by time so split mimics real forecasting
df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
df = df.sort_values("close_time").reset_index(drop=True)

# -------------------------
# 2. Train/test split
# -------------------------
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

X_train = train_df["prob_at_close"].values
y_train = train_df["resolution"].astype(int).values

X_test = test_df["prob_at_close"].values
y_test = test_df["resolution"].astype(int).values

# -------------------------
# 3. Fit isotonic regression
# -------------------------
calibrator = IsotonicRegression(
    y_min=0.0,
    y_max=1.0,
    out_of_bounds="clip",
)

calibrator.fit(X_train, y_train)

# -------------------------
# 4. Evaluate
# -------------------------
raw_test_probs = X_test
cal_test_probs = calibrator.predict(X_test)

print("Raw market Brier:", round(brier_score_loss(y_test, raw_test_probs), 4))
print("Calibrated Brier:", round(brier_score_loss(y_test, cal_test_probs), 4))

print("Raw market LogLoss:", round(log_loss(y_test, raw_test_probs, labels=[0, 1]), 4))
print("Calibrated LogLoss:", round(log_loss(y_test, cal_test_probs, labels=[0, 1]), 4))

# -------------------------
# 5. Save calibrator
# -------------------------
joblib.dump(calibrator, "market_calibrator.joblib")
print("Saved market_calibrator.joblib")
