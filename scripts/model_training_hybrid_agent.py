import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss


# def calibrate_prob(p: float) -> float:
#     if p >= 0.85:
#         return p - 0.08
#     elif p >= 0.70:
#         return p - 0.05
#     elif p >= 0.55:
#         return p - 0.03
#     else:
#         return p


# -------------------------
# 1. Load and sort data
# -------------------------
df = pd.read_csv("omen_resolved_snapshots.csv")

df["category"] = df["category"].fillna("unknown")
df = df.dropna(subset=["prob_at_close", "resolution"])
df = df[(df["prob_at_close"] >= 0) & (df["prob_at_close"] <= 1)]

# Load learned calibrator
calibrator = joblib.load("market_calibrator.joblib")

# Create learned calibrated probability feature
df["calibrated_prob"] = calibrator.predict(df["prob_at_close"].values)

# Sort by time so train/test mimics real forecasting
df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
df = df.sort_values("close_time").reset_index(drop=True)


# -------------------------
# 2. Features and target
# -------------------------
feature_cols = [
    "prob_at_close",
    "calibrated_prob",
    "volume",
    "duration_hours",
    "category",
]

target_col = "resolution"

X = df[feature_cols]
y = df[target_col].astype(int)

# -------------------------
# 3. Time-based split
# -------------------------
split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# -------------------------
# 4. Preprocessing
# -------------------------
numeric_features = [
    "prob_at_close",
    "calibrated_prob",
    "volume",
    "duration_hours",
]

categorical_features = ["category"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# -------------------------
# 5. Model
# -------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000)),
    ]
)

# -------------------------
# 6. Train
# -------------------------
model.fit(X_train, y_train)

# -------------------------
# 7. Evaluate
# -------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Rows used:", len(df))
print("Train rows:", len(X_train))
print("Test rows:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 4))
print("Log Loss:", round(log_loss(y_test, y_prob), 4))
print("Brier Score:", round(brier_score_loss(y_test, y_prob), 4))

# -------------------------
# 8. Save model
# -------------------------
joblib.dump(model, "ml_model.joblib")
print("Saved model to ml_model.joblib")
