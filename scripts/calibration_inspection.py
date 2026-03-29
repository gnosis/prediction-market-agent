import joblib
import numpy as np

calibrator = joblib.load("market_calibrator.joblib")

grid = np.linspace(0.05, 0.95, 19)
calibrated = calibrator.predict(grid)

for raw, cal in zip(grid, calibrated):
    print(f"raw={raw:.2f} -> calibrated={cal:.3f}")
