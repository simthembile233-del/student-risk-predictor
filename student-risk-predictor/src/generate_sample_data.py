"""Creates data/students.csv with synthetic data so the project runs out of the box.
Replace it with your real file using the same columns."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 600
base = rng.normal(60, 15, n).clip(15, 95)
trend = rng.normal(0, 5, n)
df = pd.DataFrame({
    "student_id": range(1, n + 1),
    "attendance_pct": (rng.normal(82, 14, n) + (base - 60) * 0.2).clip(30, 100).round(1),
    "mark_1": (base - trend + rng.normal(0, 4, n)).clip(0, 100).round(1),
    "mark_2": (base + rng.normal(0, 4, n)).clip(0, 100).round(1),
    "mark_3": (base + trend + rng.normal(0, 4, n)).clip(0, 100).round(1),
})
df["final_mark"] = (df["mark_3"] + trend * 0.8
                    + (df["attendance_pct"] - 80) * 0.25
                    + rng.normal(0, 5, n)).clip(0, 100).round(1)
df.to_csv("data/students.csv", index=False)
print("Saved data/students.csv with", n, "students")
