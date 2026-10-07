"""Check one student:  python src/predict.py 85 60 55 48"""
import sys
import joblib
import pandas as pd
from train import build_features

att, m1, m2, m3 = map(float, sys.argv[1:5])
row = pd.DataFrame([{"attendance_pct": att, "mark_1": m1, "mark_2": m2, "mark_3": m3}])
p = joblib.load("model.joblib").predict_proba(build_features(row))[0][1]
print(f"Risk of failing: {p:.0%} -> {'check in with this student' if p > 0.5 else 'on track'}")
