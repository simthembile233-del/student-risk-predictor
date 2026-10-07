"""Train an explainable at-risk model from attendance and previous marks.

Expected columns in data/students.csv:
  attendance_pct, mark_1, mark_2, mark_3 (oldest -> latest), final_mark
A student is "at risk" if final_mark < PASS_MARK.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PASS_MARK = 50
MARK_COLS = ["mark_1", "mark_2", "mark_3"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    marks = df[MARK_COLS].to_numpy()
    x = np.arange(len(MARK_COLS))
    slope = np.polyfit(x, marks.T, 1)[0]          # mark trend per exam
    out = pd.DataFrame({
        "avg_mark": marks.mean(axis=1),
        "latest_mark": marks[:, -1],
        "mark_trend": slope,
        "attendance_pct": df["attendance_pct"].to_numpy(),
    })
    out["low_att_x_decline"] = ((out["attendance_pct"] < 75) & (out["mark_trend"] < 0)).astype(int)
    return out


def main():
    df = pd.read_csv("data/students.csv")
    X = build_features(df)
    y = (df["final_mark"] < PASS_MARK).astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=1000))
    model.fit(X_tr, y_tr)

    print(classification_report(y_te, model.predict(X_te), target_names=["OK", "At risk"]))
    print("Feature effects (positive = raises risk):")
    coefs = model[-1].coef_[0]
    for name, c in sorted(zip(X.columns, coefs), key=lambda t: -abs(t[1])):
        print(f"  {name:20s} {c:+.2f}")

    joblib.dump(model, "model.joblib")


if __name__ == "__main__":
    main()
