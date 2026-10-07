# Student At-Risk Predictor

A simple, explainable machine learning project that flags students at risk of
failing, using only **attendance** and **previous marks**.

## Why
Teachers can't watch every student closely. A declining mark trend combined
with falling attendance is an early warning sign, and catching it early lets a
teacher step in before the final result.

## How it works
- Features: average mark, latest mark, mark trend (slope), attendance %, and a
  "low attendance + declining marks" flag
- Model: logistic regression (easy to explain, works on small datasets)
- Output: a risk percentage per student

## Quick start
```bash
pip install -r requirements.txt
python src/generate_sample_data.py   # skip if you have real data
python src/train.py
python src/predict.py 85 60 55 48    # attendance, mark_1, mark_2, mark_3
```

## Using your own data
Put a `data/students.csv` with these columns:
`attendance_pct, mark_1, mark_2, mark_3, final_mark`
(`mark_1` is the oldest exam, `mark_3` the latest). Change `PASS_MARK` in
`src/train.py` if your pass mark differs.

## Notes on responsible use
Predictions are a prompt for a teacher to check in, not a verdict on a
student. The model can't see the reasons behind struggles (health, home
situation, teaching quality). The included data is synthetic.

## Ideas for next steps
- Streamlit app for teachers
- Add more exams or subject-level marks
- Compare with a decision tree or random forest
- Check performance fairness across groups
