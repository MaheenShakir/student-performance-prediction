import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

CSV_PATH = "student_performance.csv"
PASS_MARK = 50
SEED = 42


def make_synthetic_data(n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "gender": rng.choice(["Male", "Female"], n),
        "study_hours_per_day": np.clip(rng.normal(3.5, 1.5, n), 0, 10).round(1),
        "attendance_pct": np.clip(rng.normal(80, 12, n), 30, 100).round(1),
        "previous_score": np.clip(rng.normal(65, 15, n), 10, 100).round(1),
        "sleep_hours": np.clip(rng.normal(7, 1.2, n), 3, 10).round(1),
        "parental_education": rng.choice(["High School", "Bachelors", "Masters", "PhD"], n, p=[0.4, 0.35, 0.18, 0.07]),
        "internet_access": rng.choice(["Yes", "No"], n, p=[0.85, 0.15]),
        "extracurricular": rng.choice(["Yes", "No"], n, p=[0.45, 0.55]),
    })
    edu = df["parental_education"].map({"High School": 0, "Bachelors": 2, "Masters": 3, "PhD": 4})
    score = (0.45 * df["previous_score"] + 4.0 * df["study_hours_per_day"] + 0.25 * df["attendance_pct"]
             + 1.0 * (df["sleep_hours"] - 6).clip(-3, 2) + edu
             + np.where(df["internet_access"] == "Yes", 1.5, 0) + rng.normal(0, 4, n))
    df["final_score"] = np.clip(score, 0, 100).round(1)
    return df


@st.cache_resource
def train_models():
    df = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else make_synthetic_data()
    df["result"] = (df["final_score"] >= PASS_MARK).astype(int)
    X = df.drop(columns=["final_score", "result"])
    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat),
    ])
    Xtr, Xte, ytr, yte, ctr, cte = train_test_split(
        X, df["final_score"], df["result"], test_size=0.2, random_state=SEED, stratify=df["result"])
    reg = Pipeline([("prep", prep), ("m", RandomForestRegressor(n_estimators=200, random_state=SEED))]).fit(Xtr, ytr)
    clf = Pipeline([("prep", prep), ("m", RandomForestClassifier(n_estimators=200, random_state=SEED))]).fit(Xtr, ctr)
    metrics = {"r2": r2_score(yte, reg.predict(Xte)), "acc": accuracy_score(cte, clf.predict(Xte))}
    return reg, clf, metrics


reg, clf, metrics = train_models()

st.title("🎓 Student Performance Predictor")
st.write("Enter a student's details to predict their **final score** and **chance of passing**.")

c1, c2 = st.columns(2)
with c1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    study = st.slider("Study hours per day", 0.0, 10.0, 3.5, 0.5)
    attendance = st.slider("Attendance (%)", 30, 100, 80)
    prev = st.slider("Previous score", 10, 100, 65)
with c2:
    sleep = st.slider("Sleep hours", 3.0, 10.0, 7.0, 0.5)
    edu = st.selectbox("Parental education", ["High School", "Bachelors", "Masters", "PhD"])
    internet = st.selectbox("Internet access", ["Yes", "No"])
    extra = st.selectbox("Extracurricular activities", ["Yes", "No"])

if st.button("Predict", type="primary", use_container_width=True):
    row = pd.DataFrame([{
        "gender": gender, "study_hours_per_day": study, "attendance_pct": attendance,
        "previous_score": prev, "sleep_hours": sleep, "parental_education": edu,
        "internet_access": internet, "extracurricular": extra,
    }])
    score = float(np.clip(reg.predict(row)[0], 0, 100))
    p_pass = float(clf.predict_proba(row)[0][1])
    m1, m2 = st.columns(2)
    m1.metric("Predicted final score", f"{score:.1f} / 100")
    m2.metric("Chance of passing", f"{p_pass * 100:.0f}%")
    st.progress(min(max(p_pass, 0.0), 1.0))
    if p_pass >= 0.8:
        st.success("On track to pass. Keep it up!")
    elif p_pass >= 0.5:
        st.warning("Borderline. More study time or better attendance could help.")
    else:
        st.error("At risk of failing. Extra support is recommended.")

st.divider()
st.caption(f"Model: Random Forest · R² = {metrics['r2']:.2f} · Pass/Fail accuracy = {metrics['acc']:.0%} "
           f"(pass mark {PASS_MARK}). Trained on a synthetic dataset for demonstration.")
