# 🎓 Student Performance Analysis & Prediction

A data science mini project that analyzes what affects student performance and predicts
a student's **final score** and **pass/fail** outcome using machine learning.

## Features
- Exploratory data analysis (distributions, correlations, group comparisons)
- Regression (Linear Regression, Random Forest) to predict the final score
- Classification (Logistic Regression, Random Forest) to flag students at risk of failing
- Feature importance to see what drives results
- Interactive Streamlit web app for live predictions

## Run the notebook
Open `Student_Performance_Analysis.ipynb` in Google Colab and click **Runtime → Run all**.

## Run the web app
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Streamlit

## Note
The dataset is synthetic (generated in code) to demonstrate the workflow.
To use real data, replace `student_performance.csv` with your own file using the same columns.
