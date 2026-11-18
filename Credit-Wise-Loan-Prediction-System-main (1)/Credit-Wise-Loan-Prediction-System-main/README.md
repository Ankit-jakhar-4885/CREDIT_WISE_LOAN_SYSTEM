# CreditWise — Loan Approval Prediction System

## 📌 Overview
**CreditWise** is a Machine Learning–based credit evaluation system that predicts whether a loan application should be **Approved (1)** or **Rejected (0)**. The system analyzes historical applicant features and builds a binary classifier using supervised ML models including **KNN, Logistic Regression, and Naive Bayes**.

---

## 🌐 Live Deployment
🚀 **Try the deployed application here:**  
🔗 [https://creditwiseloan.streamlit.app/](https://credit-wise-loan-prediction-system-tt6dcttafuygafd9yvhrkk.streamlit.app/)

---


## 🏦 Use Case
Traditional loan processing involves manual review of credit and financial documents. This process is:

- Slow  
- Human-biased  
- Inconsistent  
- Risk-prone  

CreditWise helps financial institutions automate preliminary decisioning to improve turnaround time and reduce default risk.

---

## 🎯 Project Objectives
- Predict loan approval outcome (Binary Classification)
- Improve the accuracy of credit decisioning
- Reduce manual workload for loan officers
- Support risk mitigation using applicant data

---

## 📊 Dataset Information
The dataset contains **1000 loan applications** and the following feature distribution:

| Type | Count |
|---|---|
| Numerical Features | 12 |
| Categorical Features | 7 |
| Target Variable | 1 |

**Target Label:**

1 → Approved
0 → Rejected

---

## 🛠 ML Pipeline & Methodology

### 1. Data Preprocessing
- Filled missing numerical values using **mean imputation**
- Filled missing categorical values using **mode imputation**
- Removed non-essential features like Applicant_ID

### 2. Exploratory Data Analysis (EDA)
Conducted visual analysis using Matplotlib & Seaborn:
- Class distribution (pie charts)
- Category frequencies (bar charts)
- Income distributions (histograms)
- Outlier inspection (box plots)
- Correlation heatmap

### 3. Feature Engineering
Enhanced predictive signal by transforming features:

python
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2
df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])

### 4. Encoding

Applied encoding strategies based on feature semantics:

* **LabelEncoder** → ordinal data (Education level)
* **OneHotEncoder** → nominal categorical data

### 5. Train-Test Split

80% → Training
20% → Testing

### 6. Feature Scaling

Standardized numerical features using:

StandardScaler()

### 7. Model Training

Trained the following models:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Gaussian Naive Bayes

---

## 📈 Model Evaluation

Models were evaluated using:

* Precision
* Recall
* F1 Score
* Accuracy
* Confusion Matrix

> For credit decisioning, **Precision** is important since false approvals lead to default risk.

---

## 🏆 Model Performance Summary

| Model               | Precision   | Recall   | F1       | Suitable For                |
| ------------------- | ----------- | -------- | -------- | --------------------------- |
| Logistic Regression | Moderate    | High     | Balanced | Baseline decisioning        |
| KNN                 | Moderate    | Moderate | Moderate | Similar applicant profiling |
| **Naive Bayes**     | **Highest** | Moderate | Good     | **Risk-sensitive approval** |

> **Best Performing Model → Naive Bayes (Based on Precision)**

---

## 📦 Tech Stack

**Language:** Python

**Libraries:**
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn

---

## 🚀 Future Enhancements

Planned improvements:

* Deployment as Streamlit/Flask web app
* Integration with real-time credit scoring APIs
* Explainable AI (SHAP/LIME)
* ROC-AUC benchmarking across classifiers
* Model monitoring & data drift detection
* Feature selection optimization

---

## 📄 Conclusion

CreditWise demonstrates the application of Machine Learning in modern credit underwriting by making loan approval:

✔ Faster
✔ Fairer
✔ More consistent
✔ Less risky

The system offers significant benefits for both **loan institutions** and **borrowers**, improving decision quality and customer satisfaction.

---
