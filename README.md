# 🏥 Hospital Readmission Analysis & Prediction

## 📌 Project Overview

Hospital readmission is a major challenge in healthcare because repeated admissions increase treatment costs and can indicate gaps in patient care.

This project analyzes patient hospital records to:

* Identify factors influencing readmission
* Understand patient risk patterns
* Build a machine learning model to predict high-risk patients
* Create a dashboard for business insights

---

## 🎯 Business Problem

Hospitals need to identify patients who are likely to be readmitted within 30 days so they can:

* Improve patient care
* Reduce avoidable admissions
* Lower operational costs
* Improve hospital performance metrics

---

## 📂 Dataset

**Dataset Used:** Diabetes Hospital Readmission Dataset

The dataset includes:

* Patient demographics
* Admission details
* Medication counts
* Diagnoses
* Laboratory procedures
* Previous hospital visits

Target variable:

* `readmitted_flag`

  * `1` → readmitted within 30 days
  * `0` → not readmitted within 30 days

---

## 🛠 Tools & Technologies

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**
* **Tableau**
* **GitHub**

---

## 🔍 Exploratory Data Analysis

The analysis revealed several important insights:

### Key Findings

✅ Longer hospital stays increase readmission risk

✅ Patients taking more medications have higher risk

✅ Certain diagnoses are linked with repeated admissions

✅ Age is a strong factor in readmission probability

---

## 🤖 Machine Learning Models

Two models were used:

### 1. Logistic Regression

Used as a baseline model for binary classification.

### 2. Random Forest Classifier

Used to improve prediction performance and identify feature importance.

---

## 📊 Model Evaluation

Evaluation metrics used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC AUC

### Important Note

Because healthcare datasets are imbalanced, **Recall** was prioritized to better identify high-risk patients.

---

## 📈 Most Important Predictors

The strongest predictors of readmission were:

* Time in hospital
* Number of medications
* Previous inpatient visits
* Emergency visits

---

## 📊 Dashboard

A Tableau dashboard was created to visualize:

* Readmission rate
* Age distribution
* Medication impact
* Diagnosis patterns
* Patient risk trends

---

## 📁 Project Structure

```bash
hospital-readmission-analysis/
│
├── data/
│   └── hospital_dashboard_data.csv
│
├── notebooks/
│   └── hospital_readmission_analysis.ipynb
│
├── dashboard/
│   └── Hospital_Readmission_Dashboard
│
├── images/
│   └── dashboard_screenshot.png
│
└── README.md
```

---

## 🚀 Business Impact

This solution can help healthcare providers:

* Identify high-risk patients early
* Improve discharge planning
* Reduce readmission costs
* Improve patient outcomes

---




## 📌 Future Improvements

Possible enhancements:

* Hyperparameter tuning
* XGBoost model
* SHAP explainability
* Deployment as web app

---

## 👤 Author

**Jotsna sonar**
Aspiring Data Analyst | Healthcare Analytics Enthusiast

---

## ⭐ If you found this useful

Please consider starring this repository.

