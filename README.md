# 🛡️ ChurnGuard-AI
### Customer Churn Prediction & Retention Strategy
**DataNova Hackathon · Problem Statement: DN-BU-01 (Business Track)**

---

## 📌 Project Overview
ChurnGuard-AI helps subscription and telecom businesses solve a major problem: **customers leaving unexpectedly**. 

Instead of waiting for customers to cancel, this project uses machine learning to spot at-risk customers early, understands why they are unhappy, and recommends simple, cost-effective offers to help retain them.

### 🌟 Key Highlights
- **Smart Machine Learning:** Built and compared 3 models (Gradient Boosting with 80.3% accuracy, plus a high-recall Logistic Regression that catches 78.3% of churners).
- **Clear Insights:** Discovered why customers leave (Month-to-month contracts have a 15x higher churn rate, and manual payment methods cause 3x more churn).
- **Action Playbook:** Recommends specific, practical offers (like annual upgrade discounts or free tech support) based on each customer's situation.
- **Interactive Dashboard:** A clean Streamlit web app with 6 modules including live customer testing and an ROI savings calculator.

---

## 📁 Repository Structure
```
ChurnGuard-AI/
├── app.py                         # Streamlit Web Dashboard
├── requirements.txt               # Required Python packages
├── data/
│   └── customer_churn.csv         # Dataset (7,043 customer records)
├── models/
│   ├── churn_model.pkl            # Trained Gradient Boosting Model
│   ├── logistic_regression.pkl    # Trained Logistic Regression Model
│   ├── random_forest.pkl          # Trained Random Forest Model
│   ├── scaler.pkl                 # Feature Scaler
│   ├── model_metadata.json        # Accuracy scores and feature weights
│   └── scored_test_customers.csv  # Test customers labeled by risk level
├── src/
│   ├── data_cleaning.py           # Prepares and cleans the raw data
│   ├── train_model.py             # Trains and tests the machine learning models
│   └── playbook.py                # Action playbook and customer grouping logic
└── reports/
    ├── insights_report.md         # Detailed findings and business recommendations
    └── presentation_slides.md     # Presentation slides and speaking notes
```

---

## ⚡ How to Run the Project

### 1. Set Up the Environment
```bash
# Clone the repository
git clone https://github.com/your-username/ChurnGuard-AI.git
cd ChurnGuard-AI

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the App
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501` to view the dashboard!

---

## 📊 Dashboard Modules
1. **Overview & Key Stats:** High-level summary of customer count, overall churn rate, and revenue at risk.
2. **Customer Risk Checker:** Test any customer details live to see their churn chance and recommended next steps.
3. **Customer Groups & Insights:** Explore customer groups by risk level and download high-risk lists as CSV.
4. **Action Guide & Offers:** Four clear, practical retention campaigns ready for teams to roll out.
5. **Model Accuracy & Drivers:** See how our models performed and which factors influence customer churn most.
6. **Cost & Savings Calculator:** Test different scenarios to see how much money the business can save.

---

## 👥 Authors
- **Competition:** DataNova Datathon
- **Track:** Business Track (DN-BU-01)
