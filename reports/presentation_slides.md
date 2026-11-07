# DataNova Pitch Presentation: ChurnGuard-AI
## Problem Statement: DN-BU-01 · Customer Churn Prediction and Retention Strategy

---

### Slide 1: Welcome & Introduction
- **Project Title:** ChurnGuard-AI: Catching Churn Early & Keeping Customers Happy
- **Track:** DataNova Business Track (DN-BU-01)
- **Opening Question:** "Every month, thousands of customers quietly cancel their subscriptions. What if your team knew who was thinking of leaving weeks ahead of time, and knew the exact offer to make them stay?"

---

### Slide 2: The Problem We Are Solving
- **The Numbers:**
  - 7,043 customers analyzed in our dataset.
  - Overall churn rate: **26.5%** (~1,869 customers left).
  - Over **$1.4 Million** in yearly subscription revenue lost.
- **Why Traditional Retention Fails:**
  - Most companies wait until a user clicks "Cancel" before trying to save them.
  - Sending discounts to everyone wastes money on loyal customers who never planned to leave.
- **What We Built:** An early-warning prediction system combined with targeted, cost-effective customer offers.

---

### Slide 3: What We Learned from the Data (Why People Leave)
- **1. Contract Type:** Month-to-month customers leave at a rate of **42.7%**, compared to only **2.8%** for customers on 2-year contracts.
- **2. Payment Method:** Paying with manual electronic checks has a **45.3% churn rate**, while auto-pay customers leave at only ~15%.
- **3. Tech Support:** Customers without tech support leave at **41.6%** vs. only **15.2%** for those with tech support.
- **4. New Sign-ups:** Over half of all cancellations happen in a customer's first 6 months.

---

### Slide 4: Our Machine Learning Models
- We compared three algorithms on the same test data:
  - **Gradient Boosting (Best Overall):** **80.3% Accuracy**, **0.8440 ROC-AUC**. Best when we want to make sure every offer goes to the right person.
  - **Logistic Regression (High Recall):** Catches **78.3% of all churners**. Best for aggressive campaigns where we don't want to miss anyone.
  - **Random Forest:** Balanced performance with a 75.8% accuracy.
- **Key Factors:** Customer tenure, month-to-month contracts, and payment method have the biggest weight in the predictions.

---

### Slide 5: Practical Action Plan (How to Keep Customers)
- **Plan 1: Move Customers to Annual Plans**
  - Offer month-to-month users a 15% discount or 1 free month if they upgrade to an annual plan.
- **Plan 2: Free Tech Support Bundle**
  - Offer 3 months of free tech support and security help to high-bill fiber optic customers.
- **Plan 3: Promote Automatic Payments**
  - Give a $10 bill credit to switch from manual checks to automatic card or bank payment.
- **Plan 4: First 90 Days Welcome Check-in**
  - Proactive check-in call on Day 14 and Day 45 to solve early setup headaches.

---

### Slide 6: Live Dashboard Demo
- **Module 1:** High-level metrics showing churn rate and revenue at risk.
- **Module 2:** Customer Risk Checker where you can test different details and get live recommendations.
- **Module 3:** Customer Groups & downloadable high-risk list for support teams.
- **Module 4:** Step-by-step action guide for campaigns.
- **Module 5:** Model accuracy comparisons and feature charts.
- **Module 6:** Cost and savings calculator showing expected return on investment.

---

### Slide 7: Expected Business Return
- **Example Scenario with 10,000 Customers:**
  - Customers at risk: 2,650
  - Total cost for offers: $66,250 ($25 average offer)
  - If we save just 25% of at-risk customers (662 people):
  - **Yearly revenue saved:** **$516,360**
  - **Net savings after costs:** **+$450,110**
  - **Expected Return (ROI):** **~580%**

---

### Slide 8: Conclusion
- ChurnGuard-AI is more than a predictive model:
  - It spots who is leaving.
  - It explains why.
  - It gives the business a clear, realistic playbook to save revenue.
- **Thank you! We'd love to take your questions.**
