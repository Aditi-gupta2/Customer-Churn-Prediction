# DataNova Insights Report: Customer Churn Prediction & Retention Strategy
**Problem Statement:** DN-BU-01 (Business Track)  
**Project:** ChurnGuard-AI  

---

## 1. Project Overview
Getting a new customer costs 5 to 7 times more than keeping an existing one. In this project, we looked at records from **7,043 customers** to solve a real business challenge:
1. Predict which customers are planning to leave before they actually cancel.
2. Find out why they want to leave.
3. Give the business a clear, realistic plan to help them stay.

In our dataset, **26.5% of customers left** (1,869 customers). That represents more than **$1.4 million in lost yearly revenue**. Using machine learning and data analysis, we built a system that flags high-risk customers and suggests specific offers to save the account.

---

## 2. What We Discovered in the Data (Why Customers Leave)

When we analyzed the customer habits, four clear patterns stood out:

| Factor | What the Data Shows | Churn Rate | What This Means for the Business |
| :--- | :--- | :--- | :--- |
| **Contract Length** | Month-to-month contracts | **42.7%** | Customers on monthly plans leave **15 times more often** than customers on 2-year plans (2.8%). Without a long-term commitment, it is too easy for people to cancel. |
| **Payment Method** | Manual electronic checks | **45.3%** | People paying with electronic checks churn three times as much as people on automatic card or bank payments (~15%). Manual payments create friction and remind users to cancel every month. |
| **Tech Support** | No tech support or security add-on | **41.6%** | When customers don't have tech support, small technical issues turn into cancellations. With tech support, churn drops to **15.2%**. |
| **Plan Cost** | Fiber optic with high monthly bills (>$80) | **41.9%** | Fiber optic is faster but costs more. Customers on these plans need extra attention to make sure they feel they are getting their money's worth. |

---

## 3. How the Machine Learning Models Performed

We trained and compared three different models so we could see which one works best:

| Model | Accuracy | Precision | Recall (Churners Caught) | ROC-AUC | Best Used For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting (Best Overall)** | **80.3%** | **66.2%** | 52.9% | **0.8440** | High precision when we want to make sure we don't waste discounts on customers who were going to stay anyway. |
| **Random Forest** | 75.8% | 53.1% | 76.2% | 0.8411 | A solid, balanced model that handles complex patterns well. |
| **Logistic Regression (High Recall)** | 74.2% | 50.9% | **78.3%** | 0.8420 | Great when we want to catch as many possible churners as possible (almost 80% of all leavers). |

**Top Factors Driving Churn in the Model:**
1. How long the person has been a customer (`tenure`)
2. Having a month-to-month plan (`Contract_Month-to-month`)
3. Monthly and total charges (`MonthlyCharges`, `TotalCharges`)
4. Having fiber optic internet (`InternetService_Fiber optic`)
5. Paying by electronic check (`PaymentMethod_Electronic check`)

---

## 4. Our Practical Retention Plan

Rather than just predicting churn, our project tells teams what actions to take based on the customer's risk level:

| Customer Group | Risk Level | Common Profile | Recommended Action | Cost to Company | Expected Benefit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **High Risk** | > 65% chance of leaving | Month-to-month plan, new customer (< 6 months), electronic check | • Direct check-in call from support<br>• 15% discount if they move to a 1-year contract<br>• 90 days of free tech support | $35–$45 | Retains 30% to 40% of this group |
| **Medium Risk** | 35% – 65% chance of leaving | Mid-tenure, fiber optic, no add-on services | • $10 credit to switch to automatic payments<br>• Free check on internet speed & customized package | $15–$25 | Retains 20% to 25% of this group |
| **Low Risk** | < 35% chance of leaving | Long-term customer (2+ years), auto-pay, multiple services | • Enrollment in customer loyalty rewards<br>• Early access to new features | ~$5 | Keeps loyal customers happy long-term |

---

## 5. Potential Savings and Financial Return

Let's look at an example company with **10,000 customers**:
- **Expected churners without action:** 2,650 customers (at a 26.5% churn rate)
- **Average yearly value per customer:** $780 ($65/month × 12)
- **Cost of our targeted offers:** $25 per at-risk customer = **$66,250**
- **Realistic retention success:** 25% of at-risk customers stay (662 customers saved)
- **Yearly revenue saved:** 662 × $780 = **$516,360**
- **Net profit saved after campaign costs:** $516,360 - $66,250 = **+$450,110**
- **Estimated Return on Investment (ROI):** **~580%**

---

## 6. Summary
ChurnGuard-AI gives companies a complete, working solution:
1. It predicts who might leave with **84.4% ROC-AUC**.
2. It clearly explains the root causes (contract terms, payment methods, support).
3. It provides an easy-to-use **Streamlit dashboard** for customer teams.
4. It includes a built-in business calculator showing how small offers turn into large revenue savings.
