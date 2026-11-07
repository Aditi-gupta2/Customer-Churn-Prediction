"""
Retention Playbook & Action Recommendation Engine for ChurnGuard-AI.
Maps customer profiles and predicted churn risk into practical, revenue-saving business actions.
"""
import pandas as pd
import numpy as np

def generate_customer_retention_playbook(customer_data, churn_prob):
    """
    Given a customer's details and predicted churn chance,
    returns the main reasons for risk and clear steps the team can take.
    """
    risk_tier = "High Risk" if churn_prob >= 0.65 else ("Medium Risk" if churn_prob >= 0.35 else "Low Risk")
    
    triggers = []
    actions = []
    estimated_cost = 0
    
    # 1. Contract Analysis
    contract = customer_data.get('Contract', '')
    if contract == 'Month-to-month':
        triggers.append("Month-to-month plan (Customers leave much faster than on 1 or 2-year plans)")
        actions.append({
            "category": "Contract Offer",
            "action": "Offer an annual plan with a 15% discount or 1 free month to lock in commitment.",
            "roi_priority": "Immediate"
        })
        estimated_cost += 30
    
    # 2. Payment Method Analysis
    payment = customer_data.get('PaymentMethod', '')
    if payment == 'Electronic check':
        triggers.append("Paying via manual electronic check (Historical churn is high at ~45%)")
        actions.append({
            "category": "Payment Setup",
            "action": "Give a $10 bill credit if the customer switches to automatic card or bank payment.",
            "roi_priority": "Immediate"
        })
        estimated_cost += 10
        
    # 3. Tech Support / Security
    tech_support = customer_data.get('TechSupport', '')
    online_sec = customer_data.get('OnlineSecurity', '')
    if tech_support == 'No' or online_sec == 'No':
        triggers.append("No Tech Support or Online Security attached to the account")
        actions.append({
            "category": "Service Add-on",
            "action": "Give 3 months of free priority tech support and security protection.",
            "roi_priority": "Medium"
        })
        estimated_cost += 15
        
    # 4. Fiber Optic Bills
    internet = customer_data.get('InternetService', '')
    monthly = float(customer_data.get('MonthlyCharges', 0))
    if internet == 'Fiber optic' and monthly > 80:
        triggers.append(f"High monthly bill on Fiber Optic (${monthly:.2f}/mo)")
        actions.append({
            "category": "Plan Review",
            "action": "Reach out to check service quality and offer a customized package to fit their budget.",
            "roi_priority": "High"
        })
        
    # 5. Tenure / New Customers
    tenure = int(customer_data.get('tenure', 0))
    if tenure <= 6:
        triggers.append(f"New customer (Only with us for {tenure} months)")
        actions.append({
            "category": "Welcome Call",
            "action": "Have customer support make a quick check-in call to solve any early setup problems.",
            "roi_priority": "Immediate"
        })
        estimated_cost += 20
    elif tenure >= 48:
        actions.append({
            "category": "Loyalty Perk",
            "action": "Send a thank-you note and loyalty reward for being a customer for 4+ years.",
            "roi_priority": "Long-term"
        })
        
    if not actions:
        actions.append({
            "category": "Customer Care",
            "action": "Send regular quarterly updates and keep service quality steady.",
            "roi_priority": "Low"
        })
        
    monthly_rev = float(customer_data.get('MonthlyCharges', 65.0))
    annual_revenue_at_risk = monthly_rev * 12 * churn_prob
    
    return {
        'risk_tier': risk_tier,
        'churn_probability': round(churn_prob * 100, 1),
        'annual_revenue_at_risk': round(annual_revenue_at_risk, 2),
        'key_risk_triggers': triggers if triggers else ["Stable customer with no major warning signs"],
        'recommended_actions': actions,
        'estimated_intervention_cost': estimated_cost,
        'net_revenue_protection': round(annual_revenue_at_risk - estimated_cost, 2)
    }

def calculate_cohort_summary(df):
    """
    Groups customers by risk tier to see average bill, tenure, and churn rates.
    """
    summary = df.groupby('risk_tier').agg(
        customer_count=('customerID', 'count'),
        avg_monthly_charges=('MonthlyCharges', 'mean'),
        avg_tenure=('tenure', 'mean'),
        actual_churn_rate=('actual_churn', 'mean'),
        total_monthly_revenue=('MonthlyCharges', 'sum')
    ).reset_index()
    
    summary['at_risk_annual_runrate'] = summary['total_monthly_revenue'] * 12
    return summary
