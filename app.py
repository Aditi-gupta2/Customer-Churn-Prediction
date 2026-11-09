"""
ChurnGuard-AI: Customer Churn Prediction & Retention Dashboard
DataNova DN-BU-01 · Business Track
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import sys

# Add src to path
sys.path.append(os.path.abspath('src'))
from playbook import generate_customer_retention_playbook, calculate_cohort_summary
from data_cleaning import feature_engineering

st.set_page_config(
    page_title="ChurnGuard-AI | Customer Retention Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for a clean, professional look
st.markdown("""
<style>
    .high-risk-badge {
        background-color: #e74c3c;
        color: white;
        padding: 4px 10px;
        border-radius: 5px;
        font-weight: 600;
    }
    .med-risk-badge {
        background-color: #f39c12;
        color: white;
        padding: 4px 10px;
        border-radius: 5px;
        font-weight: 600;
    }
    .low-risk-badge {
        background-color: #27ae60;
        color: white;
        padding: 4px 10px;
        border-radius: 5px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_metadata():
    try:
        model = joblib.load('models/churn_model.pkl')
        rf_model = joblib.load('models/random_forest.pkl')
        lr_model = joblib.load('models/logistic_regression.pkl')
        scaler = joblib.load('models/scaler.pkl')
        with open('models/model_metadata.json', 'r') as f:
            meta = json.load(f)
    except Exception as e:
        # Fallback: train freshly on Streamlit Cloud Python environment
        from train_model import train_and_evaluate_models
        meta = train_and_evaluate_models()
        model = joblib.load('models/churn_model.pkl')
        rf_model = joblib.load('models/random_forest.pkl')
        lr_model = joblib.load('models/logistic_regression.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
    return {
        'model': model,
        'rf_model': rf_model,
        'lr_model': lr_model,
        'scaler': scaler,
        'metadata': meta
    }

@st.cache_data
def load_datasets():
    df_raw = pd.read_csv('data/customer_churn.csv')
    df_scored = pd.read_csv('models/scored_test_customers.csv')
    return df_raw, df_scored

artifacts = load_models_and_metadata()
df_raw, df_scored = load_datasets()

# ----------------- SIDEBAR NAVIGATION -----------------
st.sidebar.title("🛡️ ChurnGuard-AI")
st.sidebar.caption("DataNova Project · Business Track")

page = st.sidebar.radio(
    "Go to",
    [
        "📊 1. Overview & Key Stats",
        "🔍 2. Customer Risk Checker",
        "👥 3. Customer Groups & Insights",
        "📋 4. Action Guide & Offers",
        "⚙️ 5. Model Accuracy & Drivers",
        "💰 6. Cost & Savings Calculator"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("**Quick Project Summary:**")
st.sidebar.write("• **Total customers:** 7,043")
st.sidebar.write("• **Current churn rate:** 26.5%")
st.sidebar.write("• **Main model:** Gradient Boosting (80.3% accuracy)")
st.sidebar.write("• **Goal:** Spot at-risk customers early and offer the right incentives to make them stay.")

# ----------------- PAGE 1: OVERVIEW -----------------
if page == "📊 1. Overview & Key Stats":
    st.title("📊 Customer Churn Overview")
    st.write("A clear summary of our customer base, how many people are leaving, and how much money is at risk.")
    
    total_customers = len(df_raw)
    raw_churn_yes = (df_raw['Churn'] == 'Yes').sum()
    overall_churn_rate = (raw_churn_yes / total_customers) * 100
    
    # Financial metrics
    at_risk_customers = len(df_scored[df_scored['risk_tier'] == 'High Risk'])
    high_risk_pct = (at_risk_customers / len(df_scored)) * 100
    annual_rev_at_risk = df_scored[df_scored['risk_tier'] == 'High Risk']['MonthlyCharges'].sum() * 12
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Customers", f"{total_customers:,}")
    with c2:
        st.metric("Overall Churn Rate", f"{overall_churn_rate:.1f}%")
    with c3:
        st.metric("High-Risk Accounts (Test Set)", f"{at_risk_customers} ({high_risk_pct:.1f}%)")
    with c4:
        st.metric("Yearly Revenue at Risk", f"${annual_rev_at_risk:,.0f}")
        
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Churn Rate by Contract Type")
        contract_churn = df_raw.groupby('Contract')['Churn'].value_counts(normalize=True).unstack() * 100
        fig_contract = px.bar(
            contract_churn, 
            barmode='group',
            labels={'value': 'Percentage (%)', 'Contract': 'Contract Type'},
            color_discrete_map={'No': '#27ae60', 'Yes': '#e74c3c'},
            title="Contract Type vs Churn"
        )
        st.plotly_chart(fig_contract, width="stretch")
        
    with col_right:
        st.subheader("Churn Rate by Payment Method")
        pay_churn = df_raw.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack() * 100
        fig_pay = px.bar(
            pay_churn,
            barmode='stack',
            labels={'value': 'Percentage (%)', 'PaymentMethod': 'Payment Method'},
            color_discrete_map={'No': '#2980b9', 'Yes': '#e67e22'},
            title="Payment Method vs Churn"
        )
        st.plotly_chart(fig_pay, width="stretch")
        
    st.markdown("---")
    st.subheader("Top Findings from the Data")
    st.write("""
    1. **Month-to-month contracts are the biggest risk:** Customers paying month-to-month leave at a rate of **42.7%**, compared to only **2.8%** for customers on 2-year contracts. Moving customers to longer contracts is the single best way to reduce churn.
    2. **Payment method matters:** Customers paying with **Electronic Check** leave at a rate of **45.3%**, while those with automatic bank transfers or cards leave at only ~15%.
    3. **Tech support helps retain users:** Customers who have tech support leave at only **15.2%**, compared to **41.6%** for those who don't have it.
    """)

# ----------------- PAGE 2: RISK CHECKER -----------------
elif page == "🔍 2. Customer Risk Checker":
    st.title("🔍 Check Customer Churn Risk")
    st.write("Enter a customer's details below to see how likely they are to leave, what is driving that risk, and what actions we should take to keep them.")
    
    tab_sim, tab_lookup = st.tabs(["⚡ Try Different Customer Details", "🔎 Look Up an Existing Customer"])
    
    with tab_sim:
        st.subheader("Customer Details")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            tenure = st.slider("How many months have they been with us?", min_value=0, max_value=72, value=4)
            monthly_charges = st.slider("Monthly Bill ($)", min_value=18.0, max_value=120.0, value=85.0, step=0.5)
            payment_method = st.selectbox("How do they pay?", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            
        with c2:
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online Security add-on", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support add-on", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup add-on", ["No", "Yes", "No internet service"])
            
        with c3:
            device_protection = st.selectbox("Device Protection add-on", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            senior = st.selectbox("Senior Citizen?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            
        # Build record
        total_charges = monthly_charges * max(tenure, 1)
        sim_data = {
            'gender': 'Male',
            'SeniorCitizen': senior,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': tenure,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        sim_df = pd.DataFrame([sim_data])
        sim_eng = feature_engineering(sim_df)
        
        feature_names = artifacts['metadata']['feature_names']
        numeric_cols = artifacts['metadata']['numeric_cols']
        
        encoded_sim = pd.get_dummies(sim_eng.drop(columns=['TenureGroup'] + [f'has_{s}' for s in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'] if f'has_{s}' in sim_eng.columns]))
        vector = pd.DataFrame(0, index=[0], columns=feature_names)
        for col in encoded_sim.columns:
            if col in vector.columns:
                vector[col] = encoded_sim[col].values[0]
                
        vector[numeric_cols] = artifacts['scaler'].transform(vector[numeric_cols])
        
        prob = artifacts['model'].predict_proba(vector)[0, 1]
        playbook = generate_customer_retention_playbook(sim_data, prob)
        
        st.markdown("---")
        res1, res2, res3 = st.columns([1, 1, 1.2])
        
        with res1:
            st.metric("Predicted Churn Chance", f"{playbook['churn_probability']}%")
            badge_class = "high-risk-badge" if playbook['risk_tier'] == "High Risk" else ("med-risk-badge" if playbook['risk_tier'] == "Medium Risk" else "low-risk-badge")
            st.markdown(f"**Risk Level:** <span class='{badge_class}'>{playbook['risk_tier']}</span>", unsafe_allow_html=True)
            
        with res2:
            st.metric("Yearly Revenue at Risk", f"${playbook['annual_revenue_at_risk']:,.2f}")
            st.metric("Estimated Cost of Action", f"${playbook['estimated_intervention_cost']}")
            
        with res3:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=playbook['churn_probability'],
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#e74c3c" if prob >= 0.65 else ("#f39c12" if prob >= 0.35 else "#27ae60")},
                    'steps': [
                        {'range': [0, 35], 'color': "#eafaf1"},
                        {'range': [35, 65], 'color': "#fef9e7"},
                        {'range': [65, 100], 'color': "#fdedec"}
                    ],
                    'threshold': {'line': {'color': "#2c3e50", 'width': 3}, 'thickness': 0.75, 'value': 50}
                }
            ))
            fig_gauge.update_layout(height=180, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_gauge, width="stretch")
            
        st.subheader("💡 Recommended Actions to Retain This Customer")
        col_t, col_a = st.columns(2)
        with col_t:
            st.markdown("**Why are they at risk?**")
            for t in playbook['key_risk_triggers']:
                st.warning(f"• {t}")
        with col_a:
            st.markdown("**What should we do?**")
            for act in playbook['recommended_actions']:
                st.success(f"**{act['category']}** ({act['roi_priority']} Priority)\n\n{act['action']}")

    with tab_lookup:
        st.subheader("Look Up a Customer from the Test Dataset")
        cust_id = st.selectbox("Pick Customer ID", df_scored['customerID'].head(100).tolist())
        cust_row = df_scored[df_scored['customerID'] == cust_id].iloc[0]
        
        c_l, c_r = st.columns([1, 2])
        with c_l:
            st.write(f"**Customer ID:** `{cust_row['customerID']}`")
            st.write(f"**Months with us:** {cust_row['tenure']}")
            st.write(f"**Contract:** {cust_row['Contract']}")
            st.write(f"**Monthly Bill:** ${cust_row['MonthlyCharges']}")
            st.write(f"**Payment Method:** {cust_row['PaymentMethod']}")
            st.write(f"**Did they actually churn?:** {'Yes' if cust_row['actual_churn'] == 1 else 'No'}")
        
        with c_r:
            score = cust_row['churn_probability']
            pb = generate_customer_retention_playbook(cust_row.to_dict(), score)
            st.metric("Model's Churn Probability", f"{pb['churn_probability']}%")
            st.write(f"**Risk Level:** {pb['risk_tier']}")
            st.write("**Recommended Steps:**")
            for a in pb['recommended_actions']:
                st.info(f"• **{a['category']}**: {a['action']}")

# ----------------- PAGE 3: SEGMENTS -----------------
elif page == "👥 3. Customer Groups & Insights":
    st.title("👥 Customer Groups & Insights")
    st.write("Dividing customers into High, Medium, and Low risk groups helps teams focus where it matters most.")
    
    cohort_summary = calculate_cohort_summary(df_scored)
    
    col1, col2, col3 = st.columns(3)
    for i, row in cohort_summary.iterrows():
        tier = row['risk_tier']
        col = col1 if tier == 'High Risk' else (col2 if tier == 'Medium Risk' else col3)
        with col:
            st.markdown(f"### {tier}")
            st.metric("Number of Customers", f"{row['customer_count']}")
            st.metric("Avg Monthly Bill", f"${row['avg_monthly_charges']:.2f}")
            st.metric("Avg Tenure", f"{row['avg_tenure']:.1f} months")
            st.metric("Actual Churn Rate", f"{row['actual_churn_rate']*100:.1f}%")
            
    st.markdown("---")
    
    st.subheader("Tenure vs Monthly Charges Across Risk Groups")
    fig_scatter = px.scatter(
        df_scored,
        x='tenure',
        y='MonthlyCharges',
        color='risk_tier',
        size='churn_probability',
        hover_data=['customerID', 'Contract', 'PaymentMethod', 'actual_churn'],
        color_discrete_map={'High Risk': '#e74c3c', 'Medium Risk': '#f39c12', 'Low Risk': '#27ae60'},
        title="Tenure vs Monthly Charges (Larger bubble = higher churn chance)"
    )
    st.plotly_chart(fig_scatter, width="stretch")
    
    st.subheader("High-Risk Customer List (Ready for Customer Support)")
    st.write("Support teams can review or download this list to reach out to customers before they cancel.")
    filtered = df_scored[df_scored['risk_tier'] == 'High Risk'][['customerID', 'tenure', 'Contract', 'MonthlyCharges', 'PaymentMethod', 'churn_probability', 'actual_churn']]
    st.dataframe(filtered.head(15), width="stretch")
    
    csv_export = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download High-Risk Customer List (.csv)",
        data=csv_export,
        file_name="high_risk_customers.csv",
        mime="text/csv"
    )

# ----------------- PAGE 4: PLAYBOOK -----------------
elif page == "📋 4. Action Guide & Offers":
    st.title("📋 Practical Retention Playbook")
    st.write("Specific, realistic offers that our team can roll out to keep customers from leaving.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Strategy 1: Move Customers to Annual Plans")
        st.info("""
        **Who to target:** Month-to-month customers in High or Medium risk.
        - **Why:** Month-to-month customers leave at a rate of 42.7%.
        - **What to do:** Show a banner in their account offering a 15% discount or 1 month free if they switch to a 1-year plan.
        - **Expected result:** Cuts customer loss in this group by around 35%.
        - **Cost:** ~$25–$35 one-time discount.
        """)
        
        st.subheader("Strategy 2: Free Tech Support Bundle")
        st.info("""
        **Who to target:** Fiber optic users who don't have tech support.
        - **Why:** They pay higher bills ($70–$100+) and get frustrated quickly when setup or Wi-Fi issues happen.
        - **What to do:** Give 3 months of free priority tech support and security help.
        - **Expected result:** Lowers fiber churn by up to 45%.
        - **Cost:** ~$15 service cost.
        """)
        
    with col2:
        st.subheader("Strategy 3: Promote Automatic Payments")
        st.warning("""
        **Who to target:** Customers paying by manual electronic check.
        - **Why:** 45.3% churn rate — manual payments add monthly friction and reminders to cancel.
        - **What to do:** Give an instant $10 bill credit if they turn on automatic card or bank payments.
        - **Expected result:** Lowers churn from 45.3% down to ~15%.
        - **Cost:** $10 one-time credit.
        """)
        
        st.subheader("Strategy 4: New Customer Welcome Program")
        st.warning("""
        **Who to target:** New sign-ups in their first 6 months.
        - **Why:** Over half of all cancellations happen in the first 6 months.
        - **What to do:** Call or message them on Day 14 and Day 45 to make sure everything is working smoothly.
        - **Expected result:** Improves 1st-year customer retention by ~20%.
        - **Cost:** ~$20 in support time.
        """)

# ----------------- PAGE 5: MODEL ACCURACY -----------------
elif page == "⚙️ 5. Model Accuracy & Drivers":
    st.title("⚙️ Machine Learning Models & Important Factors")
    st.write("We tested three different algorithms to find the best way to predict customer churn.")
    
    meta = artifacts['metadata']
    perf_df = pd.DataFrame(meta['models_performance']).T
    
    st.subheader("Model Comparison")
    st.dataframe(perf_df[['roc_auc', 'recall', 'precision', 'f1_score', 'accuracy']], width="stretch")
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("What Factors Influence Churn the Most?")
        top_feat_df = pd.DataFrame(meta['top_features'])
        fig_feat = px.bar(
            top_feat_df.sort_values('importance', ascending=True),
            x='importance',
            y='feature',
            orientation='h',
            title="Top Factors Affecting Churn",
            color='importance',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_feat, width="stretch")
        
    with c2:
        st.subheader("Confusion Matrix (Gradient Boosting)")
        gb_cm = meta['models_performance']['Gradient Boosting']['confusion_matrix']
        fig_cm = go.Figure(data=go.Heatmap(
            z=gb_cm,
            x=['Predicted to Stay', 'Predicted to Churn'],
            y=['Actually Stayed', 'Actually Churned'],
            colorscale='Blues',
            text=gb_cm,
            texttemplate="%{text}",
            textfont={"size": 18}
        ))
        fig_cm.update_layout(title="Predictions vs Reality")
        st.plotly_chart(fig_cm, width="stretch")
        
    st.info("""
    **Why did we pick Gradient Boosting?**
    - **Gradient Boosting** had the best overall score with an **ROC-AUC of 0.8440** and **80.3% accuracy**.
    - If the company wants to be extra aggressive and catch every possible customer who might leave, **Logistic Regression** is also available, which catches **78.3% of all churners**.
    """)

# ----------------- PAGE 6: ROI CALCULATOR -----------------
elif page == "💰 6. Cost & Savings Calculator":
    st.title("💰 Business Cost & Savings Calculator")
    st.write("Estimate how much money the company can save by running targeted retention offers.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Assumptions")
        cust_pool = st.number_input("Total Number of Customers", value=10000, step=1000)
        baseline_churn = st.slider("Current Churn Rate (%)", 10.0, 40.0, 26.5, 0.5) / 100.0
        avg_monthly_bill = st.slider("Average Monthly Bill ($)", 30.0, 120.0, 65.0, 5.0)
        retention_success = st.slider("How many at-risk customers can we save? (%)", 10, 50, 25, 5) / 100.0
        intervention_cost = st.slider("Cost per customer offer ($)", 10, 60, 25, 5)
        
    total_churners = cust_pool * baseline_churn
    retained_customers = total_churners * retention_success
    gross_annual_rev_saved = retained_customers * avg_monthly_bill * 12
    campaign_cost = total_churners * intervention_cost
    net_profit_saved = gross_annual_rev_saved - campaign_cost
    roi_ratio = (net_profit_saved / campaign_cost) * 100 if campaign_cost > 0 else 0
    
    with c2:
        st.subheader("Expected Savings")
        st.metric("Customers at Risk of Leaving", f"{int(total_churners):,}")
        st.metric("Customers We Expect to Save", f"{int(retained_customers):,}")
        st.metric("Yearly Revenue Saved", f"${gross_annual_rev_saved:,.2f}")
        st.metric("Total Budget for Offers", f"${campaign_cost:,.2f}")
        st.metric("Net Profit Saved", f"${net_profit_saved:,.2f}", delta=f"{roi_ratio:.0f}% ROI")
        
    st.markdown("---")
    st.success(f"""
    💡 **Bottom Line:** By spending **${campaign_cost:,.0f}** on smart, targeted customer offers, the business saves **${gross_annual_rev_saved:,.0f}** in lost yearly revenue. That gives a **Net Return of {roi_ratio:.0f}%**.
    """)
