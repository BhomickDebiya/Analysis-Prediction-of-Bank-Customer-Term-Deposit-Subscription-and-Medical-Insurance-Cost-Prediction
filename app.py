import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page styling
st.set_page_config(
    page_title="Bank Customer & Medical Insurance Analytics Suite",
    layout="wide"
)

# Top navigation bar
tab_profile, tab_bank, tab_model, tab_predictor = st.tabs([
    "📋 Data Profile & Overview",
    "📊 Bank Dataset Dashboard",
    "🔬 Statistics & Model Selection",
    "⚕️ Insurance Premium Predictor"
])

# Load Data Helper functions
@st.cache_data
def load_bank_data():
    bank_csv_path = 'bank_data/bank-full.csv'
    if os.path.exists(bank_csv_path):
        return pd.read_csv(bank_csv_path, sep=";")
    return None

@st.cache_data
def load_insurance_data():
    ins_csv_path = 'insurance.csv'
    if os.path.exists(ins_csv_path):
        return pd.read_csv(ins_csv_path)
    return None

df_bank = load_bank_data()
df_ins = load_insurance_data()

# =====================================================================
# TAB 1: DATA PROFILE & OVERVIEW
# =====================================================================
with tab_profile:
    st.title("📋 Datasets Profile & Overview")
    st.write("Comprehensive summary of structural, quality, and meta metrics for both underlying research datasets.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏦 Dataset 1: Bank Marketing Dataset")
        if df_bank is not None:
            # Metric Row
            m1, m2, m3 = st.columns(3)
            m1.metric("Row Count", f"{df_bank.shape[0]:,}")
            m2.metric("Column Count", f"{df_bank.shape[1]}")
            m3.metric("Duplicate Rows", f"{df_bank.duplicated().sum()}")

            # Null info
            st.markdown("**Missing Values per Column:**")
            nulls = df_bank.isnull().sum().to_frame(name="Null Counts")
            st.dataframe(nulls.T, use_container_width=True)

            # Data Preview
            st.markdown("**Data Head Preview:**")
            st.dataframe(df_bank.head(5), use_container_width=True)
        else:
            st.warning("Bank dataset CSV not found!")

    with col2:
        st.subheader("⚕️ Dataset 2: Medical Insurance Dataset")
        if df_ins is not None:
            # Metric Row
            m1, m2, m3 = st.columns(3)
            m1.metric("Row Count", f"{df_ins.shape[0]:,}")
            m2.metric("Column Count", f"{df_ins.shape[1]}")
            m3.metric("Duplicate Rows", f"{df_ins.duplicated().sum()}")

            # Null info
            st.markdown("**Missing Values per Column:**")
            nulls_ins = df_ins.isnull().sum().to_frame(name="Null Counts")
            st.dataframe(nulls_ins.T, use_container_width=True)

            # Data Preview
            st.markdown("**Data Head Preview:**")
            st.dataframe(df_ins.head(5), use_container_width=True)
        else:
            st.warning("Insurance dataset CSV not found!")

# =====================================================================
# TAB 2: BANK MARKETING DASHBOARD
# =====================================================================
with tab_bank:
    st.title("📊 Bank Customer Term Deposit Subscription Analysis")
    st.write("Exploratory visual representations of customer financial variables and demographics.")

    if df_bank is not None:
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Customers Included", f"{len(df_bank):,}")
        sub_rate = (df_bank['y'] == 'yes').mean() * 100
        col_m2.metric("Average Term Subscription", f"{sub_rate:.2f}%")
        col_m3.metric("Avg Customer Balance", f"${df_bank['balance'].mean():,.2f}")

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.subheader("Term Deposit Subscription Ratio")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.countplot(data=df_bank, x='y', palette='Set2', ax=ax)
            st.pyplot(fig)
            st.info("**Insight:** The dataset is highly imbalanced. Only ~11.7% of contacted customers chose to subscribe to a term deposit, indicating target marketing refinement is essential.")

            st.subheader("Age Distribution Profile")
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            sns.histplot(data=df_bank, x='age', bins=20, kde=True, color='skyblue', ax=ax2)
            st.pyplot(fig2)
            st.info("**Insight:** The customer distribution peaks in the active career window between 30 and 45 years. Outreach to retired demographics is low, despite their high relative subscription rates.")

        with col_v2:
            st.subheader("Job Type Subscriptions Comparison")
            fig3, ax3 = plt.subplots(figsize=(6, 3.5))
            sns.countplot(data=df_bank, x='job', hue='y', palette='Set1', ax=ax3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig3)
            st.info("**Insight:** Management and technician classes form the largest segment volume, but students and retired individuals exhibit higher individual conversion ratios.")

            st.subheader("Outlier-Removed Balances by Subscription")
            fig4, ax4 = plt.subplots(figsize=(6, 3.5))
            sns.boxplot(data=df_bank, x='y', y='balance', palette='pastel', ax=ax4, showfliers=False)
            st.pyplot(fig4)
            st.info("**Insight:** Subscribed customers maintain significantly higher median standing ledger balances than non-subscribers, reflecting a strong correlation between capital reserves and commitment capability.")
    else:
        st.warning("No Bank data loaded.")

# =====================================================================
# TAB 3: STATISTICS & MODEL SELECTION
# =====================================================================
with tab_model:
    st.title("🔬 Statistical Testing & Predictive Model Selection")

    st.subheader("1. Key Hypotheses & Takeaways")
    st.markdown(
        "* **One-Sample t-test on Customer Age:**\n"
        "  * *Hypothesis:* H0: Mean Age = 40 vs H1: Mean Age != 40\n"
        "  * *Result:* t-stat = **0.3492**, p-value = **0.7284**\n"
        "  * *Takeaway:* **Fail to Reject H0**. The average customer age does not deviate significantly from 40.\n\n"
        "* **Two-Sample Independent Welch's t-test (Balances):**\n"
        "  * *Hypothesis:* Equal means between subscribers and non-subscribers.\n"
        "  * *Result:* t-stat = **9.9335**, p-value = **4.3837e-23**\n"
        "  * *Takeaway:* **Reject H0**. Term deposit subscribers possess significantly higher ledger account balances.\n\n"
        "* **Chi-Square Test of Independence (Job vs Subscription):**\n"
        "  * *Result:* Chi2-stat = **836.1055**, p-value = **3.3371e-172**\n"
        "  * *Takeaway:* **Reject H0**. Job classifications are highly correlated with subscription rates.\n\n"
        "* **Two-Way ANOVA (Dependent: Age | Marital, Education):**\n"
        "  * *Takeaway:* Marital status, education background, and their respective interaction coefficients all have highly significant (p < 0.05) effects on customer age."
    )

    st.markdown("---")
    st.subheader("2. Machine Learning Regression Model Performance Comparison")
    st.write("Below are the calculated validation metrics for various regression estimators trained on the Medical Insurance Dataset.")

    comparison_data = pd.DataFrame({
        'Model': [
            'Random Forest Regression',
            'Decision Tree Regression',
            'Multiple Linear Regression',
            'Lasso Regression',
            'Ridge Regression'
        ],
        'R2 Score': [0.8735, 0.8336, 0.7836, 0.7835, 0.7833],
        'MAE ($)': ['2,560.40', '2,930.77', '4,181.19', '4,182.22', '4,193.20'],
        'RMSE ($)': ['4,431.79', '5,082.51', '5,796.28', '5,797.04', '5,800.46']
    })

    col_table, col_chart = st.columns([1, 1.2])

    with col_table:
        st.markdown("**Performance Table:**")
        st.dataframe(comparison_data, use_container_width=True)
        st.success("🏆 **Best Performing Model Selected:** **Random Forest Regression** with a validation R2 score of **87.35%**.")

    with col_chart:
        st.markdown("**Model $R^2$ Comparison Chart:**")
        fig_compare, ax_compare = plt.subplots(figsize=(7, 4.3))
        chart_df = pd.DataFrame({
            'Model': ['Random Forest', 'Decision Tree', 'Linear Reg.', 'Lasso', 'Ridge'],
            'R2': [0.8735, 0.8336, 0.7836, 0.7835, 0.7833]
        })
        sns.barplot(data=chart_df, x='R2', y='Model', palette='Blues_r', ax=ax_compare)
        ax_compare.set_xlim(0, 1.0)
        ax_compare.set_xlabel("R-squared Score (Higher is Better)")
        ax_compare.set_ylabel("Algorithm")
        for i, v in enumerate(chart_df['R2']):
            ax_compare.text(v + 0.01, i, f"{v:.4f}", va='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_compare)

# =====================================================================
# TAB 4: INSURANCE PREMIUM PREDICTOR
# =====================================================================
with tab_predictor:
    st.title("⚕️ Medical Insurance Premium Predictor")
    st.write("Input target customer attributes below to execute the saved production pipeline and determine estimates.")

    model_path = 'best_insurance_model.joblib'
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
    else:
        st.error("Inference model 'best_insurance_model.joblib' missing. Execute modeling tasks first.")
        st.stop()

    with st.form("predictor_form"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
            bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
            children = st.number_input("Number of Dependents / Children", min_value=0, max_value=10, value=0, step=1)
        with col_f2:
            sex = st.selectbox("Biological Sex", options=["male", "female"])
            smoker = st.selectbox("Smoker Status", options=["no", "yes"])
            region = st.selectbox("US Geographic Region", options=["southwest", "southeast", "northwest", "northeast"])

        submit_button = st.form_submit_button("Predict Premium")

    if submit_button:
        input_df = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region
        }])
        prediction = pipeline.predict(input_df)[0]
        st.success(f"### Estimated Annual Premium Charges: **${prediction:,.2f}**")