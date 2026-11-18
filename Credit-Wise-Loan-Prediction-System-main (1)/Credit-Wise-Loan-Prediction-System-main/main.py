import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# --- CONFIGURATION ---
st.set_page_config(page_title="LoanGuard AI", page_icon="🏦", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0px 2px 10px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def train_model():
    # Load data (Assuming loan_approval_data.csv exists)
    try:
        df = pd.read_csv("loan_approval_data.csv")
    except FileNotFoundError:
        # Dummy data creation for demonstration if file is missing
        return None, None, None, None, None, None, None

    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols = df.select_dtypes(include=["number"]).columns

    # Imputation
    num_imp = SimpleImputer(strategy="mean")
    df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])
    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

    # Preprocessing
    df = df.drop("Applicant_ID", axis=1)
    le_target = LabelEncoder()
    df["Loan_Approved"] = le_target.fit_transform(df["Loan_Approved"])
    le_edu = LabelEncoder()
    df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])

    cols_to_ohe = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(df[cols_to_ohe])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols_to_ohe), index=df.index)
    df = pd.concat([df.drop(columns=cols_to_ohe), encoded_df], axis=1)

    # Feature engineering
    df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2
    df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])

    X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio", "Applicant_Income"])
    y = df["Loan_Approved"]
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    return model, scaler, le_edu, ohe, cols_to_ohe, numerical_cols, categorical_cols

# Initialize
log_model, scaler, le_edu, ohe, ohe_cols, num_cols, cat_cols = train_model()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/100/bank.png")
    st.title("LoanGuard AI")
    st.info("This AI evaluates loan applications based on historical credit trends.")
    st.divider()
    st.header("Settings")
    theme = st.toggle("Dark Mode Look", value=False)

# --- MAIN UI ---
st.title("🏦 Loan Approval Intelligence")
st.write("Fill out the applicant profile below to generate a real-time risk assessment.")

tab1, tab2 = st.tabs(["📝 Application Form", "📊 Model Insights"])

with tab1:
    with st.expander("👤 Applicant Personal Details", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            applicant_id = st.text_input("Application Ref #", value="APP-1001")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            age = st.slider("Age", 18, 100, 30)
        with col2:
            marital = st.selectbox("Marital Status", ["Married", "Single"])
            edu = st.selectbox("Education Level", ["Graduate", "Undergraduate"])
        with col3:
            dependents = st.number_input("Dependents", 0, 10, 0)
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    with st.expander("💼 Financial Profile", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            income = st.number_input("Monthly Income ($)", min_value=0, value=5000)
            co_income = st.number_input("Co-applicant Income ($)", min_value=0, value=0)
        with c2:
            emp_status = st.selectbox("Employment", ["Employed", "Self-Employed", "Unemployed"])
            credit_score = st.select_slider("Credit Score", options=range(300, 851), value=650)
        with c3:
            dti = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3, help="Total monthly debt payments divided by gross monthly income.")

    with st.expander("💰 Loan Request Details", expanded=True):
        l1, l2, l3 = st.columns(3)
        with l1:
            loan_amt = st.number_input("Requested Amount ($)", min_value=0, value=20000)
        with l2:
            loan_term = st.selectbox("Term (Months)", [12, 18, 24, 36, 48, 60, 120])
        with l3:
            purpose = st.selectbox("Purpose", ["Home", "Education", "Personal", "Debt Consolidation"])

    predict_btn = st.button("Analyze Application")

if predict_btn:
    # Processing (Simplified for the example)
    input_data = pd.DataFrame([{
        "Applicant_ID": 0, "Applicant_Income": income, "Coapplicant_Income": co_income,
        "Employment_Status": emp_status, "Age": age, "Marital_Status": marital,
        "Dependents": dependents, "Credit_Score": credit_score, "Existing_Loans": 0,
        "DTI_Ratio": dti, "Savings": 0.0, "Collateral_Value": 0.0,
        "Loan_Amount": loan_amt, "Loan_Term": loan_term, "Loan_Purpose": purpose,
        "Property_Area": property_area, "Education_Level": edu, "Gender": gender,
        "Employer_Category": "A", "Loan_Approved": 0 
    }])

    # Encoding & Feature Engineering (Keep your logic here)
    input_data["Education_Level"] = le_edu.transform(input_data["Education_Level"])
    encoded_input = ohe.transform(input_data[ohe_cols])
    encoded_df_input = pd.DataFrame(encoded_input, columns=ohe.get_feature_names_out(ohe_cols), index=input_data.index)
    input_final = pd.concat([input_data.drop(columns=ohe_cols), encoded_df_input], axis=1)
    input_final["DTI_Ratio_sq"] = input_final["DTI_Ratio"] ** 2
    input_final["Credit_Score_sq"] = input_final["Credit_Score"] ** 2
    input_final["Applicant_Income_log"] = np.log1p(input_final["Applicant_Income"])
    X_input = input_final.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio", "Applicant_Income", "Applicant_ID"])
    
    # Prediction
    X_input_scaled = scaler.transform(X_input)
    prob = log_model.predict_proba(X_input_scaled)[0][1]

    # Display Results with flair
    st.subheader("Analysis Result")
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if prob > 0.5:
            st.success("### APPROVED")
            st.balloons()
        else:
            st.error("### DECLINED")
            
    with res_col2:
        st.write(f"Approval Probability: **{prob:.1%}**")
        st.progress(prob)

with tab2:
    st.subheader("Model Performance")
    st.write("This model uses Logistic Regression to determine creditworthiness.")
    # You could add feature importance charts here later!
