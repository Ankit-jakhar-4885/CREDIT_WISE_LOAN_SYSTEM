import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load data
df = pd.read_csv("loan_approval_data.csv")

# Separate categorical and numerical columns
categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["number"]).columns

# Imputation
num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

cat_imp = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

# Drop ID
df = df.drop("Applicant_ID", axis=1)

# Encode target
le_target = LabelEncoder()
df["Loan_Approved"] = le_target.fit_transform(df["Loan_Approved"])

# Encode Education_Level
le_edu = LabelEncoder()
df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])

# One-hot encode other categorical features
cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = ohe.fit_transform(df[cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=df.index)
df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)

# Feature engineering
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2
df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])

# Split features and target
X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio", "Applicant_Income"])
y = df["Loan_Approved"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

y_pred = log_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------
# USER INPUT PROCESSING
# -------------------------

user_input = {
    "Applicant_ID": int(input("Enter Applicant ID: ")),
    "Applicant_Income": float(input("Enter Applicant Income: ")),
    "Coapplicant_Income": float(input("Enter Coapplicant Income: ")),
    "Employment_Status": input("Enter Employment Status: "),
    "Age": int(input("Enter Age: ")),
    "Marital_Status": input("Enter Marital Status: "),
    "Dependents": int(input("Enter Number of Dependents: ")),
    "Credit_Score": int(input("Enter Credit Score: ")),
    "Existing_Loans": int(input("Enter Number of Existing Loans: ")),
    "DTI_Ratio": float(input("Enter DTI Ratio: ")),
    "Savings": float(input("Enter Savings Amount: ")),
    "Collateral_Value": float(input("Enter Collateral Value: ")),
    "Loan_Amount": float(input("Enter Loan Amount: ")),
    "Loan_Term": int(input("Enter Loan Term (months): ")),
    "Loan_Purpose": input("Enter Loan Purpose: "),
    "Property_Area": input("Enter Property Area: "),
    "Education_Level": input("Enter Education Level: "),
    "Gender": input("Enter Gender: "),
    "Employer_Category": input("Enter Employer Category: "),
    "Loan_Approved": input("Enter Loan_Approved: ")
}

df1 = pd.DataFrame([user_input])

# Apply imputers
df1[numerical_cols] = num_imp.transform(df1[numerical_cols])
df1[categorical_cols] = cat_imp.transform(df1[categorical_cols])

# Drop ID
df1 = df1.drop("Applicant_ID", axis=1)

# Encode Education_Level using same encoder
df1["Education_Level"] = le_edu.transform(df1["Education_Level"])

# One-hot encode using same encoder
encoded_input = ohe.transform(df1[cols])
encoded_df_input = pd.DataFrame(encoded_input, columns=ohe.get_feature_names_out(cols), index=df1.index)
df1 = pd.concat([df1.drop(columns=cols), encoded_df_input], axis=1)

# Feature engineering
df1["DTI_Ratio_sq"] = df1["DTI_Ratio"] ** 2
df1["Credit_Score_sq"] = df1["Credit_Score"] ** 2
df1["Applicant_Income_log"] = np.log1p(df1["Applicant_Income"])

# Final feature set (same as training)
X_input = df1.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio", "Applicant_Income"])

# Scale
X_input_scaled = scaler.transform(X_input)

# Predict
y_pred_output = log_model.predict(X_input_scaled)
print("Prediction (0=Not Approved, 1=Approved):", y_pred_output)