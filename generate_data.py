
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

n = 100

applicant_income = np.random.randint(15000, 100001, n)
coapplicant_income = np.random.choice(
    [0, 0, 0, 5000, 10000, 15000, 20000, 30000],
    n
)
loan_amount = np.random.randint(100000, 1500001, n)
loan_term = np.random.choice([60, 120, 180, 240, 300, 360], n)
credit_score = np.random.randint(450, 851, n)
employment_type = np.random.choice(
    ["Salaried", "Self-employed"], n
)
existing_loans = np.random.randint(0, 5, n)

# Create illustrative synthetic approval labels
total_income = applicant_income + coapplicant_income

approval_score = (
    (credit_score >= 650).astype(int)
    + (total_income >= 40000).astype(int)
    + (loan_amount <= total_income * 30).astype(int)
    + (existing_loans <= 1).astype(int)
)

loan_status = np.where(approval_score >= 3, "Approved", "Rejected")

df = pd.DataFrame({
    "applicant_income": applicant_income,
    "coapplicant_income": coapplicant_income,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "credit_score": credit_score,
    "employment_type": employment_type,
    "existing_loans": existing_loans,
    "loan_status": loan_status
})

output_path = Path("data/loan_data.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("Dataset created successfully!")
print("Total records:", len(df))
print("\nFirst 5 records:")
print(df.head())
print("\nLoan status distribution:")
print(df["loan_status"].value_counts())
print("\nSaved to:", output_path)

