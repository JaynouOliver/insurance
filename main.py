from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


model = joblib.load("xgb.joblib")

# List of categorical columns
cat_cols = ['gender', 'marital_status', 'employment_type', 'region', 'has_dependents']

# Define the data schema for incoming requests
class EmployeeData(BaseModel):
    age: int
    gender: str
    marital_status: str
    salary: float
    employment_type: str
    region: str
    has_dependents: str
    tenure_years: float

app = FastAPI()

@app.post("/predict")
def predict(data: EmployeeData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Ensure categorical columns are of 'category' dtype
    for col in cat_cols:
        input_df[col] = input_df[col].astype('category')

    # Predict
    pred = model.predict(input_df)[0]
    prob = float(model.predict_proba(input_df)[0][1])

    return {
        "predicted_class": int(pred),
        "predicted_probability": prob
    }


