import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.preprocessing import data_preprocessing
from src.feature_engineering import age_to_bins, simplify_target

class Features(BaseModel):
    features: list[float]

app = FastAPI()

model = joblib.load("./models/lr_model.pkl")

@app.post("/predict")
def predict(input_data: Features):
    try:
        # Convert raw list into DataFrame with expected columns
        df = pd.DataFrame([input_data.features], columns=[
            "age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal"
        ])

        # Apply same transformations as training
        df = data_preprocessing(df)
        df = age_to_bins(df)
        #df = simplify_target(df)

        # Predict
        prediction = model.predict(df)[0]
        confidence = model.predict_proba(df)[0].max()

        return {"prediction": int(prediction), "confidence": float(confidence)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
