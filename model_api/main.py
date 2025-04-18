from fastapi import FastAPI
from pydantic import BaseModel,Field
from model_utils import make_prediction

app = FastAPI(title = 'Customer Churn Prediction')

class CustomerInput(BaseModel):
  age: int = Field(...,example = 34)
  income: float = Field(...,example = 72000)
  gender: str = Field(...,example = 'female')

@app.get("/")
def root():
  return {"message": "ML API is running."}

@app.post("/predict")
def predict(input: CustomerInput):
  input_dict = input.dict()
  predictoin,cofidence = make_prediction(input_dict)
  return {
    "prediction": prediction,
    "confidence": round(confidence,3)
  }
  
  
