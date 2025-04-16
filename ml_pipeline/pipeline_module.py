import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

## Preprocessing pipeline

def create_pipeline(numeric_features, categorical_features):
  numeric_transformers = Pipeline(steps = [
    ('imputer',SimpleImputer(strategy = 'mean')),
    ('scaler', StandardScaler())
  ])

  categorical_transformer = Pipeline(steps= [
    ('imputer',SimpleImputer(strategy = 'most_frequent')),
    ('encoder', OneHotEncoder(handle_unkown = 'ignore'))
  ])

  preprocessor = ColumnTransformer(
    transformers = [
      ('num',numeric_transformers,numeric_features),
      ('cat',categorical_transformer,categorical_features)
    ]
  )
  model_pipeline = Pipeline(steps = [
    ('preprocessor',preprocessor),
    ('classifier', RandomForestClassifier(n_estimators = 100, random_state = 123))
  ]
  )
  return model_pipeline

def train_and_save(data: pd.DataFrame,traget_col: str, model_path: str): 
  y = data[target_col]
  X = data.drop(columns = [target_col])

  numeric_features = X.select_dtypes(include = ['int64,float64']).columns.tolist()
  categorical_features = X.select_dtypes(include = ['object','category']).columns.tolist()

  X_tran,X_test, y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
  pipeline = create_pipeline(numeric_features,categorical_features)
  pipeline.Fit(X_train,y_train)

  os.makedirs(os.path.dirname(model_path),exist_ok = True)
  joblib.dump(pipeline,model_path)

  print(f"Model pipeline saved to {model_path}")


def load_model(model_path:str):
  return joblib.load(model_path)

def predict(model,input_data: pd.DataFrame):
  return model.predict(input_data)


