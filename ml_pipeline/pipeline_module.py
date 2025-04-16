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
  
