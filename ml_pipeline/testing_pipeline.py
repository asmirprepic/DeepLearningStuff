import pytest
import pandas as pd
import numpy as np
from pipeline_module import train_and_save_model, load_model, predict

MODEL_PATH = 'models/random_forest_pipeline.joblib'


@pytest.fixture(scope='module')
def dummy_data():
    return pd.DataFrame({
        'age': [25, 32, 47, 51, 62],
        'income': [50000, 64000, 120000, 70000, 90000],
        'gender': ['male', 'female', 'female', 'male', 'female'],
        'churn': [0, 1, 0, 1, 0]
    })

def test_pipeline_train_and_predict(dummy_data):
    train_and_save_model(dummy_data, target_col='churn', model_path=MODEL_PATH)
    model = load_model(MODEL_PATH)

    test_input = pd.DataFrame({
        'age': [29],
        'income': [58000],
        'gender': ['female']
    })

    prediction = predict(model, test_input)
    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]
