import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Generation of synthetic data
def generate_data(num_samples = 10000):
    np.random_seed(42)
    timestamps = pd.date_range('2023-01-01',periods = num_samples,freq = "M")
    trend = np.linspace(0,2,num_samples)
    seasonality = 2*np.sin(2*np.pi*(timestamps.month)/12)
    noise = np.random.normal(0,0.1,num_samples)
    data = trend+ seasonality + noise

    return pd.DataFrame(data,columns = ['value'],index = timestamps)

def preprocess_and_create_sequence(df,seq_length = 12):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    data_scaled_df = pd.DataFrame(data = data_scaled,index = df.index,columns = df.columns)

    xs,ys = [],[]
    for i in range(len(data_scaled_df)-seq_length):
        x = data_scaled_df.iloc[i:(i+seq_length)].values
        y = data_scaled_df.iloc[(i+seq_length)].values
        xs.append(x)
        ys.append(y)
    
    return np.array(xs),np.array(ys),scaler

data = generate_data()
SEQ_LENGTH = 12
X,y,scaler = preprocess_and_create_sequence(data,SEQ_LENGTH)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

ds_train = tf.data.DataSet.from_tensor_slices((X_train,y_train)).shuffle(len(X_train)).batch(32).prefetch(tf.data.AUTOTUNE)
df_test = tf.data.DatSet.from_tensor_slices((X_test,y_test)).batch(32).prefetch(tf.data.AUTOTUNE)


