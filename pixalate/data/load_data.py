import pandas as pd
import numpy as np
import datetime
import os
import time
from matplotlib import pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from collections import Counter

# for the sake of simplicity i am giving in the configurations here, later might add to a separate conf/ in a YAML file
# file_path = "pixalate\data\data.csv"
    

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, parse_dates = ['click_time'])

    # the ip, app, device, os and channel are categorical variables encoded as integers.
    # setting them as categorise for analysis
    variables = ['ip', 'app', 'device', 'os', 'channel']
    for variable in variables:
        df[variable] = df[variable].astype('category')

    # extracting the day and the hour from the click_time column
    df['hour'] = df['click_time'].dt.hour
    df['day'] = df['click_time'].dt.day

    # converting date stamps to date/time type
    df['click_time'] = pd.to_datetime(df['click_time'])
    df['attributed_time'] = pd.to_datetime(df['attributed_time'])

    # counting the number of classes where 1 implies a click and 0 not a click
    count_0 = df[df['is_attributed'] == 0].shape[0]
    count_1 = df[df['is_attributed'] == 1].shape[0]
    print(f"Number of rows with 'is_attribute' == 0: {count_0}")
    print(f"Number of rows with 'is_attribute' == 1: {count_1}")

    # removing the featuers to accomodate SMOTE - i Know this is not ideal, but EDA is not the goal here
    X = df.drop(['is_attributed', 'attributed_time', 'click_time'], axis=1)
    y = df['is_attributed']

    # splitting into 80:20 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # applying SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    print('Original dataset shape:', Counter(y_train))
    print('Resampled dataset shape:', Counter(y_train_res))

    return df

# def split_data_for_federated_learning(df, n_clients = 5):
#     # split the dataset into n parts to simulate federated clients
#     client_data = np.array_split(df, n_clients)
    
#     return client_data

# if __name__ == "__main__":
#     file_path = "pixalate\data\data.csv"
#     df = load_and_preprocess_data(file_path)
#     client_data = split_data_for_federated_learning(df)

#     print(f"Loaded and preprocessed {len(df)} records.")
#     print(f"Split data into {len(client_data)} parts for federated learning.")

#     for i, data in enumerate(client_data):
#         data.to_csv(f"data\client_{i+1}_data.csv", index=False)
#         print(f"Saved data for client {i+1} with {len(data)} records.")