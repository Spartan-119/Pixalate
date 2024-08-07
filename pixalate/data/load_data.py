import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, parse_dates = ['click_time'])

    # extracting the day and the hour from the click_time column
    df['hour'] = df['click_time'].dt.hour
    df['day'] = df['click_time'].dt.day

    # convert categorical variables to numeric
    for col in ['ip', 'app', 'device', 'os', 'channel']:
        df[col] = df[col].astype('category').cat.codes
    
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