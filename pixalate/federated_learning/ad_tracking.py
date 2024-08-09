import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset
import torch
from typing import Tuple, Dict
from collections import Counter

file_path = "pixalate\data\data.csv"

def load_data(file_path: str) -> Tuple[DataLoader, DataLoader, Dict]:
    df = pd.read_csv(file_path, parse_dates=['click_time'])

    # Convert categorical variables
    variables = ['ip', 'app', 'device', 'os', 'channel']
    for variable in variables:
        df[variable] = df[variable].astype('category').cat.codes

    # Extract hour and day from click_time
    df['hour'] = df['click_time'].dt.hour
    df['day'] = df['click_time'].dt.day

    # Prepare features and target
    X = df.drop(['is_attributed', 'attributed_time', 'click_time'], axis=1)
    y = df['is_attributed']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE to the training data
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Standardize the features
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_res, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_res.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    # Create TensorDatasets
    trainset = TensorDataset(X_train_tensor, y_train_tensor)
    testset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    # Number of examples
    num_examples = {"trainset": len(trainset), "testset": len(testset)}

    return trainloader, testloader, num_examples

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

def train(
    model,  # This will be a scikit-learn model
    trainloader: torch.utils.data.DataLoader,
) -> None:
    """Train the classical machine learning model."""
    # Collect all training data
    X_train, y_train = [], []
    for data in trainloader:
        X_batch, y_batch = data
        X_train.extend(X_batch.numpy())
        y_train.extend(y_batch.numpy())

    # Fit the model
    model.fit(X_train, y_train)
    print("Training complete")

# Example usage:
# model = LogisticRegression(max_iter=1000)
# or
# model = RandomForestClassifier(n_estimators=100)
# train(model, trainloader)

def test(
    model,  # This will be a scikit-learn model
    testloader: torch.utils.data.DataLoader,
) -> Tuple[float, float]:
    """Evaluate the classical machine learning model."""
    # Collect all test data
    X_test, y_test = [], []
    for data in testloader:
        X_batch, y_batch = data
        X_test.extend(X_batch.numpy())
        y_test.extend(y_batch.numpy())

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Calculate accuracy and log loss
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)

    print(f"Test Accuracy: {accuracy:.4f}, Log Loss: {loss:.4f}")
    return loss, accuracy

# Example usage:
# loss, accuracy = test(model, testloader)

def main():
    print("Classical Machine Learning Training")
    
    # Load data
    print("Loading data...")
    trainloader, testloader, num_examples = load_data(file_path)
    
    # Choose a model
    # You can switch between Logistic Regression and Random Forest
    model = LogisticRegression(max_iter=1000)
    # model = RandomForestClassifier(n_estimators=100)
    
    # Train the model
    print("Start training...")
    train(model, trainloader)
    
    # Evaluate the model
    print("Evaluate model...")
    loss, accuracy = test(model, testloader)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

if __name__ == "__main__":
    main()