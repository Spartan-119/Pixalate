from collections import OrderedDict
from typing import *

import numpy as np
import torch
import torch.utils
import torch.utils.data

from ad_tracking import train, test, load_data
import flwr as fl

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FLClient(fl.client.NumPyClient):
    """
    Flower client implementing the EDA and model building to identify a fraud click
    """

    def __init__(
            self,
            model, # this will be a sklearn model
            trainloader: torch.utils.data.DataLoader,
            testloader: torch.utils.data.DataLoader,
            num_examples: Dict,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        # For Scikit-learn, extract coefficients and intercept
        if isinstance(self.model, LogisticRegression):
            return [self.model.coef_, self.model.intercept_]
        elif isinstance(self.model, RandomForestClassifier):
            # RandomForest doesn't have a direct parameter list like logistic regression
            return []  # Placeholder: Federated learning with RF might need different handling
        else:
            raise NotImplementedError("Model type not supported")

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        # Only applicable for models like LogisticRegression
        if isinstance(self.model, LogisticRegression):
            self.model.coef_ = parameters[0]
            self.model.intercept_ = parameters[1]
        elif isinstance(self.model, RandomForestClassifier):
            # RandomForest doesn't have a direct parameter setting method
            pass  # Placeholder for RF-specific logic
        else:
            raise NotImplementedError("Model type not supported")

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        train(self.model, self.trainloader)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}
    
def main() -> None:
    """load data, start fl_client"""
    file_path = "pixalate\data\data.csv"

    # Load data
    trainloader, testloader, num_examples = load_data(file_path)

    # Initialize model
    model = LogisticRegression(max_iter=1000)
    # model = RandomForestClassifier(n_estimators=100)  # Uncomment if using RandomForest

    # Create Flower client
    client = FLClient(model, trainloader, testloader, num_examples)

    # Start client
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)

if __name__ == "__main__":
    main()