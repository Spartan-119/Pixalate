import flwr as fl
import torch 
import torch.nn as nn
import torch.optim as optim
from pixalate.models.fraud_detection_model import FraudDetectionModel, create_model, get_model_parameters, set_model_parameters

class FraudDetectionClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters())

    def get_parameters(self):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)
        train_loader = torch.utils.data.DataLoader(
            list(zip(self.x_train, self.y_train)), batch_size=32, shuffle=True
        )
        
        self.model.train()
        for epoch in range(1):  # mention the number of epochs here
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
        
        return get_model_parameters(self.model), len(self.x_train), {}

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.x_test)
            loss = self.criterion(outputs, self.y_test.unsqueeze(1))
            accuracy = ((outputs > 0.5) == self.y_test.unsqueeze(1)).float().mean()
        return loss.item(), len(self.x_test), {"accuracy": accuracy.item()}

def client_fn(client_data):
    x_train = client_data.drop('is_attributed', axis=1)
    y_train = client_data['is_attributed']
    model = create_model(input_dim=x_train.shape[1])
    return FraudDetectionClient(model, x_train, y_train, x_train, y_train)