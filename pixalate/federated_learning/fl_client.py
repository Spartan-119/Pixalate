import flwr as fl
import torch 
import torch.nn as nn
import torch.optim as optim
from pixalate.models.fraud_detection_model import FraudDetectionModel, create_model, get_model_parameters, set_model_parameters