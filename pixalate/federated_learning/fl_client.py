from collections import OrderedDict
from typing import *

import numpy as np
import torch

import pixalate.federated_learning.ad_tracking as ad_tracking
import flwr as fl

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FLClient(fl.client.NumPyClient):
    """
    Flower client implementing the EDA and model building to identify a fraud click
    """

    def __init__(
            self,
    ):
        pass

    def get_parameters(self):
        pass

    def set_parameters(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        pass

    def fit(self):
        pass