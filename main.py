# this is the starting point of execution

import flwr as fl
from pixalate.data.load_data import load_and_preprocess_data, split_data_for_federated_learning
from pixalate.federated_learning.fl_client import client_fn

def main():
    # Load and preprocess data
    df = load_and_preprocess_data("pixalate\data\data.csv")
    client_data = split_data_for_federated_learning(df, n_clients=5)

    # Define flower client
    def client_fn(cid: str):
        return client_fn(client_data[int(cid)])

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        client_manager=fl.server.SimpleClientManager(),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            min_fit_clients=5,
            min_available_clients=5,
        ),
    )

if __name__ == "__main__":
    main()