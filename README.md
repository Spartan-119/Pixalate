The TalkingData AdTracking dataset was put up on Kaggle in 2017 as a competition by the Chinese company TalkingData, which processes three billion clicks every day, 90% of which are possibly illegitimate. [16]. Its method of detecting and preventing click fraud is to monitor and analyze users’ click journeys across their portfolios; IP addresses that generate many clicks but never install apps are flagged. A blacklist of IP addresses and devices is then created based on this information. The goal of the competition was to develop the best model for predicting whether a user would proceed to install an app after clicking on an ad and to distinguish fraudulent clicks from benign ones. The data provided were extensive, comprising a total of 203,694,359 real-time ad click records captured on a mobile platform, with an overall size of roughly 7 GB over four days. Table 4 illustrates the statistics of the TalkingData dataset.

# The Design Process

## Dataset Division

- **Global Evaluation Dataset**: 20% of the dataset, i.e., 20,000 records, will be reserved for global evaluation, encompassing both global training and global test sets.

## Simulation Setup

In this simulation, there are 5 clients, each receiving an IID (Independent and Identically Distributed) dataset portion. This results in each client obtaining 16,000 datapoints (80,000 / 5).

### Client-Side Processing

1. **Data Preprocessing**:
   - Each client will execute a script to preprocess their respective data.
   - The data will be split into training and test sets locally, with 20% allocated for testing.

2. **Local Training and Evaluation**:
   - Each client will train a machine learning model locally using their training set.
   - The model will be evaluated on the local test set, and the evaluation metrics will be printed.

3. **Model Parameter Transmission**:
   - The trained model parameters will be sent to the main server.

### Server-Side Aggregation

1. **Federated Averaging (FedAvg)**:
   - The main server will aggregate the received model parameters using the FedAvg algorithm.

2. **Global Evaluation**:
   - The aggregated model will be evaluated on the global test set.
   - The evaluation results will be printed.

## Steps to Run the Simulation

1. **Create the Federated Learning Environment Setup**:
   - Set up the environment required for federated learning.

2. **Setup the `fl_client` and `fl_model` Scripts**:
   - Develop and configure the client-side scripts (`fl_client` and `fl_model`) to handle data preprocessing, local training, and evaluation.

3. **Setup the `fl_strategy` and `fl_server` Scripts**:
   - Develop and configure the server-side scripts (`fl_strategy` and `fl_server`) to handle model aggregation and global evaluation.

4. **Run the Simulation**:
   - Execute the simulation by running the main script, which will coordinate the entire federated learning process.
