import flwr as fl
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Define evaluation function for the server
def evaluate_server(server_round: int, parameters: Dict[str, any], config: Dict[str, any]) -> Tuple[float, Dict[str, any]]:
    """Custom evaluation function to evaluate the global Random Forest model on the server."""
    
    # Load your validation dataset (this could be from a file or a database)
    validation_data = pd.read_csv("path/to/your/validation_data.csv")
    X_val = validation_data.drop('label', axis=1)
    y_val = validation_data['label']
    
    # Load the global model parameters into your Random Forest model
    global_model = RandomForestClassifier()  # Initialize your model
    global_model.set_params(**parameters)    # Set the global parameters
    global_model.fit(X_val, y_val)           # Fit on validation data
    
    # Make predictions and calculate accuracy
    y_pred = global_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)  # Calculate accuracy

    return accuracy, {}

# Define the server strategy (FedAvg) with custom evaluation
strategy = fl.server.strategy.FedAvg(
    evaluate_fn=evaluate_server,  # Server-side evaluation function
    fraction_fit=1.0,             # Use all available clients for training
    min_fit_clients=5,            # Minimum number of clients to participate in training
    min_eval_clients=5,           # Minimum number of clients for evaluation
    min_available_clients=5,      # Minimum number of clients required
    num_rounds=10                 # Number of federated learning rounds
)

# Start the Flower server
if __name__ == "__main__":
    fl.server.start_server("[::]:8080", strategy=strategy)
