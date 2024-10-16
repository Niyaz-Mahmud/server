import flwr as fl
from typing import Dict, Tuple

# Define evaluation function for the server
def evaluate_server(server_round: int, parameters: Dict[str, any], config: Dict[str, any]) -> Tuple[float, Dict[str, any]]:
    """Custom evaluation function to evaluate the global Random Forest model on the server."""
    accuracy = 1.0  # You can modify this to include server-side evaluation logic if needed
    return accuracy, {}

# Define the server strategy (FedAvg) with custom evaluation
strategy = fl.server.strategy.FedAvg(
    evaluate_fn=evaluate_server,  # Server-side evaluation function
    fraction_fit=1.0,             # Use all available clients for training
    fraction_eval=1.0,            # Use all available clients for evaluation
    min_fit_clients=5,            # Minimum number of clients to participate in training
    min_eval_clients=5,           # Minimum number of clients for evaluation
    min_available_clients=5,      # Minimum number of clients required
    num_rounds=10                 # Number of federated learning rounds
)

# Start the Flower server
if __name__ == "__main__":
    fl.server.start_server("[::]:8080", strategy=strategy)
