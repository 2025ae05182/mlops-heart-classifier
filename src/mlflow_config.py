import os
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

def setup_mlflow(experiment_name: str):
    """
    Configure MLflow tracking and experiment with a project-level folder.
    Ensures both DB and artifacts live under ./mlflow_tracking/
    """

    # Project-level tracking folder
    tracking_dir = os.path.abspath("./mlflow_tracking")

    # Ensure folder exists
    os.makedirs(tracking_dir, exist_ok=True)

    # Set backend DB
    mlflow.set_tracking_uri(f"sqlite:///{tracking_dir}/mlflow.db")

    # Check if experiment already exists
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        # Create experiment with explicit artifact location
        client.create_experiment(
            name=experiment_name,
            artifact_location=Path(tracking_dir, "artifacts").as_uri()
        )

    # Set experiment for current run context
    mlflow.set_experiment(experiment_name)

    import logging
    logging.info(f"MLflow tracking set to: {tracking_dir}")
    logging.info(f"Experiment: {experiment_name}")

    logging.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
