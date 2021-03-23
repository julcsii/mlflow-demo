import mlflow
from azureml.core import Workspace, Dataset

backend_config = {"USE_CONDA": False}

# Setup Azure ML studio workspace
ws = Workspace.from_config("azure_ml_config.json")

# Set tracking URL for mlflow
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

experiment_name = 'test_experiment_with_mlflow_2'
exp_id = mlflow.set_experiment(experiment_name)

local_env_run = mlflow.projects.run(uri=".",
                                    entry_point="train",
                                    parameters={"data_file":"data/winequality-red.csv"},
                                    backend = "azureml",
                                    use_conda=False,
                                    backend_config = backend_config, 
                                    )
# does not work with CLI