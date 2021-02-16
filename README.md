# mlflow-demo

## Setup
Activate/create new conda env:
```
conda create -n mlflow-demo python=3.8
conda activate mlflow-demo
```

Install Mlflow:
```
pip install mlflow==1.13.0
```

Export Databricks environment variables:
```
export DATABRICKS_HOST="..."
export DATABRICKS_TOKEN="..."
```

## Run training 

Locally
```
mlflow run . -e train --experiment-id 2712754460055773 -P data_file=data/winequality-red.csv
```

On Databicks
```
mlflow run https://github.com/julcsii/mlflow-demo.git -b databricks --backend-config new_cluster_spec.json -P data_file=/delta/winequality.delta
```

On Azure ML - todo