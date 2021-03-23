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

Setup authentication to Azure ML studio.
1. Download config from UI
2. Save to repository.


## Use case

Data: [Red wine quality](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/)

ML Task: Random Forest based Regression to predict the quality of red wine given some attributes.


## Run training 

### Locally 
Train
```
mlflow run . -e train --experiment-id 2712754460055773 -P data_file=data/winequality-red.csv
```


Train with hyperparam tuning
```
mlflow run . -e tune --experiment-id 2712754460055773 -P data_file=data/winequality-red.csv
```

Track on Azure ML:


### On Databicks

Make sure to have any additional dependencies on the dbfs (that is not in your specified runtime).

Train

CSV:
```
mlflow run https://github.com/julcsii/mlflow-demo.git -e train --experiment-id 2712754460055773 -b databricks --backend-config new_cluster_spec.json -P data_file=/dbfs/FileStore/tables/red_wine/winequality_red.csv
```

Delta:
```
mlflow run https://github.com/julcsii/mlflow-demo.git -e train --experiment-id 2712754460055773 -b databricks --backend-config new_cluster_spec.json -P data_file=/dbfs/FileStore/tables/winequality.delta
```


Train with hyperparam tuning
```
mlflow run https://github.com/julcsii/mlflow-demo.git -e tune --experiment-id 2712754460055773 -b databricks --backend-config new_cluster_spec.json -P data_file=/dbfs/FileStore/tables/red_wine/winequality_red.csv
```

