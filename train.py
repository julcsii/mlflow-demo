import numpy as np
import pandas as pd
import databricks.koalas as ks
import argparse
import mlflow
 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from hyperopt import hp, tpe, fmin, SparkTrials, STATUS_OK
 

def objective(space):
    pipeline = Pipeline([
            ('scaler', preprocessing.StandardScaler()), 
            ('rf', RandomForestRegressor(**space, random_state=random_state))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    
    return {'loss': rmse, 'status': STATUS_OK}


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train and tune RandomForestRegressor to predict red wine quality.')
    parser.add_argument('data_path', type=str,
                        help='Path to data. CSV or delta.')
    parser.add_argument('random_state', type=int,
                        help='Random seed')
    parser.add_argument('test_size', type=float,
                        help='Ratio of the test data')
    parser.add_argument('num_evals', type=int,
                        help='Number of models we want to evaluate')
                          
    args = parser.parse_args()

    # Track on MLflow managed by Databricks
    mlflow.set_tracking_uri("databricks")

    with mlflow.start_run() as run:
        # Parameters
        data_path = args.data_path
        random_state=args.random_state
        test_size=args.test_size
        num_evals = args.num_evals

        # Load red wine data 
        if data_path.split(".")[-1]=="csv":
            data = pd.read_csv(data_path, sep=';')
        if data_path.split(".")[-1]=="delta":
            data_ks = ks.read_delta(data_path)
            data = data_ks.to_pandas()
        else:
            print("Only CSV or Delta files are supported")
        
        # Split data into training and test sets
        y = data.quality
        X = data.drop('quality', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=test_size, 
                                                            random_state=random_state, 
                                                            stratify=y)
        # todo: log data
        
        # Declare hyperparameters to tune
        param_space= {
            "max_depth": hp.randint("max_depth", 2, 10),
            "max_features": hp.choice("max_features", ["auto", "sqrt", "log2"])
        }
        
        # Tune model using Hyperopt
        trials = SparkTrials()
        best = fmin(objective,
            space=param_space,
            algo=tpe.suggest,
            max_evals=num_evals,
            trials=trials)

        print(best)
        print(SparkTrials.results)

        best_max_depth = best["max_depth"]
        best_max_features = best["max_features"]

        # Train model on entire training data
        pipeline = Pipeline([
            ('scaler', preprocessing.StandardScaler()), 
            ('rf', RandomForestRegressor(max_depth=best_max_depth, max_features=best_max_features, random_state=random_state))
            ])
        pipeline.fit(X_train, y_train)

        # Evaluate on test data
        r2 = pipeline.score(X_test, y_test)
        
        # Log param and metric for the final model
        mlflow.log_param("max_depth", best_max_depth)
        mlflow.log_param("max_features", best_max_features)
        mlflow.log_metric("loss", r2)

        # Log shap explanations
        mlflow.shap.log_explanation(pipeline.predict_proba, X)

        signature = mlflow.models.infer_signature(X_train, pipeline.predict(X_train))

        #todo: add input examples

        # Register model
        mlflow.sklearn.log_model(
            sk_model=pipeline, 
            artifact_path="red_wine_quality", 
            registered_model_name="red_wine_quality-sklearn_RandomForestRegression_hyperopt",
            signature=signature)

