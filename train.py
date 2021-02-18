import numpy as np
import pandas as pd
import argparse
import mlflow

from pyspark.sql import SparkSession

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from hyperopt import hp, tpe, fmin, SparkTrials, STATUS_OK
 

def create_spark_session():
    spark = SparkSession \
        .builder \
        .appName("MLflow demo app") \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:0.1.0") \
        .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC") \
        .config("spark.executor.extraJavaOptions", "-Duser.timezone=UTC") \
        .config("spark.sql.session.timeZone", "UTC") \
        .config("spark.driver.host", "localhost") \
        .getOrCreate()
    return spark


def objective(space):
    pipeline = Pipeline([
            ('scaler', preprocessing.StandardScaler()), 
            ('rf', RandomForestClassifier(**space, random_state=random_state, n_jobs=8))
    ])
    pipeline.fit(X_train, y_train)
    score = cross_val_score(pipeline, X_train, y_train).mean()
    
    return {'loss': -score, 'status': STATUS_OK}


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train and tune RandomForestRegressor to predict red wine quality.')
    parser.add_argument('--data_path', type=str,
                        help='Path to data. CSV or delta.')
    parser.add_argument('--random_state', type=int,
                        help='Random seed')
    parser.add_argument('--test_size', type=float,
                        help='Ratio of the test data')
    parser.add_argument('--num_evals', type=int,
                        help='Number of models we want to evaluate')
    parser.add_argument('--use_hyperopt', dest='use_hyperopt', action='store_true')
                          
    args = parser.parse_args()

    # Track on MLflow managed by Databricks
    mlflow.set_tracking_uri("databricks")


    with mlflow.start_run() as run:
        # Parameters
        data_path = args.data_path
        random_state=args.random_state
        test_size=args.test_size
        num_evals = args.num_evals
        hyperopt = args.use_hyperopt

        # Load red wine data 
        if data_path.split(".")[-1]=="csv":
            data = pd.read_csv(data_path, sep=';')
        elif data_path.split(".")[-1]=="delta":
            spark = create_spark_session()
            df = spark.read.format("delta").load(data_path)
            data = df.toPandas()
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
        default_n_estimators = 10
        default_max_depth = 4
        default_max_features = 6
        default_criterion = "gini"

        if hyperopt:
            print("Hyper parameter tuning with Hyperopt..")
            trials = SparkTrials(6)
        
            # Declare hyperparameters to tune
            param_space = {
                'max_depth': hp.choice('max_depth', range(1,20)),
                'max_features': hp.choice('max_features', range(1,150)),
                'n_estimators': hp.choice('n_estimators', range(100,500)),
                'criterion': hp.choice('criterion', ["gini", "entropy"])}
            
            # Tune model using Hyperopt
            best = fmin(objective,
                space=param_space,
                algo=tpe.suggest,
                max_evals=num_evals,
                trials=trials)

            print(SparkTrials.results)

            best_max_depth = best["n_estimators"]
            best_max_depth = best["max_depth"]
            best_max_features = best["max_features"]
            best_criterion = best["criterion"]
        else:
            best_n_estimators = default_n_estimators
            best_max_depth = default_max_depth
            best_max_features = default_max_features
            best_criterion = default_criterion

        # Train model on entire training data
        pipeline = Pipeline([
            ('scaler', preprocessing.StandardScaler()), 
            ('rf', RandomForestClassifier(
                max_depth=best_max_depth, 
                n_estimators=best_n_estimators,
                max_features=best_max_features, 
                criterion=best_criterion,
                random_state=random_state, 
                n_jobs=8))
            ])
        pipeline.fit(X_train, y_train)

        # Evaluate on test data
        r2 = pipeline.score(X_test, y_test)
        
        # Log param and metric for the final model
        mlflow.log_param("n_estimators", best_n_estimators)
        mlflow.log_param("max_depth", best_max_depth)
        mlflow.log_param("max_features", best_max_features)
        mlflow.log_metric("loss", r2)

        # Log shap explanations
        mlflow.shap.log_explanation(pipeline.predict, X_train)

        # Infer signature
        signature = mlflow.models.infer_signature(X_train, pipeline.predict(X_train))

        # Register model
        mlflow.sklearn.log_model(
            sk_model=pipeline, 
            artifact_path="red_wine_quality", 
            registered_model_name="red_wine_quality-sklearn_RandomForestRegression_hyperopt",
            signature=signature)
        #todo: add input examples

