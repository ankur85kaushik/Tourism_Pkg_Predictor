# for data manipulation
import pandas as pd
import mlflow
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# for model serialization
import joblib
import os

Xtrain_path = "tourism_project/data/Xtrain.csv"
Xtest_path = "tourism_project/data/Xtest.csv"
ytrain_path = "tourism_project/data/ytrain.csv"
ytest_path = "tourism_project/data/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

ytrain = ytrain['ProdTaken'] if 'ProdTaken' in ytrain.columns else ytrain.squeeze()
ytest = ytest['ProdTaken'] if 'ProdTaken' in ytest.columns else ytest.squeeze()

# Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define base XGBoost model directly
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Fixed hyperparameter grid values
param_grid = {
    'n_estimators':[50, 75, 100],
    'max_depth':[2, 3, 4],
    'colsample_bytree': [0.4, 0.5, 0.6],
    'colsample_bylevel': [0.4, 0.5, 0.6],
    'learning_rate': [0.01, 0.05, 0.1],
    'reg_lambda': [0.4, 0.5, 0.6],
}

# Start MLflow parent run
with mlflow.start_run():
    # GridSearch directly on the model
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate nested run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in the main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

print("Training complete! Metrics successfully logged to MLflow.")

