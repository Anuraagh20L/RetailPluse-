import mlflow

with mlflow.start_run(run_name="Week1_Baseline_Models"):

    mlflow.log_metric("Prophet_MAE", 23105.08)
    mlflow.log_metric("Prophet_RMSE", 31294.97)

    mlflow.log_metric("LSTM_MAE", 13754.61)
    mlflow.log_metric("LSTM_RMSE", 21568.59)

    print("Metrics logged successfully!")