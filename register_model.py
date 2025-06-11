import os
import mlflow
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# ---------------------------------------
# ✅ Load from GitHub Actions environment
# ---------------------------------------
mlflow_tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
databricks_token = os.environ["DATABRICKS_TOKEN"]

# Required for Databricks MLflow authentication
os.environ["DATABRICKS_HOST"] = mlflow_tracking_uri
os.environ["DATABRICKS_TOKEN"] = databricks_token

# ---------------------------------------
# ✅ Set tracking URI and experiment
# ---------------------------------------
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("/Users/michael.wallendjack@effem.com/MLflow Tracing Tutorial")

# ---------------------------------------
# 🧪 Sample regression training
# ---------------------------------------
np.random.seed(22)
X_sqft = pd.DataFrame({"sqft": np.random.randint(500, 3500, 100)})
y = X_sqft["sqft"] * 100 + np.random.normal(0, 20000, 100)

X_train, X_test, y_train, y_test = train_test_split(X_sqft, y, test_size=0.2, random_state=22)

# ---------------------------------------
# 📦 Start tracking run
# ---------------------------------------
with mlflow.start_run() as run:
    run_id = run.info.run_id

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    input_example = X_test.iloc[:5]
    signature = infer_signature(X_test, predictions)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
        signature=signature
    )

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    print(f"✅ Run complete. RMSE: {rmse:.2f}, R2: {r2:.4f}")

# ---------------------------------------
# 📚 Register model in Databricks registry
# ---------------------------------------
model_uri = f"runs:/{run_id}/model"
model_name = "my_first_registered_model"
client = MlflowClient()

try:
    client.get_registered_model(model_name)
except MlflowException:
    client.create_registered_model(model_name)

version = client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=run_id
)

client.transition_model_version_stage(
    name=model_name,
    version=version.version,
    stage="Production"
)

print(f"✅ Model '{model_name}' version {version.version} registered and transitioned to Production.")
