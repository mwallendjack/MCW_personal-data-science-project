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

# Set your experiment
mlflow.set_experiment("my_second_mlflow_experiment")

# Load or create data (replace this with your actual dataset)
# Example: Predict house price based on square footage
np.random.seed(22)
X_sqft = pd.DataFrame({"sqft": np.random.randint(500, 3500, 100)})
y = X_sqft["sqft"] * 100 + np.random.normal(0, 20000, 100)

X_train, X_test, y_train, y_test = train_test_split(X_sqft, y, test_size=0.2, random_state=22)

# Start tracking run
with mlflow.start_run() as run:
    run_id = run.info.run_id

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # Infer input/output schema
    input_example = X_test.iloc[:5]
    signature = infer_signature(X_test, predictions)

    # Log model and metrics
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

    print(f"Run complete. RMSE: {rmse:.2f}, R2: {r2:.4f}")

# Register to Model Registry
model_uri = f"runs:/{run_id}/model"
model_name = "my_first_registered_model"
client = MlflowClient()

# Create model name if it doesn’t exist
try:
    client.get_registered_model(model_name)
except MlflowException:
    client.create_registered_model(model_name)

# Register new version
version = client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=run_id
)

# Promote to Production
client.transition_model_version_stage(
    name=model_name,
    version=version.version,
    stage="Production"
)

print(f"✅ Model '{model_name}' version {version.version} registered and moved to Production.")
