import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import time
import json

def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def run_linear_regression():
    print("\nLinear Regression")
    print("-" * 20)
    
    # Load data from the same JSON file
    data = load_data("data.json")
    x_data = np.array(data["linear"]["x"])
    y = np.array(data["linear"]["y"])
    
    # Reshape X into a 2D array with 6 features
    m = data["linear"]["m"]
    n = data["linear"]["n"]
    X = x_data.reshape(m, n)
    
    print(f"Number of examples (m): {m}")
    print(f"Number of features (n): {n}")
    print(f"Total X values length: {m * n}")
    print(f"Total Y values length: {len(y)}")
    
    # Load test data
    x_test_data = np.array(data["linear"]["x_test"])
    y_test = np.array(data["linear"]["y_test"])
    X_test = x_test_data.reshape(-1, n)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    start_time = time.time()
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Print training time
    elapsed = time.time() - start_time
    print(f"Elapsed: {elapsed:.2f}s")
    
    # Print weights (including bias)
    weights = np.concatenate(([model.intercept_], model.coef_)).reshape(-1, 1)
    print(f"Final weights: {weights.tolist()}")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    print(f"\nPredictions: {y_pred.reshape(-1, 1).tolist()[:10]} ...")
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"\nResults:")
    print(f"Total test samples: {len(y_test)}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root MSE: {rmse:.4f}")

def run_logistic_regression():
    print("\nLogistic Regression")
    print("-" * 20)
    
    # Load data
    data = load_data("data.json")
    x_data = np.array(data["logistic"]["x"])
    y = np.array(data["logistic"]["y"])
    
    # Reshape X into a 2D array with 2 features
    m = data["logistic"]["m"]
    n = data["logistic"]["n"]
    X = x_data.reshape(m, n)
    
    # Load test data
    x_test_data = np.array(data["logistic"]["x_test"])
    y_test = np.array(data["logistic"]["y_test"])
    X_test = x_test_data.reshape(-1, n)
    
    print(f"Number of examples (m): {m}")
    print(f"Number of features (n): {n}")
    print(f"Total X values length: {m * n}")
    print(f"Total Y values length: {len(y)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    start_time = time.time()
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    
    # Print training time
    elapsed = time.time() - start_time
    print(f"Elapsed: {elapsed:.2f}s")
    
    # Print weights (including bias)
    weights = np.concatenate(([model.intercept_[0]], model.coef_[0])).reshape(-1, 1)
    print(f"Final weights: {weights.tolist()}")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    print(f"\nPredictions: {y_pred.tolist()}")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nResults:")
    print(f"Total samples: {len(y_test)}")
    print(f"Correct predictions: {sum(y_pred == y_test)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    run_linear_regression()
    run_logistic_regression()
