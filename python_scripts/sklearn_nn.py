import json
import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

def run(epochs, learning_rate, data_field='linear'):
    # --- Data Preparation ---
    with open('data.json', 'r') as file:
        data = json.load(file)

    m = data[data_field]['m']
    n = data[data_field]['n']
    m_test = data[data_field]['m_test']

    # Training data
    x_train = np.array(data[data_field]['x'], dtype=np.float32).reshape(m, n)
    y_train = np.array(data[data_field]['y'], dtype=np.float32).reshape(m, 1)

    # Test data
    x_test = np.array(data[data_field]['x_test'], dtype=np.float32).reshape(m_test, n)
    y_test = np.array(data[data_field]['y_test'], dtype=np.float32).reshape(m_test, 1)

    print("Starting training...")

    if data_field == 'linear':
        # Regression neural network
        net = MLPRegressor(hidden_layer_sizes=(64, 64), 
                           max_iter=epochs, 
                           learning_rate_init=learning_rate, 
                           random_state=42)

        net.fit(x_train, y_train.ravel())
        y_pred = net.predict(x_test)

        mse_test = mean_squared_error(y_test, y_pred)
        mae_test = mean_absolute_error(y_test, y_pred)

        print(f"\nFinal Results after {epochs} epochs and learning rate {learning_rate}:")
        print(f"Test MSE: {mse_test:.4f}")
        print(f"Test MAE: {mae_test:.4f}")

    elif data_field == 'logistic':
        # Classification neural network
        net = MLPClassifier(hidden_layer_sizes=(64, 64), 
                            max_iter=epochs, 
                            learning_rate_init=learning_rate, 
                            random_state=42)

        net.fit(x_train, y_train.ravel())
        y_pred = net.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred) * 100
        correct = np.sum(y_pred == y_test.ravel())

        print(f"\nFinal Results after {epochs} epochs and learning rate {learning_rate}:")
        print(f"Test Accuracy: {accuracy:.2f}% ({correct} out of {m_test})")

    else:
        # Classification neural network
        net = MLPClassifier(hidden_layer_sizes=(64, 64), 
                            max_iter=epochs, 
                            learning_rate_init=learning_rate, 
                            random_state=42)

        net.fit(x_train, y_train.ravel())
        y_pred = net.predict(x_test)
        print(y_pred)

        accuracy = accuracy_score(y_test, y_pred) * 100
        correct = np.sum(y_pred == y_test.ravel())

        print(f"\nFinal Results after {epochs} epochs and learning rate {learning_rate}:")
        print(f"Test Accuracy: {accuracy:.2f}% ({correct} out of {m_test})")

if __name__ == "__main__":
    run(epochs=1000, learning_rate=0.0001, data_field='neural_network')