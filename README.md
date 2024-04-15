import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

file_path = 'E:\Assignment 1(445)\AirQualityUCI.xlsx'

# Load the Excel file into a DataFrame
data = pd.read_excel(file_path)
data.head()
# Convert timestamp columns to numeric
timestamp_cols = data.select_dtypes(include=['datetime64']).columns
for col in timestamp_cols:
    data[col] = pd.to_numeric(data[col])

# Drop object columns
object_cols = data.select_dtypes(include=['object']).columns
data = data.drop(object_cols, axis=1)

print(data.head())
import numpy as np
import copy
import math
import matplotlib.pyplot as plt

def compute_cost(features, targets, weights, bias, lambd):
    m = features.shape[0]
    cost = np.sum(np.abs(np.dot(features, weights) + bias - targets)) + lambd * np.sum(weights**2)  # Regularized cost
    return cost / m

def compute_gradient(features, targets, weights, bias, lambd):
    m = features.shape[0]
    predictions = np.dot(features, weights) + bias
    errors = predictions - targets

    dw = np.dot(features.T, np.sign(errors)) / m + 2 * lambd * weights  # Regularized gradient
    db = np.sum(np.sign(errors)) / m

    return dw, db


def gradient_descent(features, targets, weights_init, bias_init, alpha, num_iters, lambd):
    weights = copy.deepcopy(weights_init)
    bias = bias_init
    m = features.shape[0]
    J_history = []

    print_interval = max(1, int(num_iters / 10) + (num_iters % 10 > 0))

    for i in range(num_iters):
        dw, db = compute_gradient(features, targets, weights, bias, lambd)
        weights -= alpha * dw
        bias -= alpha * db
        cost = compute_cost(features, targets, weights, bias, lambd)
        J_history.append(cost)

        if i % print_interval == 0:
            print("Iteration {:4}: Cost {:0.2e}".format(i, cost))

    return weights, bias, J_history

# Assuming 'data' has been defined and processed as in your original code

# Split data into training and testing sets (75% training, 25% testing)
split_index = int(0.75 * len(data))
features = np.array(data.iloc[:, 0:11])
targets = np.array(data.iloc[:, 11:])

features_train, features_test = features[:split_index], features[split_index:]
targets_train, targets_test = targets[:split_index], targets[split_index:]

# Feature Scaling
std_features_train = np.std(features_train, axis=0)
std_features_test = np.std(features_test, axis=0)
std_features_train[std_features_train == 0] = 1
std_features_test[std_features_test == 0] = 1
features_train_scaled = (features_train - np.mean(features_train, axis=0)) / std_features_train
features_test_scaled = (features_test - np.mean(features_test, axis=0)) / std_features_test
features_train_scaled = np.nan_to_num(features_train_scaled)
features_test_scaled = np.nan_to_num(features_test_scaled)

# Initialize parameters
weights_init = np.zeros((features_train_scaled.shape[1], targets_train.shape[1]))
bias_init = np.zeros(targets_train.shape[1])

# Gradient Descent settings
iterations = 5000
learning_rate = 0.0001
lambd = 0.01  # Regularization parameter

# Run Gradient Descent for each target column
weights_final_list = []
bias_final_list = []
mae_list = []

for i in range(targets_train.shape[1]):
    print(f"Training for target column {i+1}")
    weights_final, bias_final, J_hist = gradient_descent(features_train_scaled, targets_train[:, i], weights_init[:, i], bias_init[i], learning_rate, iterations, lambd)
    weights_final_list.append(weights_final)
    bias_final_list.append(bias_final)

    targets_pred = np.dot(features_test_scaled, weights_final) + bias_final
    mae = np.mean(np.abs(targets_test[:, i] - targets_pred))
    mae_list.append(mae)
    print(f"Test Mean Absolute Error (Column {i+1}): ", mae)

    print(f"(weights, bias) found by gradient descent (Column {i+1}):")
    print("weights:", np.array_str(weights_final, precision=4, suppress_small=True))
    print("bias:", bias_final)

    fig, ax1 = plt.subplots(1, 1, constrained_layout=True, figsize=(12, 4))
    ax1.plot(range(iterations), J_hist)
    ax1.set_title(f"Cost vs. Iteration (Column {i+1})")
    ax1.set_ylabel('Cost')
    ax1.set_xlabel('Iteration')
    plt.show()

print("Mean Absolute Errors:", mae_list)
