import numpy as np
import matplotlib.pyplot as plt

# Load the training and testing data
train_data = np.loadtxt('adaboost-train-24.txt')
test_data = np.loadtxt('adaboost-test-24.txt')

# Separate the features (X, Y) and labels (class)
X_train = train_data[:, 0:2]  # First two columns are features
y_train = train_data[:, 2]    # Last column is the label

X_test = test_data[:, 0:2]    # First two columns are features
y_test = test_data[:, 2]      # Last column is the label

# Plot for the training data
plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='red', marker='x', label='Negative')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', marker='x', label='Positive')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training Data')
plt.legend()

# Plot for the testing data
plt.figure(figsize=(8, 6))
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='red', marker='x', label='Negative')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='blue', marker='x', label='Positive')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Testing Data')
plt.legend()

plt.show()
