import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap

# Load training data
train_data = np.loadtxt('adaboost-train-24.txt')
X_train = train_data[:, :-1]  # Feature columns
y_train = train_data[:, -1]   # Label column

# Load testing data
test_data = np.loadtxt('adaboost-test-24.txt')
X_test = test_data[:, :-1]    # Feature columns
y_test = test_data[:, -1]     # Label column

# Create and train AdaBoost classifier
adaboost_clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                                  n_estimators=50, algorithm='SAMME')
adaboost_clf.fit(X_train, y_train)

# Plotting decision boundary function
def plot_decision_boundary(X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    Z = adaboost_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create a color map
    cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF'])

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background)
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', marker='x', label='Negative')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='x', label='Positive')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

# Plot training data
plot_decision_boundary(X_train, y_train, 'AdaBoost Classifier Decision Boundary (Training Data)')

# Plot test data
plot_decision_boundary(X_test, y_test, 'AdaBoost Classifier Decision Boundary (Test Data)')

plt.show()
