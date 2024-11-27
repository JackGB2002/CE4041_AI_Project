import numpy as np
import matplotlib.pyplot as plt

class WeightedWeakLinear:
    def __init__(self):
        self.threshold = None
        self.polarity = None
        self.alpha = None
        self.feature_index = None  # Store the feature index used for the threshold

    def fit(self, X, y, weights):
        n_samples, n_features = X.shape
        best_loss = float('inf')

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    predictions[polarity * X[:, feature_index] < polarity * threshold] = -1
                    
                    # Calculate weighted error
                    error = weights[(predictions != y)].sum()
                    
                    # If error is less than best_loss, update best parameters
                    if error < best_loss:
                        best_loss = error
                        self.threshold = threshold
                        self.polarity = polarity
                        self.feature_index = feature_index  # Store the feature index

        # Calculate alpha (classifier weight)
        self.alpha = 0.5 * np.log((1 - best_loss) / (best_loss + 1e-10))

    def predict(self, X):
        predictions = np.ones(X.shape[0])
        # Use the threshold to classify based on the polarity
        if self.polarity == 1:
            predictions[X[:, self.feature_index] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_index] >= self.threshold] = -1
        return predictions

class AdaBoost:
    def __init__(self, n_classifiers):
        self.n_classifiers = n_classifiers
        self.classifiers = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        weights = np.ones(n_samples) / n_samples  # Initialize weights

        for _ in range(self.n_classifiers):
            classifier = WeightedWeakLinear()
            classifier.fit(X, y, weights)
            predictions = classifier.predict(X)

            # Calculate the classifier's error
            error = weights[(predictions != y)].sum()

            # Update weights
            weights *= np.exp(-classifier.alpha * y * predictions)
            weights /= weights.sum()  # Normalize weights

            # Store classifier and its weight
            self.classifiers.append(classifier)
            self.alphas.append(classifier.alpha)

    def predict(self, X):
        # Aggregate predictions from all classifiers
        final_predictions = np.zeros(X.shape[0])
        for alpha, classifier in zip(self.alphas, self.classifiers):
            final_predictions += alpha * classifier.predict(X)
        return np.sign(final_predictions)

# Function to load data
def load_data(train_file, test_file):
    train_data = np.loadtxt(train_file)
    test_data = np.loadtxt(test_file)

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    y_train[y_train == 0] = -1  # Convert to -1/+1
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    y_test[y_test == 0] = -1  # Convert to -1/+1

    return X_train, y_train, X_test, y_test

# Function to plot scatter of the data
def plot_scatter(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1', marker='o')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class +1', marker='o')
    plt.title('Scatter Plot of Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid()
    plt.show()

# Function to plot decision boundary
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
    plt.title('AdaBoost Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Function to evaluate model performance
def evaluate_model(X_train, y_train, X_test, y_test, n_classifiers):
    model = AdaBoost(n_classifiers)
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)

    print(f'Training Accuracy with {n_classifiers} classifiers: {train_accuracy:.2f}')
    print(f'Testing Accuracy with {n_classifiers} classifiers: {test_accuracy:.2f}')

    return train_accuracy, test_accuracy

# Main execution
if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, y_test = load_data('C:/Users/Ethan/Downloads/Artificial Intelligence/CE4041_AI_Project/Project_2/adaboost-train-24.txt',
                                                  'C:/Users/Ethan/Downloads/Artificial Intelligence/CE4041_AI_Project/Project_2/adaboost-test-24.txt')

    # Plot scatter of training data
    plot_scatter(X_train, y_train)

    # Evaluate model with varying number of classifiers
    n_values = range(1, 51)  # Testing with 1 to 50 classifiers
    train_accuracies = []
    test_accuracies = []

    for n in n_values:
        train_acc, test_acc = evaluate_model(X_train, y_train, X_test, y_test, n)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    # Plotting the accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(n_values, train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(n_values, test_accuracies, label='Testing Accuracy', marker='o')
    plt.title('Accuracy vs Number of Classifiers')
    plt.xlabel('Number of Classifiers')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

    # Determine maximum accuracy and number of classifiers needed for 100% training accuracy
    n_train_100 = next((n for n, acc in zip(n_values, train_accuracies) if acc == 1.0), None)
    max_test_accuracy = max(test_accuracies)
    n_test_max_accuracy = test_accuracies.index(max_test_accuracy) + 1  # +1 for index adjustment

    print(f'Number of classifiers for 100% training accuracy: {n_train_100}')
    print(f'Maximum achievable accuracy on testing set: {max_test_accuracy:.2f}')
    print(f'Can testing set achieve 100% accuracy? {"Yes" if max_test_accuracy == 1.0 else "No"}')
    print(f'Number of weak learners for maximum testing accuracy: {n_test_max_accuracy}')