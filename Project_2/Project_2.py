import os as os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap

#Constants 
n = 0.1



class Data_Set:
    
    def __init__(self, coordinates, category, weights): 
        self.coordinates = coordinates
        self.weights = weights.reshape(len(weights), 1)
        self.category = category.reshape(len(category), 1)
        #print(self.coordinates.shape)
        #print(self.weights.shape)
        #print(self.category.shape)

    def select(self, selection): 
        
        selection = selection.reshape(len(self.category))
        
        print(selection)
        coordinates = self.coordinates[selection == 1, :]
        weights = self.weights[selection == 1]
        category = self.category[selection == 1]

        return Data_Set(coordinates, category, weights)

class Weak_Classifier: 
    
    def __init__(self):
        self.slope = 1 
        self.offset = 0
        self.alpha = 0
    
    def train(self, dataset):
        
        weighted = dataset.coordinates*dataset.weights
        # Select all of the positive category points, them and sum them and normalize. 
        positive = weighted[(dataset.category.reshape(len(dataset.category)) == 1), :].sum(axis = 0) / dataset.weights[(dataset.category.reshape(len(dataset.category)) == 1)].sum(axis = 0)
        
        # Select all of the negative category points, them and sum them and normalize. 
        negative = weighted[(dataset.category.reshape(len(dataset.category)) == -1), :].sum(axis = 0) / dataset.weights[(dataset.category.reshape(len(dataset.category)) == -1)].sum(axis = 0)
        
        #print(positive)
        #print(negative)
        
        #Calculate the slope of the dividing line 
        slope = (-1) / (positive[1] - negative[1]) / (positive[0] - negative[0])
        #Calculate the midpoint between the two points 
        midpoint = np.array([((positive[0] + negative[0]) / 2), ((positive[1] + negative[1]) / 2)])
        
        #Calculate the "C" value in the expression "Y = MX + C" or the offset in this classifier
        C = midpoint[1] - slope*midpoint[0]
    
        
        #print(slope)
        #print(midpoint)
        #print(C)
        
        #Update the Classifier values: 
        self.slope = slope 
        self.offset = C 
        
    def classify(self, dataset): 
        
        output = np.sign(dataset.coordinates[:, 1] - self.slope*dataset.coordinates[:, 1] - self.offset) 
        
        return output.reshape(dataset.category.shape)
    
    
    def accuracy(self, dataset): 
        
        #Get the classifiers guesses on the data:
        guesses = self.classify(dataset)
        #print(guesses)
        #print(guesses.shape)
        
        #TODO: Add in the calculation for ai here and set it to self...
        
        #Select the errors and weight them: 
        error = dataset.weights[guesses == dataset.category].sum()
        print("Percentage error: ", error, "%")
        
        return error

#class AdaBoost_Classifier: 
    



# Load training data
train_data = np.loadtxt('adaboost-train-24.txt')
train_coordinates = train_data[:, :-1]  # Feature columns
train_category = train_data[:, -1]   # Label column
train_weights = (1/len(train_category))*np.ones(len(train_category))

train_dataset = Data_Set(train_coordinates, train_category, train_weights)

# Load testing data
test_data = np.loadtxt('adaboost-test-24.txt')
test_coordinates = test_data[:, :-1]  # Feature columns
test_category = test_data[:, -1]   # Label column
test_weights = (1/len(test_category))*np.ones(len(test_category))

test_dataset = Data_Set(test_coordinates, test_category, test_weights)




#Generate an initial Selection Array
rand_array = np.random.rand(len(train_category))
selection_array = np.where(rand_array > 0.95, 1, 0)

temp_train = train_dataset.select(selection_array)


weak = Weak_Classifier()
weak.train(temp_train)
error = weak.accuracy(train_dataset)
guesses = weak.classify(train_dataset)
new_weights = train_dataset.weights * (np.where((guesses == train_dataset.category), (1/(2*(1-error))), (1/(2*(error)))))
print(new_weights)
# Load testing data
test_data = np.loadtxt('adaboost-test-24.txt')
X_test = test_data[:, :-1]    # Feature columns
y_test = test_data[:, -1]     # Label column


print(adaboost_clf.score(X_train, y_train))
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
    plt.scatter(X[y == Z][:, 0], X[y == Z][:, 1], color='red', marker='+', label='Negative')
    plt.scatter(X[y == -Z][:, 0], X[y == -Z][:, 1], color='blue', marker='.', label='Positive')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

# Plot training data
#plot_decision_boundary(X_train, y_train, 'AdaBoost Classifier Decision Boundary (Training Data)')

# Plot test data
plot_decision_boundary(X_test, y_test, 'AdaBoost Classifier Decision Boundary (Test Data)')

plt.show()
#Save the plots to a file
os.makedirs("./Output_Plots", exist_ok=True)
plt.savefig("./Output_Plots/Plot_1.png")