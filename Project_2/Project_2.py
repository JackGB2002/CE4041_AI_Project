import os as os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#Constants 
n = 0.1 #NOTE Not sure if this is required 
N = 50#N samples 

plt.figure(figsize=(10, 10))

class Data_Set:
    
    def __init__(self, coordinates, category, weights): 
        self.coordinates = coordinates
        self.weights = weights.reshape(len(weights), 1)
        self.category = category.reshape(len(category), 1)
        self.summed_weights = np.zeros(self.weights.shape)
        
        self.weight_summation()
        #print(self.coordinates.shape)
        #print(self.weights.shape)
        #print(self.category.shape)

    def select(self, selection): 
        
        #Just create a duplicate array
        if selection is None: 
            coordinates = self.coordinates
            weights = self.weights
            category = self.category

            return Data_Set(coordinates, category, weights)
        
        selection = selection.reshape(len(self.category))
        
        #print(selection)
        coordinates = self.coordinates[selection == 1, :]
        weights = self.weights[selection == 1]
        category = self.category[selection == 1]

        return Data_Set(coordinates, category, weights)

    def plot(self, category, lines):
        
        
        plt.scatter(self.coordinates[self.category[:,0] == 1, 0],  self.coordinates[self.category[:,0] == 1, 1], color='red', marker='.', label='Positive')
        plt.scatter(self.coordinates[self.category[:,0] == -1, 0], self.coordinates[self.category[:,0] == -1,1], color='blue', marker='.', label='Negative')
        plt.title("Data_Set Plot")
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.legend()

        plt.xlim(-2 , 2)
        plt.ylim(-2 , 2)

        plt.show()
        #Save the plots to a file
        os.makedirs("./Output_Plots", exist_ok=True)
        plt.savefig("./Output_Plots/Data_Set.png")
        plt.figure(figsize=(10, 10))

    def weight_summation(self):
        indices = np.arange(len(self.summed_weights))
        
        for index in indices: 
            if(index == 0):
                self.summed_weights[index] = self.weights[0]
                
            else: 
                self.summed_weights[index] = self.summed_weights[index - 1] + self.weights[index]
        
    def select_weighted_random(self, n_elements): 
                
        selection_matrix = np.zeros(self.weights.shape)
        indecies = []
        
        for element in range(n_elements): 
            
            random = np.random.rand() * self.summed_weights[-1]
            
            for weight in range(len(self.weights)): 
               
               
                if(self.summed_weights[weight] > random): 
                    
                    
                    
                    offset = 0
                    while(selection_matrix[(weight + offset)%(len(self.weights))] == 1):
                        offset = offset + 1
                    
                    indecies.append((weight + offset)%(len(self.weights)))
                    
                    selection_matrix[(weight + offset)%(len(self.weights))] = 1
                    break
        
        #print(indecies)
        return self.select(selection_matrix)     
                
class Weak_Classifier: 
    
    def __init__(self):
        self.slope = 1 
        self.offset = 0
        self.alpha = 0
        self.normal = -1
    
    def train(self, dataset):
        
        
        #TODO: Re look at this and ensure that it is producing an average location of the points and not a vector which would mess up the offset term
        weighted = dataset.coordinates*dataset.weights
        # Select all of the positive category points, them and sum them and normalize. 
        positive = weighted[(dataset.category.reshape(len(dataset.category)) == 1), :].sum(axis = 0) / dataset.weights[(dataset.category.reshape(len(dataset.category)) == 1)].sum(axis = 0)
        
        # Select all of the negative category points, them and sum them and normalize. 
        negative = weighted[(dataset.category.reshape(len(dataset.category)) == -1), :].sum(axis = 0) / dataset.weights[(dataset.category.reshape(len(dataset.category)) == -1)].sum(axis = 0)
        
        #print(positive)
        #print(negative)
        
        #Calculate the slope of the dividing line 
        slope = (-1) / ((positive[1] - negative[1]) / (positive[0] - negative[0]))
      

        #Calculate the midpoint between the two points 
        midpoint = np.array([((positive[0] + negative[0]) / 2), ((positive[1] + negative[1]) / 2)])
        
        #Calculate the "C" value in the expression "Y = MX + C" or the offset in this classifier
        C = midpoint[1] - slope*midpoint[0]

        normal = np.sign(positive[1] - slope*positive[0] - C)
    
        #Update the Classifier values: 
        self.slope = slope 
        self.offset = C 
        self.normal = normal
    
        """
        x1 = -2
        x2 = 2
        y1 = slope*x1 + C
        y2 = slope*x2 + C
    
        guesses = self.classify(dataset)
        
        #dataset.plot(lines = [positive[0], positive[1], negative[0], negative[1]], category = None)
        plt.plot(positive[0], positive[1], color='red', marker='o')
        plt.plot(negative[0], negative[1], color='blue', marker='o')
        plt.plot( [positive[0], negative[0]], [positive[1], negative[1]])
        plt.plot( [x1, x2], [y1, y2])
        
        plt.scatter(dataset.coordinates[dataset.category[:,0] != guesses[:, 0], 0],  dataset.coordinates[dataset.category[:,0] != guesses[:, 0], 1], color='green', marker='x')
        
        dataset.plot(lines = None, category = None)
        
        #print(np.where(dataset.category == guesses, 0, 1))
        #print(dataset.category)
        #print(guesses)
        
        #print(slope)
        #print(midpoint)
        #print(C)
        """

        
    def classify(self, dataset): 
        
        output =  self.normal * np.sign(dataset.coordinates[:, 1] - self.slope*dataset.coordinates[:, 0] - self.offset) 
        
        return output.reshape(dataset.category.shape)
    
    
    def accuracy(self, dataset): 
        
        #Get the classifiers guesses on the data:
        guesses = self.classify(dataset)
        #print(guesses)
        #print(guesses.shape)
        
      
        
        #Select the errors and weight them: 
        error = dataset.weights[guesses != dataset.category].sum()
        
        print("Weak Error: ", error)
        
        self.alpha = (1/2)*np.log((1-error) / error)
        return error

    def plot_decision_boundary(self):
        
        x1 = -2
        x2 = 2
        y1 = self.slope*x1 + self.offset
        y2 = self.slope*x2 + self.offset
        
        plt.plot( [x1, x2], [y1, y2])
        
        plt.title("Decision Boundaries")
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        #plt.legend()

        plt.xlim(-2 , 2)
        plt.ylim(-2 , 2)


class AdaBoost_Classifier: 
    
    def __init__(self, itterations, data_set):
        self.itterations = itterations
        self.data_set = data_set
        self.weak_classifiers = []

    def train(self):
        
        training_data = self.data_set.select(None)
        
        for itteration in range(self.itterations): 
            
                      
            training_sample_set = training_data.select_weighted_random(N)
                        
            weak = Weak_Classifier()
            weak.train(training_sample_set)
            error = weak.accuracy(train_dataset)
            guesses = weak.classify(train_dataset)
            
            new_weights = train_dataset.weights * (np.where((guesses == train_dataset.category), (1/(2*(1-error))), (1/(2*(error)))))

            training_data = Data_Set(train_dataset.coordinates, train_dataset.category, new_weights)
            
            self.weak_classifiers.append(weak)           

    def classify(self, data_set): 
        
        total_classification = np.zeros(data_set.category.shape)
        
        for weak_classifier in self.weak_classifiers: 
            total_classification += weak_classifier.alpha*weak_classifier.classify(data_set)
        
        #print(total_classification)
        
        
        return np.sign(total_classification)

    def accuracy(self, dataset): 
        
        #Get the classifiers guesses on the data:
        guesses = self.classify(dataset)
        #print(guesses)
        #print(guesses.shape)
        
      
        
        #Select the errors and weight them: 
        error = dataset.weights[guesses != dataset.category].sum()
        
        print("Final Error: ", error)
        
        self.alpha = (1/2)*np.log((1-error) / error)
        return error
    	
    def plot_decision_boundaries(self):
        
        plt.figure(figsize=(10, 10))
        
        for weak in self.weak_classifiers: 
            weak.plot_decision_boundary()
        
        os.makedirs("./Output_Plots", exist_ok=True)
        plt.savefig("./Output_Plots/Decision_Boundaries.png")
        #plt.figure(figsize=(10, 10))
      
    def plot_points(self, data_set):
        
        guesses = self.classify(data_set)
        
        plt.scatter(data_set.coordinates[data_set.category[:,0] != guesses[:, 0], 0],  data_set.coordinates[data_set.category[:,0] != guesses[:, 0], 1], color='green', marker='x')
       
        plt.scatter(data_set.coordinates[data_set.category[:,0] == 1, 0],  data_set.coordinates[data_set.category[:,0] == 1, 1], color='red', marker='.', label='Positive')
        plt.scatter(data_set.coordinates[data_set.category[:,0] == -1, 0], data_set.coordinates[data_set.category[:,0] == -1,1], color='blue', marker='.', label='Negative')
       
       
        os.makedirs("./Output_Plots", exist_ok=True)
        plt.savefig("./Output_Plots/Incorrect_Points.png")
            
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

ada = AdaBoost_Classifier(10, train_dataset)
ada.train()
print(ada.accuracy(train_dataset))
ada.plot_decision_boundaries()
ada.plot_points(test_dataset.select_weighted_random(20))

# Plotting decision boundary function
"""
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
"""
# Plot training data
#plot_decision_boundary(X_train, y_train, 'AdaBoost Classifier Decision Boundary (Training Data)')

# Plot test data
#plot_decision_boundary(X_test, y_test, 'AdaBoost Classifier Decision Boundary (Test Data)')

"""
plt.show()
#Save the plots to a file
os.makedirs("./Output_Plots", exist_ok=True)
plt.savefig("./Output_Plots/Plot_1.png")
"""