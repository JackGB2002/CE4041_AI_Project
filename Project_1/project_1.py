##===========================================================
#
#Title: CE4041_AI Project 1
#@Author: Jack Browne, Syed Hassaan Shah, Ethan o’Brien  
#@Date : 15/11/2024
#
##===========================================================




## Train a convolutional neural network on the MNIST dataset.
## Uses two convolutional layers, max-pooling, and dropout layers.
##
## SGD parameters: ETA = 0.15, momentum (alpha) = 0.9
## RMSprop optimizer, categorical cross-entropy loss
## 20 epochs, 80%/20% training/validation split, with early stopping


# Constants
EPOCHS = 20
SPLIT = 0.2
SHUFFLE = True
BATCH = 32
OPT = 'Adam'#'rmsprop' #'Adam'




import os as os
import numpy as np
from tensorflow.random import set_seed
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout # type: ignore
from tensorflow.keras.initializers import RandomNormal # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore

#Setting up enviornment random seeds: 
np.random.seed(1)                    # Initialise system RNG.

set_seed(2)        # Initialize the seed of the Tensorflow backend.




# Load and preprocess the MNIST dataset
(tra, tralab), (tes, teslab) = mnist.load_data()
tra_vec = np.reshape(tra, (len(tra), 28, 28, 1)).astype('float32') / 255.0
tes_vec = np.reshape(tes, (len(tes), 28, 28, 1)).astype('float32') / 255.0
cat_tralab = to_categorical(tralab)
cat_teslab = to_categorical(teslab)

# Defining the CNN model
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    #Dropout(0.25),

    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"),
    MaxPooling2D(pool_size=(2, 2)),
    #Dropout(0.25),

    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.1),

    Flatten(),

    Dense(128, activation='relu'),
    #Dropout(0.4),
    Dense(10, activation='softmax')
])

print("The Keras network model")
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=OPT, metrics=['accuracy'])

# Define early stopping callback
stop = EarlyStopping(monitor='val_loss', min_delta=0.0 , patience=5, mode='min', restore_best_weights=True)

# Train the model with augmented data
history = model.fit(tra_vec, cat_tralab, epochs=EPOCHS, validation_split=SPLIT, callbacks=[stop])


###############################################################################################
# Plot graphs of results
###############################################################################################

import matplotlib.pyplot as plt

def plot_loss(history):
    """
    Plots the training and validation loss over epochs.
    
    Parameters:
    - history: Training history object from model.fit(), containing loss per epoch.
    """
    # Define range of epochs for x-axis
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Plot Training and Validation Loss
    plt.figure('Training and validation loss history')
    plt.plot(epochs, history.history['loss'], label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.title('Training and validation loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Adjust layout to fit subplots nicely and display
    plt.tight_layout()
    
    #Save the plots to a file
    os.makedirs("./Project_1/Output_Plots", exist_ok=True)
    plt.savefig("./Project_1/Output_Plots/Plot_1.png")
    
    plt.show()

def evaluate_model_performance(model, test_data, test_labels):
    """
    Evaluates the model on a separate test dataset and prints performance metrics.
    
    Parameters:
    - model: Trained model to evaluate.
    - test_data: Test dataset to evaluate the model on.
    - test_labels: True labels for the test dataset.
    """
    # Evaluate model on test data and retrieve loss and accuracy
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    print("\nModel Performance on Testing Set:")
    print(f"Accuracy on Testing Data: {test_acc * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}\n")
    
    # Retrieve final training and validation accuracy from training history
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    print("Final Training Accuracy:   {:6.2f}%".format(train_acc * 100))
    print("Final Validation Accuracy: {:6.2f}%".format(val_acc * 100))
    print("Final Testing Accuracy:    {:6.2f}%".format(test_acc * 100))

def plot_training_validation_accuracy(history):
    """
    Plots the training and validation accuracy over epochs.
    
    Parameters:
    - history: Training history object from model.fit(), containing accuracy per epoch.
    """
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    # Plot Training and Validation Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history.history['accuracy'], label='Training Accuracy', marker='o', color='orange')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy', marker='o', color='red')
    plt.title('Training and Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Project_1/Output_Plots/Accuracy_Plot.png")
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(model, test_data, test_labels):
    """
    Plots a normalized confusion matrix of the model's predictions on the test dataset.
    
    Parameters:
    - model: Trained model to evaluate.
    - test_data: Test dataset to evaluate the model on.
    - test_labels: True labels for the test dataset.
    """
    # Predict the class labels on the test set
    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Normalize the confusion matrix to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.title('Confusion Matrix for Model Predictions')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig("./Project_1/Output_Plots/Confusion_Matrix.png")
    plt.show()

# Call the functions after training
evaluate_model_performance(model, tes, cat_teslab)
plot_loss(history)

