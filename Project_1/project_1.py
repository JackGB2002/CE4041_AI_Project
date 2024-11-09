## kerasMNISTv2.py
##
## Train a convolutional neural network on the MNIST dataset.
## Uses two convolutional layers, max-pooling, and dropout layers.
##
## SGD parameters: ETA = 0.15, momentum (alpha) = 0.9
## RMSprop optimizer, categorical cross-entropy loss
## 20 epochs, 80%/20% training/validation split, with early stopping

import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout # type: ignore
from tensorflow.keras.initializers import RandomNormal # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore

# Load and preprocess the MNIST dataset
(tra, tralab), (tes, teslab) = mnist.load_data()
tra_vec = np.reshape(tra, (len(tra), 28, 28, 1)).astype('float32') / 255.0
tes_vec = np.reshape(tes, (len(tes), 28, 28, 1)).astype('float32') / 255.0
cat_tralab = to_categorical(tralab)
cat_teslab = to_categorical(teslab)

# Defining the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.1),

    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    # Dropout(0.5),

    Flatten(),

    Dense(256, activation='relu'),
    # Dropout(0.3),
    Dense(10, activation='softmax')
])

print("The Keras network model")
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Define early stopping callback
stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='min')

# Train the model with augmented data
history = model.fit(tra_vec, cat_tralab, epochs=20, validation_split=0.2, callbacks=[stop])


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

# Call the functions after training
evaluate_model_performance(model, tes, cat_teslab)
plot_loss(history)

