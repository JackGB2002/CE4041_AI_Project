##
##  optdigits.py
##
##  Utilities to load and display the Optdigits.
##
import numpy as np
import matplotlib.pyplot as plt
import os


def load_dataset(dset='train'):
    """Load an optdigits data set (dset == "train"|"test")
       Return tuple (vectors, labels), where 
       vectors = array N x 64 of normalised (0.0..1.0) inputs (float32), and
       labels  = vector of output labels (0..9) (uint8).
    """
    def buildpath():
        "Build appropriate path + filename base depending on current location."
        if os.getcwd().split('/')[-1] == 'optdigits':  # in optdigits folder
            path = ''
        else:      # No, so must include this at start of relative filename.
            path = 'optdigits/'
        return path + 'optdigits.'
    dataset_name = buildpath() + dset[:3]
    dataset = np.loadtxt(dataset_name,dtype='float32',delimiter=',')
    vectors = dataset[:,:64] / 16.0 
    labels  = dataset[:,64].astype('uint8')
    return vectors,labels


def make_digits_image(digit_vectors,rows=10,cols=10,start_at=0):
    "Create an image of rows x cols optdigits from set 'digit_vectors'."
    assert digit_vectors.shape[1] == 64, "1st argument must be N x 64 array."
    im = np.zeros((rows*10,cols*10),dtype=digit_vectors.dtype)
    i = start_at
    for r in range(rows):
        for c in range(cols):
            if i >= 0 and i < digit_vectors.shape[0]:
                im[r*10:r*10+8,c*10:c*10+8] = digit_vectors[i].reshape((8,8))
                i += 1
    return im

    
def display_labels(labels_vector,rows=10,cols=10,title=" ",start_at=0):
    "Display rows x cols of the optdigits labels from the set 'labels_vector'."
    i = 0
    if title != " ": print(title)
    for r in range(rows):
        for c in range(cols):
            print(labels_vector[i], end=' ')
            i += 1
        print()


def display_digits(digit_vectors,rows=10,cols=10,title=" ",start_at=0):
    "Display rows x cols of the optdigits digits from the set 'digit_vectors'."
    plt.figure(title)
    plt.imshow(make_digits_image(digit_vectors,rows,cols,start_at),
               cmap='gray_r', interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()


if __name__ == "__main__":
    print("Loading OPTDIGITS training and testing sets")
    train_vectors, train_labels = load_dataset("train")
    test_vectors, test_labels = load_dataset("test")
    print("There are {} images and {} labels in the training set.".format(
        train_vectors.shape[0], train_labels.shape[0]))
    print("There are {} images and {} labels in the testing set.".format(
        test_vectors.shape[0], test_labels.shape[0]))
    print("The training set images are {}-pixel vectors.".format(
        train_vectors.shape[1]))
    print("The testing set images are {}-pixel vectors.".format(
        test_vectors.shape[1]))
    print("Max and min pixel values in the training set are {} and {}.".format(
        train_vectors.max(), train_vectors.min()))
    print("Max and min pixel values in the testing set are {} and {}.".format(
        test_vectors.max(), test_vectors.min()))
        
    print("Here are the first 100 labels of the training set, and their"
          " associated images")
    
    display_labels(train_labels[:100],title="Training vector 0...99 labels")

    display_digits(train_vectors[:100],title="Training vectors 0...99 as images")
    
    plt.figure("Training set digit counts")
    plt.bar(range(10), np.bincount(train_labels))
    train_class_avg = len(train_labels)/10
    plt.plot([-.5,9.5],[train_class_avg,train_class_avg],'r:')
    plt.xticks(range(10))
    
    plt.show()
        
