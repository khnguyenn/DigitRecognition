# DigitRecognition

The Handwritten Digit Recognition is a project developed to classify digits drawn by the user in an interface.


### The Data Set 
The Data Set : MNIST 

The Training and Calculations of The Algorithm were implemented by Numpy Library.

The main objective of this project is to gain an in-depth understanding of how a neural network functions.

## How it works
### Algorithm
The algorithm is a `neural network` with dense layers, following this configuration:

- **Input Size**: 784   
- **Hidden Layer 1**: 50 neurons, activation function **ReLU**  
- **Hidden Layer 2**: 20 neurons, activation function **ReLU**  
- **Output Layer**: 10 neurons, activation function **Softmax**

**Backpropagation**: The `batch gradient descent` method was used with a batch size of 40 and 100 epochs, with a learning rate of 0.005.