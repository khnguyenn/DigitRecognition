import numpy as np 


# Implementation of Activation Function 

# ReLU Function
def ReLU(x):
    return np.maximum(0,x)

# ReLU_deriv
def ReLU_Deriv(x):
    return np.where(x > 0 , 1 , 0)

# Sigmoid Function
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Softmax function
def SoftMax(x):
    return np.exp(x)/np.sum(np.exp(x),axis = 1,keepdims=True)


def initialize_parameters(input_size,hidden_layer1_size,hidden_layer2_size, output_size):
    # Initialize w small enough -> avoid 

    weights = {
        'W1': np.random.randn(input_size, hidden_layer1_size) * 0.01,
        'b1': np.zeros((1, hidden_layer1_size)),
        'W2': np.random.randn(hidden_layer1_size, hidden_layer2_size) * 0.01,
        'b2': np.zeros((1, hidden_layer2_size)),
        'W3': np.random.randn(hidden_layer2_size, output_size) * 0.01,
        'b3': np.zeros((1, output_size)),
    }
    
    return weights

def forward_prop(X,weights,return_back = 'forward'):
    #Input Layer 
    Z1 = np.dot(X, weights['W1']) + weights['b1']  #X in shape(m, 28 x 28) -> W(28x28,hidden_layers_1)
    A1 = ReLU(Z1)
    #Hidden Layer 
    Z2 = np.dot(A1,weights['W2']) + weights['b2']
    A2 = ReLU(Z2)
    # Output Layer
    Z3 = np.dot(A2,weights['W3']) + weights['b3']
    A3 = SoftMax(Z3)
    #Return type1
    forward_data = (Z1, A1, Z2, A2, Z3, A3)
    if return_back == 'forward':
        return forward_data
    # Return type 2 when using prediction, since we'll only need the A3 results to predict
    elif return_back == 'predict':
        return A3

def backward_prop(...):
    






