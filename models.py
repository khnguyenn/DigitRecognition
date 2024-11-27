import numpy as np 
# Load the DataSet


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

def onehot_coder(Y):
    n_classes = np.max(Y) + 1
    y_hot = np.eye(n_classes,dtype=int)[Y]
    return y_hot.squeeze()
    

def initialize_parameters(input_size,hidden_layer1_size,hidden_layer2_size, output_size):
    """Initialize 

    Args:
        input_size (_type_): _description_
        hidden_layer1_size (_type_): _description_
        hidden_layer2_size (_type_): _description_
        output_size (_type_): _description_

    Returns:
        weights : 
            W1 (784,hidden_layer1_size)
            b1 (1,hidden_layer1_size)
    """    
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
    """Forward_Propagation

    Args:
        X (_type_): (60000,784)
        weights (_type_): _description_
        return_back (str, optional): _description_. Defaults to 'forward'.

    Returns:
        _type_: _description_
    """    
    #Input Layer 
    Z1 = np.dot(X, weights['W1']) + weights['b1']  #X in shape(m, 784) -> W(784,hidden_layers_1)
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
    
def compute_cost(AL,Y):
    m = Y.shape[0] ## Or Shape [0] Depends on the Data Set.
    return -np.sum(Y * np.log(AL + 1e-8)) / m 

def backward_prop(X,Y,foward_data,weights):
    """_summary_

    Args:
        X (_type_): _description_
        Y (_type_): _description_
        foward_data (_type_): _description_
        weights (_type_): _description_

    Returns:
        _type_: _description_
    """       
    # Loads the results of the last forward propagation.
    Z1, A1, Z2,A2, Z3,A3 = foward_data
    # Number of samples in training.
    m = X.shape[0] 

    #Process
    #Output Layer
    dZ3 = A3 - Y # Derivative of cost regarding Z3 
    gradient_W3 = np.dot(A2.T,dZ3) / m 
    gradient_b3 = np.sum(dZ3, axis = 0, keepdims= True) / m

    #Second Layer
    dA2 = np.dot(dZ3,weights['W3'].T)
    dZ2 = dA2 * ReLU_Deriv(Z2) 
    gradient_W2 = np.dot(A1.T,dZ2) / m 
    gradient_b2 = np.sum(dZ2, axis = 0, keepdims= True) / m 

    # Input Layer 
    dA1 = np.dot(dZ2,weights['W2'].T)
    dZ1 = dA1 * ReLU_Deriv(Z1)
    gradient_W1 = np.dot(X.T,dZ1)/m
    gradient_b1 = np.sum(dZ1, axis = 0, keepdims= True)

    # Save gradients as dictionary
    gradients = {
        'gradient_W3': gradient_W3, 'gradient_b3': gradient_b3,
        'gradient_W2': gradient_W2, 'gradient_b2': gradient_b2,
        'gradient_W1': gradient_W1, 'gradient_b1': gradient_b1
    }
    return gradients
                 
# Updates the weights using gradient descent
def update_weights(weights, gradients, learning_rate):
    """Update_Weights Gradient Descent

    Args:
        weights (_type_): _description_
        gradients (_type_): _description_
        learning_rate (_type_): _description_

    Returns:
        _type_: _description_
    """    
    weights['W3'] -= gradients['gradient_W3'] * learning_rate
    weights['b3'] -= gradients['gradient_b3'] * learning_rate
    weights['W2'] -= gradients['gradient_W2'] * learning_rate
    weights['b2'] -= gradients['gradient_b2'] * learning_rate
    weights['W1'] -= gradients['gradient_W1'] * learning_rate
    weights['b1'] -= gradients['gradient_b1'] * learning_rate

    return weights


# Prediction Function 
def predict(X,weights):
    """_summary_

    Args:
        X (_type_): _description_
        weights (_type_): _description_

    Returns:
        np
    """    
    A3 = forward_prop(X,weights,return_back = 'predict')
    return np.argmax(A3,axis = 1)


def train(X,Y,input_size, hidden_layer1_size, hidden_layer2_size, output_size, learning_rate, epochs, batch_size, weights):
    #Inputs Data
    m = X.shape[0]
    Y = onehot_coder(Y)
    cost = []
    epoch_list = []

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]

        # Using Mini-Batch:
        for i in range(0,m,batch_size):
            #Batch Sample
            X_batch = X_shuffled[i:i+batch_size] #i + batch_size - i = batch_size
            Y_batch = Y_shuffled[i:i+batch_size]
            # Forward propagation with the batch sample
            forward_data = forward_prop(X_batch,weights)
            # Perform backpropagation
            gradients = backward_prop(X_batch,Y_batch,forward_data,weights)
            # Update weights
            weights = update_weights(weights,gradients,learning_rate)

        #Calculate cost for each iterations:
        forward_data_full = forward_prop(X,weights)
        _,_,_,_,_, A3 = forward_data_full
        epoch_cost = compute_cost(A3,Y)
        #Print cost for each multiple of 10 epoch count:
        if epoch % 10 == 0:
            print(f'Epoch {epoch},cost: {epoch_cost}')
            cost.append(epoch_cost)
            epoch.append(epoch)
    
    return weights


    



    






    


    






