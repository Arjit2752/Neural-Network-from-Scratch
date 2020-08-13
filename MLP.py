# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 01:01:42 2020

@author: Arjit2752
"""

import numpy as np


## Activation function classes
class Identity(object):
    
    def __init__(self):
        self.state = None
    
    def __call__(self,x):
        return self.forward(x)
    
    def forward(self,x):
        self.state = x
        return self.state
    
    def derivative(self):
        return 1.0
    
class Sigmoid(object):
    
    def __init__(self):
        self.state = None
    
    def __call__(self,x):
        return self.forward(x)
    
    def forward(self,x):
        self.state = 1.0/(1.0 + np.exp(-x))
        return self.state
    
    def derivative(self):
        return self.state * (1 - self.state)
    
class Tanh(object):
    
    def __init__(self):
        self.state = None
    
    def __call__(self,x):
        return self.forward(x)
    
    def forward(self,x):
        self.state = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
        return self.state
    
    def derivative(self):
        return 1.0 - (self.state * self.state)
    
class Relu(object):
    
    def __init__(self):
        self.state = None
    
    def __call__(self,x):
        return self.forward(x)
    
    def forward(self,x):
        self.state = 0.0
        if x>0.0:
            self.state = x
        return self.state
    
    def derivative(self):
        if self.state > 0.0:
            return 1.0
        else:
            return 0.0
    
class LeakyRelu(object):
    
    def __init__(self):
        self.state = None
    
    def __call__(self,x):
        return self.forward(x)
    
    def forward(self,x):
        if x >= 0.0:
            self.state = x
        else:
            self.state = 0.1*x
        return self.state
    
    def derivative(self):
        if self.state >= 0.0:
            return 1.0
        else:
            return 0.1


class Criterion(object):
    # The following Criterion class can be used again as basis for 
    # number of loss functions(which are in the form of classes so that they can be exchanged easily)
    
    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self,x,y):
        return self.forward(x,y)
    
    def forward(self,x,y):
        raise NotImplementedError
        
    def derivative(self):
        raise NotImplementedError

class SoftmaxCrossEntropy(Criterion):
    """ Softmax Cross Entropy loss"""
    
    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None
#    def __call__(self,x,y):
#        return self.forward(x,y)
    
    def softmax(self, x):
        ''' this function is to return prediction probabilities'''
        self.logits = x
        mx = np.max(self.logits, axis=1).reshape(-1, 1)
        subtracted = self.logits - mx
        self.exp_logits = np.exp(subtracted)
        exp_sum = self.exp_logits.sum(axis=1).reshape(-1, 1)
        self.sm = self.exp_logits/exp_sum
        return self.sm
    
    def forward(self,x,y):
        """ 
        x is the matrix of pre-activations calculated in last Softmax layer for whole batch(mini batch) i.e. predicted matrix
        y is the matrix of actual labels for that particular input batch(mini batch) i.e. true labels matrix
        loss is the Softmax Cross entropy loss vector for whole batch(mini batch)
        Arguments: x (np.array) = (batch_size, 10)
                   y (np.array) = (batch_size, 10)
        Return:
            loss (np.array) = (batch_size, )
        """
        self.logits = x
        self.labels = y
        # Numerically stable Softmax
        mx = np.max(self.logits, axis=1).reshape(-1, 1)
        subtracted = self.logits - mx
        self.exp_logits = np.exp(subtracted)
        exp_sum = self.exp_logits.sum(axis=1).reshape(-1, 1)
        self.sm = self.exp_logits/exp_sum

        # Cross Entropy
        first = -(self.logits * self.labels).sum(axis=1).reshape(-1,1)
        second = mx + np.log(exp_sum)
        self.loss = first + second
        return self.loss
    
    def derivative(self):
        """
        Derivative Loss function
        
        Return:
        out (np.array): (batch size, 10)
        """
        return self.sm - self.labels


class Layer(object):
    ''' This class represents a layer of neural network, i.e. each instance of this 
        class is a layer
        Initialization Argument:
            in_feature : represents the number of inputs(neurons) from previous layer
            out_feature : represents the number of neuron in this particular layer(i.e. layer instance)
            self.W (np.array): Weight matrix for this particular layer, shape-> (in_feature, out_feature)
            self.dW (np.array): Derivative of Loss function w.r.t. Weight matrix(self.W), shape-> (in_feature, out_feature)
            self.b : Biases matrix this particular layer, shape-> (1,out_feature)
            self.db : Derivative of Loss function w.r.t. Biases matrix(self.b), shape-> (1,out_feature)
            self.momentum_W : Weight momentum matrix for momentum update equation
            self.momentum_b : Biases momentum matrix for momentum update equation
            (NOTE:- self.momentum_W and self.momentum_b are used only when MLP.is_momentum = True )
    '''
    
    def __init__(self, in_feature, out_feature):
        # in_feature is actually number of artificial neurons in previous layer
        # out_ feature is actually number of artificial neurons in this particular working layer
        self.in_feature = in_feature
        self.out_feature = out_feature
        # Intializing weight and weight derivative matrix
        self.W = np.random.rand(in_feature, out_feature)
        self.dW = np.zeros_like(self.W, dtype= np.float64)
        # Initializing bias and bias derivative matrix
        self.b = np.random.rand(1,out_feature)
        self.db = np.zeros_like(self.b, dtype= np.float64)
        
        self.momentum_W = np.zeros(self.W.shape)
        self.momentum_b = np.zeros(self.b.shape)
        
    def forward(self, x):
        """
        Arguments:
            x (np.array) = input batch (batch_size, in_feature)
        Return:
            out (np.array) = pre-activation matrix for next layer (batch_size, out_feature)
        """
        self.x = x
        out = np.matmul(self.x, self.W) + self.b
        return out
    
    def backward(self, delta):
        """
        This function calculates self.dW and self.db for each epoch
        and return derivative of self.W w.r.t. inputs from previous layer.
        Arguments:
            delta (np.array) = (batch_size, out_feature)
        Returns:
            dh (np.array) = it is derivative of Weight matrix w.r.t. input features from previous layer
                            shape-> (batch_size, in-feature)
        """
        # self.dW and self.db ,represent the gradients of the loss w.r.t self.W and self.b, are averaged across the batch
        self.dW = np.dot(self.x.T, delta)/delta.shape[0]
        self.db = np.sum(delta, axis=0, keepdims=True)/delta.shape[0]
        dh = np.dot(delta, self.W.T)
        return dh
        
  
class MLP(object):
    """
    This is the main class to be imported for creating and executing neural network.
    ## OPTIMIZER USED IN THIS CLASS IS STOCHASTIC GRADIENT DESCENT FOR MINI BATCHES ##
    
    (NOTE:- THIS CLASS IS BASICALLY DESIGNED FOR CLASSIFICATION PURPOSE,
             SO FOR REGRESSION MEAN SQUARE ERROR FUNCTION SHOULD BE ADDED AND SOME MINOR CHANGES ACCORDING TO ALGORITHM
             SHOULD BE DONE)
    Example : 
        from MLP import *
        nn = MLP(input_size= 784, output_size= 10, hidden= [64, 64, 32], activations=[Sigmoid(), Sigmoid(), Sigmoid(), Identity()],
                 batch_size=1000, epochs=100, learning_rate=0.01, replace=True)
    Arguments:
        input_size = number of input features(i.e. number of neurpns in input layer)
        output_size = number of output classes in neural network classification
        hidden-> list = list of neurons in each hidden layer(for reference see the above example)
        activations-> list = list of activation functions for each hidden layer & output layer
        criterion = it is the loss function, by default it is SoftmaxCrossEntropy(), if other loss function are added then they
                    can be used.(default = SoftmaxCrossEntropy())
        batch_size = size of batch to be selected for training for each epoch(default = 20)
        learning_rate = learning rate for SGD optimizer(default = 0.01)
        epochs =  number of epochs for training(default = 100)
        is_momentum-> bool = if want to use momentum update equation for optimizer updation(default = False)
        momentum = momentum value, should given only if is_momentum = True (default = None)
        replace = for replacement of rows while randomly batch selection(default = False)
        
    Methdos:
        forward(x): for forward propagation of input and returns the pre-activation values that to be passed to SoftmaxCrossEntropy()
        step(): for weight and biases matrix updation by SGD optimizer
        backward(labels): for backward propagation and to obtain dW and db for each layer(just according to neural network algorithm)
        get_parameters(): gathers all weight matrix & biases matrix from each layer and appends them to self.weights & self.biases
                         respectively, making tensors for weights and biases and, Returns =  [self.weights, self.biases]
        accuracy(pred, labels): pred= predicted labels vector for test data(i.e. predicted probabilities)
                                 labels= true labels vector for test data
                                 calculates the number of correct prediction in self.correct
                                 Returns accuracy for test data
        fit(x,y): the main method for training the neural network, provide,
                  x = input features for training as numpy.array in shape-> (batch_size, input_size)
                  y = true labels corresponding to each input features for training as numpy.array in shape-> (batch_size, output_size)
        predict(test): returns vector of predicted labels for test data, shape-> (test.shape[0], )
        predict_prob(test): returns probability matrix for each test data, shape-> (test.shape[0], output_size)
    
    """
    
    
    
    def __init__(self, input_size, output_size, hidden, activations,criterion= SoftmaxCrossEntropy(),
                 batch_size=20, epochs=100,learning_rate=0.01, is_momentum= False, momentum= None, 
                 replace= False):
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden = hidden
        self.activations = activations
        self.criterion = criterion
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epochs = epochs
        self.is_momentum = is_momentum
        self.momentum = momentum
        self.replace = replace
        self.weights = []
        self.biases = []
        self.probs = None
        self.loss = None
        self.list_indx = None
        self.correct = None
        self.layers = [Layer(in_feat, out_feat) for in_feat,out_feat in zip([self.input_size] + hidden, hidden + [self.output_size])]
        
        
    def forward(self, x):
        """
        Arguments:
            x (np.array) = input_data of shape-> (batch_size, input_size)
        Returns:
            out (np.array) = output/predicted values given by last layer,
                            which should be later passed to softmax cross entropy for Loss calculation
                            shape-> (batch_size, out_size)
        """
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            x = self.activations[i](x)
        return x
    
    def step(self):
        # Apply step function of Stochastic Gradient Descent on weights and biases matrix of each layer
        # to reach or get the weight and biases matrix providing least Loss
        # if want to use momentum for step function according to momentum update equation, then
        # provide is_momentum = True and provide some value for 'momentum' attribute while initializing class
        
        if self.is_momentum == False:
            for layer in self.layers:
                layer.W = layer.W - self.lr * layer.dW
                layer.b = layer.b - self.lr * layer.db
        else:
            for layer in self.layers:
                layer.momentum_W = self.momentum * layer.momentum_W - self.lr * layer.dW
                layer.W = layer.W + layer.momentum_W
                layer.momentum_b = self.momentum * layer.momentum_b - self.lr * layer.db
                layer.b = layer.b + layer.momentum_b
    
    def backward(self, labels):
        # Backpropagation through activation functions and linear layers
        
        final_layer = self.activations[-1]
        final_output = final_layer.state
        self.loss = self.criterion(final_output, labels)
        self.probs = self.criterion.sm
        delta = self.criterion.derivative()
        
        # This would return nothing but finds and assigns dW and db for each layer
        # to update Weight and biases matrix with minimum Loss for each layer
        for i in range(len(self.hidden), -1, -1):
            delta = delta * self.activations[i].derivative()
            delta = self.layers[i].backward(delta)
    
    def get_parameters(self):
        """ Stores all weight and biases matrix of all layers into self.weights and self.biases tensor
        Returns: updated self.weights and self.biases in the form -> [self.weights, self.biases]
        """
        for i, layer in enumerate(self.layers):
            w = layer.W
            b = layer.b
            self.weights.append(w)
            self.biases.append(b)
        return [self.weights, self.biases]
    
    def accuracy(self,pred,labels):
#        pred = probs.argmax(axis=1).reshape(-1,1)
#        lab = labels.argmax(axis=1)
        self.correct = np.count_nonzero(pred == labels)
        accu = self.correct/labels.shape[0]
        return accu
    
    def fit(self, x, y):
        
        for i in range(self.epochs):
            # batch selection
            self.list_indx = list(np.random.choice(np.arange(0,x.shape[0]), self.batch_size, replace = self.replace))
            x_batch = x[self.list_indx]
            y_batch = y[self.list_indx]
            # forward propagation
            # xo stores final output returned by forward func i.e. the output given by last layer just before providing to SoftmaxCrossEntropy()
            xo = self.forward(x_batch) 
            # back propagation
            self.backward(y_batch)
            self.step()
            acc = self.accuracy(self.probs.argmax(axis=1), y_batch.argmax(axis=1))
            # self.loss = (batch_size,1)
            # self.loss.sum())/self.batch_size = averaged loss of whole batch (i.e. sum of loss in all batch rows/batch_size)
            print('Epoch = {}/{}, Loss = {}, Correct_predicted = {}/{} Accuracy = {}'.format(i+1, self.epochs, 
                  (self.loss.sum())/self.batch_size, self.correct, self.batch_size, acc, end='\n'))
    
    def predict(self, test):
        '''returns vector of predicted labels for test data, shape-> (test.shape[0], )'''
        x = self.forward(test)
        soft = SoftmaxCrossEntropy()
        y = soft.softmax(x)
        y = y.argmax(axis=1)
        # y is an numpy array(vector) of shape(test.shape[0], ), where test.shape[0] = number of input rows
        return y
    
    def predict_prob(self, test):
        '''returns probability matrix for each test data, shape-> (test.shape[0], output_size)'''
        x = self.forward(test)
        soft = SoftmaxCrossEntropy()
        y = soft.softmax(x)
        return y
        
