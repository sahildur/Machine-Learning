import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import random
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import math
import timeit

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    each=np.array(z)
#    len(each)
#    print each
    sig_ans=1 / (1 + np.exp(-each))
    return sig_ans
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    
    #Pick a reasonable size for validation data
    train_data=[]
    train_label = []
    validation_data = []
    validation_label = []
    test_data = []
    test_label = []
    
    count=0    
    for count in range(0,10):
        for i in range(len(mat['train'+str(count)])):
            h=0

            train_data.append(mat['train'+str(count)][i])
            temp=[0,0,0,0,0,0,0,0,0,0]
            temp[count]=1
            train_label.append(count)
        
        count=count+1
    count=0
    
    #print len(train_data)
    
    for count in range(0,10):
        for i in range(len(mat['test'+str(count)])):
            h=0
            test_data.append(mat['test'+str(count)][i])
            temp=[0,0,0,0,0,0,0,0,0,0]
            temp[count]=1
            test_label.append(count)


        count=count+1


##Shuffle the training data and labels together
    c = list(zip(train_data, train_label))


    random.shuffle(c)
    train_data, train_label = zip(*c)

    
    validation_data=train_data[50000:]
    validation_label=train_label[50000:]

    train_data=train_data[:50000]
    train_label=train_label[:50000]
        
        #### To lower the no of sample uncomment below
    #train_data=train_data[1:1300]
    #train_label=train_label[1:1300]
    #validation_data=validation_data[1:300]
    #validation_label=validation_label[1:300]
    #test_data=test_data[1:300]
    #test_label=test_label[1:300]

    
    train_data = np.array(train_data,dtype='float32')
    train_data=train_data/255 ##normalize the data
    train_label = np.array(train_label,dtype='float32')
    validation_data = np.array(validation_data,dtype='float32')
    validation_data=validation_data/255
    validation_label = np.array(validation_label,dtype='float32')
    test_data = np.array(test_data,dtype='float32')
    test_data=test_data/255
    test_label = np.array(test_label,dtype='float32')    

####uncomment to see the number as an 28X28 image

#    b=np.split(train_data[10],28)
#    plt.imshow(b,cmap = cm.Greys_r)
#
#    plt.show()

    
    #train_data = np.array([])
    #train_label = np.array([])
    #validation_data = np.array([])
    #validation_label = np.array([])
    #test_data = np.array([])
    #test_label = np.array([])    
    
    #Your code here

    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    obj_val = 0  
    
    #Your code here
    #
    #
    #
    #
    #
    
    
    ####labels back to 10X1
    big_training_label=[]
    for i in range(len(training_label)):

        count=int(training_label[i])
        temp=[0,0,0,0,0,0,0,0,0,0]
        temp[count]=1
        big_training_label.append(temp)        

    
    ##########
    
    ##feed forward 
    training_label=np.array(big_training_label,dtype='float32')
    samples=len(training_data)
    training_data=np.c_[training_data, np.ones(samples)] #adding bias node to input

    
    z=np.dot(training_data,np.transpose(w1)) #hidden output
    z=sigmoid(z) #hidden sigmoid
    z=np.c_[z, np.ones(len(z))] #adding bias node to hidden
    out=np.dot(z,np.transpose(w2)) # output node output
    out=sigmoid(out) #sigmoid final output
    delta_error=out-training_label#output - training #differenciation of error with respect to output

#OLD code for total error with LOOP
    ################obj val and regularization
    #    total_error=0
    #for i in range(len(delta_error)):
    #    temp_square_row=delta_error[i]**2
    #    temp_square_sum=sum(temp_square_row)
    #    temp_square_sum_half=temp_square_sum/2
    #    total_error=total_error+temp_square_sum_half
    #    
    #error_func_avg=total_error/len(training_data)
    
    #total average error of the output
    error_func_avg=(sum(sum(delta_error**2))/2)/samples

#Regularization of error function
#dampen the effect in change of larger weights
    s1=np.sum(w1*w1)
    s2=np.sum(w2*w2)

    reg = (lambdaval/(2*samples))*(s1 + s2)


    obj_val=error_func_avg+reg;
#####3    
    
#OLD CODE with loops to calculate Grad_w1 Grad_w2    #It takes around 6 hours to run(learn)
#    total_delta_w_out=np.zeros([10,n_hidden+1], dtype='float32', order='C')
#    total_delta_w_hid=np.zeros([n_hidden,n_input+1], dtype='float32', order='C')

#    for i in range(samples):
#        out_minus_label=delta_error[i]
#        diff_of_sig_output=out[i]*(1-out[i])
#        delta_out=out_minus_label*diff_of_sig_output
#        delta_out=np.vstack(delta_out)
#        z_each=z[i]
#        z_each=np.vstack(z[i])
##        print delta_out.shape
#        #print np.transpose(z_each).shape
#        delta_w_out=np.dot(delta_out,np.transpose(z_each))
#        total_delta_w_out=total_delta_w_out+delta_w_out
#    
#        
#        sigma_delta_into_w=delta_out*w2
#        
#        diff_of_sig_hidden=z_each*(1-z_each) #hidden/net
#        vertical_output_countri_sum=np.sum(sigma_delta_into_w,0)
#        vertical_output_countri_sum=np.vstack(vertical_output_countri_sum)
#        mat_training_each=np.vstack(training_data[i])
#        
#        
#        delta_hidden=vertical_output_countri_sum*diff_of_sig_hidden#delta_hidden
#
#        delta_hidden_removed_last_row=np.delete(delta_hidden,len(delta_hidden)-1,0)
#        delta_w1_each=np.dot(delta_hidden_removed_last_row,np.transpose(mat_training_each))
#        
#        total_delta_w_hid=total_delta_w_hid+delta_w1_each
#        
#    
#    total_delta_w_out_avg=total_delta_w_out/samples
#    total_delta_w_hid_avg=total_delta_w_hid/samples
#
#    grad_w2=total_delta_w_out_avg
#    grad_w1=total_delta_w_hid_avg
#    
    
    ##Calculation of grad_w2
    out_minus_label_new=delta_error
    diff_of_sig_output_new=out*(1-out)
    delta_out_new=out_minus_label_new*diff_of_sig_output_new 
    
    grad_w2_new=np.dot(np.transpose(delta_out_new),z)
    grad_w2_new=grad_w2_new/samples
    
    ####Calculation of grad_w1

    
    sigma_delta_multiply_with_into_w_new=np.dot(delta_out_new,w2) # It multiplies 
    #each row of w2 with delta_out_new and then sum it vertical-which is stored as a scaler as wij
    # and this is for all the training data
    
    
    diff_of_sig_hidden_new=z*(1-z) #hidden/net #differenciation of sigmoid

    delta_hidden_new=sigma_delta_multiply_with_into_w_new*diff_of_sig_hidden_new#delta_hidden
    #1Xhidden 
    #for all data hidden X training data length

    total_delta_w_hid_new=np.dot(np.transpose(delta_hidden_new),training_data)
    total_delta_w_hid_new_last_row_removed_new=np.delete(total_delta_w_hid_new,len(total_delta_w_hid_new)-1,0)    

#    total_delta_w_hid_new_avg=total_delta_w_hid_new/samples
    grad_w1_new=total_delta_w_hid_new_last_row_removed_new
    grad_w1_new=total_delta_w_hid_new_last_row_removed_new/samples
    
    print obj_val
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1_new.flatten(), grad_w2_new.flatten()),0)
#    obj_grad = np.array([])
    
    return (obj_val,obj_grad)

def nnPredict(w1,w2,data):


    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    labels = np.array([])
    #Your code here
    data=np.c_[data, np.ones(len(data))]
    z=np.dot(data,np.transpose(w1))
    z=sigmoid(z)
    z=np.c_[z, np.ones(len(z))]
    out=np.dot(z,np.transpose(w2))
    out=sigmoid(out)
    label_list=[]    
    for i in range(len(out)):        
        label_list.append(np.argmax(out[i])*1.0)
    labels=np.array(label_list)
    return labels
    


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 20;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.1;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

start = timeit.default_timer()
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
stop = timeit.default_timer()

print 'Time for minimize function: '+str(stop - start)+' seconds'
#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


print "||||||||"
#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset
#train_label_1d=[]
#for i in range(len(train_label)):
#    train_label_1d.append(np.argmax(train_label[i])*1.0)


print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)
#
#validation_label_1d=[]
#for i in range(len(validation_label)):
#    validation_label_1d.append(np.argmax(validation_label[i])*1.0)


#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset
#test_label_1d=[]
#for i in range(len(test_label)):
#    test_label_1d.append(np.argmax(test_label[i])*1.0)
    
print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

import pickle

# obj0, obj1, obj2 are created here...

# Saving the objects:
with open('objs.pickle', 'w') as f:
    pickle.dump([n_hidden, w1, w2,lambdaval], f)

# Getting back the objects:
with open('objs.pickle') as f:
    obj0, obj1, obj2, obj3 = pickle.load(f)