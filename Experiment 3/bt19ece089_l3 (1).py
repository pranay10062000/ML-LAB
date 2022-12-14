# -*- coding: utf-8 -*-
"""BT19ECE089_L3

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1efYSo73WfKqNL4LMuDrHR7k4bCilz3H9
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import matplotlib.pyplot as plt
import random
import math

data= loadmat('/content/Matlab_cancer.mat')

data_x = data['x']
data_y = data['t']
trnspose = np.transpose(np.vstack([data_y, data_x]))
np.random.shuffle(trnspose)
trnspose = np.transpose(trnspose)
data_y = trnspose[0, :]
data_x = trnspose[2:, :]
split_ratio = 0.7
p = math.ceil(data_x.shape[1]*split_ratio)
print("training dataset:-",p)
train_x = data_x[:, :p//2]
train_y = data_y[:p//2]
test_x = data_x[:, p:]
test_y = data_y[p:]
val_x = data_x[:, p//2:p]
val_y = data_y[p//2:p]
np.count_nonzero(train_x)

class Neural_Network():

    def __init__(self, model, learning_rate):
        self.architecture = model
        self.neural_layers = len(model)
        self.learning_rate = learning_rate
        self.dw = []
        self.db = []
        self.biases = []
        self.weights = []
        self.cost = []
        self.test_accuracy = []
        self.train_accuracy = []
        self.validation_accuracy = []
        
        for i,j in zip(model[1:], model[:-1]):
            w = np.random.randn(i, j) 
            b = np.random.randn(i, 1) 
            dw = np.zeros([i, j])
            db = np.zeros([i, 1])
            self.dw.append(dw)
            self.db.append(db)
            self.weights.append(w)
            self.biases.append(b)
        self.activation = []
        for i in model:
            a = np.zeros(i)
            self.activation.append(a)

    def sigmoid(self, z):
        activation = 1/(1 + np.exp(-z))
        return activation
    def cost_function(self, Y):
        L = (Y * np.log(self.activation[-1]) + (1-Y)*np.log(1-self.activation[-1]))
        L = -L
        J = np.sum(L)/Y.shape[0]
        self.cost.append(J)
    def forward_propagation(self, ip):

        
        activation = ip
        self.activation[0] = activation
        save_num = list(range(1,self.neural_layers))

        for i,w,b in zip(save_num, self.weights, self.biases):
            z = np.matmul(w, activation) + b
            activation = self.sigmoid(z) 
            self.activation[i] = activation 
    
    def backward_propagation(self, batch_size, Y):

        dz = self.activation[-1] - Y
        dw = np.matmul(dz, self.activation[-2].T) / batch_size
        db = np.sum(dz, axis=1)/batch_size
        self.dw[-1] = dw
        self.db[-1] = db.reshape([-1, 1])

        for i in range(2, self.neural_layers):
            
            sis = self.activation[-i] * (1 - self.activation[-i])
            dz = np.matmul(self.weights[-i+1].T, dz) 
            dz = dz * sis
            
            dw = np.matmul(dz, self.activation[-i-1].T) / batch_size
            db = np.sum(dz, axis=1) / batch_size

            self.dw[-i] = dw
            self.db[-i] = db.reshape([-1, 1])
            self.cost_function(Y)


    def gradient_descent(self):
        for i in range(self.neural_layers-1):
            self.weights[i] = self.weights[i] - self.learning_rate * self.dw[i]
            self.biases[i] = self.biases[i] - self.learning_rate * self.db[i]

    def accuracy(self, ip, op, threshold=0.5, confusion=False):
        activation = ip
        save_n = list(range(1,self.neural_layers))
        accuracy = 0

        for i,w,b in zip(save_n, self.weights, self.biases):
            z = np.matmul(w, activation) + b   
            activation = self.sigmoid(z)  

        activation = activation.reshape(-1,)
        activation[activation>threshold] = 1
        activation[activation<=threshold] = 0

        if confusion==True:
            return activation

        for i,j in enumerate(activation):
            if j==op[i]:
                accuracy = accuracy+1
        return accuracy/ip.shape[1]


    def confusion_matrix(self, ip, op, threshold_list):
        T_p = []
        T_n = []
        F_p = []
        F_n = []
        for i in threshold_list:
            activation = self.accuracy(ip, op, i, True)
            c = activation - 2 * op
            T_p.append(np.count_nonzero(c==-1))
            T_n.append(np.count_nonzero(c==0))
            F_p.append(np.count_nonzero(c==1))
            F_n.append(np.count_nonzero(c==-2))
        return T_p,T_n,F_p,F_n
    def train_function(self, ip, op, epochs, Validation_Set=None):
        batch_size = ip.shape[1]
        for i in range(epochs):
            self.forward_propagation(ip)
            self.backward_propagation(batch_size, op)
            self.gradient_descent()
            train_acc1 = self.accuracy(ip, op)
            print('accuracy-\t', train_acc1*100, '%')
            self.train_accuracy.append(train_acc1)
            if Validation_Set!=None:
                test_acc1 = self.accuracy(Validation_Set[0], Validation_Set[1])
                self.test_accuracy.append(test_acc1)

ANN_obj = Neural_Network([100, 32, 1], 3e-2)

threshold_list = np.arange(0, 1, 0.005)
T_p,T_n,F_p,F_n = ANN_obj.confusion_matrix(train_x, train_y, threshold_list)
T_p = np.array(T_p)
T_n = np.array(T_n)
F_p = np.array(F_p)
F_n = np.array(F_n)

#without validation
ANN_obj.neural_layers
ANN_obj.train_function(train_x, train_y, 600)

#with validation
ANN_obj.train_function(train_x, train_y, 600, [val_x, val_y])

specificity = T_n/(F_p+T_n)
sensitivity = T_p/(T_p+F_n)
f_r = 1-specificity

print(specificity)
print(sensitivity)

from mlxtend.plotting import plot_confusion_matrix
threshold_a = np.arange(0, 1, 0.1)
Tp,Tn,Fp,Fn = ANN_obj.confusion_matrix(train_x, train_y, threshold_a)
Tp = np.array(Tp)
Tn = np.array(Tn)
Fp = np.array(Fp)
Fn = np.array(Fn)
for i,j,k,l,threshold_value in zip(Tp, Tn, Fp, Fn, threshold_list):
    print(i,j,k,l)
    confusion_matrix = np.array([[j, k], [l, i]])
    fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix, figsize=(3, 3), cmap=plt.cm.Blues)
    plt.xlabel('prediction values')
    plt.ylabel('true values')
    plt.title('confusion - matrix,threshold-- {}'.format(round(threshold_value,3)), fontsize=15)
    plt.show()

plt.plot(ANN_obj.train_accuracy)

ANN_obj.accuracy(test_x, test_y)