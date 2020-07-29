# Learn to use CARET!
import numpy as np
import math
import time
import statistics
import CARET
from concaveEval import *
import matplotlib.pyplot as plt
import os
import random

#p = np.array([0.0680, 0.0160, 0.1707, 0.1513, 0.1790, 0.2097, 0.0548, 0.0337, 0.0377, 0.0791])
p = np.array([1.9, 0.9, 0.0680, -1.60])
p = p.reshape(4, -1)


nIID = 100

num = [500]
Mvec = [100]

numTrials = 1

mse = np.zeros((numTrials, len(num)))
mae = np.zeros((numTrials, len(num)))
ti = np.zeros((numTrials, len(num)))

yHatOut = []

xSaveTrain = []
ySaveTrain = []
ySaveTest = []
xSaveTest = []

counter = 0
np.random.seed(42)
a_0 = np.hstack((np.ones((max(num),1))*-1.5, np.ones((max(num),1))*-1, np.zeros((max(num),1)), np.ones((max(num),1))))
for m in range(numTrials):
    print("=======================")
    print("m: ", m)
    print("=======================")
    
    xTrain = np.random.random((max(num),p.shape[1]))
    yTrain = np.dot(xTrain, p.T) + a_0
    yTrain = yTrain.min(axis=1).reshape(-1,1)
    xTest = np.random.random((100, p.shape[1]))
    yTest = np.dot(xTest, p.T)
    yTest = yTest.min(axis=1)
    xSaveTest.append(xTest)
    ySaveTest.append(yTest)
    xSaveTrain.append(xTrain)
    ySaveTrain.append(yTrain)
    print(yTrain.shape)
    for i in range(len(num)):
        n = num[i]
        n += 1
        
        tstart = time.time()
        dic = CARET.CARET(x=xTrain[1:n,:],
                             y=yTrain[1:n],
                             convexFlag=1)
        alpha = dic['alpha'][-1]
        beta = dic['beta'][-1]
        tend = time.time()
        print('CARET done')
        print(np.array(beta).shape)
        print(np.array(alpha).shape)
        yHat_train = concaveEval(alpha, beta, xTrain)
        yHat = concaveEval(alpha, beta, xTest)
        res = yTest[1:nIID] - yHat[1:nIID]
        ti[m, i] = tend
        mse[m, i] = np.mean(res**2)
        mae[m, i] = np.mean(abs(res))
        ex_time =  tend - tstart
        print('time', ex_time)

Experiment_path = "./Images"

if not os.path.exists(Experiment_path):
    os.makedirs(Experiment_path)
for i in range(p.shape[1]):
    plt.figure()
    plt.plot(xTrain[:, i], yTrain,  '*', label="data") 
    plt.plot(xTrain[:, i], yHat_train, '.', label="Approximated PWLC function") 
    plt.ylabel("Dependant variable")
    plt.xlabel("independent variable")
    plt.legend()
    plt.savefig('%s/dimension%s.png'%(Experiment_path, i))
plt.show()