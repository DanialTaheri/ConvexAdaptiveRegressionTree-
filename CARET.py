import numpy as np
from sklearn import linear_model
import collections
import random
import math
import statistics
import time

def find_index(List, target, direction):
    if direction == "eq":
        res = []
        for i in range(len(List)):
            if List[i] == target:
                res.append(i)
        return res
    else:
        res1 = []
        res2 = []
        for i in range(len(List)):
            if List[i] >= target:
                res1.append(i)
            else:
                res2.append(i)
        return res1, res2
        
        
    
def CARET(x, y, convexFlag=1, minNode=10):
    
    nb, dim = x.shape[0], x.shape[1]
    
    

    resStruct = collections.defaultdict(list)
    abStruct = collections.defaultdict(list)
    
    Flag = True
    
    #initialize with one partition
    
    if (nb > minNode):
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        beta = regr.coef_
        alpha = [regr.intercept_]
    else:
        beta = np.zeros((1,dim))
        alpha = np.mean(y)
        
    pred_y = np.dot(x, beta.T) + alpha
    res = y - pred_y

    
    idList = [1 for _ in range(nb)]
    
    abStruct["alpha"].append(alpha)
    abStruct["beta"].append(beta)
    abStruct["residuals"].append(res.reshape(-1))
    abStruct["idList"].append(idList)
    

    MSEvec = []
    MSEvec.append(np.mean(res**2))

    
    if nb < minNode:
        Flag = False
        
    resStruct["residual"].append(res)
    
    iteration = 1
    K = 2
    while Flag:
        idListOld = abStruct["idList"][iteration-1]
        tempStruct = collections.defaultdict(list)
        resVec = []
        idListNew = idListOld.copy()
        counter = 1
        for k in range(1, K):
            
            idVec = find_index(idListOld, k, "eq")
            nk = len(idVec)

            xHat = x[idVec, :]
            yHat = y[idVec]

            if len(idVec) == 0:

                continue
            
            regressor = Node(x[idVec, :], y[idVec], np.array(np.arange(len(y[idVec]))), minNode)
            ret = regressor.get_rhs_lhs()
            if  ret == False:
                regr = linear_model.LinearRegression()
                regr.fit(xHat, yHat)
                beta = regr.coef_
                alpha = regr.intercept_

                pred_y = np.dot(xHat, beta.T) + alpha
                res = yHat - pred_y

                tempStruct["alpha"].extend([alpha])
                tempStruct["beta"].extend([beta])

                for ID in idVec:
                    idListNew[ID] = counter
                counter += 1
                continue
            else:
                xHat1Ind, xHat2Ind, var_idx = ret[1:]
                n1 = len(xHat1Ind)
                n2 = len(xHat2Ind)
                #print("n1, n2", n1,n2, var_idx)
                regr1 = linear_model.LinearRegression()

                regr1.fit( xHat[xHat1Ind, :], yHat[xHat1Ind])
                beta1 = regr1.coef_
                alpha1 = [regr1.intercept_]

                regr2 = linear_model.LinearRegression()
                regr2.fit(xHat[xHat2Ind, :], yHat[xHat2Ind])
                beta2 = regr2.coef_
                alpha2 = [regr2.intercept_]

                betaHat = np.vstack((beta1, beta2))
                alphaHat = [alpha1, alpha2]

                y_pred = np.zeros((nb, betaHat.shape[0]))
                for i in range(betaHat.shape[0]):
                    y_pred[:, i] = np.dot(x, betaHat[i, :].T) + alphaHat[i]
                
                #print("Refit")
                if convexFlag == 0:
                    raise ValueError
                else:
                    gg = y_pred.min(axis = 1)
                    iList = []
                    for i in range(y_pred.shape[0]):
                        for j in range(y_pred.shape[1]):
                            if y_pred[i,j] == gg[i]:
                                iList.append(counter + j)

                for ID in idVec:
                    idListNew[ID] = iList[ID]


                tempStruct["alpha"].extend(alphaHat)
                tempStruct["beta"].extend(betaHat)
                counter += 2

                
        abStruct["alpha"].append(tempStruct["alpha"])
        
        abStruct["beta"].append(tempStruct["beta"])
        abStruct["idList"].append(idListNew)
        
            


        
        K = counter 
        iteration += 1
        counter += 1 
        for item in range(len(abStruct["idList"][iteration-1])):
            if abStruct["idList"][iteration-1][item] == abStruct["idList"][iteration-2][item]:
                Flag = False
            else: 
                Flag = True
                break
    return abStruct

    
    
    




class Node:
    def __init__(self, x, y, idxs, min_leaf):
        self.x = x 
        self.y = y
        self.idxs = idxs 
        self.min_leaf = min_leaf
        self.n_data = len(idxs)
        self.n_features = x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
      
    def find_varsplit(self):
        for c in range(self.n_features): 
            self.find_better_split(c)
        if self.is_leaf: 
            return False
        x = self.split_col
        left = np.nonzero(x <= self.split)[0]
        right = np.nonzero(x > self.split)[0]

        return True, left, right, self.var_idx

        
    def find_better_split(self, var_idx):
        x = self.x[self.idxs, var_idx]

        self.threshold, self.Y = zip(*sorted(zip(x, self.y[self.idxs]) ))
        
        self.threshold = np.array(self.threshold)
        self.Y = np.array(self.Y)
        lhs = np.zeros(self.n_data, dtype=bool)
        rhs = np.ones(self.n_data, dtype=bool)
        for r in range(self.n_data):
            lhs[r] = True 
            rhs[r] = False

            if rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf:
                continue

            curr_score = self.find_score(rhs, lhs)
            if self.threshold[r] == self.threshold[r-1]:
                continue
            if curr_score < self.score: 
                self.var_idx = var_idx
                self.score = curr_score
                self.split = (self.threshold[r] + self.threshold[r-1])/2
               

    def get_rhs_lhs(self):
        return self.find_varsplit()
                
                
    def find_score(self, lhs, rhs):
        lhs_std = self.Y[lhs].std()
        rhs_std = self.Y[rhs].std()
        return lhs_std * lhs.sum() + rhs_std * rhs.sum()
                
    @property
    def split_col(self): return self.x[self.idxs,self.var_idx]
                
    @property
    def is_leaf(self): return self.score == float('inf')                


        
class DecisionTreeRegressor(Node):
    def fit(self, x, y):
        n_classes = np.array(np.arange(len(y)))
        self.dtree = Node(x, y, n_classes, min_leaf)
        return self
