import numpy as np
from math import factorial
from itertools import permutations
import random
from collections import Counter
import math

def Permutation_Entropy(data,m,delay):#Permutation entropy function
    def key(a):
        return a[0]
    X=[[]for i in range(len(data)-(m-1)*delay)]
    for i in range(len(X)):#Map data to m-dimensional space
        for j in range(m):
            X[i].append([data[i+j*delay],j])
        X[i].sort(key=key)
    ordinal_patterns=[]
    for i in range(len(X)):
        s=''
        for j in range(m):
            s+=str(X[i][j][1])
        ordinal_patterns.append(s)

    ordinal_patterns=Counter(ordinal_patterns)
    P=[]
    for key in ordinal_patterns.keys():
        P.append(ordinal_patterns[key]/len(X))#Calculate the probability distribution of the sequence model
    H=0
    for i in range(len(P)):
        H+=P[i]*math.log(P[i])
    H*=-1.0
    return H/math.log(math.factorial(m))