import math

def sigmoid(x):
    return 1/(1+math.exp(-x))

def relu(x):
    return max(x, 0)

def tanh(x):
    return 2/(1+math.exp(-2*x))-1

def swish(x):
    return x*sigmoid(x)