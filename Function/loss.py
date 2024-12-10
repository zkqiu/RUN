import math
import numpy as np

def L1loss(preds, labels):
    return np.mean(abs(preds-labels))

def NLLloss(preds, labels):
    pass

def MSEloss(preds, labels):
    return np.mean((preds-labels)**2)

def BCEloss(preds, labels):
    return -np.mean(labels * np.log(preds) + (1 - labels) * np.log(1 - preds))

def CEloss(preds, labels):
    return -np.mean(labels * np.log(preds))
        