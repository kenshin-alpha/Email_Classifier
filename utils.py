#Any extra functionality that need to be reused will go here

import numpy as np
from scipy.sparse import hstack

def concat_features(X, y_encoded):
    y_2d = np.array(y_encoded).reshape(-1, 1)
    if hasattr(X, "tocsr"):
        return hstack([X, y_2d]).tocsr()
    else:
        return np.hstack([X, y_2d])

def encode_safe(le, labels):
    classes = list(le.classes_)
    return np.array([classes.index(l) if l in classes else 0 for l in labels])