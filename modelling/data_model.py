import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
from utils import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame = None,
                 X_train=None, X_test=None, y_train=None, y_test=None,
                 train_df=None, test_df=None) -> None:
        if X_train is not None and y_train is not None:
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.train_df = train_df
            self.test_df = test_df
            self.embeddings = X
            self.df = df
            self.y = y_train if y_train is not None else None
        else:
            self.embeddings = X
            self.df = df
            self.y = df[Config.TYPE_COLS]
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, self.y, test_size=0.2, random_state=seed
            )
            self.train_df, self.test_df = train_test_split(
                df, test_size=0.2, random_state=seed
            )

    def get_type(self):
        return  self.y
    def get_X_train(self):
        return  self.X_train
    def get_X_test(self):
        return  self.X_test
    def get_type_y_train(self):
        return  self.y_train
    def get_type_y_test(self):
        return  self.y_test
    def get_train_df(self):
        return  self.train_df
    def get_embeddings(self):
        return  self.embeddings
    def get_type_test_df(self):
        return  self.test_df


