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
            # Handling Low Frequency Classes
            min_instances = 5
            mask = np.ones(df.shape[0], dtype=bool)
            for col in Config.TYPE_COLS:
                if col in df.columns:
                    counts = df[col].value_counts()
                    valid_classes = counts[counts >= min_instances].index
                    mask &= df[col].isin(valid_classes).values
                    
            df = df[mask].reset_index(drop=True)
            if hasattr(X, "tocsr"):
                X = X.tocsr()[mask]
            else:
                X = X[mask]
                
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
    
    # Ensuring Maximum Consistent Input Data Format Handling
    def get_chained_data(self, appended_train_feature, appended_test_feature, next_y_train, next_y_test):
        y_train_feat = np.array(appended_train_feature).reshape(-1, 1)
        y_test_feat = np.array(appended_test_feature).reshape(-1, 1)
        
        X_tr = hstack([self.X_train, y_train_feat]).tocsr() if hasattr(self.X_train, "tocsr") else np.hstack([self.X_train, y_train_feat])
        X_te = hstack([self.X_test, y_test_feat]).tocsr() if hasattr(self.X_test, "tocsr") else np.hstack([self.X_test, y_test_feat])
        
        return Data(X=self.embeddings, df=self.df, X_train=X_tr, X_test=X_te, 
                    y_train=next_y_train, y_test=next_y_test, train_df=self.train_df, test_df=self.test_df)
                    
    def get_filtered_data(self, train_mask, next_y_train):
        X_tr = self.X_train.tocsr()[train_mask.values] if hasattr(self.X_train, "tocsr") else self.X_train[train_mask.values]
        return Data(X=None, df=None, X_train=X_tr, X_test=self.X_test, y_train=next_y_train, y_test=None)



