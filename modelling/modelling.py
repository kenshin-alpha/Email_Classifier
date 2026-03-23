from model.randomforest import RandomForest
from modelling.data_model import Data
from Config import Config
from sklearn.preprocessing import LabelEncoder
from utils import concat_features, encode_safe
import pandas as pd
import numpy as np


def model_predict(data, df, name):
    # Here we need to call the methods related to the model e.g., random forest 
    if name == 'chained':
        y2_train = data.y_train[Config.TYPE_COLS[0]]
        
        # Modelling for y2
        model_y2 = RandomForest("y2", data.embeddings, y2_train)
        d_y2 = Data(X=data.embeddings, df=data.df, X_train=data.X_train, X_test=data.X_test,
                    y_train=y2_train, y_test=data.y_test[Config.TYPE_COLS[0]],
                    train_df=data.train_df, test_df=data.test_df)
        model_y2.train(d_y2)
        model_y2.predict(data.X_test)
        preds_y2 = model_y2.predictions
        
        
        return {
            "name": "chained",
            "models": [model_y2],
            "predictions": pd.DataFrame({
                Config.TYPE_COLS[0]: preds_y2
              })
        }
    return None



def model_evaluate(model, data):
    model.print_results(data)