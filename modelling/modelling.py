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
        y3_train = data.y_train[Config.TYPE_COLS[1]]
        
        # Modelling for y2
        model_y2 = RandomForest("y2", data.embeddings, y2_train)
        d_y2 = Data(X=data.embeddings, df=data.df, X_train=data.X_train, X_test=data.X_test,
                    y_train=y2_train, y_test=data.y_test[Config.TYPE_COLS[0]],
                    train_df=data.train_df, test_df=data.test_df)
        model_y2.train(d_y2)
        model_y2.predict(data.X_test)
        preds_y2 = model_y2.predictions
        
        # Encode for next stage
        le_y2 = LabelEncoder()
        y2_enc_train = le_y2.fit_transform(y2_train)
        preds_y2_enc = encode_safe(le_y2, preds_y2)
        
        # 2. Model for y3
        X_train_y3 = concat_features(data.X_train, y2_enc_train)
        X_test_y3 = concat_features(data.X_test, preds_y2_enc)
        
        model_y3 = RandomForest("y3", data.embeddings, y3_train)
        d_y3 = Data(X=data.embeddings, df=data.df, X_train=X_train_y3, X_test=X_test_y3,
                    y_train=y3_train, y_test=data.y_test[Config.TYPE_COLS[1]],
                    train_df=data.train_df, test_df=data.test_df)
        model_y3.train(d_y3)
        model_y3.predict(X_test_y3)
        preds_y3 = model_y3.predictions

         # Encode for next stage
        le_y3 = LabelEncoder()
        y3_enc_train = le_y3.fit_transform(y3_train)
        preds_y3_enc = encode_safe(le_y3, preds_y3)
        
        X_train_y4 = concat_features(X_train_y3, y3_enc_train)
        X_test_y4 = concat_features(X_test_y3, preds_y3_enc)

        return {
            "name": "chained",
            "models": [model_y2, model_y3],
            "predictions": pd.DataFrame({
                Config.TYPE_COLS[0]: preds_y2,
                Config.TYPE_COLS[1]: preds_y3
            })
        }
    return None



def model_evaluate(model, data):
    model.print_results(data)