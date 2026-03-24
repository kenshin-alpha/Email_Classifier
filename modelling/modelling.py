from model.randomforest import RandomForest
from modelling.data_model import Data
from Config import Config
from sklearn.preprocessing import LabelEncoder
from utils import concat_features, encode_safe
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
    

def model_predict(data, df, name):
    # Here we need to call the methods related to the model e.g., random forest 
    if name == 'chained':
        y2_train = data.y_train[Config.TYPE_COLS[0]]
        y3_train = data.y_train[Config.TYPE_COLS[1]]
        y4_train = data.y_train[Config.TYPE_COLS[2]]
        
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
        
        # Modelling for y3
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

        # Modelling for y4
        model_y4 = RandomForest("y4", data.embeddings, y4_train)
        d_y4 = Data(X=data.embeddings, df=data.df, X_train=X_train_y4, X_test=X_test_y4,
                    y_train=y4_train, y_test=data.y_test[Config.TYPE_COLS[2]],
                    train_df=data.train_df, test_df=data.test_df)
        model_y4.train(d_y4)
        model_y4.predict(X_test_y4)
        preds_y4 = model_y4.prediction

        return {
            "name": "chained",
            "models": [model_y2, model_y3, model_y4],
            "predictions": pd.DataFrame({
                Config.TYPE_COLS[0]: preds_y2,
                Config.TYPE_COLS[1]: preds_y3,
                Config.TYPE_COLS[2]: preds_y4
            })
        }
    elif name == 'hierarchical':
        print("\nExecuting Hierarchical Architecture...")
        y2_train = data.y_train[Config.TYPE_COLS[0]]
        model_y2 = RandomForest("y2_top", data.embeddings, y2_train)
        d_y2 = Data(X=data.embeddings, df=data.df, X_train=data.X_train, X_test=data.X_test, 
                    y_train=y2_train, y_test=data.y_test[Config.TYPE_COLS[0]],
                    train_df=data.train_df, test_df=data.test_df)
        model_y2.train(d_y2)
        model_y2.predict(data.X_test)
        preds_y2 = model_y2.predictions
        
        models_y3 = {}
        for c in y2_train.unique():
            if pd.isna(c): continue
            mask = y2_train == c
            X_train_sub = data.X_train.tocsr()[mask.values] if hasattr(data.X_train, "tocsr") else data.X_train[mask.values]
            y3_train_sub = data.y_train.loc[mask, Config.TYPE_COLS[1]]
            
            if len(y3_train_sub) > 0:
                m3 = RandomForest(f"y3_{c}", data.embeddings, y3_train_sub)
                d3 = Data(X=None, df=None, X_train=X_train_sub, X_test=data.X_test,
                          y_train=y3_train_sub, y_test=None, train_df=None, test_df=None)
                m3.train(d3)
                models_y3[c] = m3
                
            
        return {
            "name": "hierarchical",
            "models": [model_y2],
            "predictions": pd.DataFrame({
                Config.TYPE_COLS[0]: preds_y2
            })

    return None



def model_evaluate(model, data):
    if isinstance(model, dict):
        print(f"\n=== {model['name'].capitalize()} Model Evaluation ===")
        preds = model["predictions"]
        for col in Config.TYPE_COLS:
            print(f"\nClassification Report for {col}:")
            y_true = data.y_test[col]
            y_pred = preds[col]
            print(classification_report(y_true, y_pred, zero_division=0))
