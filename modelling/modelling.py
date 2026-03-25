from model.randomforest import RandomForest
from modelling.data_model import Data
from Config import Config
from sklearn.preprocessing import LabelEncoder
from utils import encode_safe
import pandas as pd
import numpy as np

def model_predict(data: Data, df: pd.DataFrame, name: str, model_class=RandomForest):
    if name == 'chained':
        print("\nExecuting Chained Multi-outputs Architecture...")
        y2_train = data.y_train[Config.TYPE_COLS[0]]
        y3_train = data.y_train[Config.TYPE_COLS[1]]
        y4_train = data.y_train[Config.TYPE_COLS[2]]
        
        # Modelling for y2
        model_y2 = model_class("y2", data.embeddings, y2_train)
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
        d_y3 = data.get_chained_data(y2_enc_train, preds_y2_enc, y3_train, data.y_test[Config.TYPE_COLS[1]])
        model_y3 = model_class("y3", data.embeddings, y3_train) 
        model_y3.train(d_y3)
        model_y3.predict(d_y3.X_test)
        preds_y3 = model_y3.predictions
        
        # Encode for next stage
        le_y3 = LabelEncoder()
        y3_enc_train = le_y3.fit_transform(y3_train)
        preds_y3_enc = encode_safe(le_y3, preds_y3)
        
        # Modelling for y4
        d_y4 = d_y3.get_chained_data(y3_enc_train, preds_y3_enc, y4_train, data.y_test[Config.TYPE_COLS[2]])
        model_y4 = model_class("y4", data.embeddings, y4_train)
        model_y4.train(d_y4)
        model_y4.predict(d_y4.X_test)
        preds_y4 = model_y4.predictions
        
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
        model_y2 = model_class("y2_top", data.embeddings, y2_train)
        d_y2 = Data(X=data.embeddings, df=data.df, X_train=data.X_train, X_test=data.X_test, 
                    y_train=y2_train, y_test=data.y_test[Config.TYPE_COLS[0]],
                    train_df=data.train_df, test_df=data.test_df)
        model_y2.train(d_y2)
        model_y2.predict(data.X_test)
        preds_y2 = model_y2.predictions
        
        y3_train = data.y_train[Config.TYPE_COLS[1]]
        models_y3 = {}
        for c in y2_train.unique():
            if pd.isna(c): continue
            train_mask = y2_train == c
            d3 = data.get_filtered_data(train_mask=train_mask, next_y_train=y3_train.loc[train_mask])
            
            if len(d3.y_train) > 0:
                m3 = model_class(f"y3_{c}", data.embeddings, d3.y_train)
                m3.train(d3)
                models_y3[c] = m3
                
        y4_train = data.y_train[Config.TYPE_COLS[2]]
        models_y4 = {}
        for c in y3_train.unique():
            if pd.isna(c): continue
            train_mask = y3_train == c
            d4 = data.get_filtered_data(train_mask=train_mask, next_y_train=y4_train.loc[train_mask])
            
            if len(d4.y_train) > 0:
                m4 = model_class(f"y4_{c}", data.embeddings, d4.y_train)
                m4.train(d4)
                models_y4[c] = m4
                
        # Proper Hierarchical Test Vector Mask Filtering
        preds_y3 = np.array(["Unknown/Other"] * data.X_test.shape[0], dtype=object)
        for c, m3 in models_y3.items():
            mask = (preds_y2 == c)
            if mask.any():
                X_test_sub = data.X_test.tocsr()[mask] if hasattr(data.X_test, "tocsr") else data.X_test[mask]
                m3.predict(X_test_sub)
                preds_y3[mask] = m3.predictions

        preds_y4 = np.array(["Unknown/Other"] * data.X_test.shape[0], dtype=object)
        for c, m4 in models_y4.items():
            mask = (preds_y3 == c)
            if mask.any():
                X_test_sub = data.X_test.tocsr()[mask] if hasattr(data.X_test, "tocsr") else data.X_test[mask]
                m4.predict(X_test_sub)
                preds_y4[mask] = m4.predictions
                
        return {
            "name": "hierarchical",
            "models": [model_y2, models_y3, models_y4],
            "predictions": pd.DataFrame({
                Config.TYPE_COLS[0]: preds_y2,
                Config.TYPE_COLS[1]: preds_y3,
                Config.TYPE_COLS[2]: preds_y4
            })
        }
    return None

def model_evaluate(model, data):
    if isinstance(model, dict):
        print(f"\n================ {model['name'].upper()} STRICT DEPENDENT EVALUATION ================")
        preds = model["predictions"]
        
        y_true_y2 = data.y_test[Config.TYPE_COLS[0]].values
        y_true_y3 = data.y_test[Config.TYPE_COLS[1]].values
        y_true_y4 = data.y_test[Config.TYPE_COLS[2]].values
        
        y_pred_y2 = preds[Config.TYPE_COLS[0]].values
        y_pred_y3 = preds[Config.TYPE_COLS[1]].values
        y_pred_y4 = preds[Config.TYPE_COLS[2]].values
        
        # Accuracy depends strictly on previous nodes satisfying constraints sequentially
        correct_y2 = (y_pred_y2 == y_true_y2)
        correct_y3 = correct_y2 & (y_pred_y3 == y_true_y3)
        correct_y4 = correct_y3 & (y_pred_y4 == y_true_y4)
        
        acc_y2 = np.mean(correct_y2)
        acc_y3 = np.mean(correct_y3)
        acc_y4 = np.mean(correct_y4)
        
        print(f"Strict Accuracy Level 1 ({Config.TYPE_COLS[0]}): {acc_y2:.2%}")
        print(f"Strict Accuracy Level 2 Chained ({Config.TYPE_COLS[1]}): {acc_y3:.2%}")
        print(f"Strict Accuracy Level 3 Full Depth ({Config.TYPE_COLS[2]}): {acc_y4:.2%}")
        print("==========================================================================")
        return {"Type 2": acc_y2, "Type 3": acc_y3, "Type 4": acc_y4}
    else:
        model.print_results(data)
        return None