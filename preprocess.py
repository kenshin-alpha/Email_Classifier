#Methods related to data loading and all pre-processing steps will go here

import os
import pandas as pd
from Config import Config

def get_input_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df1 = pd.read_csv(os.path.join(base_dir, "data/AppGallery.csv"))
    df2 = pd.read_csv(os.path.join(base_dir, "data/Purchasing.csv"))
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.dropna(subset=Config.TYPE_COLS)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].fillna("")
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].fillna("")
    return df

def remove_duplicates(df):
    return df.drop_duplicates()

def to_lower(df):
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].str.lower()
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].str.lower()
    return df