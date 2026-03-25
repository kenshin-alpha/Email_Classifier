#This is a main file: The controller. All methods will directly on directly be called here
from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)


def load_data():
    #load the input data
    df = get_input_data()
    return  df

def preprocess_data(df):
    # De-duplicate input data
    df =  de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame, name):
    model = model_predict(data, df, name)
    if model is not None:
        return model_evaluate(model, data)
    return None
# Code will start executing from following line
if __name__ == '__main__':
    
    # pre-processing steps
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    
    # data transformation
    X, group_df = get_embeddings(df)
    # data modelling
    data = get_data_object(X, df)
    # modelling
    print("\nStarting Assessment Validation Pipeline...")
    res_chained = perform_modelling(data, df, 'chained')
    res_hierarchical = perform_modelling(data, df, 'hierarchical')
    
    print("\n============= FINAL ARCHITECTURE COMPARISON ==============")
    if res_chained and res_hierarchical:
        print(f"Design 1 (Chained Multi-Output) Final Sequential Accuracy: {res_chained.get('Type 4', 0):.2%}")
        print(f"Design 2 (Hierarchical Filter) Final Sequential Accuracy: {res_hierarchical.get('Type 4', 0):.2%}")
        winner = "Design 1 (Chained)" if res_chained.get('Type 4', 0) >= res_hierarchical.get('Type 4', 0) else "Design 2 (Hierarchical)"
        print(f"Optimal Design Decision for Full CA Scenario: {winner}")

