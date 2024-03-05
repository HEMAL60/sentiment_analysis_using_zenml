from zenml import pipeline
from steps.ingest_data import ingest_data_from_path
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from steps.clean_data import clean_data_df

@pipeline(enable_cache=True)
def train_pipeline(data_path:str):
    df = ingest_data_from_path(data_path)
    X_train,X_test,y_train,y_test = clean_data_df(df)
    model = train_model(X_train,X_test,y_train,y_test)
    r2_score, rmse = evaluate_model(model,X_test,y_test)
    
    