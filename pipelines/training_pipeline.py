from zenml import pipeline
from steps.ingest_data import ingest_data_from_path
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from steps.clean_data import clean_data_df

@pipeline
def train_pipeline(data_path:str):
    df = ingest_data_from_path(data_path)
    clean_data_df(df)
    train_model(df)
    evaluate_model(df)