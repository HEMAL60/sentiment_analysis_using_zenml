import logging
import pandas as pd
from zenml import step


class Ingest_data:  #this class is responsible to ingest data and get data in pd.Dataframe format
    def __init__(self,data_path: str):
        self.data_path = data_path

    def get_data(self):
        logging.info(f"ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step        
 #this steps is used to ingest data ,in simple terms it is used to load  dataset from a particular path
def ingest_data_from_path(data_path: str)->pd.DataFrame:
    try:
        return Ingest_data(data_path).get_data()
    except Exception as e:
        logging.error(f"error while ingesting the data {e}")
        raise e