import logging 
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning,DataPreProcessStrategy,DataDivideStartegy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_data_df(df:pd.DataFrame)->Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"],
]:
    
    """
    cleans the data and divides the dataset into train and test

    Arguments : df(Raw data)
    returns: X_train,X_test, y_train,y_test
    """
    try:
        process_startegy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df,process_startegy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStartegy()
        data_cleaning = DataCleaning(processed_data,divide_strategy)
        X_train,X_test,y_train,y_test = data_cleaning.handle_data()
        return X_train,X_test,y_train,y_test
    except Exception as e:
        logging.error(f"erroring in cleaning data {e}")
        raise e