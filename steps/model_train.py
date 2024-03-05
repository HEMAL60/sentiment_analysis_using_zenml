import logging
import pandas as pd

from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test:pd.Series,
    config: ModelNameConfig,
                )->RegressorMixin:
    
    """
    trains the model on ingested data 

    args : X_train,X_test,y_train,y_test
    """

    try:
        model = None
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            train_model = model.train(X_train,y_train)
            return trained_model

        else:
            raise ValueError("model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error("Error in training model :{}".format(e))
        raise e
   