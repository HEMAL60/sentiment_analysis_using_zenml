import pandas as pd
import logging
from zenml import step

@step
def evaluate_model(df:pd.DataFrame)->None:
    pass