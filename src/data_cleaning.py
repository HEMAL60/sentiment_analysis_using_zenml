import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    #this is an abstract method which will be inherited by other strategy classes
    @abstractmethod
    def handle_data(self,data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]: #this is a constructor that will be inherited by other child strategy classes
        pass

class DataPreProcessStrategy(DataStrategy):
    # this strategy is responsible for data pre processing steps like removing unwanted columns and filling up all the null values
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:  #ovriden method of abstract strategy class
        
        try:
            data= data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp"
                ],
                axis=1
            )  #droping unwanted columns

            
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["review_comment_message"].fillna("No review", inplace=True)
            #filling null values

            data = data.select_dtypes(include=[np.number])
            cols_to_drop =["customer_zip_code_prefix" ,"order_item_id"]  #droping unwanted columns
            data = data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error("Error in preprocessing data: f)".format(e))
            raise e
        
class DataDivideStartegy(DataStrategy):
    #this is a strategy to divide and split our dataset into training and testing

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]: #here expected return type is expected a union of data frame and series as Xtrain and Xtest is 
        #data frame and y_train and y_test is pd.Series

        try:
            X= data.drop(["review_score"], axis=1)
            y= data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data: f)".format(e))
            raise e

class DataCleaning:
    """
    this class will make use of above created strategies, it will preprocess the data and also it will split the dataset into two 
    parts i.e. tarining and testing 
    """
    def __init__(self,data: pd.DataFrame,strategy: DataStrategy)->None:
        self.df = data
        self.strategy = strategy

    def handle_data(self)->Union[pd.DataFrame,pd.Series]:
        return self.strategy.handle_data(self.df)
        