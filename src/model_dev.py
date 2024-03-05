import logging
from abc import ABC, abstractmethod
from sklearn import LinearRegression 

class Model(ABC):
    """
    Abstract class for all the models
    """
    @abstractmethod
    def train(self,X_train,y_train):
        pass

class LinearRegressionModel(Model):

    def train(self, X_train, y_train,**kwargs):
        """
        Trains the model 
        args: X_train, y_train

        returns : None
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train,y_train)
            logging.info("model training completed")
            return reg
        except Exception as e:
            logging.error(f"error while training the model {e}")
            raise e
        
"""we can add other classes related to algorithms and can use them for training"""