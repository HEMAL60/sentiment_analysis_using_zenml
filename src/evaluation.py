import logging 
from abc import ABC ,abstractmethod
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """Abstract class defining strategy for evaluation our models"""

    @abstractmethod
    def calcualte_scores(self,y_true: np.ndarray,y_pred : np.ndarray):
        """
        calculates the scores of the model
        args : 
            y_true: True labels
            y_pred: predicted labels
        returns:
            None
        """

    pass

class MSE(Evaluation):

    """evalution strategy that uses Mean Squared Error """

    def calcualte_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("calculating MSE")
            mse = mean_squared_error(y_true,y_pred)
            logging.info(f"MSE : {mse}")
            return mse
        except Exception as e:
            logging.error(f"error in calculating MSE {e}")
            raise e
        

class R2(Evaluation):

    def calcualte_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("calculating R2 score")
            r2 = r2_score(y_true,y_pred)
            logging.info(f"r2 score : {r2}")
            return r2
        except Exception as e:
            logging.error(f"error in calculating r2 {e}")
            raise e
        
class RMSE(Evaluation):

    def calcualte_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("calculating MSE")
            mse = mean_squared_error(y_true,y_pred,squared=False)
            logging.info(f"MSE : {mse}")
            return mse
        except Exception as e:
            logging.error(f"error in calculating MSE {e}")
            raise e