import os,sys

from diabetes_prediction.exception.exception import DiabetesPredictionException
from diabetes_prediction.logging.logger import logging 

from diabetes_prediction.constant.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME

class NetworkModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise DiabetesPredictionException(e,sys)
        
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise DiabetesPredictionException(e,sys)
        
        