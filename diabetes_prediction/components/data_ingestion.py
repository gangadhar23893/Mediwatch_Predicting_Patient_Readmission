import os
import sys
import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()


from diabetes_prediction.exception.exception import DiabetesPredictionException
from diabetes_prediction.logging.logger import logging

from diabetes_prediction.entity.config_entity import DataIngestionConfig
from diabetes_prediction.entity.artifact_entity import DataIngestionArtifact


MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        
        except Exception as e:
            raise DiabetesPredictionException(e,sys)
        

    def export_collection_as_dataframe(self):
        #read the data from mongodb

        try:
            database_name = self.data_ingestion_config.database_name
            collection_name  = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df= df.drop(columns=["_id"],axis=1)

            #df.replace({'?': np.nan},inplace=True)
            #df = df.drop(columns=['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty',
                                   #'repaglinide', 'nateglinide', 'chlorpropamide', 'acarbose', 'miglitol', 
                                   #'troglitazone', 'tolazamide', 'examide', 'citoglipton',
                                    # 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 
                                    # 'metformin-rosiglitazone','metformin-pioglitazone', 'acetohexamide',
                                     #  'tolbutamide','diag_1',
                    #  'diag_2','diag_3'],axis=1)
            return df
        
        except Exception as e:
            raise DiabetesPredictionException(e,sys)
        
    def export_data_into_feature_store(self,dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path =os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)

            return dataframe
        

        except Exception as e:
            raise DiabetesPredictionException(e,sys)
        
    def split_data_train_test(self,dataframe : pd.DataFrame):

        try:
            train_set,test_set = train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio)

            logging.info("performed train test split on dataframe")
            logging.info("Exited split_data_as_train_test method of Data_Ingestion class")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)

            logging.info(f"Exporting train and test file path.")

            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)

            logging.info(f"Exported train and test file path.")

        except Exception as e:
            raise DiabetesPredictionException(e,sys)
        
    def initiate_data_ingestion(self):

        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_train_test(dataframe)
            dataingestionartifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                        test_file_path=self.data_ingestion_config.testing_file_path)
            return dataingestionartifact

        except Exception as e:
            raise DiabetesPredictionException(e,sys)
