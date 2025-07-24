import os
import sys
import json
import numpy as np
import pandas as pd
import certifi
import pymongo
from diabetes_prediction.exception.exception import DiabetesPredictionException
from diabetes_prediction.logging.logger import logging

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where()

class DiabetesDataExtract():

    def __init__(self):

        try:
            pass
        
        except Exception as e:
            raise DiabetesPredictionException(e,sys)
        
    def csv_to_json_converter(self,file_path):

        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        
        except Exception as e:
            raise DiabetesPredictionException(e,sys)
        
    def insert_data_mongodb(self,records,database,collection,batch_size=1000,retries=3):
        
        try:
            self.database = database
            self.collection = collection
            self.records = records

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, tls=True, tlsAllowInvalidCertificates=True, tlsCAFile=ca)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            #self.collection.insert_many(self.records)
            # Optional: clear old data
            self.collection.delete_many({})

            # Insert in batches
            for i in range(0, len(self.records), batch_size):
                batch = self.records[i:i + batch_size]
                for attempt in range(retries):
                    try:
                        self.collection.insert_many(batch)
                        break  # Success
                    except pymongo.errors.AutoReconnect as e:
                        print(f"[Retry {attempt+1}/{retries}] AutoReconnect: Retrying in 3s...")
                        time.sleep(3)
                        if attempt == retries - 1:
                            raise e
    
            return len(self.records)
        
        except Exception as e:
            raise DiabetesPredictionException(e,sys)
        
if __name__ == '__main__':

    FILE_PATH = "Diabestes_patient_data\\diabetic_data.csv"
    DATABASE = "MYPROJECTDB"
    Collection="DIABETIC_PREDICTION_DATA"

    dataobj = DiabetesDataExtract()
    records = dataobj.csv_to_json_converter(file_path=FILE_PATH)
    #print(records)
    no_of_records=dataobj.insert_data_mongodb(records,DATABASE,Collection)
    print(no_of_records)


        
