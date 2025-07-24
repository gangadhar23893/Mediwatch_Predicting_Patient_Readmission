import sys
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
#from diabetes_prediction.constant.training_pipeline import TARGET_COLUMN,COLUMNS_DROP,GENDER_COL,AGE_COL,IMPUTER_COL,OHE_COL,ORDINAL_COL,ORDINAL_ORDER
from diabetes_prediction.constant.training_pipeline import TARGET_COLUMN,COLUMNS_DROP,GENDER_COL,AGE_COL,OHE_COL,ORDINAL_COL,ORDINAL_ORDER,RACE_COL
from diabetes_prediction.entity.config_entity import DataTransformationConfig
from diabetes_prediction.entity.artifact_entity import DataTransformationArtifact,DataValidationArtifact
from diabetes_prediction.logging.logger import logging
from diabetes_prediction.utils.main_utils.utils import save_numpy_array_data,save_object
from diabetes_prediction.exception.exception import DiabetesPredictionException
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise DiabetesPredictionException(e,sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise DiabetesPredictionException(e, sys)
        

    def get_data_transformer_object(self) -> Pipeline:

        try:
            logging.info("entered data preprocessing pipeline")
            """
            preprocessor = ColumnTransformer([('impute',SimpleImputer(strategy ='most_frequent'),IMPUTER_COL),
                                             ('ohe',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),OHE_COL),
                                             ('ord_cat_col',OrdinalEncoder(),ORDINAL_COL)],remainder='passthrough')

            """
            

            preprocessor = ColumnTransformer([('impute',SimpleImputer(strategy ='most_frequent'),RACE_COL),
                                               ('ohe',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),OHE_COL),
                                             ('ord_cat_col',OrdinalEncoder(),ORDINAL_COL)],remainder='passthrough')

            pipe = Pipeline([
                ('preprocessing',preprocessor)
                ])

            return pipe


        except Exception as e:
            raise DiabetesPredictionException(e,sys)
        
        

    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            train_df.replace({'?': np.nan},inplace=True)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            test_df.replace({'?': np.nan},inplace=True)

            #trainig file 
            input_feature_train_df = train_df.drop(columns =TARGET_COLUMN + COLUMNS_DROP,axis=1)
            input_feature_train_df[GENDER_COL]=input_feature_train_df[GENDER_COL].replace({'Unknown/Invalid':0,'Female':0,'Male' : 1})
            input_feature_train_df[AGE_COL]=input_feature_train_df[AGE_COL].replace({'[70-80)':75,
                                                                                     '[60-70)':65,
                                                                                     '[50-60)':55,
                                                                                     '[80-90)':75,
                                                                                     '[40-50)':45,
                                                                                     '[30-40)':35,
                                                                                     '[90-100)':95,
                                                                                     '[20-30)':25,
                                                                                     '[10-20)':15,
                                                                                     '[0-10)':5})
            input_feature_train_df[RACE_COL]=input_feature_train_df[RACE_COL].replace({'Caucasian':0,'AfricanAmerican':1,'Hispanic':2,'Asian':3,'Other':4})
            
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train = preprocessor_object.transform(input_feature_train_df)


            target_feature_train_df = train_df[TARGET_COLUMN].copy()
            target_feature_train_df[TARGET_COLUMN] = target_feature_train_df[TARGET_COLUMN].replace({">30" : 1 , "<30" : 1,'NO':0})

            




            #test file
            input_feature_test_df = test_df.drop(columns =TARGET_COLUMN + COLUMNS_DROP,axis=1)
            input_feature_test_df[GENDER_COL]=input_feature_test_df[GENDER_COL].replace({'Unknown/Invalid':0,'Female':0,'Male' : 1})
            input_feature_test_df[AGE_COL]=input_feature_test_df[AGE_COL].replace({'[70-80)':75,
                                                                                     '[60-70)':65,
                                                                                     '[50-60)':55,
                                                                                     '[80-90)':75,
                                                                                     '[40-50)':45,
                                                                                     '[30-40)':35,
                                                                                     '[90-100)':95,
                                                                                     '[20-30)':25,
                                                                                     '[10-20)':15,
                                                                                     '[0-10)':5})
            input_feature_test_df[RACE_COL]=input_feature_test_df[RACE_COL].replace({'Caucasian':0,'AfricanAmerican':1,'Hispanic':2,'Asian':3,'Other':4})
            
            transformed_input_test = preprocessor_object.transform(input_feature_test_df)

            target_feature_test_df = test_df[TARGET_COLUMN].copy()
            target_feature_test_df[TARGET_COLUMN] = target_feature_test_df[TARGET_COLUMN].replace({">30" : 1 , "<30" : 1,'NO':0})

            train_arr = np.c_[transformed_input_train,np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test,np.array(target_feature_test_df)]

            # save numpy array data

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path,preprocessor_object)

            #preparing artifacts

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )

            return data_transformation_artifact





        except Exception as e:
            raise DiabetesPredictionException(e, sys)
