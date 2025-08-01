from diabetes_prediction.components.data_ingestion import DataIngestion
from diabetes_prediction.components.data_validation import DataValidation
from diabetes_prediction.components.model_trainer import ModelTrainer
from diabetes_prediction.exception.exception import DiabetesPredictionException
from diabetes_prediction.components.data_transformation import DataTransformation
from diabetes_prediction.logging.logger import logging
from diabetes_prediction.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from diabetes_prediction.entity.config_entity import TrainingPipelineConfig,ModelTrainerConfig
import sys

if __name__ == '__main__':
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        print(dataingestionartifact)

        data_validation_config=DataValidationConfig(trainingpipelineconfig)
        data_validation=DataValidation(dataingestionartifact,data_validation_config)
        logging.info("Initiate the data Validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("data Validation Completed")
        print(data_validation_artifact)

        data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
        logging.info("Data transformation started")
        data_transformation = DataTransformation(data_validation_artifact,data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed")
        print(data_transformation_artifact)
        

        logging.info("Model Training started")
        model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model Training Artifact created")

    except Exception as e:
        raise DiabetesPredictionException(e,sys)