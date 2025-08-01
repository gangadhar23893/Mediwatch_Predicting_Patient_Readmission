import sys
import os

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()

mongo_db_url = os.getenv("MONGODB_URL_KEY")
print(mongo_db_url)

import pymongo
from diabetes_prediction.logging.logger import logging
from diabetes_prediction.exception.exception import DiabetesPredictionException
from diabetes_prediction.pipeline.training_pipeline import TrainingPipeline
from diabetes_prediction.utils.ml_utils.model.estimator import NetworkModel

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from diabetes_prediction.utils.main_utils.utils import load_object

client = pymongo.MongoClient(mongo_db_url, tlsCAFile = ca)

from diabetes_prediction.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME,DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory=r"C:\Users\gthatava\OneDrive - Capgemini\Desktop\PYTHON\1.SWITCHUP\MLOPS\Diabetes_prediction\templates")

@app.get("/", tags = ["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is succesful")
    except Exception as e:
        raise DiabetesPredictionException(e,sys)
    
@app.post("/predict")
async def predict_route(request:Request,file: UploadFile=File(...)):
    try:
        df = pd.read_csv(file.file)

        preprocessor = load_object(r"C:\Users\gthatava\OneDrive - Capgemini\Desktop\PYTHON\1.SWITCHUP\MLOPS\Diabetes_prediction\final_models\preprocessor.pkl")
        final_model= load_object(r"C:\Users\gthatava\OneDrive - Capgemini\Desktop\PYTHON\1.SWITCHUP\MLOPS\Diabetes_prediction\final_models\model.pkl")

        network_model = NetworkModel(preprocessor=preprocessor,model=final_model)

        y_pred = network_model.predict(df)

        df['predicted_column'] = y_pred
        print(df['predicted_column'])

        df.to_csv(r"C:\Users\gthatava\OneDrive - Capgemini\Desktop\PYTHON\1.SWITCHUP\MLOPS\Diabetes_prediction\prediction_output\output.csv")

        table_html = df.to_html(classes="table table-striped")

        return templates.TemplateResponse("table.html",{"request":request, "table": table_html})
    
    except Exception as e:
        raise DiabetesPredictionException(e,sys)


    

if __name__ == "__main__":
    app_run(app,host = "localhost" , port = 8000)

    
