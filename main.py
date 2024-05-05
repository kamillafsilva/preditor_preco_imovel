from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import Literal
#import os
#import sys

from pyspark.sql import SparkSession

from pyspark.sql import functions as f
from pyspark.sql.types import StringType

from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml import PipelineModel

class ModelPayload(BaseModel):
    unit: Literal['Apartamento', 'Casa']
    bathrooms: int
    bedrooms: int
    parkingSpaces: int
    suites: int
    usableAreas: float
    zone: Literal['Zona Oeste', 'Zona Norte', 'Zona Sul', 'Zona Central']
    condo: float

spark = SparkSession.builder.master('local[*]').getOrCreate()


loaded_pipeline = PipelineModel.load('pipeline_fitted')
loaded_model = RandomForestRegressionModel.load('rf_fitted')

app = FastAPI()

@app.post('/predict_price')
async def predict(payload: ModelPayload):

  input =  [(payload.unit, payload.bathrooms, 
             payload.bedrooms, payload.parkingSpaces, payload.suites, 
             payload.usableAreas, payload.zone, payload.condo, 0)] 
          
  col_names = ['unit', 'bathrooms', 'bedrooms', 'parkingSpaces', 'suites', 'usableAreas', 'zone', 'condo_imputed', 'target']           

  x = spark.createDataFrame(data = input, schema = col_names)

  num = [f.name for f in x.schema.fields if not isinstance(f.dataType, StringType)] 

  x_transformed = x
  for col in num:
    x_transformed = x_transformed.withColumn(col, f.log(x[col] + 1).alias(col))

  x_scaled = loaded_pipeline.transform(x).select('features_minmax', 'target')
  pred = loaded_model.transform(x_scaled)

  inv_pred = pred
  inv_pred = pred.withColumn('prediction', (f.exp(pred['prediction']) - 1).alias('inv_prediction'))

  y = round(inv_pred.collect()[0][2], 0)

  return y

if __name__=='__main__':
    uvicorn.run('main:app', reload=True)


