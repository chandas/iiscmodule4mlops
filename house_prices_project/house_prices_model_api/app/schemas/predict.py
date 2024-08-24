from typing import Any, List, Optional
import datetime

from pydantic import BaseModel
from house_prices_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                "size": 800,
                "bedrooms": 1,
                "age": 10,	
                "distance": 1,
                    }
                ]
            }
        }
