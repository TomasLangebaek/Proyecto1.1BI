from joblib import load
from pydantic import BaseModel
import pandas as pd


class SvmModel:

    def __init__(self):
        self.model = load("assets/svm_pipeline.joblib")

    def make_predictions(self, data):
        data = [data['study_and_condition'].squeeze()]
        result = self.model.predict(data)
        return result


class DataModel(BaseModel):
    study_and_condition: str
