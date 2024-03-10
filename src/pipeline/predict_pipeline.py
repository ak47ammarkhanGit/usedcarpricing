import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        length: float,
        width: float,
        curb_weight: int,
        engine_size: int,
        horsepower: int,
        city_L_per_100km: float,
        highway_L_per_100km: float,
        wheel_base: float,
        bore: float,
        drive_wheels: str
        ):

        self.length = length

        self.width = width

        self.curb_weight = curb_weight

        self.engine_size = engine_size

        self.horsepower = horsepower

        self.city_L_per_100km = city_L_per_100km

        self.highway_L_per_100km = highway_L_per_100km

        self.wheel_base = wheel_base

        self.bore = bore

        self.drive_wheels = drive_wheels

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "length": [self.length],
                "width": [self.width],
                "curb_weight": [self.curb_weight],
                "engine_size": [self.engine_size],
                "horsepower": [self.horsepower],
                "city_L_per_100km": [self.city_L_per_100km],
                "highway_L_per_100km": [self.highway_L_per_100km],
                "wheel_base": [self.wheel_base],
                "bore": [self.bore],
                "drive_wheels": [self.drive_wheels],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)