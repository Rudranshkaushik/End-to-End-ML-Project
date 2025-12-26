import sys 
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass


    def prediction(self, features: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("before loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("after loading")

        # Normalize missing markers to NaN so SimpleImputer can handle them
            features = features.copy()
            features = features.replace({None: np.nan, "None": np.nan, "": np.nan})

        # (Optional) enforce expected column order
        # expected_cols = [
        #     "gender", "race_ethnicity", "parental_level_of_education",
        #     "lunch", "test_preparation_course", "reading_score", "writing_score"
        # ]
        # features = features.reindex(columns=expected_cols)

        # Debug: inspect encoder categories to confirm what was learned during fit
        # (Run this once or under a DEBUG flag)
        # onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        # print([list(c) for c in onehot.categories_])

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)

        
class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)



