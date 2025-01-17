import pandas as pd
from typing import Union, List
import joblib
import os
import numpy as np
from satorilib.logging import info, debug, warning
from satoriengine.veda.adapters.tinytimemixer.preprocess import ttmDataPreprocess
from satoriengine.veda.adapters.interface import ModelAdapter, TrainingResult
from sktime.forecasting.ttm import TinyTimeMixerForecaster


class SimpleTTMAdapter(ModelAdapter):

    @staticmethod
    def condition(*args, **kwargs) -> float:
        if (
            isinstance(kwargs.get('availableRamGigs'), float)
            and kwargs.get('availableRamGigs') < .1
        ):
            return 1.0
        if len(kwargs.get('data', [])) < 10:
            return 1.0
        return 0.0

    def __init__(self, **kwargs):
        super().__init__()
        self.model: TinyTimeMixerForecaster = None
        self.dataset: pd.DataFrame = None
        

    def load(self, modelPath: str, **kwargs) -> Union[None, TinyTimeMixerForecaster]:
        """loads the model model from disk if present"""
        try:
            saved_state = joblib.load(modelPath)
            self.model = saved_state['stableModel']
            return self.model
        except Exception as e:
            debug(f"Error Loading Model File : {e}", print=True)
            if os.path.isfile(modelPath):
                os.remove(modelPath)
            return None

    def save(self, modelpath: str, **kwargs) -> bool:
        """saves the stable model to disk"""
        try:
            os.makedirs(os.path.dirname(modelpath), exist_ok=True)
            self.modelError = self.score
            state = {
                'stableModel' : self.model,
            }
            joblib.dump(state, modelpath)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def fit(self, data: pd.DataFrame, **kwargs) -> TrainingResult:
        self.dataset, _ = self._manageData(data)
        if self.model is None:
            self.model = TinyTimeMixerForecaster()
        self.model = self.model.fit(y=self.dataset, fh=[1]) # fh is hard_coded here but would be based on forecasting window in the most comprehensive implementation
        return TrainingResult(1, self)

    def compare(self, other: ModelAdapter, **kwargs) -> bool:
        # debug("self.score",self.score(), print=True)
        isImproved = self.score() < other.score()
        # debug(isImproved, print=True)
        return isImproved # You don't need to worry about compare for tinytimemixer

    def score(self, **kwargs) -> float:
        if self.model is None:
            return np.inf
        return 0.0

    def predict(self, data, **kwargs) -> Union[None, pd.DataFrame]:
        """prediction without refit"""
        self.dataset, _ = self._manageData(data)
        predictions = self.model.predict(y=self.dataset)
        # Note: if you wish for a prediction one day in advance and the frequency is less than 1 day, it would require a longer prediction window
        # and taking the prediction result from the date/time that corresponds to one day later.
        resultDf = pd.DataFrame(
            {
            "ds": predictions.index, 
            "pred": predictions.iloc[0][0]
            }
        )
        # debug("Prediction for TTM", resultDf, print=True)
        return resultDf
    
    def _manageData(self, data: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        procData = ttmDataPreprocess(data)
        procData.dataset.drop(['id'], axis=1, inplace=True)
        return procData.dataset, procData.sampling_frequency
        
        