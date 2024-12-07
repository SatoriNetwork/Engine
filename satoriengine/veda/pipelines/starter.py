import pandas as pd
from typing import Union, Optional, Any
from collections import namedtuple
from satoriengine.veda.pipelines.interface import PipelineInterface, TrainingResult


class StarterPipeline(PipelineInterface):

    @staticmethod
    def condition(*args, **kwargs) -> float:
        if kwargs.get('dataCount', 0) < 5:
            return 1.0
        return 0.0

    def __init__(self, **kwargs):
        self.model = None

    def load(self, modelPath: str, **kwargs) -> Union[None, "PipelineInterface"]:
        """loads the model model from disk if present"""

    def save(self, modelpath: str, **kwargs) -> bool:
        """saves the stable model to disk"""
        return True

    def fit(self, data: pd.DataFrame, **kwargs) -> TrainingResult:
        if self.model is None:
            status, model = StarterPipeline.starterEnginePipeline(data)
            if status == 1:
                self.model = model
                return TrainingResult(status, self)
        else:
            return TrainingResult(0, self)

    def compare(self, other: PipelineInterface, **kwargs) -> bool:
        return True

    def score(self, **kwargs) -> float:
        return 0.0

    def predict(self, data, **kwargs) -> Union[None, pd.DataFrame]:
        """prediction without training"""
        status, predictorModel = StarterPipeline.starterEnginePipeline(data)
        if status == 1:
            return predictorModel[0].forecast
        return None

    @staticmethod
    def starterEnginePipeline(starterDataset: pd.DataFrame) -> tuple[int, list]:
        """Starter Engine function for the Satori Engine"""
        result = namedtuple(
            "Result",
            ["forecast", "backtest_error", "model_name", "unfitted_forecaster"])
        forecast = None
        if len(starterDataset) == 0:
            return 1, [0]
        elif len(starterDataset) == 1:
            # If dataset has only 1 row, return the same value in the forecast dataframe
            value = starterDataset.iloc[0, 1]
            forecast = pd.DataFrame({
                "ds": [pd.Timestamp.now() + pd.Timedelta(days=1)],
                "pred": [value]})
        else:
            # If dataset has 2 or more rows, return the average of the last 2
            value = starterDataset.iloc[-2:, 1].mean()
            forecast = pd.DataFrame({
                "ds": [pd.Timestamp.now() + pd.Timedelta(days=1)],
                "pred": [value]})
        starterResult = result(
            forecast=forecast,
            backtest_error=20000,
            model_name="starterDataset_model",
            unfitted_forecaster=None)
        return 1, [starterResult]
