import pandas as pd
from typing import Union, Optional, Any
from collections import namedtuple
from satoriengine.veda.pipelines.interface import PipelineInterface, TrainingResult


class StarterPipeline(PipelineInterface):

    def __init__(self, **kwargs):
        self.model = None

    @staticmethod
    def load(modelPath: str, **kwargs) -> Union[None, "PipelineInterface"]:
        """loads the model model from disk if present"""

    def save(self, modelpath: str, **kwargs) -> bool:
        """saves the stable model to disk"""
        return True

    def fit(self, **kwargs) -> TrainingResult:
        if self.model is None:
            status, model = StarterPipeline.starterEnginePipeline(kwargs["data"])
            if status == 1:
                self.model = model
                return TrainingResult(status, model, False)
        else:
            return TrainingResult(1, self.model, True)

    def compare(self, other: Union[PipelineInterface, None] = None, **kwargs) -> bool:
        """true indicates this model is better than the other model"""
        return True

    def score(self, **kwargs) -> float:
        pass

    def predict(self, **kwargs) -> Union[None, pd.DataFrame]:
        """prediction without training"""
        status, predictor_model = StarterPipeline.starterEnginePipeline(kwargs["data"])
        if status == 1:
            return predictor_model[0].forecast
        return None

    @staticmethod
    def starterEnginePipeline(starter_dataset: pd.DataFrame):
        """Starter Engine function for the Satori Engine"""
        result = namedtuple(
            "Result",
            ["forecast", "backtest_error", "model_name", "unfitted_forecaster"],
        )

        forecast = None

        if len(starter_dataset) == 1:
            # If dataset has only 1 row, return the same value in the forecast dataframe
            value = starter_dataset.iloc[0, 1]
            forecast = pd.DataFrame(
                {"ds": [pd.Timestamp.now() + pd.Timedelta(days=1)], "pred": [value]}
            )
        elif len(starter_dataset) == 2:
            # If dataset has 2 rows, return their average
            value = starter_dataset.iloc[:, 1].mean()
            forecast = pd.DataFrame(
                {"ds": [pd.Timestamp.now() + pd.Timedelta(days=1)], "pred": [value]}
            )
        starter_result = result(
            forecast=forecast,
            backtest_error=20,
            model_name="starter_dataset_model",
            unfitted_forecaster=None,
        )
        return 1, [starter_result]
