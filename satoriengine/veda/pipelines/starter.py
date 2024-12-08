import pandas as pd
from typing import Union
from collections import namedtuple
from sklearn.linear_model import LinearRegression
from satoriengine.veda.pipelines.interface import PipelineInterface, TrainingResult


class StarterPipeline(PipelineInterface):

    @staticmethod
    def condition(*args, **kwargs) -> float:
        if len(kwargs.get('data', [])) < 10:
            return 1.0
        return 0.0

    def __init__(self, **kwargs):
        super().__init__()
        self.model = None

    def load(self, modelPath: str, **kwargs) -> Union[None, "PipelineInterface"]:
        """loads the model model from disk if present"""

    def save(self, modelpath: str, **kwargs) -> bool:
        """saves the stable model to disk"""
        return True

    def fit(self, data: pd.DataFrame, **kwargs) -> TrainingResult:
        # we don't need to fit anything
        #forecast = StarterPipeline.starterEnginePipeline(data)
        #if self.model is None:
        #    self.model = forecast
        return TrainingResult(-1, self)

    def compare(self, other: PipelineInterface, **kwargs) -> bool:
        return True

    def score(self, **kwargs) -> float:
        return 0.0

    def predict(self, data, **kwargs) -> Union[None, pd.DataFrame]:
        """prediction without training"""
        return StarterPipeline.starterEnginePipeline(data)

    @staticmethod
    def starterEnginePipeline(starterDataset: pd.DataFrame) -> pd.DataFrame:
        """Starter Engine function for the Satori Engine"""
        #result = namedtuple(
        #    "Result",
        #    ["forecast", "backtest_error", "model_name", "unfitted_forecaster"])
        if starterDataset is None or len(starterDataset) == 0:
            return pd.DataFrame({
                "ds": [pd.Timestamp.now() + pd.Timedelta(days=1)],
                "pred": [0]})
        if len(starterDataset) == 1:
            # If dataset has only 1 row, return the same value in the forecast dataframe
            value = starterDataset.iloc[0, 1]
            return pd.DataFrame({
                "ds": [pd.Timestamp.now() + pd.Timedelta(days=1)],
                "pred": [value]})
        if len(starterDataset) <= 4:
            # If dataset has 2-4 rows, return the average of the last 2
            value = starterDataset.iloc[-2:, 1].mean()
            return pd.DataFrame({
                "ds": [pd.Timestamp.now() + pd.Timedelta(days=1)],
                "pred": [value]})
        # If dataset has more than 4 rows, use linear regression
        x = starterDataset.index.values.reshape(-1, 1)
        y = starterDataset.iloc[:, 1].values
        model = LinearRegression()
        model.fit(x, y)
        next_time = starterDataset.index[-1] + 1
        predicted_value = model.predict([[next_time]])[0]
        return pd.DataFrame({
            "ds": [pd.Timestamp.now() + pd.Timedelta(days=1)],
            "pred": [predicted_value]})
