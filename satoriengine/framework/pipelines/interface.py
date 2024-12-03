import pandas as pd
from typing import Union, Optional, Any
import joblib
import os
from satorilib.logging import error, debug


class TrainingResult:

    def __init__(self, status, model: "PipelineInterface", stagnated: bool = False):
        self.status = status
        self.model = model
        self.stagnated = stagnated


class PipelineInterface:

    def __init__(self, *args, **kwargs):
        self.model = None

    def load(self, modelPath: str, **kwargs) -> Union[None, "PipelineInterface"]:
        """
        loads the model model from disk if present

        Args:
            modelpath: Path where the model should be loaded from

        Returns:
        PipelineInterface: Model if load successful, None otherwise
        """

    def save(self, modelpath: str, **kwargs) -> bool:
        """
        Save the model to disk.

        Args:
            model: The model to save
            modelpath: Path where the model should be saved

        Returns:
            bool: True if save successful, False otherwise
        """
        pass

    def fit(self, **kwargs) -> TrainingResult:
        """
        Train a new model.

        Args:
            **kwargs: Keyword arguments including datapath and stable model

        Returns:
            TrainingResult: Object containing training status and model
        """
        pass

    def compare(self, stable: Optional[Any] = None, **kwargs) -> bool:
        """
        Compare stable (model) and pilot models based on their backtest error.

        Args:
            stable: The current stable model
            replace: Whether to replace stable with pilot if pilot performs better

        Returns:
            bool: True if pilot should replace stable, False otherwise
        """
        pass

    def score(self, **kwargs) -> float:
        """
        will score the model.
        """
        pass

    def predict(self, **kwargs) -> Union[None, pd.DataFrame]:
        """
        Make predictions using the stable model

        Args:
            **kwargs: Keyword arguments including datapath and stable model

        Returns:
            Optional[pd.DataFrame]: Predictions if successful, None otherwise
        """
        pass
