import pandas as pd
from typing import Union, Optional, Any
import joblib
import os
from satorilib.logging import error, debug, info


class TrainingResult:

    def __init__(self, status, model: "PipelineInterface", stagnated: bool = False):
        self.status = status
        self.model = model
        self.stagnated = stagnated


class PipelineInterface:

    def __init__(self, *args, **kwargs):
        self.model = None

    def load(self, modelPath: str, *args, **kwargs) -> Union[None, "PipelineInterface"]:
        """
        loads the model model from disk if present

        Args:
            modelpath: Path where the model should be loaded from

        Returns:
        PipelineInterface: Model if load successful, None otherwise
        """
        pass

    def save(self, modelpath: str, *args, **kwargs) -> bool:
        """
        Save the model to disk.

        Args:
            model: The model to save
            modelpath: Path where the model should be saved

        Returns:
            bool: True if save successful, False otherwise
        """
        pass

    def fit(self, *args, **kwargs) -> TrainingResult:
        """
        Train a new model.

        Args:
            **kwargs: Keyword arguments including datapath and stable model

        Returns:
            TrainingResult: Object containing training status and model
        """
        pass

    def compare(self, other: Optional[Any] = None, *args, **kwargs) -> bool:
        """
        Compare other (model) and pilot models based on their backtest error.
        Args:
            other: The model to compare against, typically the "stable" model
        Returns:
            bool: True if pilot should replace other, False otherwise
            this should return a comparison object which has a bool expression
        """
        if not isinstance(other, self.__class__):
            return True
        this_score = self.score()
        other_score = other.score()
        is_improved = this_score < other_score
        if is_improved:
            info(
                'model improved!'
                f'\n  stable score: {other_score}'
                f'\n  pilot  score: {this_score}',
                color='green')
        else:
            debug(
                f'\nstable score: {other_score}'
                f'\npilot  score: {this_score}',
                color='yellow')
        return is_improved

    def score(self, *args, **kwargs) -> float:
        """
        will score the model.
        """
        pass

    def predict(self, *args, **kwargs) -> Union[None, pd.DataFrame]:
        """
        Make predictions using the stable model

        Args:
            **kwargs: Keyword arguments including datapath and stable model

        Returns:
            Optional[pd.DataFrame]: Predictions if successful, None otherwise
        """
        pass
