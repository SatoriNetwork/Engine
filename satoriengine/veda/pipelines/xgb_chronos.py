'''
run chronos on the data
produce a feature of predictions
feed data and chronos predictions into xgboost
'''

import pandas as pd
from typing import Union
import numpy as np
import joblib
import os

from satoriengine.veda.pipelines.interface import PipelineInterface, TrainingResult
#from satoriengine.veda.pipelines.chronos import ChronosVedaPipeline
from satoriengine.veda.process import process_data
from satorilib.logging import info, debug, error, warning, setup, DEBUG

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


class XgbChronosVedaPipeline(PipelineInterface):

    def __init__(self, **kwargs):
        self.model: Union[XGBRegressor, None] = None
        #self.chronos: Union[ChronosVedaPipeline, None] = None
        self.hyperparameters: Union[dict, None] = None
        self.train_x: pd.DataFrame = None
        self.test_x: pd.DataFrame = None
        self.train_y: np.ndarray = None
        self.test_y: np.ndarray = None
        self.X_full: pd.DataFrame = None
        self.y_full: pd.Series = None
        self.split: float = None
        self.model_error: float = None

    def load(self, modelPath: str, **kwargs) -> Union[None, XGBRegressor]:
        """loads the model model from disk if present"""
        try:
            saved_state = joblib.load(modelPath)
            self.model = saved_state['stable_model']
            self.model_error = saved_state['model_error']
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
            self.model_error = self.score()
            state = {
                'stable_model' : self.model,
                'model_error' : self.model_error
            }
            joblib.dump(state, modelpath)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False


    def fit(self, **kwargs) -> TrainingResult:
        """Train a new model"""
        proc_data = process_data(kwargs["data"], quick_start=False)
        pre_train_x, pre_test_x, self.train_y, self.test_y = train_test_split(
            proc_data.dataset.index.values,
            proc_data.dataset['value'],
            test_size=self.split or 0.2,
            shuffle=False,
        )
        self.train_x = self._prepare_time_features(pre_train_x)
        self.test_x = self._prepare_time_features(pre_test_x)
        self.hyperparameters = self._prep_params()
        # append the chronos predictions to the training data
        #   hour  day  month  year  day_of_week
        # 0       19    1     10  2024            1
        # 1       19    1     10  2024            1
        # 2       19    1     10  2024            1
        # 3       19    1     10  2024            1
        # 4       20    1     10  2024            1
        # ...    ...  ...    ...   ...          ...
        # 7300    12   21     11  2024            3
        # 7301    12   21     11  2024            3
        # 7302    12   21     11  2024            3
        # 7303    12   21     11  2024            3
        # 7304    12   21     11  2024            3
        self.model = XGBRegressor(**self.hyperparameters)
        self.model.fit(
            self.train_x,
            self.train_y,
            eval_set=[(self.train_x, self.train_y), (self.test_x, self.test_y)],
            verbose=False,
        )
        return TrainingResult(1, self, False)

    def compare(self, other: Union[PipelineInterface, None] = None, **kwargs) -> bool:
        """
        Compare other (model) and pilot models based on their backtest error.
        Returns True if pilot model performs better than other model.
        """
        if not isinstance(other, self.__class__):
            return True
        this_score = self.score()
        other_score = other.model_error or other.score()
        is_improved = this_score < other_score
        if is_improved:
            info(
                'model improved!'
                f'\n  other score: {other_score}'
                f'\n  pilot  score: {this_score}'
                f'\n  Parameters: {self.hyperparameters}',
                color='green')
        else:
            debug(
                f'\nother score: {other_score}'
                f'\npilot  score: {this_score}',
                color='yellow')
        return is_improved

    def score(self, **kwargs) -> float:
        """will score the model"""
        if self.model is None:
            return np.inf
        self.model_error = mean_absolute_error(self.test_y, self.model.predict(self.test_x))
        return self.model_error

    def predict(self, **kwargs) -> pd.DataFrame:
        """Make predictions using the stable model"""
        proc_data = process_data(kwargs["data"], quick_start=False)
        self.X_full = self._prepare_time_features(proc_data.dataset.index.values)
        self.y_full = proc_data.dataset['value']
        self.model.fit(
            self.X_full,
            self.y_full,
            verbose=False,
        )
        last_date = pd.Timestamp(proc_data.dataset.index[-1])
        future_predictions = self._predict_future(
            self.model, last_date, proc_data.sampling_frequency
        )
        return future_predictions

    def _predict_future(
        self,
        model: XGBRegressor,
        last_date: pd.Timestamp,
        sf: str = 'H',
        periods: int = 168,
    ) -> pd.DataFrame:
        """Generate predictions for future periods"""
        future_dates = pd.date_range(
            start=pd.Timestamp(last_date) + pd.Timedelta(sf), periods=periods, freq=sf
        )
        future_features = self._prepare_time_features(future_dates)
        predictions = model.predict(future_features)
        results = pd.DataFrame({'date_time': future_dates, 'pred': predictions})
        return results

    @staticmethod
    def _prepare_time_features(dates: np.ndarray) -> pd.DataFrame:
        """Convert datetime series into numeric features for XGBoost"""
        df = pd.DataFrame({'date_time': pd.to_datetime(dates)})
        df['hour'] = df['date_time'].dt.hour
        df['day'] = df['date_time'].dt.day
        df['month'] = df['date_time'].dt.month
        df['year'] = df['date_time'].dt.year
        df['day_of_week'] = df['date_time'].dt.dayofweek
        return df.drop('date_time', axis=1)

    @staticmethod
    def _prep_params() -> dict:
        """
        Generates randomized hyperparameters for XGBoost within reasonable ranges.
        Returns a dictionary of hyperparameters.
        """
        param_ranges = {
            'n_estimators': (100, 1000),
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'min_child_weight': (1, 7),
            'gamma': (0, 0.5),
        }

        params = {
            'random_state': np.random.randint(0, 10000),
            'eval_metric': 'mae',
            'learning_rate': np.random.uniform(
                param_ranges['learning_rate'][0], param_ranges['learning_rate'][1]
            ),
            'subsample': np.random.uniform(
                param_ranges['subsample'][0], param_ranges['subsample'][1]
            ),
            'colsample_bytree': np.random.uniform(
                param_ranges['colsample_bytree'][0], param_ranges['colsample_bytree'][1]
            ),
            'gamma': np.random.uniform(
                param_ranges['gamma'][0], param_ranges['gamma'][1]
            ),
            'n_estimators': np.random.randint(
                param_ranges['n_estimators'][0], param_ranges['n_estimators'][1]
            ),
            'max_depth': np.random.randint(
                param_ranges['max_depth'][0], param_ranges['max_depth'][1]
            ),
            'min_child_weight': np.random.randint(
                param_ranges['min_child_weight'][0], param_ranges['min_child_weight'][1]
            ),
        }

        return params
