from typing import Union
import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from satorilib.logging import info, debug, warning
from satoriengine.veda.process import process_data
from satoriengine.veda.pipelines.interface import PipelineInterface, TrainingResult


class XgbPipeline(PipelineInterface):

    @staticmethod
    def condition(*args, **kwargs) -> float:
        if kwargs.get('cpu', 0) == 1 or len(kwargs.get('data', [])) >= 10_000:
            return 1.0
        return 0.0

    def __init__(self, **kwargs):
        super().__init__()
        self.model: XGBRegressor = None
        self.modelError: float = None
        self.hyperparameters: Union[dict, None] = None
        self.dataset: pd.DataFrame = None
        self.trainX: pd.DataFrame = None
        self.testX: pd.DataFrame = None
        self.trainY: np.ndarray = None
        self.testY: np.ndarray = None
        self.fullX: pd.DataFrame = None
        self.fullY: pd.Series = None
        self.split: float = None
        self.rng = np.random.default_rng(37)

    def load(self, modelPath: str, **kwargs) -> Union[None, XGBRegressor]:
        """loads the model model from disk if present"""
        try:
            saved = joblib.load(modelPath)
            self.model = saved['stableModel']
            self.modelError = saved['modelError']
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
            self.modelError = self.score()
            state = {
                'stableModel' : self.model,
                'modelError' : self.modelError}
            joblib.dump(state, modelpath)
            return True
        except Exception as e:
            warning(f"Error saving model: {e}")
            return False
        

    def compare(self, other: Union[PipelineInterface, None] = None, **kwargs) -> bool:
        """
        Compare other (model) and this models based on their backtest error.
        Returns True if this model performs better than other model.
        """
        if not isinstance(other, self.__class__):
            return True
        thisScore = self.score()
        otherScore = other.modelError or other.score()
        isImproved = thisScore < otherScore
        if isImproved:
            info(
                'model improved!'
                f'\n  stable score: {otherScore}'
                f'\n  pilot  score: {thisScore}'
                f'\n  Parameters: {self.hyperparameters}',
                color='green')
        else:
            debug(
                f'\nstable score: {otherScore}'
                f'\npilot  score: {thisScore}')
        return isImproved

    def score(self, **kwargs) -> float:
        """will score the model"""
        if self.model is None:
            return np.inf
        self.modelError = mean_absolute_error(self.testY, self.model.predict(self.testX))
        return self.modelError

    def fit(self, data: pd.DataFrame, **kwargs) -> TrainingResult:
        """ Train a new model """
        _, _ = self._manageData(data)
        # todo: get ready to combine features from different sources (merge)
        # todo: keep a running dataset and update incrementally w/ process_data
        # todo: linear, if not fractal, interpolation
        preTrainX, preTestX, self.trainY, self.testY = train_test_split(
            self.dataset.index.values,
            self.dataset['value'],
            test_size=self.split or 0.2,
            shuffle=False,
            random_state=37)
        self.trainX = self._prepareTimeFeatures(preTrainX)
        self.testX = self._prepareTimeFeatures(preTestX)
        self.hyperparameters = self._mutateParams(
            prevParams=self.hyperparameters,
            rng=self.rng)
        if self.model is None:
            self.model = XGBRegressor(**self.hyperparameters)
        else:
            self.model.set_params(**self.hyperparameters)
        self.model.fit(
            self.trainX,
            self.trainY,
            eval_set=[(self.trainX, self.trainY), (self.testX, self.testY)],
            verbose=False)
        return TrainingResult(1, self)

    def predict(self, data: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, None]:
        """Make predictions using the stable model"""
        if self.model is None or self.dataset is None:
            return None
        _, samplingFrequency = self._manageData(data)
        self.fullX = self._prepareTimeFeatures(self.dataset.index.values)
        self.fullY = self.dataset['value']
        self.model.fit(self.fullX, self.fullY, verbose=False)
        lastDate = pd.Timestamp(self.dataset.index[-1])
        futurePredictions = self._predictFuture(
            self.model,
            lastDate,
            samplingFrequency)
        return futurePredictions

    def _predictFuture(
        self,
        model: XGBRegressor,
        lastDate: pd.Timestamp,
        sf: str = 'H',
        periods: int = 168,
    ) -> pd.DataFrame:
        """Generate predictions for future periods"""
        futureDates = pd.date_range(
            start=pd.Timestamp(lastDate) + pd.Timedelta(sf),
            periods=periods,
            freq=sf)
        futureFeatures = self._prepareTimeFeatures(futureDates)
        predictions = model.predict(futureFeatures)
        results = pd.DataFrame({'date_time': futureDates, 'pred': predictions})
        return results

    def _manageData(self, data: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        '''
        here we need to merge the chronos predictions with the data, but it
        must be done incrementally because it takes too long to do it on the
        whole dataset everytime so we save the processed data and
        incrementally add to it over time.
        '''

        def updateData(data: pd.DataFrame) -> pd.DataFrame:
            procData = process_data(data, quick_start=False)
            procData.dataset.drop(['id'], axis=1, inplace=True)
            # incrementally add missing processed data rows to the self.dataset
            if self.dataset is None:
                self.dataset = procData.dataset
                self.dataset['chronos'] = np.nan
            else:
                # Identify rows in procData.dataset not present in self.dataset
                missingRows = procData.dataset[~procData.dataset.index.isin(self.dataset.index)]
                # Append only the missing rows to self.dataset
                self.dataset = pd.concat([self.dataset, missingRows])
            return self.dataset, procData.sampling_frequency

        def addPercentageChange(df: pd.DataFrame) -> pd.DataFrame:

            def calculatePercentageChange(df, past):
                return ((df['value'] - df['value'].shift(past)) / df['value'].shift(past)) * 100

            for past in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
                df[f'percent{past}'] = calculatePercentageChange(df, past)
            return df

        self.dataset, samplingFrequency = updateData(data)
        self.dataset = addPercentageChange(self.dataset)
        return self.dataset, samplingFrequency


    @staticmethod
    def _prepareTimeFeatures(dates: np.ndarray) -> pd.DataFrame:
        """Convert datetime series into numeric features for XGBoost"""
        df = pd.DataFrame({'date_time': pd.to_datetime(dates)})
        df['hour'] = df['date_time'].dt.hour
        df['day'] = df['date_time'].dt.day
        df['month'] = df['date_time'].dt.month
        df['year'] = df['date_time'].dt.year
        df['day_of_week'] = df['date_time'].dt.dayofweek
        return df.drop('date_time', axis=1)

    @staticmethod
    def paramBounds() -> dict:
        return {
            'n_estimators': (100, 2000),
            'max_depth': (3, 10),
            'learning_rate': (0.005, 0.3),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'min_child_weight': (1, 10),
            'gamma': (0, 1),
            'scale_pos_weight': (0.5, 10)}

    @staticmethod
    def _prepParams(rng: Union[np.random.Generator, None] = None) -> dict:
        """
        Generates randomized hyperparameters for XGBoost within reasonable ranges.
        Returns a dictionary of hyperparameters.
        """
        paramBounds: dict = XgbPipeline.paramBounds()
        rng = rng or np.random.default_rng(37)
        params = {
            'random_state': rng.integers(0, 10000),
            'eval_metric': 'mae',
            'learning_rate': rng.uniform(
                paramBounds['learning_rate'][0],
                paramBounds['learning_rate'][1]),
            'subsample': rng.uniform(
                paramBounds['subsample'][0],
                paramBounds['subsample'][1]),
            'colsample_bytree': rng.uniform(
                paramBounds['colsample_bytree'][0],
                paramBounds['colsample_bytree'][1]),
            'gamma': rng.uniform(
                paramBounds['gamma'][0],
                paramBounds['gamma'][1]),
            'n_estimators': rng.integers(
                paramBounds['n_estimators'][0],
                paramBounds['n_estimators'][1]),
            'max_depth': rng.integers(
                paramBounds['max_depth'][0],
                paramBounds['max_depth'][1]),
            'min_child_weight': rng.integers(
                paramBounds['min_child_weight'][0],
                paramBounds['min_child_weight'][1]),
            'scale_pos_weight': rng.uniform(
                paramBounds['scale_pos_weight'][0],
                paramBounds['scale_pos_weight'][1])}
        return params

    @staticmethod
    def _mutateParams(
        prevParams: Union[dict, None] = None,
        rng: Union[np.random.Generator, None] = None,
    ) -> dict:
        """
        Tweaks the previous hyperparameters for XGBoost by making random adjustments
        based on a squished normal distribution that respects both boundaries and the
        relative position of the current value within the range.
        Args:
            prevParams (dict): A dictionary of previous hyperparameters.
        Returns:
            dict: A dictionary of tweaked hyperparameters.
        """
        rng = rng or np.random.default_rng(37)
        prevParams = prevParams or XgbPipeline._prepParams(rng)
        paramBounds: dict = XgbPipeline.paramBounds()
        mutatedParams = {}
        for param, (minBound, maxBound) in paramBounds.items():
            currentValue = prevParams[param]
            rangeSpan = maxBound - minBound
            # Generate a symmetric tweak centered on the current value
            stdDev = rangeSpan * 0.1  # 10% of the range as standard deviation
            tweak = rng.normal(0, stdDev)
            # Adjust the parameter and ensure it stays within bounds
            newValue = currentValue + tweak
            newValue = max(minBound, min(maxBound, newValue))
            # Ensure integers for appropriate parameters
            if param in ['n_estimators', 'max_depth', 'min_child_weight']:
                newValue = int(round(newValue))
            mutatedParams[param] = newValue
        # to handle static parameters... we should keep random_state static
        # because we're exploring the hyperparameter state space relative to it
        mutatedParams['random_state'] = prevParams['random_state']
        mutatedParams['eval_metric'] = 'mae'
        return mutatedParams


    @staticmethod
    def _straight_line_interpolation(df, valueColumn, step='10T', scale=0.0, rng: Union[np.random.Generator, None] = None):
        """
        This would probably be better to use than the stepwise pattern as it
        atleast points in the direction of the trend.
        Performs straight line interpolation on missing timestamps.
        Parameters:
        - df: DataFrame with a datetime index and a column to interpolate.
        - valueColumn: The column name with values to interpolate.
        - step: The frequency to use for resampling (e.g., '10T' for 10 minutes).
        Returns:
        - DataFrame with interpolated values.
        """
        # Ensure the DataFrame has a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date_time' in df.columns:
                df['date_time'] = pd.to_datetime(df['date_time'])
                df.set_index('date_time', inplace=True)
            else:
                raise ValueError("The DataFrame must have a DatetimeIndex or a 'date_time' column.")
        # Sort the index and resample
        df = df.sort_index()
        df = df.resample(step).mean()  # Resample to fill in missing timestamps with NaN
        # Perform fractal interpolation
        rng = rng or np.random.default_rng(seed=37)
        for _ in range(5):  # Number of fractal iterations
            filled = df[valueColumn].interpolate(method='linear')  # Linear interpolation
            perturbation = rng.normal(scale=scale, size=len(filled))  # Small random noise
            df[valueColumn] = filled + perturbation  # Add fractal-like noise
        return df

    @staticmethod
    def merge(dfs: list[pd.DataFrame], targetColumn: Union[str, tuple[str]]):
        ''' Layer 1
        combines multiple mutlicolumned dataframes.
        to support disparate frequencies,
        outter join fills in missing values with previous value.
        filters down to the target column observations.
        '''
        from functools import reduce
        import pandas as pd
        if len(dfs) == 0:
            return None
        if len(dfs) == 1:
            return dfs[0]
        for ix, item in enumerate(dfs):
            if targetColumn in item.columns:
                dfs.insert(0, dfs.pop(ix))
                break
            # if we get through this loop without hitting the if
            # we could possibly use that as a trigger to use the
            # other merge function, also if targetColumn is None
            # why would we make a dataset without target though?
        for df in dfs:
            df.index = pd.to_datetime(df.index)
        return reduce(
            lambda left, right:
                pd.merge_asof(left, right, left_index=True, right_index=True),
            dfs)
