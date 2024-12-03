from typing import Iterable, Union
import traceback
import os
import time
import copy
import random
import pandas as pd
import numpy as np
from reactivex import merge
from reactivex.subject import BehaviorSubject
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from satorilib import logging
from satorilib.concepts import StreamId, StreamOverview
from satorilib.disk import Cached
from satorilib.interfaces.model import ModelMemoryApi
from satoriengine.concepts import HyperParameter
from satoriengine.model.pilot import PilotModel
from satoriengine.model.stable import StableModel

from satoriengine.model.chronos_adapter import ChronosAdapter
from satoriengine.model.ttm_adapter import TTMAdapter
from typing import Union, Optional, Any
from collections import namedtuple
from satoriengine.veda.pipelines.interface import PipelineInterface, TrainingResult
from satoriengine.utils import clean


class XGBRegressorPipeline(PipelineInterface):

    def __init__(self, *args, **kwargs):
        # unused
        # useGPU: bool
        # self.hyperParameters: list[HyperParameter] # unused
        # xgbParams: list = []
        self.model: XGBRegressor = XGBRegressor(eval_metric='mae')

    def fit(self, **kwargs) -> TrainingResult:
        ''' train model - in place '''
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(
            data, df_target, test_size=self.split or 0.2, shuffle=False)
        # using data to train the model
        self.model.fit(
            self.trainX,
            self.trainY,
            eval_set=[(self.trainX, self.trainY), (self.testX, self.testY)],
            verbose=False)
        return TrainingResult(1, self.model, True)

    def save(self, modelpath: str, **kwargs) -> bool:
        """saves the stable model to disk"""
        return True

    def compare(self, other: Union[PipelineInterface, None] = None, **kwargs) -> bool:
        """true indicates the pilot model is better than the stable model"""
        if isinstance(other, self.__class__):
            return True
        return self._score(self.model) < self._score(other)

    def predict(self, **kwargs) -> Union[None, pd.DataFrame]:
        """prediction without training"""
        return self.model.predict(kwargs["data"])

    def _produceTrainingSet(self):
        df = self.testFeatureSet.copy()
        df = df.iloc[0:-1, :]
        df = df.replace([np.inf, -np.inf], np.nan)
        df.columns = clean.columnNames(df.columns)
        # df = df.reset_index(drop=True)
        # df = coerceAndFill(df)
        df = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))

        lookback_len = next(
            (param.test for param in self.hyperParameters if param.name == 'lookback_len'), 1)
        data = df.to_numpy(dtype=np.float64).flatten()
        data = np.concatenate(
            [np.zeros((lookback_len,), dtype=data.dtype), data])
        data = [
            data[i-lookback_len:i]
            for i in range(lookback_len, data.shape[0])]
        df = pd.DataFrame(data)

        df_target = self.target.iloc[0:df.shape[0], :]
        data = df_target.to_numpy(dtype=np.float64).flatten()
        df_target = pd.DataFrame(data)

    def _score(self, model: XGBRegressor, testX: Iterable, testY: Iterable):
        return mean_absolute_error(testY, model.predict(testX))

    def _updateHyperParameters(self):
        # assum self.model is XGBRegressor
        # instead of explicitly holding the hyperparameters, outside the model
        # we can just randomize the hyperparameters in the model in a similar,
        # adjustable-learning-rate way as we used to:
        # def radicallyRandomize():
        #    for param in self.hyperParameters:
        #        x = param.min + (random.random() * (param.max - param.min))
        #        if param.kind == int:
        #            x = int(x)
        #        param.test = x
        #
        # def incrementallyRandomize():
        #    for param in self.hyperParameters:
        #        x = (
        #            (random.random() * param.limit * 2) +
        #            (param.value - param.limit))
        #        if param.min < x < param.max:
        #            if param.kind == int:
        #                x = int(round(x))
        #            param.test = x
        #
        # x = random.random()
        # if x >= .9:
        #    radicallyRandomize()
        # elif .1 < x < .9:
        #    incrementallyRandomize()
        # something like
        self.model.set_params(
            learning_rate=min(max(self.model.learning_rate *
                              random.uniform(0.8, 1.25), .001), 1),
            n_estimators=min(
                max(int(self.model.n_estimators * random.uniform(0.8, 1.25)), 10), 1000),
            max_depth=min(max(int(self.model.max_depth * random.uniform(0.8, 1.25)), 3), 10))
        # {
        #    'objective': 'reg:squarederror',
        #    'base_score': 0.5,
        #    'booster': 'gbtree',
        #    'colsample_bylevel': 1,
        #    'colsample_bynode': 1,
        #    'colsample_bytree': 1,
        #    'gamma': 0,
        #    'importance_type': 'gain',
        #    'learning_rate': 0.01,      # Updated value, if modified
        #    'max_delta_step': 0,
        #    'max_depth': 5,             # Updated value, if modified
        #    'min_child_weight': 1,
        #    'missing': None,
        #    'n_estimators': 200,        # Updated value, if modified
        #    'n_jobs': 1,
        #    'nthread': None,
        #    'random_state': 0,
        #    'reg_alpha': 0,
        #    'reg_lambda': 1,
        #    'scale_pos_weight': 1,
        #    'seed': None,
        #    'silent': None,
        #    'subsample': 1,
        #    'verbosity': 1,
        #    'eval_metric': 'mae'        # Specified during instantiation
        # }
