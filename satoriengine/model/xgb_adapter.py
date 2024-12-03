from typing import Union
import traceback
import os
import time
import copy
import pandas as pd
import numpy as np
from reactivex import merge
from reactivex.subject import BehaviorSubject
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_absolute_error
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


class XGBRegressorAdapter():

    def __init__(self, useGPU: bool, hyperParameters: list[HyperParameter], xgbParams: list = []):
        self.model = XGBRegressor(
            eval_metric='mae',
            **{param.name: param.value for param in hyperParameters if param.name in xgbParams})

    def fit(self, trainX, trainY, eval_set, verbose):
        ''' online learning '''
        pass

    def forecast(self, current) -> np.ndarray:
        ''' produce a forecast '''
        return np.asarray([], dtype=np.float32)

    def predict(self, current) -> float:
        ''' produce a prediction '''
        return 0

    def score(self, testX, testY):
        return mean_absolute_error(testY, self.model.predict(testX))


class XGBClassifierAdapter():

    def __init__(self, useGPU):
        self.model = XGBClassifier()

    def fit(self, trainX, trainY, eval_set, verbose):
        ''' online learning '''
        pass

    def forecast(self, current) -> np.ndarray:
        ''' produce a forecast '''
        return np.asarray([], dtype=np.float32)

    def predict(self, current) -> float:
        ''' produce a prediction '''
        return 0

    def score(self, testX, testY):
        return self.model.score(testX, testY)
