'''
Basic Reponsibilities of the StableModel:
1. keep a record of the datasets, features, and parameters of the best model.
2. retrain the best model on new data available, generate and report prediction.
'''
import numpy as np
import pandas as pd
from itertools import product
from functools import partial
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from satorilib import logging
from satoriengine.model.interfaces.stable import StableModelInterface
from satoriengine.utils.impute import coerceAndFill
from satoriengine.utils import clean


class StableModel(StableModelInterface):

    ### TARGET ######################################################################

    def _produceTarget(self):
        # logging.debug('self.dataset')
        # logging.debug(self.dataset)
        series = self.dataset.loc[:, self.id].shift(-1)
        # logging.debug('series')
        # logging.debug(series)
        self.target = pd.DataFrame(series)
        # logging.debug('self.target')
        # logging.debug(self.target)

    ### FEATURES ####################################################################

    def _produceFeatureStructure(self):
        self.features = {
            **self.features,
            **{
                metric(column=col): partial(metric, column=col)
                for metric, col in product(self.metrics.values(), self.dataset.columns)}
        }

    def _produceFeatureSet(self):
        producedFeatures = []
        # logging.debug('SELF.CHOSENFEATURES', self.chosenFeatures)
        for feature in self.chosenFeatures:
            # logging.debug('FEATURE', feature)
            fn = self.features.get(feature)
            # logging.debug('FN', fn, callable(fn))
            if callable(fn):
                producedFeatures.append(fn(self.dataset))
        if len(producedFeatures) > 0:
            self.featureSet = pd.concat(
                producedFeatures,
                axis=1,
                keys=[s.name for s in producedFeatures])
        # if self.featureSet == None:
        #    #logging.debug('WE NEED A FEATURESET')

    def _produceFeatureImportance(self):
        try:
            feature_imports = self.xgbStable.feature_importances_ if hasattr(
                self, 'xgbStable') else np.ones(self.featureSet.columns.shape)
            self.featureImports = {
                name: fimport
                for fimport, name in zip(feature_imports, self.featureSet.columns)
            }
        except Exception as e:
            logging.info('race ignored in feature importance:', e)
            self.featureImports = {name: fimport for fimport, name in zip(
                np.ones(self.featureSet.columns.shape), self.featureSet.columns)}
            # not sure how this error can happen because .fit is run before this function is called.
            # but it only happens rarely on startup so it's a race condition and has no effect.
            # File "/usr/local/lib/python3.9/threading.py", line 980, in _bootstrap_inner
            #    self.run()
            # File "/usr/local/lib/python3.9/threading.py", line 917, in run
            #    self._target(*self._args, **self._kwargs)
            # File "/Satori/Neuron/satorineuron/init/start.py", line 107, in start
            #    self.buildEngine()
            # File "/Satori/Neuron/satorineuron/init/start.py", line 240, in buildEngine
            #    self.engine.run()
            # File "/Satori/Engine/satoriengine/engine.py", line 130, in run
            #    predictor(model)
            # File "/Satori/Engine/satoriengine/engine.py", line 105, in predictor
            #    model.runPredictor()
            # File "/Satori/Engine/satoriengine/managers/model.py", line 472, in runPredictor
            #    makePrediction(isVariable=True, private=True)
            # File "/Satori/Engine/satoriengine/managers/model.py", line 404, in makePrediction
            #    if isVariable and self.stable.build():
            # File "/Satori/Engine/satoriengine/model/stable.py", line 181, in build
            #    self._produceFeatureImportance()
            # File "/Satori/Engine/satoriengine/model/stable.py", line 60, in _produceFeatureImportance
            #    for fimport, name in zip(self.xgbStable.feature_importances_, self.featureSet.columns)
            # File "/usr/local/lib/python3.9/site-packages/xgboost-1.7.2-py3.9-linux-x86_64.egg/xgboost/sklearn.py", line 1304, in feature_importances_
            #    b: Booster = self.get_booster()
            # File "/usr/local/lib/python3.9/site-packages/xgboost-1.7.2-py3.9-linux-x86_64.egg/xgboost/sklearn.py", line 649, in get_booster
            #    raise NotFittedError("need to call fit or load_model beforehand")
            # sklearn.exceptions.NotFittedError: need to call fit or load_model beforehand
            # except sklearn.exceptions.NotFittedError as e:

    def leastValuableFeature(self):
        ''' called by pilot '''
        if len(self.xgbStable.feature_importances_) == len(self.chosenFeatures):
            matched = [(val, idx) for idx, val in enumerate(
                self.xgbStable.feature_importances_)]
            candidates = []
            for pair in matched:
                if pair[0] not in self.pinnedFeatures:
                    candidates.append(pair)
            if len(candidates) > 0:
                return self.chosenFeatures[min(candidates)[1]]
        return None

    ### FEATURE DATA ####################################################################

    def _produceFeatureData(self):
        '''
        produces our feature data map:
        {feature: (feature importance, [raw inputs])}
        '''
        for k in self.featureSet.columns:
            self.featureData[k] = (
                # KeyError: ('streamrSpoof', 'simpleEURCleanedHL', 'RollingHigh43median')
                self.featureImports[k],
                self.featureData[k][1] if k in self.featureData.keys() else [] + (
                    self.features[k].keywords.get('columns', None)
                    or [self.features[k].keywords.get('column')]))

    def showFeatureData(self):
        '''
        returns true raw feature importance
        example: {
            'Close': 0.6193444132804871,
            'High': 0.16701968474080786,
            'Low': 0.38159190578153357}
        '''
        rawImportance = {}
        for importance, features in self.featureData.values():
            for name in features:
                rawImportance[name] = (
                    importance / len(features)) + rawImportance.get(name, 0)
        return rawImportance

    ### CURRENT ####################################################################

    def _producePredictable(self):
        if self.featureSet.shape[0] > 0:
            df = self.featureSet.copy()
            df = df.iloc[0:-1, :]
            df = df.replace([np.inf, -np.inf], np.nan)
            df.columns = clean.columnNames(df.columns)
            # df = df.reset_index(drop=True) # why?
            # df = coerceAndFill(df)
            df = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))

            lookback_len = next(
                (param.value for param in self.hyperParameters if param.name == 'lookback_len'), 1)
            data = df.to_numpy(dtype=np.float64).flatten()
            data = data[-lookback_len:]
            if self.manager.predictor == 'xgboost' and data.shape[0] < lookback_len:
                data = np.pad(
                    data, (lookback_len - data.shape[0], 0), mode='constant', constant_values=0.0)
            self.current = pd.DataFrame([data])
            # self.current = pd.DataFrame(
            #     self.featureSet.iloc[-1, :]).T  # .dropna(axis=1)
            # logging.debug('\nself.dataset\n', self.dataset.tail(2))
            # logging.debug('\nself.featureSet\n', self.featureSet.tail(2))
            # logging.debug('\nself.current\n', self.current)
            # logging.debug('\nself.prediction\n', self.prediction)

    def producePrediction(self):
        ''' called by manager '''
        # self.current = self.current.apply(
        #     lambda col: pd.to_numeric(col, errors='ignore'))
        df = self.current.copy()
        # df.columns = clean.columnNames(df.columns)
        # todo: maybe this should be done on broadcast? saving it to memory
        #       and we should save this to disk so we have a history
        if hasattr(self, 'prediction') and self.prediction is not None:
            if not hasattr(self, 'predictions'):
                self.predictions = []
            self.predictions.append(self.prediction)
        self.prediction = self.xgb.predict(df)[0]

    ### TRAIN ######################################################################

    def _produceTrainingSet(self):
        df = self.featureSet.copy()
        df = df.iloc[0:-1, :]
        df = df.replace([np.inf, -np.inf], np.nan)
        df.columns = clean.columnNames(df.columns)
        # df = df.reset_index(drop=True) # why?
        # df = coerceAndFill(df)
        df = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))

        lookback_len = next(
            (param.value for param in self.hyperParameters if param.name == 'lookback_len'), 1)
        data = df.to_numpy(dtype=np.float64).flatten()
        data = np.concatenate(
            [np.zeros((lookback_len,), dtype=data.dtype), data])
        data = [data[i-lookback_len:i]
                for i in range(lookback_len, data.shape[0])]
        df = pd.DataFrame(data)

        df_target = self.target.iloc[0:df.shape[0], :]

        # handle nans
        df_target = df_target.apply(lambda col: pd.to_numeric(col, errors='coerce'))
        df_target = df_target.replace([np.inf, -np.inf], np.nan).fillna(0)  # or use another imputation strategy

        data = df_target.to_numpy(dtype=np.float64).flatten()
        df_target = pd.DataFrame(data)

        self.trainX, self.testX, self.trainY, self.testY = train_test_split(
            df, df_target, test_size=self.split or 0.2, shuffle=False)
        # self.trainX = self.trainX.apply(
        #     lambda col: pd.to_numeric(col, errors='coerce'))
        # self.testX = self.testX.apply(
        #     lambda col: pd.to_numeric(col, errors='coerce'))
        # self.trainY = self.trainY.apply(
        #     lambda col: pd.to_numeric(col, errors='coerce'))
        # self.testY = self.testY.apply(
        #     lambda col: pd.to_numeric(col, errors='coerce'))
        # self.trainY = self.trainY.astype(
        #     self.trainX[self.trainX.columns[self.trainX.columns.isin(self.trainY.columns)]].dtypes)
        # self.testY = self.testY.astype(
        #     self.testX[self.testX.columns[self.testX.columns.isin(self.testY.columns)]].dtypes)

    def _produceFit(self):
        self.xgbInUse = True
        # if all(isinstance(y[0], (int, float)) for y in self.trainY.values):
        # self.xgb = XGBRegressor(
        #     eval_metric='mae',
        #     **{param.name: param.value for param in self.hyperParameters})
        # else:
        #    # todo: Classifier untested
        #    self.xgb = XGBClassifier(
        #        eval_metric='mae',
        #        **{param.name: param.value for param in self.hyperParameters})
        # for param in self.hyperParameters:
        #     setattr(self.xgb, param.name, param.value)
        self.xgb.fit(
            self.trainX,
            self.trainY,
            eval_set=[(self.trainX, self.trainY), (self.testX, self.testY)],
            verbose=False)
        # self.xgbStable = copy.deepcopy(self.xgb) ## didn't fix it.
        if self.manager.predictor == 'xgboost':
            self.xgbStable = self.xgb  # turns on pilot
        self.xgbInUse = False

    ### MAIN PROCESSES #################################################################

    def build(self):
        if (
            self.dataset is not None and
            not self.dataset.empty and
            self.dataset.shape[0] > 10
        ):
            self._produceTarget()
            self._produceFeatureStructure()
            self._produceFeatureSet()
            self._producePredictable()
            self._produceTrainingSet()
            self._produceFit()
            self._produceFeatureImportance()
            self._produceFeatureData()
            return True
        return False
