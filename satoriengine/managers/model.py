# TODO: refactor see issue #24

'''
Basic Reponsibilities of the ModelManager:
1. keep a record of the datasets, features, and parameters of the best model.
2. retrain the best model on new data available, generate and report prediction.
3. continuously generate new models to attempt to find a better one.
    A. search the parameter space smartly
    B. search the engineered feature space smartly
    C. evaluate new datasets when they become available
4. save the best model details and load them upon restart
'''
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


class ModelManager(Cached):

    config = None

    @classmethod
    def setConfig(cls, config):
        cls.config = config

    def __init__(
        self,
        variable: StreamId,
        output: StreamId,
        targets: list[StreamId] = None,
        memory: ModelMemoryApi = None,
        modelPath: str = None,
        xgbParams: list = None,
        hyperParameters: list[HyperParameter] = None,
        metrics: dict = None,
        features: dict = None,
        chosenFeatures: list[str] = None,
        pinnedFeatures: list[str] = None,
        exploreFeatures: bool = True,
        split: Union[int, float] = .2,
        override: bool = False,
    ):
        '''
        variable: the response variable - that which we are trying to predict.
                Is a StreamId object which must be entirely specified.
        disk: the disk interface
        memory: the memory interface
        modelPath: the path of the model
        hyperParameters: a list of HyperParameter objects
        metrics: a dictionary of functions that each produce a feature
                (from 1 dynamic column) example: year over year, rolling average
        features: a dictionary of functions that each take in multiple columns
                of the raw data and ouput a feature (cols known ahead of time)
                example: high minus low, x if y > 1 else 2**z
        chosenFeatures: list of feature names to start with
        pinnedFeatures: list of feature names to keep in model
        exploreFeatures: change features or not
        id: column name of response variable
        split: train test split percentage or count
        override: override the existing model saved to disk if there is one
        '''
        self.variable = variable
        self.output = output
        self.key = self.variable.id
        self.id = self.variable.id

        self.memory = memory
        # self.modelPath = modelPath or self.disk.defaultModelPath(self.variable)
        self.targets: list[StreamId] = targets
        self.setupFlags()
        self.get()

        self.useGPU = os.environ.get('GPU_FLAG', default='off') == 'on'
        self.predictor = os.environ.get(
            'PREDICTOR', default='xgboost')  # xgboost chronos ttm
        if self.predictor not in ['chronos', 'ttm']:
            self.predictor = 'xgboost'
        self.xgbParams = xgbParams or []
        self.stable = StableModel(
            manager=self,
            hyperParameters=hyperParameters or [],
            metrics=metrics,
            features=features or {},
            chosenFeatures=chosenFeatures or [],
            pinnedFeatures=pinnedFeatures or [],
            split=split)
        if not override:
            self.load()
        self.pilot = PilotModel(
            manager=self,
            stable=self.stable,
            exploreFeatures=exploreFeatures)

        self.lastOverview: Union[StreamOverview, None] = None
        # not even necessary right now.
        # self.syncManifest()

    @property
    def streamId(self) -> StreamId:
        return self.variable

    @property
    def prediction(self):
        ''' gets prediction from the stable model '''
        # this should convert from np.float64, etc to float
        if (
            isinstance(self.stable.prediction, np.float64) or
            isinstance(self.stable.prediction, np.float32) or
            isinstance(self.stable.prediction, np.float16)
        ):
            return float(self.stable.prediction)
        if (
            isinstance(self.stable.prediction, np.int64) or
            isinstance(self.stable.prediction, np.int32) or
            isinstance(self.stable.prediction, np.int16) or
            isinstance(self.stable.prediction, np.int8)
        ):
            return int(self.stable.prediction)
        return self.stable.prediction

    def buildStable(self):
        self.stable.build()

    def overview(self):
        # def fullData():
        #    try:
        #        return self.disk.gather(
        #            streamIds=self.targets,
        #            targetColumn=self.id)
        #    except Exception as e:
        #        return pd.DataFrame()

        def getRows():
            try:
                return self.dataset.dropna().iloc[-20:].loc[:, (self.variable.source, self.variable.author, self.variable.stream, self.variable.target)]
            except Exception as e:
                traceback.print_exc()
                logging.error('error in overview', e)
                return []

        def getValues(rows: pd.DataFrame):
            try:
                return rows.values.flatten().tolist()
            except Exception as e:
                traceback.print_exc()
                logging.error('error in overview', e)
                return []

        def getPredictions(rows: pd.DataFrame):
            try:
                df = self.diskOf(self.output).cache.copy()
                if df is None or df.empty:
                    return []
                # this isn't enough, we need to match predictions times
                # return df.iloc[-20:].value.values.tolist()
                predictions = []
                closestTimes = []
                df.index = pd.to_datetime(df.index)
                rows.index = pd.to_datetime(rows.index)
                for t in rows.index:
                    closestTime = df.index[df.index > t].min()
                    if pd.notna(closestTime) and closestTime not in closestTimes:
                        closestTimes.append(closestTime)
                        predictions.append(df.loc[closestTime, 'value'])
                return predictions[:-1]
            except Exception as e:
                traceback.print_exc()
                logging.error('error in overview', e)
                return []

        rows = getRows()
        
        # Get the price from the output stream (prediction stream)
        price_per_obs = 0.0
        try:
            # The output stream is the prediction stream that we're pricing
            output_stream = self.output
            # We need to get the stream from the database to access its price
            # This is a bit of a hack, but we'll try to get the price from the disk cache
            output_disk = self.diskOf(output_stream)
            if hasattr(output_disk, 'stream') and hasattr(output_disk.stream, 'price_per_obs'):
                price_per_obs = output_disk.stream.price_per_obs or 0.0
        except Exception as e:
            # If we can't get the price, default to 0.0
            price_per_obs = 0.0
        
        self.lastOverview = StreamOverview(
            streamId=self.variable,
            value=self.stable.current.values[-1][0] if hasattr(
                self.stable, 'current') else '',
            prediction=self.stable.prediction if hasattr(
                self.stable, 'prediction') and self.stable.prediction != None else False,
            # dataset=fullData(),
            values=getValues(rows),
            predictions=getPredictions(rows),
            # 'predictions': self.stable.predictions if hasattr(self.stable, 'predictions') else [],
            # this isn't the accuracy we really care about (historic accuracy),
            # it's accuracy of this current model on historic data.
            accuracy=f'{str(self.stableScore*100)[0:5]} %' if hasattr(
                self, 'stableScore') else '',
            errs=self.errs if hasattr(self, 'errs') else [],
            subscribers='none',
            price_per_obs=price_per_obs)
        return self.lastOverview

    def miniOverview(self):
        ''' only contains the streamId '''
        # really small
        # return self.lastOverview or StreamOverview(streamId=self.variable, value='')
        return self.lastOverview or StreamOverview(
            streamId=self.variable,
            value=self.stable.current.values[-1][0] if hasattr(
                self.stable, 'current') else '',
            prediction=self.stable.prediction if hasattr(
                self.stable, 'prediction') else '',)

    def syncManifest(self):
        # todo: fix this to work with the data from the server
        #       the manifest is only needed for the pilot. server knows the
        #       inputs for the actual stable model. so at this time lest ignore.
        #       model after relay config
        manifest = ModelManager.config.manifest()
        manifest[self.key] = {
            'targets': [x.id for x in self.targets],
            'purged': manifest.get(self.key, {}).get('purged', [])}
        ModelManager.config.put('manifest', data=manifest)

    ### FLAGS ################################################################################

    def setupFlags(self):
        self.modelUpdated = BehaviorSubject(None)
        self.variableUpdated = BehaviorSubject(None)
        self.targetUpdated = BehaviorSubject(None)
        self.inputsUpdated = BehaviorSubject(None)
        self.newAvailableInput = BehaviorSubject(None)
        self.predictionUpdate = BehaviorSubject(None)
        self.privatePredictionUpdate = BehaviorSubject(None)
        self.anyPredictionUpdate = merge(
            self.predictionUpdate,
            self.privatePredictionUpdate,)

    ### GET DATA ####################################################################

    # @staticmethod
    # def addFeatureLevel(df:pd.DataFrame):
    #    ''' adds a feature level to the multiindex columns'''
    #    return pd.MultiIndex.from_tuples([c + ('Raw',)  for c in df.columns])

    def get(self):
        ''' gets the raw data from disk '''

        def handleEmpty():
            '''
            todo: what should we do if no data available yet?
            should self.dataset be None? or should it be an empty dataframe without our target columns?
            or should it be an empty dataframe with our target columns?
            It seems like it should just be None and that we should halt behavior until it has a
            threshold amount of data.
            '''
            self.dataset = self.dataset if self.dataset is not None else pd.DataFrame(
                {x.key: [] for x in set(self.targets)})
        try:
            self.dataset = self.disk.gather(
                streamIds=self.targets,
                targetColumn=self.id)
        except Exception as e:
            logging.warning('Error in self.disk.gather for', self.id, e)
            self.dataset = None
        # logging.debug('SETTING DATA:', color='yellow')
        # logging.debug('self.targets', self.targets, color='yellow')
        # logging.debug('self.id', self.id, color='yellow')
        # logging.debug('self.dataset', self.dataset, color='yellow')
        handleEmpty()

    ### FEATURE DATA ####################################################################

    def showFeatureData(self):
        '''
        returns true raw feature importance
        example: {
            'Close': 0.6193444132804871,
            'High': 0.16701968474080786,
            'Low': 0.38159190578153357}
        '''
        return self.stable.showFeatureData()

    ### META TRAIN ######################################################################

    def evaluateCandidate(self):
        '''
        model consists of the hyperParameter values and the chosenFeatures,
        these are saved to disk. we also replace the model object itself.
        '''
        def scoreRegressiveModels():
            '''
            MAE of 0:
                This would indicate a perfect model, meaning the predictions
                perfectly match the true values. However, achieving an MAE of
                exactly 0 is rare and often unlikely in practical scenarios.
            Small MAE:
                A smaller MAE indicates that the model's predictions are, on
                average, closer to the true values. The closer the MAE is to 0,
                the better the model's performance. However, what is considered
                a "small" MAE varies depending on the problem and context.
            MAE relative to the scale of the target variable:
                The interpretation of MAE should be considered relative to the
                scale of your target variable. If the target variable has a
                large range or high variance, a relatively small MAE might still
                represent a good model performance.
            Comparison between models:
                MAE is often more interpretable and intuitive when used for
                comparing different models. If you have multiple models, you can
                compare their MAE values directly. A model with a lower MAE is
                generally considered to have better predictive performance
            '''
            if (
                not hasattr(self.pilot, 'testY') or
                not hasattr(self.stable, 'testY')
            ):
                return False
            self.stableScore = mean_absolute_error(
                self.stable.testY,
                self.stable.xgb.predict(self.stable.testX))
            self.pilotScore = mean_absolute_error(
                self.pilot.testY,
                self.pilot.xgb.predict(self.pilot.testX))
            return self.pilotScore < self.stableScore
            # if result:
            #    logging.debug(self.variable.stream, self.variable.target, 'scores:',
            #                  self.stableScore, self.pilotScore)

        def scoreClassificationModels():
            '''
            The R2 score typically ranges from -∞ to 1.
            1 indicates a perfect fit.
            0 indicates the model performs no better than randomly guessing.
            negative values indicate that the model's predictions are worse than
            simply guessing the mean of the target variable.
            '''
            if (
                not hasattr(self.pilot, 'testY') or
                not hasattr(self.stable, 'testY')
            ):
                return False
            self.stableScore = self.stable.xgb.score(
                self.stable.testX,
                self.stable.testY)
            self.pilotScore = self.pilot.xgb.score(
                self.pilot.testX,
                self.pilot.testY)
            return self.pilotScore > self.stableScore

        def replaceStableModel():
            for param in self.stable.hyperParameters:
                # is this right? it looks right but I don't think the stable model ever updates from the pilot
                param.value = param.test
            self.stable.chosenFeatures = self.pilot.testFeatures
            self.stable.featureSet = self.pilot.testFeatureSet
            self.stable.xgb = copy.deepcopy(self.pilot.xgb)
            self.save()
            return True

        if isinstance(self.stable.xgb, XGBRegressor):
            if scoreRegressiveModels():
                self.saveErr()
                return replaceStableModel()
        elif isinstance(self.stable.xgb, XGBClassifier):
            if scoreClassificationModels():
                self.saveErr()
                return replaceStableModel()
        return False

    ### SAVE ###########################################################################

    def saveErr(self):
        # todo: save this to disk so we have a history
        if not hasattr(self, 'errs'):
            self.errs = []
        self.errs.append(self.pilotScore)

    def save(self):
        ''' save the current model '''
        if self.disk is None:
            return False
        self.disk.saveModel(
            self.stable.xgb,
            # modelPath=self.modelPath,
            streamId=self.variable,
            hyperParameters=self.stable.hyperParameters,
            chosenFeatures=self.stable.chosenFeatures)

    def load(self):  # -> bool:
        ''' loads the model - happens on init so we automatically load our progress '''
        if self.predictor == 'chronos':
            self.stable.xgb = ChronosAdapter(self.useGPU)
        elif self.predictor == 'ttm':
            self.stable.xgb = TTMAdapter(self.useGPU)
        else:
            if self.disk is None:
                return False
            try:
                xgb = self.disk.loadModel(
                    # modelPath=self.modelPath,
                    streamId=self.variable)
                # logging.debug('LOADING STABLE', xgb)
                if xgb is None or xgb == False:
                    self.stable.xgb = XGBRegressor(
                        eval_metric='mae',
                        **{param.name: param.value for param in self.stable.hyperParameters if param.name in self.xgbParams})
                    return False
                if (
                    all([scf in self.stable.features.keys() for scf in xgb.savedChosenFeatures]) and
                    # all([shp in self.stable.hyperParameters for shp in xgb.savedHyperParameters])
                    True
                ):
                    self.stable.xgb = xgb
                    self.stable.hyperParameters = [next((param2 for param2 in xgb.savedHyperParameters if param2.name == param.name), param)
                                                for param in self.stable.hyperParameters]
                    self.stable.chosenFeatures = xgb.savedChosenFeatures
            except Exception as e:
                #logging.warning('error loading model', e)
                self.stable.xgb = XGBRegressor(
                    eval_metric='mae',
                    **{param.name: param.value for param in self.stable.hyperParameters if param.name in self.xgbParams})
                return False
        if self.predictor == 'chronos' or self.predictor == 'ttm':
            lb_idx = next((i for i in range(len(self.stable.hyperParameters))
                          if self.stable.hyperParameters[i].name == 'lookback_len'), -1)
            if lb_idx >= 0:
                self.stable.hyperParameters[lb_idx].value = self.stable.xgb.ctx_len
        return True

    ### LIFECYCLE ######################################################################

    def runPredictor(self):
        def makePrediction(isVariable=False, private=False):
            '''
            we make predictions on startup, so that our 'prediction' variable
            in the model is not empty, and we can show a prediction to the user,
            but these predictions might be using a very old model so we do not
            actually broadcast or save these predictions to the database. that's
            when private is True.
            '''
            # logging.debug('makePrediction', self.variable,
            #              isVariable, private, color="yellow")
            # why do I rebuild each time? (would this be sufficient? self.stable.xgb is not None and self.stable.xgb.isFitted)
            if isVariable and self.stable.build():
                self.stable.producePrediction()
                # logging.debug('prediction produced',
                #              self.stable.prediction, color='yellow')
                if private:
                    self.privatePredictionUpdate.on_next(self)
                else:
                    logging.info(
                        'prediction produced! '
                        f'{self.output.stream} {self.variable.target}:',
                        self.stable.prediction,
                        color='green')
                    self.stable.metric_prediction = self.stable.prediction
                    self.predictionUpdate.on_next(self)

        def makePredictionFromNewModel():
            logging.info(
                f'model improved! {self.variable.stream} {self.variable.target}'
                f'\n  stable score: {self.stableScore}'
                f'\n  pilot  score: {self.pilotScore}'
                "\n  parameters: {}".format(
                    {param.name: param.value for param in self.stable.hyperParameters}),
                color='green')
            makePrediction(isVariable=True, private=True)

        def makePredictionFromNewInputs():
            '''
            go get the entire dataset from memory. this is triggered at the end
            of gathering and merging the ipfs history data.
            '''
            self.get()
            makePrediction(isVariable=True)

        def makePredictionFromNewTarget(incremental):
            # note: on disk we remove all duplicates, but in the model, datasets
            #       are combined and often have duplicates because they show
            #       what the data was in the interim. However, we probably still
            #       don't want to generally blindly append a new row as it might
            #       be an entire duplicate row. (each row should have at least
            #       one change somewhere unless the model uses an algorithm that
            #       wants it at constant time intervals or something). so we
            #       remove duplicates here instead of in self.memory because
            #       it's really the model's perogative.
            # logging.debug('in makePredictionFromNewVariable',
            #              incremental.columns, print=True)
            self.dataset = self.memory.dropDuplicates(
                self.memory.appendInsert(
                    df=self.dataset,
                    incremental=incremental))
            makePrediction(isVariable=True)

        def makePredictionFromNewVariable(incremental):
            # logging.debug('in makePredictionFromNewVariable',
            #              incremental.columns, print=True)
            for col in incremental.columns:
                if col not in self.dataset.columns:
                    incremental = incremental.drop(col, axis=1)
            # incremental.columns = ModelManager.addFeatureLevel(df=incremental)
            if hasattr(self.stable, 'metric_prediction'):
                incremental_np = np.float32(incremental[self.id][-1])
                self.stable.metric_loss = np.abs(
                    self.stable.metric_prediction - incremental_np)
                self.stable.metric_loss_acc = 100 - \
                    self.stable.metric_loss / (incremental_np + 1e-10) * 100
                alpha = 0.01
                if hasattr(self.stable, 'metric_loss_ema'):
                    self.stable.metric_loss_ema = alpha * self.stable.metric_loss + \
                        (1 - alpha) * self.stable.metric_loss_ema
                else:
                    self.stable.metric_loss_ema = self.stable.metric_loss
                self.stable.metric_loss_ema_acc = 100 - \
                    self.stable.metric_loss_ema / \
                    (incremental_np + 1e-10) * 100
                logging.info(
                    f'metrics for {self.variable.stream} {self.variable.target}'
                    f'\n  loss {self.stable.metric_loss} acc {self.stable.metric_loss_acc}'
                    f'\n  loss ema {self.stable.metric_loss_ema} acc {self.stable.metric_loss_ema_acc}',
                    color='green')
            self.dataset = self.memory.dropDuplicates(
                self.memory.appendInsert(
                    df=self.dataset,
                    incremental=incremental))
            makePrediction(isVariable=True)

        self.modelUpdated.subscribe(
            lambda x: makePredictionFromNewModel() if x is not None else None)
        self.inputsUpdated.subscribe(
            lambda x: makePredictionFromNewInputs() if x else None)
        self.variableUpdated.subscribe(
            lambda x: makePredictionFromNewVariable(x) if x is not None else None)
        self.targetUpdated.subscribe(
            lambda x: makePredictionFromNewTarget(x) if x is not None else None)
        # we make a new prediction on startup, but its not trustworthy:
        makePrediction(isVariable=True, private=True)

    def runExplorer(self):
        if hasattr(self.stable, 'target') and hasattr(self.stable, 'xgbStable'):
            try:
                self.pilot.build()
                if self.evaluateCandidate():
                    self.modelUpdated.on_next(True)
            except NotFittedError as e:
                '''
                this happens on occasion...
                maybe making  self.xgbStable a deepcopy would fix
                '''
                # logging.debug('not fitted', e)
            # except AttributeError as e:
            #    '''
            #    this happens at the beginning of running when we have not set
            #    self.xgbStable yet.
            #    '''
            #    #logging.debug('Attribute', e)
            #    pass
            except Exception as e:
                logging.error('UNEXPECTED', e)
            time.sleep(1)
        else:
            time.sleep(1)

    def syncAvailableInputs(self):

        def sync(x):
            '''
            add the new datastreams and histories to the top
            of the list of things to explore and evaluate
            '''
            # something like this?
            # self.features.append(x)
            #
            # self.targets.append(StreamId(x))  or something
            # self.syncManifest()  then sync manifest when you change targets.
            # maybe remove targets that aren't being used as any features.. somewhere?

        self.newAvailableInput.subscribe(
            lambda x: sync(x) if x is not None else None)
