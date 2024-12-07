from satoriengine.veda.pipelines import PipelineInterface, SKPipeline, StarterPipeline, XgbPipeline, XgbChronosPipeline
from satoriengine.veda.data import StreamForecast
from satorilib.logging import INFO, setup, debug, info, warning, error
from satorilib.disk.filetypes.csv import CSVManager
from satorilib.concepts import Stream, StreamId, Observation
from satorilib.disk import getHashBefore
from satorilib.utils.system import getProcessorCount
from satorilib.utils.time import datetimeToTimestamp, now
from satorilib.utils.hash import hashIt, generatePathId
from reactivex.subject import BehaviorSubject
import pandas as pd
import threading
import json
import copy
import time
import os
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

setup(level=INFO)


class Engine:
    def __init__(self, streams: list[Stream], pubstreams: list[Stream]):
        self.streams = streams
        self.pubstreams = pubstreams
        self.streamModels: Dict[StreamId, StreamModel] = {}
        self.new_observation: BehaviorSubject = BehaviorSubject(None)
        self.predictionProduced: BehaviorSubject = BehaviorSubject(None)
        self.setup_subscriptions()
        self.initialize_models()

    def pause(self):
        for streamModel in self.streamModels.values():
            streamModel.pause()

    def resume(self):
        for streamModel in self.streamModels.values():
            streamModel.resume()

    def setup_subscriptions(self):
        self.new_observation.subscribe(
            on_next=lambda x: self.handleNewObservation(
                x) if x is not None else None,
            on_error=lambda e: self.handle_error(e),
            on_completed=lambda: self.handle_completion())

    def initialize_models(self):
        for stream, pubStream in zip(self.streams, self.pubstreams):
            self.streamModels[stream.streamId] = StreamModel(
                streamId=stream.streamId,
                predictionStreamId=pubStream.streamId,
                predictionProduced=self.predictionProduced)
            self.streamModels[stream.streamId].choosePipeline(inplace=True)
            self.streamModels[stream.streamId].run_forever()
            #break  # only one stream for testing

    def handleNewObservation(self, observation: Observation):
        streamModel = self.streamModels.get(observation.streamId)
        streamModel.handleNewObservation(observation)
        if streamModel.thread is None or not streamModel.thread.is_alive():
            streamModel.choosePipeline(inplace=True)
            streamModel.run_forever()
        if streamModel is not None and len(streamModel.data) > 1:
            debug(
                f'Making prediction based on new observation using {streamModel.pipeline.__name__}', color='teal')
            self.pause()
            streamModel.producePrediction()
            self.resume()
        else:
            info(f"No model found for stream {observation.streamId}")

    def handle_error(self, error):
        print(f"An error occurred new_observaiton: {error}")

    def handle_completion(self):
        print("new_observation completed")


class StreamModel:
    def __init__(
        self,
        streamId: StreamId,
        predictionStreamId: StreamId,
        predictionProduced: BehaviorSubject,
    ):
        self.preferredPipelines: list[PipelineInterface] = [StarterPipeline, XgbPipeline, XgbChronosPipeline, SKPipeline]
        self.defaultPipelines: list[PipelineInterface] = [XgbPipeline, XgbPipeline, StarterPipeline]
        self.failedPipelines = []
        self.thread: threading.Thread = None
        self.streamId: StreamId = streamId
        self.predictionStreamId: StreamId = predictionStreamId
        self.predictionProduced: StreamId = predictionProduced
        self.data: pd.DataFrame = self.loadData()
        self.pipeline: PipelineInterface = self.choosePipeline()
        self.pilot: PipelineInterface = self.pipeline(uid=streamId)
        self.pilot.load(self.modelPath())
        self.stable: PipelineInterface = copy.deepcopy(self.pilot)
        self.paused: bool = False
        debug(f'StreamModel {generatePathId(streamId=self.streamId)} initialized with {self.pipeline.__name__}', color='teal')

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def handleNewObservation(self, observation: Observation):
        """extract the data and save it to self.data"""
        parsedData = json.loads(observation.raw)
        self.data = pd.concat(
            [
                self.data,
                pd.DataFrame({
                    "date_time": [str(parsedData["time"])],
                    "value": [float(parsedData["data"])],
                    "id": [str(parsedData["hash"])]}),
            ],
            ignore_index=True)

    def producePrediction(self, updatedModel=None):
        """
        triggered by
            - model replaced with a better one
            - new observation on the stream
        """
        updatedModel = updatedModel or self.stable
        if updatedModel is not None:
            forecast = updatedModel.predict(data=self.data)
            if isinstance(forecast, pd.DataFrame):
                observationTime = datetimeToTimestamp(now())
                prediction = StreamForecast.firstPredictionOf(forecast)
                observationHash = hashIt(
                    getHashBefore(pd.DataFrame(), observationTime)
                    + str(observationTime)
                    + str(prediction))
                self.save_prediction(
                    observationTime, prediction, observationHash)
                streamforecast = StreamForecast(
                    streamId=self.streamId,
                    predictionStreamId=self.predictionStreamId,
                    currentValue=self.data,
                    forecast=forecast,  # maybe we can fetch this value from predictionHistory
                    observationTime=observationTime,
                    observationHash=observationHash,
                    predictionHistory=CSVManager().read(self.prediction_data_path()))
                self.predictionProduced.on_next(streamforecast)
            else:
                error("Forecast failed, retrying with Quick Model")
                debug("Model Path to be deleted : ",
                      self.modelPath(), color="teal")
                if os.path.isfile(self.modelPath()):
                    try:
                        os.remove(self.modelPath())
                        debug("Deleted failed model file", color="teal")
                    except Exception as e:
                        error(f"Failed to delete model file: {str(e)}")
                self.stable = None
                pipelineClass = self.choosePipeline()
                rollbackModel = pipelineClass()
                try:
                    trainingResult = rollbackModel.fit(data=self.data)
                    if trainingResult.status == 1:
                        debug(
                            f'New model trained: '
                            f'{trainingResult.model[0].model_name}',
                            color="teal")
                        self.stable = copy.deepcopy(rollbackModel)
                        self.producePrediction(self.stable)
                    else:
                        error(
                            f"Failed to train alternative model (status: {trainingResult.status})")
                except Exception as e:
                    error(f"Error training new model: {str(e)}")

    def save_prediction(
        self,
        observationTime: str,
        prediction: float,
        observationHash: str,
    ) -> pd.DataFrame:
        # alternative - use data manager: self.predictionUpdate.on_next(self)
        df = pd.DataFrame(
            {"value": [prediction], "hash": [observationHash]},
            index=[observationTime])
        df.to_csv(
            self.prediction_data_path(),
            float_format="%.10f",
            mode="a",
            header=False)
        return df

    def loadData(self) -> pd.DataFrame:
        try:
            return pd.read_csv(
                self.data_path(),
                names=["date_time", "value", "id"],
                header=None)
        except FileNotFoundError:
            return pd.DataFrame(columns=["date_time", "value", "id"])

    def data_path(self) -> str:
        return (
            '/Satori/Neuron/data/'
            f'{generatePathId(streamId=self.streamId)}/aggregate.csv')

    def prediction_data_path(self) -> str:
        return (
            '/Satori/Neuron/data/'
            f'{generatePathId(streamId=self.predictionStreamId)}/aggregate.csv')

    def modelPath(self) -> str:
        return (
            '/Satori/Neuron/models/veda/'
            f'{generatePathId(streamId=self.streamId)}/'
            f'{self.pipeline.__name__}.joblib')

    def choosePipeline(self, inplace: bool = False) -> PipelineInterface:
        """
        everything can try to handle some cases
        Engine
            - low resources available - SKPipeline
            - few observations - SKPipeline
            - (mapping of cases to suitable pipelines)
        examples: StartPipeline, SKPipeline, XGBoostPipeline, ChronosPipeline, DNNPipeline
        """
        if False: # for testing specific pipelines
            pipeline = XgbChronosPipeline
        else:
            pipeline = None
            for p in self.preferredPipelines:
                if p in self.failedPipelines:
                    continue
                if p.condition(dataCount=len(self.data)) == 1:
                    pipeline = p
                    break
            if pipeline is None:
                for pipeline in self.defaultPipelines:
                    if pipeline not in self.failedPipelines:
                        break
                if pipeline is None:
                    pipeline = self.defaultPipelines[-1]
        if (
            inplace and (
                not hasattr(self, 'pilot') or
                not isinstance(self.pilot, pipeline))
        ):
            info(
                f'Switching from {self.pipeline.__name__} '
                f'to {pipeline.__name__} on {self.streamId}',
                color='blue')
            self.pipeline = pipeline
            self.pilot = pipeline(uid=self.streamId)
            self.pilot.load(self.modelPath())
        return pipeline

    def run(self):
        """
        main loop for generating models and comparing them to the best known
        model so far in order to replace it if the new model is better, always
        using the best known model to make predictions on demand.
        Breaks if backtest error stagnates for 3 iterations.
        """
        while len(self.data) > 0:
            if self.paused:
                time.sleep(1)
                continue
            self.choosePipeline(inplace=True)
            trainingResult = self.pilot.fit(data=self.data)
            if trainingResult.status == 1:
                if self.pilot.compare(self.stable):
                    if self.pilot.save(self.modelPath()):
                        self.stable = copy.deepcopy(self.pilot)
                        info(
                            "Stable Model Updated for stream:",
                            self.streamId.cleanId,
                            print=True)
                        self.producePrediction(self.stable)
            else:
                debug(f'Model Training Failed on {self.streamId} waiting 10 minutes to retry')
                self.failedPipelines.append(self.pilot)
                time.sleep(60*10)

    def run_forever(self):
        self.thread = threading.Thread(target=self.run, args=(), daemon=True)
        self.thread.start()
