import warnings
warnings.filterwarnings('ignore')
from typing import Union, Dict
import copy
import json
import threading
import pandas as pd

import pickle
import os
from reactivex.subject import BehaviorSubject
from satorilib.utils.hash import hashIt, generatePathId
from satorilib.utils.time import datetimeToTimestamp, now
from satorilib.utils.system import getProcessorCount
from satorilib.disk import getHashBefore
from satorilib.concepts import Stream, StreamId, Observation
from satorilib.disk.filetypes.csv import CSVManager
from satorilib.logging import debug, info, error, setup, DEBUG, INFO
from satoriengine.veda.Data import StreamForecast
from satoriengine.veda.pipelines import PipelineInterface, SKPipeline, StarterPipeline, XgbPipeline

setup(level=INFO)

setup(level=INFO)


class Engine:
    def __init__(self, streams: list[Stream], pubstreams: list[Stream]):
        self.streams = streams
        self.pubstreams = pubstreams
        self.streamModels: Dict[StreamId, StreamModel] = {}
        self.new_observation: BehaviorSubject = BehaviorSubject(None)
        self.prediction_produced: BehaviorSubject = BehaviorSubject(None)
        self.setup_subscriptions()
        self.initialize_models()

    def setup_subscriptions(self):
        self.new_observation.subscribe(
            on_next=lambda x: self.handle_new_observation(x) if x is not None else None,
            on_error=lambda e: self.handle_error(e),
            on_completed=lambda: self.handle_completion())

    def initialize_models(self):
        for stream, pubStream in zip(self.streams, self.pubstreams):
            self.streamModels[stream.streamId] = StreamModel(
                streamId=stream.streamId,
                predictionStreamId=pubStream.streamId,
                prediction_produced=self.prediction_produced)
            self.streamModels[stream.streamId].choose_pipeline(inplace=True)
            self.streamModels[stream.streamId].run_forever()

    def handle_new_observation(self, observation: Observation):
        streamModel = self.streamModels.get(observation.streamId)
        streamModel.handle_new_observation(observation)
        if streamModel.thread is None or not streamModel.thread.is_alive():
            streamModel.choose_pipeline(inplace=True)
            streamModel.run_forever()
        if streamModel is not None and len(streamModel.data) > 1:
            debug(f'Making prediction based on new observation using {streamModel.pipeline.__name__}', color='teal')
            streamModel.produce_prediction()
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
        prediction_produced: BehaviorSubject,
    ):
        self.thread = None
        self.streamId = streamId
        self.predictionStreamId = predictionStreamId
        self.prediction_produced = prediction_produced
        self.data: pd.DataFrame = self.load_data()
        self.pipeline: PipelineInterface = self.choose_pipeline()
        self.pilot: PipelineInterface = self.pipeline()  # Create instance first
        self.pilot.load(self.model_path())  # Then load model on the instance
        self.stable: PipelineInterface = copy.deepcopy(self.pilot)


    def handle_new_observation(self, observation: Observation):
        """extract the data and save it to self.data"""
        parsed_data = json.loads(observation.raw)
        self.data = pd.concat(
            [
                self.data,
                pd.DataFrame({
                    "date_time": [str(parsed_data["time"])],
                    "value": [float(parsed_data["data"])],
                    "id": [str(parsed_data["hash"])]}),
            ],
            ignore_index=True)

    def produce_prediction(self, updated_model=None):
        """
        triggered by
            - model replaced with a better one
            - new observation on the stream
        """
        debug('1 predict', print=True)
        updated_model = updated_model or self.stable
        if updated_model is not None:
            debug('2 predict', print=True)
            forecast = updated_model.predict(data=self.data)
            debug('3 predict', print=True)
            if isinstance(forecast, pd.DataFrame):
                observationTime = datetimeToTimestamp(now())
                prediction = StreamForecast.firstPredictionOf(forecast)
                observationHash = hashIt(
                    getHashBefore(pd.DataFrame(), observationTime)
                    + str(observationTime)
                    + str(prediction))
                self.save_prediction(observationTime, prediction, observationHash)
                streamforecast = StreamForecast(
                    streamId=self.streamId,
                    predictionStreamId=self.predictionStreamId,
                    currentValue=self.data,
                    forecast=forecast, # maybe we can fetch this value from predictionHistory
                    observationTime=observationTime,
                    observationHash=observationHash,
                    predictionHistory=CSVManager().read(self.prediction_data_path()))
                self.prediction_produced.on_next(streamforecast)
            else:
                error("Forecast failed, retrying with Quick Model")
                debug("Model Path to be deleted : ", self.model_path(), color="teal")
                if os.path.isfile(self.model_path()):
                    try:
                        os.remove(self.model_path())
                        debug("Deleted failed model file", color="teal")
                    except Exception as e:
                        error(f"Failed to delete model file: {str(e)}")
                self.stable = None
                pipeline_class = self.choose_pipeline()
                rollback_model = pipeline_class()
                try:
                    training_result = rollback_model.fit(data=self.data)
                    if training_result.status == 1:
                        debug(f"New model trained: {training_result.model[0].model_name}", color="teal")
                        self.stable = copy.deepcopy(rollback_model)
                        self.produce_prediction(self.stable)
                    else:
                        error(f"Failed to train alternative model (status: {training_result.status})")
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

    def load_data(self) -> pd.DataFrame:
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

    def model_path(self) -> str:
        debug(
            '/Satori/Neuron/models/veda/'
            f'{generatePathId(streamId=self.streamId)}/'
            f'{self.pipeline.__name__}.joblib',
            color="teal")
        return (
            '/Satori/Neuron/models/veda/'
            f'{generatePathId(streamId=self.streamId)}/'
            f'{self.pipeline.__name__}.joblib')

    def choose_pipeline(self, inplace: bool = False) -> PipelineInterface:
        """
        everything can try to handle some cases
        Engine
            - low resources available - SKPipeline
            - few observations - SKPipeline
            - (mapping of cases to suitable pipelines)
        examples: StartPipeline, SKPipeline, XGBoostPipeline, ChronosPipeline, DNNPipeline
        """
        if self.data is None or len(self.data) < 3:
            if inplace and not isinstance(self.pilot, StarterPipeline):
                self.pilot = StarterPipeline()
            return StarterPipeline
        # if getProcessorCount() < 4:
        #     if inplace and not isinstance(self.pilot, XgbPipeline):
        #         self.pilot = XgbPipeline()
        #     return XgbPipeline
        # if 3 <= len(self.data) < 40:
        #     if inplace and not isinstance(self.pilot, XgbPipeline):
        #         self.pilot = XgbPipeline()
        #     return XgbPipeline
        # at least 4 processors and
        # at least 40 observations
        # still debugging
        if inplace and not isinstance(self.pilot, SKPipeline):
            self.pilot = SKPipeline()
        return SKPipeline
        # if inplace and not isinstance(self.pilot, XgbPipeline):
        #     self.pilot = XgbPipeline()
        # return XgbPipeline


    def run(self):
        """
        main loop for generating models and comparing them to the best known
        model so far in order to replace it if the new model is better, always
        using the best known model to make predictions on demand.
        Breaks if backtest error stagnates for 3 iterations.
        """
        # still have a "problem?" where the model makes predictions right away
        # wasn't sure SKPipeline was working so just using XgbPipeline for now
        while len(self.data) > 0:
            trainingResult = self.pilot.fit(data=self.data)
            if trainingResult.status == 1 and not trainingResult.stagnated:
                if self.pilot.compare(self.stable):
                    if self.pilot.save(self.model_path()):
                        self.stable = copy.deepcopy(self.pilot)
                        info("Stable Model Updated for stream:", self.streamId, print=True)
                        self.produce_prediction(self.stable)
            else:
                if not trainingResult.stagnated:
                    debug("Starter Pipeline", print=True)
                else:
                    error("Model Training Failed, Breaking out of the Loop")
                break 

    def run_forever(self):
        self.thread = threading.Thread(target=self.run, args=(), daemon=True)
        self.thread.start()
