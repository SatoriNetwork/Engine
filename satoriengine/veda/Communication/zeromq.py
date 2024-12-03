import threading
import joblib
import zmq
from datetime import datetime
import random
from typing import Union, Optional, Any, List, Dict
import os
import signal
import sys
import time

from process import process_data
from determine_features import determine_feature_set
from model_creation import model_create_train_test_and_predict

class Engine:
    def __init__(self, streams: list[str]):
        self.streams = streams 
        self.models: Dict[str, Model] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.context = zmq.Context()
        self.sockets: Dict[str, zmq.Socket] = {}
        self.running = True
        self.initialize_sockets()
        self.initialize_models()
        self.setup_signal_handling()
        self.run()

    def initialize_sockets(self):
        for stream in self.streams:
            socket = self.context.socket(zmq.PAIR)
            socket.bind(f"inproc://{stream}")
            self.sockets[stream] = socket

    def initialize_models(self):
        for stream in self.streams:
            stream_id = os.path.splitext(os.path.basename(stream))[0]
            self.models[stream] = Model(
                streamId=stream_id,
                socket=self.context.socket(zmq.PAIR),
                datapath_override=stream
            )
            self.models[stream].socket.connect(f"inproc://{stream}")

    def setup_signal_handling(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        print("\nShutting down gracefully...")
        self.running = False
        for model in self.models.values():
            model.stop()
        self.context.term()
        sys.exit(0)

    def handle_model_update(self, stream: str):
        try:
            message = self.sockets[stream].recv_string(flags=zmq.NOBLOCK)
            print(f"Model updated for stream {stream}: {message}")
            self.models[stream].predict()
        except zmq.Again:
            pass  

    def run_model(self, stream: str):
        model = self.models[stream]
        model.run()

    def listen_for_updates(self, stream: str):
        while self.running:
            self.handle_model_update(stream)
            time.sleep(0.1)  # Small delay to prevent CPU overuse

    def start_threads(self):
        for stream, _ in self.models.items():
            model_thread = threading.Thread(target=self.run_model, args=(stream,))
            update_thread = threading.Thread(target=self.listen_for_updates, args=(stream,))
            self.threads[stream] = (model_thread, update_thread)
            model_thread.start()
            update_thread.start()

    def wait_for_completion(self):
        try:
            while self.running:
                all_finished = True
                for model_thread, _ in self.threads.values():
                    if model_thread.is_alive():
                        all_finished = False
                        break
                if all_finished:
                    print("All models have completed execution.")
                    self.running = False
                time.sleep(1)  # Check every second
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Shutting down...")
            self.running = False

        # Wait for all threads to finish
        for model_thread, update_thread in self.threads.values():
            model_thread.join()
            update_thread.join()

    def run(self):
        self.start_threads()
        self.wait_for_completion()
        print("Engine shutdown complete.")


class Model:
    def __init__(self, streamId: str, socket: zmq.Socket = None, datapath_override: str = None, modelpath_override: str = None):
        self.streamId = streamId
        self.datapath = datapath_override or self.data_path()
        self.modelpath = modelpath_override or self.model_path()
        self.stable: list = self.load()
        self.socket = socket
        self.running = True

    def stop(self):
        self.running = False

    def data_path(self) -> str:
        return f'./data/{self.streamId}/aggregate.csv'

    def model_path(self) -> str:
        return f'./models/{self.streamId}'

    def load(self) -> Union[None, list]:
        try:
            self.stable = joblib.load(self.modelpath)
            return self.stable
        except FileNotFoundError:
            return None

    def save(self):
        os.makedirs(os.path.dirname(self.modelpath), exist_ok=True)
        joblib.dump(self.stable, self.modelpath)
        self.socket.send_string(f"Model updated: {self.stable[0].model_name}")

    def compare(self, model, replace:bool = False) -> bool:
        print(f"***** Comparison for {self.streamId} *****")
        print(f"Pilot score : {model[0].backtest_error}")
        print(f"Stable score : {self.stable[0].backtest_error}")
        print("*****************************************")
        compared = model[0].backtest_error < self.stable[0].backtest_error
        if replace and compared:
            self.stable = model
            print(f"The New Stable model for {self.streamId} is : {self.stable[0].model_name}")
            return True
        return compared

    def predict(self):
        print(f"Predicting with model for {self.streamId}: {self.stable[0].model_name}")
        status, predictor_model = engine(
            filename=self.datapath, 
            list_of_models=[self.stable[0].model_name],
            mode='predict',
            unfitted_forecaster=self.stable[0].unfitted_forecaster
        )
        if status == 1:
            print(f"Prediction for {self.streamId}:")
            print(f"Model: {predictor_model[0].model_name}")
            print(f"Forecast: {predictor_model[0].forecast}")
        if status == 4:
            print(f"Error in prediction for {self.streamId}:")
            print(predictor_model)

    def run(self):
        if self.stable is None:
            print(f"Initializing model for {self.streamId}")
            status, model = engine(self.datapath, ['quick_start'])
            if status == 1:
                self.stable = model
                self.save()
                print(f"Initial stable model for {self.streamId}:")
                print(f"Model: {model[0].model_name}")
                print(f"Backtest error: {model[0].backtest_error}")

        iterations = 0
        while iterations < 3 and self.running: 
            print(f"Running iteration {iterations + 1} for {self.streamId}")
            status, pilot = engine(self.datapath, ['random_model'])
            if status == 4:
                print(f"Error in model creation for {self.streamId}:")
                print(pilot)
            elif status == 1:
                if self.compare(pilot, replace=True):
                    self.save()
                    self.predict()
            iterations += 1
            time.sleep(1)  # Add a small delay between iterations

        print(f"Model run complete for {self.streamId}")

    def run_forever(self):
        self.thread = threading.Thread(target=self.run, args=(), daemon=True)
        self.thread.start()

    def run_specific(self):
        status, model = engine(self.datapath, [self.modelpath])
        self.stable = model
        print(status)
        print(model)
        print(model[0].backtest_error)
        # Trigger a prediction after running a specific model
        self.predict()


def check_model_suitability(list_of_models, allowed_models, dataset_length):
    suitable_models = []
    unsuitable_models = []

    for model in list_of_models:
        if model in allowed_models:
            suitable_models.append(True)
        else:
            suitable_models.append(False)
            reason = f"Not allowed for dataset size of {dataset_length}"
            unsuitable_models.append((model, reason))

    return suitable_models, unsuitable_models

def engine(
    filename: str,
    list_of_models: List[str],
    interval: List[int] = [10, 90],
    feature_set_reduction: bool = False,
    exogenous_feature_type: str = 'ExogenousFeaturesBasedonSeasonalityTestWithAdditivenMultiplicative',
    feature_set_reduction_method: str = 'RFECV',
    random_state_hyperr: int = 123,
    metric: str = 'mase',
    mode: str = 'train',
    unfitted_forecaster: Optional[Any] = None
):
    ''' Engine function for the Satori Engine '''

    list_of_models = [model.lower() for model in list_of_models]

    quick_start_present = "quick_start" in list_of_models
    random_model_present = "random_model" in list_of_models
    random_state_hyper = random_state_hyperr

    # Process data first to get allowed_models
    proc_data = process_data(
        filename,
        quick_start=quick_start_present
    )

    # if quick_start_present and random_model_present:
    #     warnings.warn(
    #         "Both 'quick_start' and 'random_model' are present. 'quick_start' will take precedence.")

    if random_model_present and not quick_start_present:
        current_time = datetime.now()
        seed = int(current_time.strftime("%Y%m%d%H%M%S%f"))
        random.seed(seed)
        print(f"Using random seed: {seed}")

        # Randomly select options
        feature_set_reduction = random.choice([True, False])
        exogenous_feature_type = random.choice(["NoExogenousFeatures", "Additive", "AdditiveandMultiplicativeExogenousFeatures",
                                               "ExogenousFeaturesBasedonSeasonalityTestWithAdditivenMultiplicative"])
        feature_set_reduction_method = random.choice(["RFECV", "RFE"])
        random_state_hyper = random.randint(0, 2**32 - 1)

        # Replace 'random_model' with a randomly selected model from allowed_models
        list_of_models = [random.choice(
            proc_data.allowed_models) if model == "random_model" else model for model in list_of_models]
        print(f"Randomly selected models: {list_of_models}")
        print(f"feature_set_reduction: {feature_set_reduction}")
        print(f"exogenous_feature_type: {exogenous_feature_type}")
        print(f"feature_set_reduction_method: {feature_set_reduction_method}")
        print(f"random_state_hyper: {random_state_hyper}")

    if quick_start_present:
        feature_set_reduction = False
        exogenous_feature_type = "NoExogenousFeatures"
        list_of_models = proc_data.allowed_models

    if proc_data.if_small_dataset:
        return 2, "Status = 2 (insufficient amount of data)"

    # Check if the requested models are suitable based on the allowed_models
    suitable_models, unsuitable_models = check_model_suitability(
        list_of_models, proc_data.allowed_models, len(proc_data.dataset))

    if unsuitable_models:
        print("The following models are not allowed due to insufficient data:")
        for model, reason in unsuitable_models:
            print(f"- {model}: {reason}")

    if not any(suitable_models):
        return 3, "Status = 3 (none of the requested models are suitable for the available data)"

    # Filter the list_of_models to include only suitable models
    list_of_models = [model for model, is_suitable in zip(
        list_of_models, suitable_models) if is_suitable]

    try:
        features = None
        for model_name in list_of_models:
            if model_name in ['baseline', 'arima']:
                features = determine_feature_set(
                    dataset=proc_data.dataset,
                    data_train=proc_data.data_subsets['train'],
                    end_validation=proc_data.end_times['validation'],
                    end_train=proc_data.end_times['train'],
                    dataset_with_features=proc_data.dataset_withfeatures,
                    dataset_start_time=proc_data.dataset_start_time,
                    dataset_end_time=proc_data.dataset_end_time,
                    initial_lags=proc_data.lags,
                    weight_para=proc_data.use_weight,
                    exogenous_feature_type=exogenous_feature_type,
                    feature_set_reduction=feature_set_reduction,
                    feature_set_reduction_method=feature_set_reduction_method,
                    bayesian_trial=20,
                    random_state_hyper=random_state_hyper,
                    frequency=proc_data.sampling_frequency,
                    backtest_steps=proc_data.backtest_steps,
                    prediction_steps=proc_data.forecasting_steps,
                    hyper_flag=False
                )
            else:
                features = determine_feature_set(
                    dataset=proc_data.dataset,
                    data_train=proc_data.data_subsets['train'],
                    end_validation=proc_data.end_times['validation'],
                    end_train=proc_data.end_times['train'],
                    dataset_with_features=proc_data.dataset_withfeatures,
                    dataset_start_time=proc_data.dataset_start_time,
                    dataset_end_time=proc_data.dataset_end_time,
                    initial_lags=proc_data.lags,
                    weight_para=proc_data.use_weight,
                    exogenous_feature_type=exogenous_feature_type,
                    feature_set_reduction=feature_set_reduction,
                    feature_set_reduction_method=feature_set_reduction_method,
                    bayesian_trial=20,
                    random_state_hyper=random_state_hyper,
                    frequency=proc_data.sampling_frequency,
                    backtest_steps=proc_data.backtest_steps,
                    prediction_steps=proc_data.forecasting_steps,
                    hyper_flag=True
                )

        list_of_results = []
        for model_name in list_of_models:
            result = model_create_train_test_and_predict(
                model_name=model_name,
                dataset=proc_data.dataset,
                dataset_train=proc_data.data_subsets['train'],
                end_validation=proc_data.end_times['validation'],
                end_test=proc_data.end_times['test'],
                sampling_freq=proc_data.sampling_frequency,
                differentiation= features.differentiation,
                selected_lags=features.selected_lags,
                selected_exog=features.selected_exog,
                dataset_selected_features=features.dataset_selected_features,
                data_missing=features.missing_values,
                weight=features.weight,
                select_hyperparameters=True,
                default_hyperparameters=None,
                random_state_hyper=random_state_hyper,
                backtest_steps=proc_data.backtest_steps,
                interval=interval,
                metric=metric,
                forecast_calendar_features=features.forecast_calendar_features,
                forecasting_steps=proc_data.forecasting_steps,
                hour_seasonality=features.hour_seasonality,
                dayofweek_seasonality=features.dow_seasonality,
                week_seasonality=features.week_seasonality,
                baseline_1=proc_data.time_metric_baseline,
                baseline_2=proc_data.forecasterequivalentdate,
                baseline_3=proc_data.forecasterequivalentdate_n_offsets,
                mode=mode,
                forecaster=unfitted_forecaster
            )
            list_of_results.append(result)

        return 1, list_of_results  # Status = 1 (ran correctly)

    except Exception as e:
        # Additional status code for unexpected errors
        return 4, f"An error occurred: {str(e)}"

if __name__ == "__main__":
    
    streams = ["NATGAS1D.csv", "modifiedkaggletraffic2.csv"]
    # streams = ["NATGAS1D.csv"]
    engine1 = Engine(streams)

    # model1 = Model(streamId="NATGAS1D",
    #                datapath_override="NATGAS1D.csv",
    #                modelpath_override="baseline")
    # model1.run_specific()

    print("Main program exiting.")



