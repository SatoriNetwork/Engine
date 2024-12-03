import threading
import joblib
from reactivex.subject import BehaviorSubject
import pandas as pd
# from satorilib.utils.hash import generatePathId
# from satorilib.concepts import Stream, StreamId

# testing purposes
import os
# end

from datetime import datetime
import random
from typing import Union, Optional, Any, List, Dict

from process import process_data
from determine_features import determine_feature_set
from model_creation import model_create_train_test_and_predict


class Engine:
    # behaviour subject to send the prediction back to the neuron
    def __init__(self, streams: list[str]):
        self.streams = streams
        self.models: Dict[str, Model] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.model_updated_subjects: Dict[str, BehaviorSubject] = {}
        self.setup_subjects()
        self.setup_subscriptions()
        self.initialize_models()
        self.run()

    def setup_subjects(self):
        for stream in self.streams:
            self.model_updated_subjects[stream] = BehaviorSubject(None)

    def setup_subscriptions(self):
        for stream, subject in self.model_updated_subjects.items():
            subject.subscribe(
                on_next=lambda x, s=stream: self.handle_model_update(s, x),
                on_error=lambda e, s=stream: self.handle_error(s, e),
                on_completed=lambda s=stream: self.handle_completion(s)
            )

    def initialize_models(self):
        for stream in self.streams:
            stream_id = os.path.splitext(os.path.basename(stream))[0]
            self.models[stream] = Model(
                streamId=stream_id,
                modelUpdated=self.model_updated_subjects[stream],
                datapath_override=stream
            )

    def handle_model_update(self, stream: str, updated_model):
        if updated_model is not None:
            print(f"Model updated for stream {stream}: {updated_model[0].model_name}")
            self.models[stream].predict(updated_model)

    def handle_error(self, stream: str, error):
        print(f"An error occurred in stream {stream}: {error}")

    def handle_completion(self, stream: str):
        print(f'''
              ******************************************************
              Model update stream completed for {stream}
              ******************************************************
              ''')

    def run_model(self, stream: str):
        model = self.models[stream]
        model.run()

    def start_threads(self):
        for stream, _ in self.models.items():
            thread = threading.Thread(target=self.run_model, args=(stream,))
            self.threads[stream] = thread
            thread.start()

    def wait_for_completion(self):
        for thread in self.threads.values():
            thread.join()

    def run(self):
        self.start_threads()
        self.wait_for_completion()

class Model:
    def __init__(self, streamId: str, modelUpdated: BehaviorSubject, datapath_override: str = None, modelpath_override: str = None):
        self.streamId = streamId
        self.datapath = datapath_override or self.data_path()
        self.modelpath = modelpath_override or self.model_path()
        self.stable: list = self.load()
        self.modelUpdated = modelUpdated

    def data_path(self) -> str:
        return f'./data/{self.streamId}/aggregate.csv'

    def model_path(self) -> str:
        return f'./models/{self.streamId}'

    def load(self) -> Union[None, list]:
        ''' loads the stable model from disk if present'''
        try:
            self.stable = joblib.load(self.modelpath)
            return self.stable
        except FileNotFoundError:
            return None

    def save(self):
        ''' saves the stable model to disk '''
        os.makedirs(os.path.dirname(self.modelpath), exist_ok=True)
        print(self.stable)
        print(self.modelpath)
        joblib.dump(self.stable, self.modelpath)
        self.modelUpdated.on_next(self.stable)

    def compare(self, model, replace:bool = False) -> bool:
        ''' compare the stable model to the heavy model '''
        print("******************************************************************************************")
        print(f"Pilot score : {model[0].backtest_error}")
        print(f"Stable score : {self.stable[0].backtest_error}")
        print("******************************************************************************************")
        compared  = model[0].backtest_error < self.stable[0].backtest_error
        if replace and compared:
            self.stable = model
            print(f"The New Stable model is : {self.stable[0].model_name}")
            return True
        return compared


    def predict(self, data=None): # only needs to fit the whole data ( rn fit for training + fit for the whole dataset )
        ''' prediction without training '''
        # print(self.stable[0].unfitted_forecaster)
        print("Here")
        status, predictor_model = engine( filename=self.datapath,
                               list_of_models=[self.stable[0].model_name],
                               mode='predict',
                               unfitted_forecaster=self.stable[0].unfitted_forecaster
                               )
        if status == 1:
            print(predictor_model[0].model_name)
            print(predictor_model[0].forecast)
            print(predictor_model[0].forecast['pred'].iloc[0])

        if status == 4:
            print(predictor_model)

    def run(self): # it only needs to fit the training set ( rn fit for training + fit for the whole dataset )
        '''
        main loop for generating models and comparing them to the best known
        model so far in order to replace it if the new model is better, always
        using the best known model to make predictions on demand.
        '''
        if self.stable is None:
            status, model = engine(self.datapath, ['quick_start'])
            print(model)
            if status == 1:
                self.stable = model
                print(model[0].backtest_error)
                print(f"The Stable model is : {self.stable[0].model_name}")
                if self.stable[0].model_name != "starter_dataset_model":
                    self.save()

        i = 0
        while True:
            status, pilot = engine(self.datapath, ['random_model'])
            if status == 4:
                print("*************** Error ******************")
                print(pilot)
                print("*************** Error ******************")

            if status == 1:
                # if status 1 only then below
                if self.compare(pilot, replace=True):
                    self.save()
            i += 1

    def run_forever(self):
        self.thread = threading.Thread(target=self.run, args=(), daemon=True)
        self.thread.start()

    def run_specific(self):
        ''' To pass in a model and run only that ( testing purposes) '''
        # start_time = time.time()
        status, model = engine(self.datapath, [self.modelpath])
        self.stable = model
        print(status)
        print(model)
        # print(model[0].model_name)
        print(model[0].backtest_error)
        # print(type(model[0]))

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
    exogenous_feature_type: str = "ExogenousFeaturesBasedonSeasonalityTestWithAdditivenMultiplicative",
    feature_set_reduction_method: str = "RFECV",
    random_state_hyperr: int = 123,
    metric: str = "mase",
    mode: str = "train",
    unfitted_forecaster: Optional[Any] = None,
):
    """Engine function for the Satori Engine"""

    col_names_starter = ["date_time", "value", "id"]
    starter_dataset = pd.read_csv(filename, names=col_names_starter, header=None)
    if len(starter_dataset) < 3:
        from collections import namedtuple

        Result = namedtuple(
            "Result",
            ["forecast", "backtest_error", "model_name", "unfitted_forecaster"],
        )
        if len(starter_dataset) == 1:
            # If dataset has only 1 row, return the same value in the forecast dataframe
            value = starter_dataset.iloc[0, 1]
            forecast = pd.DataFrame(
                {"ds": [pd.Timestamp.now() + pd.Timedelta(days=1)], "pred": [value]}
            )
        elif len(starter_dataset) == 2:
            # If dataset has 2 rows, return their average
            value = starter_dataset.iloc[:, 1].mean()
            forecast = pd.DataFrame(
                {"ds": [pd.Timestamp.now() + pd.Timedelta(days=1)], "pred": [value]}
            )

        starter_result = Result(
            forecast=forecast,
            backtest_error=20,
            model_name="starter_dataset_model",
            unfitted_forecaster=None,
        )

        return 1, [starter_result]

    list_of_models = [model.lower() for model in list_of_models]

    quick_start_present = "quick_start" in list_of_models
    random_model_present = "random_model" in list_of_models
    random_state_hyper = random_state_hyperr

    # Process data first to get allowed_models
    proc_data = process_data(filename, quick_start=quick_start_present)

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
        feature_set_reduction = True # testing
        exogenous_feature_type = random.choice(
            [
                "NoExogenousFeatures",
                "Additive",
                "AdditiveandMultiplicativeExogenousFeatures",
                "ExogenousFeaturesBasedonSeasonalityTestWithAdditivenMultiplicative",
            ]
        )
        feature_set_reduction_method = random.choice(["RFECV", "RFE"])
        random_state_hyper = random.randint(0, 2**32 - 1)

        # Replace 'random_model' with a randomly selected model from allowed_models
        list_of_models = [
            (
                random.choice(proc_data.allowed_models)
                if model == "random_model"
                else model
            )
            for model in list_of_models
        ]
        print(f"Randomly selected models: {list_of_models}")
        print(f"feature_set_reduction: {feature_set_reduction}")
        print(f"exogenous_feature_type: {exogenous_feature_type}")
        print(f"feature_set_reduction_method: {feature_set_reduction_method}")
        print(f"random_state_hyper: {random_state_hyper}")

    if quick_start_present:
        feature_set_reduction = False
        exogenous_feature_type = "NoExogenousFeatures"
        list_of_models = proc_data.allowed_models

    if proc_data.if_invalid_dataset:
        return 2, "Status = 2 (insufficient amount of data)"

    # Check if the requested models are suitable based on the allowed_models
    suitable_models, unsuitable_models = check_model_suitability(
        list_of_models, proc_data.allowed_models, len(proc_data.dataset)
    )

    if unsuitable_models:
        print("The following models are not allowed due to insufficient data:")
        for model, reason in unsuitable_models:
            print(f"- {model}: {reason}")

    if not any(suitable_models):
        return (
            3,
            "Status = 3 (none of the requested models are suitable for the available data)",
        )

    # Filter the list_of_models to include only suitable models
    list_of_models = [
        model
        for model, is_suitable in zip(list_of_models, suitable_models)
        if is_suitable
    ]

    try:
        features = None
        for model_name in list_of_models:
            if model_name in ["baseline", "arima"]:
                features = determine_feature_set(
                    dataset=proc_data.dataset,
                    data_train=proc_data.data_subsets["train"],
                    end_validation=proc_data.end_times["validation"],
                    end_train=proc_data.end_times["train"],
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
                    hyper_flag=False,
                )
            else:
                features = determine_feature_set(
                    dataset=proc_data.dataset,
                    data_train=proc_data.data_subsets["train"],
                    end_validation=proc_data.end_times["validation"],
                    end_train=proc_data.end_times["train"],
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
                    hyper_flag=True,
                )

        list_of_results = []
        for model_name in list_of_models:
            result = model_create_train_test_and_predict(
                model_name=model_name,
                dataset=proc_data.dataset,
                dataset_train=proc_data.data_subsets["train"],
                end_validation=proc_data.end_times["validation"],
                end_test=proc_data.end_times["test"],
                sampling_freq=proc_data.sampling_frequency,
                differentiation=features.differentiation,
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
                forecaster=unfitted_forecaster,
            )
            list_of_results.append(result)

        return 1, list_of_results  # Status = 1 (ran correctly)

    except Exception as e:
        # Additional status code for unexpected errors
        return 4, f"An error occurred: {str(e)}"

# csv_files = ["NATGAS1D.csv", "modifiedkaggletraffic2.csv"]
csv_files = ["aggregatee.csv"]
engine = Engine(csv_files)
