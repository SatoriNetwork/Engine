from typing import Union, List, Optional, Any
import datetime
import random
import pandas as pd
from process import process_data
from determine_features import determine_feature_set
from model_creation import model_create_train_test_and_predict

class AIEngineOutput:
    def __init__(self, 
                 status: int,
                 list_of_models: Union[List, str]):
        self.status = status
        self.list_of_models = list_of_models

def skEnginePipeline(
        data: pd.DataFrame,
        list_of_models: List[str],
        interval: List[int] = [10, 90],
        feature_set_reduction: bool = False,
        exogenous_feature_type: str = "ExogenousFeaturesBasedonSeasonalityTestWithAdditivenMultiplicative",
        feature_set_reduction_method: str = "RFECV",
        random_state_hyperr: int = 123,
        metric: str = "mase",
        mode: str = "train",
        unfitted_forecaster: Optional[Any] = None,
    ) -> AIEngineOutput:
        """Engine function for the Satori Engine"""

        # if data.empty():
        #     return 2, "Empty Dataset"

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

        list_of_models = [model.lower() for model in list_of_models]

        quick_start_present = "quick_start" in list_of_models
        random_model_present = "random_model" in list_of_models
        random_state_hyper = random_state_hyperr

        # Process data first to get allowed_models
        proc_data = process_data(data, quick_start=quick_start_present)

        if random_model_present and not quick_start_present:
            current_time = datetime.now()
            seed = int(current_time.strftime("%Y%m%d%H%M%S%f"))
            random.seed(seed)
            print(f"Using random seed: {seed}")

            # Randomly select options
            feature_set_reduction = random.choice([True, False])
            if feature_set_reduction and len(proc_data.dataset) < 50:
                feature_set_reduction = False
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
            # return 2, "Status = 2 (insufficient amount of data)"
            return AIEngineOutput(
                2,
                "Status = 2 (insufficient amount of data)"
            )

        # Check if the requested models are suitable based on the allowed_models
        suitable_models, unsuitable_models = check_model_suitability(
            list_of_models, proc_data.allowed_models, len(proc_data.dataset)
        )

        if unsuitable_models:
            print("The following models are not allowed due to insufficient data:")
            for model, reason in unsuitable_models:
                print(f"- {model}: {reason}")

        if not any(suitable_models):
            # return (
            #     3,
            #     "Status = 3 (none of the requested models are suitable for the available data)",
            # )
            return AIEngineOutput(
                3,
                "Status = 3 (none of the requested models are suitable for the available data)",
            )

        # Filter the list_of_models to include only suitable models
        list_of_models = [
            model
            for model, is_suitable in zip(list_of_models, suitable_models)
            if is_suitable
        ]

        print(1)

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
            
            print(2)
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

            # return 1, list_of_results  # Status = 1 (ran correctly)
            return AIEngineOutput(
                1,
                list_of_results
            )

        except Exception as e:
            # Additional status code for unexpected errors
            # return 4, f"An error occurred: {str(e)}"
            return AIEngineOutput(
                4,
                f"An error occurred: {str(e)}"
            )