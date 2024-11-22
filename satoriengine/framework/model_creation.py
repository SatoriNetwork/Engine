import warnings
warnings.filterwarnings('ignore')

from typing import Callable, Union, Any, Dict, List, Optional
import copy

# Data processing
# ==============================================================================
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# skforecast wrappers/interfaces that simply the use of a combination of different capabilities
from skforecast.ForecasterBaseline import ForecasterEquivalentDate
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import bayesian_search_forecaster
from skforecast.model_selection import backtesting_forecaster

# import optuna

from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.model_selection_sarimax import backtesting_sarimax
from pmdarima import auto_arima

from skforecast.model_selection import (
    grid_search_forecaster,
    random_search_forecaster,
    bayesian_search_forecaster,
)

# linear regressors : LinearRegression(), Lasso() or Ridge()
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from lineartree import LinearBoostRegressor

from sklearn.preprocessing import StandardScaler

from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.neuralforecast import NeuralForecastLSTM
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.forecasting.model_selection import ForecastingOptunaSearchCV
from sktime.performance_metrics.forecasting import (
    mean_absolute_scaled_error,
    mean_squared_error,
    mean_absolute_error,
)
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteScaledError,
)  # check if this is needed
import optuna

from .determine_features import GeneralizedHyperparameterSearch
from satorilib.logging import debug, info, error


class ForecastModelResult:
    def __init__(
        self,
        model_name: str,
        sampling_freq: str,
        differentiation: int,
        selected_lags: int,
        selected_exog: list,
        dataset_selected_features: pd.DataFrame,
        selected_hyperparameters: Union[Dict[str, Any], None],
        backtest_steps: Optional[int] = None,
        backtest_prediction_interval: Optional[List[int]] = None,
        backtest_predictions: Optional[pd.DataFrame] = None,
        backtest_error: Optional[float] = None,
        backtest_interval_coverage: Optional[Union[float, str]] = None,
        model_trained_on_all_data: Optional[Any] = None,
        forecasting_steps: Optional[int] = None,
        forecast: Optional[pd.DataFrame] = None,
        unfitted_forecaster: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.sampling_freq = sampling_freq
        self.differentiation = differentiation
        self.selected_lags = selected_lags
        self.selected_exog = selected_exog
        self.dataset_selected_features = dataset_selected_features
        self.selected_hyperparameters = selected_hyperparameters
        self.backtest_steps = backtest_steps
        self.backtest_prediction_interval = backtest_prediction_interval
        self.backtest_predictions = backtest_predictions
        self.backtest_error = backtest_error
        self.backtest_interval_coverage = backtest_interval_coverage
        self.model_trained_on_all_data = model_trained_on_all_data
        self.forecasting_steps = forecasting_steps
        self.forecast = forecast
        self.unfitted_forecaster = unfitted_forecaster


def create_forecaster(
    model_type,
    if_exog=None,
    random_state=None,
    verbose=None,
    lags=None,
    differentiation=None,
    custom_params=None,
    weight=None,
    steps=None,
    time_metric_baseline="days",
    forecasterequivalentdate=1,
    forecasterequivalentdate_n_offsets=7,
    y=None,
    start_p=24,
    start_q=0,
    max_p=24,
    max_q=1,
    seasonal=True,
    test="adf",
    m=24,
    d=None,
    D=None,
):
    forecaster_params = {
        "lags": lags,
        "differentiation": (
            differentiation if differentiation and differentiation > 0 else None
        ),
        "weight_func": weight,
        "transformer_y": StandardScaler(),
        "transformer_exog": StandardScaler() if if_exog else None,
    }
    forecaster_params = {k: v for k, v in forecaster_params.items() if v is not None}

    regressor_params = {"random_state": random_state} if random_state else {}

    if custom_params:
        regressor_params.update(custom_params)

    def create_autoreg(regressor_class, **extra_params):
        params = {**regressor_params, **extra_params}
        return lambda: ForecasterAutoreg(
            regressor=regressor_class(**params), **forecaster_params
        )

    def create_autoreg_direct(regressor_class, **extra_params):
        params = {**regressor_params, **extra_params}
        return lambda: ForecasterAutoregDirect(
            regressor=regressor_class(**params), steps=steps, **forecaster_params
        )

    model_creators = {
        "baseline": lambda: ForecasterEquivalentDate(
            offset=pd.DateOffset(**{time_metric_baseline: forecasterequivalentdate}),
            n_offsets=forecasterequivalentdate_n_offsets,
        ),
        "arima": lambda: ForecasterSarimax(
            regressor=auto_arima(
                y=y,
                start_p=start_p,
                start_q=start_q,
                max_p=max_p,
                max_q=max_q,
                seasonal=seasonal,
                test=test or "adf",
                m=m,
                d=d,
                D=D,
                trace=True,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
            ),
            **forecaster_params,
        ),
        "direct_linearregression": create_autoreg_direct(LinearRegression),
        "direct_ridge": create_autoreg_direct(Ridge),
        "direct_lasso": create_autoreg_direct(Lasso),
        "direct_linearboost": create_autoreg_direct(
            LinearBoostRegressor, base_estimator=LinearRegression()
        ),
        "direct_lightgbm": create_autoreg_direct(LGBMRegressor),
        "direct_xgb": create_autoreg_direct(XGBRegressor),
        "direct_catboost": create_autoreg_direct(CatBoostRegressor),
        "direct_histgradient": create_autoreg_direct(HistGradientBoostingRegressor),
        "autoreg_linearregression": create_autoreg(LinearRegression),
        "autoreg_ridge": create_autoreg(Ridge),
        "autoreg_lasso": create_autoreg(Lasso),
        "autoreg_linearboost": create_autoreg(
            LinearBoostRegressor, base_estimator=LinearRegression()
        ),  # test
        "autoreg_lightgbm": create_autoreg(LGBMRegressor, verbose=verbose),
        "autoreg_randomforest": create_autoreg(RandomForestRegressor),
        "autoreg_xgb": create_autoreg(XGBRegressor),
        "autoreg_catboost": create_autoreg(
            CatBoostRegressor,
            verbose=False,
            allow_writing_files=False,
            boosting_type="Plain",
            leaf_estimation_iterations=10,
        ),
        "autoreg_histgradient": create_autoreg(
            HistGradientBoostingRegressor, verbose=0 if verbose == -1 else verbose
        ),
    }

    if model_type not in model_creators:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_type == "arima" and y is None:
        raise ValueError("For ARIMA model, 'y' parameter is required.")

    return model_creators[model_type]()


def perform_backtesting(
    forecaster,
    y,
    end_validation,
    exog=None,
    steps=24,
    metric="mean_absolute_scaled_error",
    refit=False,
    interval=None,
    n_boot=0,
    in_sample_residuals=True,
    binned_residuals=False,
    is_sarimax=False,
    fixed_train_size=False,
    suppress_warnings_fit=True,
):
    # debug(len(y))
    # debug(len(y.loc[:end_validation]))
    if is_sarimax:
        backtesting_params = {
            "forecaster": forecaster,
            "y": y,
            "initial_train_size": len(y.loc[:end_validation]),
            "fixed_train_size": fixed_train_size,
            "steps": steps,
            "metric": metric,
            "refit": False,
            "n_jobs": "auto",
            "suppress_warnings_fit": suppress_warnings_fit,
            "verbose": False,
            "show_progress": False,
        }
        backtest_metric, backtest_predictions = backtesting_sarimax(
            **backtesting_params
        )
    else:
        backtesting_params = {
            "forecaster": forecaster,
            "y": y,
            "exog": exog,
            "steps": steps,
            "metric": metric,
            "initial_train_size": len(y.loc[:end_validation]),
            "refit": refit,
            "interval": interval,
            "n_boot": n_boot,
            "in_sample_residuals": in_sample_residuals,
            "binned_residuals": binned_residuals,
            "n_jobs": "auto",
            "verbose": False,
            "show_progress": True,
        }
        backtest_metric, backtest_predictions = backtesting_forecaster(
            **backtesting_params
        )

    debug(f"Backtest error ({metric}): {backtest_metric}")
    return backtest_metric, backtest_predictions


def predict_interval_custom(
    forecaster, steps=24, interval=[10, 90], n_boot=20, exog=None, alpha=None
):
    if isinstance(forecaster, ForecasterSarimax):
        # Case for ARIMA model
        if alpha is None:
            alpha = 0.05  # Default value if not provided
        return forecaster.predict_interval(steps=steps, alpha=alpha)
    elif exog is not None:
        # Case with exogenous features for other models
        return forecaster.predict_interval(
            exog=exog, steps=steps, interval=interval, n_boot=n_boot
        )
    else:
        # Case without exogenous features for other models
        return forecaster.predict_interval(
            steps=steps, interval=interval, n_boot=n_boot
        )


def calculate_theoretical_coverage(interval):
    if len(interval) != 2:
        raise ValueError("Interval must contain exactly two values")

    lower, upper = interval
    if not (isinstance(lower, (int, float)) and isinstance(upper, (int, float))):
        raise ValueError("Interval values must be numbers")

    if lower >= upper:
        raise ValueError("Lower bound must be less than upper bound")

    coverage = upper - lower
    # print(f"Theoretical coverage: {coverage}%")
    return coverage


def calculate_interval_coverage(
    satoridataset, interval_predictions, end_validation, end_test, value
):
    # Ensure both datasets have the same index
    common_index = satoridataset.loc[end_validation:end_test].index.intersection(
        interval_predictions.index
    )

    # Align the datasets
    satori_aligned = satoridataset.loc[common_index, value]
    predictions_aligned = interval_predictions.loc[common_index]

    # Calculate coverage
    coverage = np.mean(
        np.logical_and(
            satori_aligned >= predictions_aligned["lower_bound"],
            satori_aligned <= predictions_aligned["upper_bound"],
        )
    )

    # pseudo-code
    # a df which contains a common index [date-time], 3 columns[  predictions_aligned["lower_bound"], satori_aligned["value"], predictions_aligned["upper_bound"] ]
    coverage_df = pd.DataFrame(
        {
            "lower_bound": predictions_aligned["lower_bound"],
            value: satori_aligned,
            "upper_bound": predictions_aligned["upper_bound"],
            "if_in_range": (satori_aligned >= predictions_aligned["lower_bound"])
            & (satori_aligned <= predictions_aligned["upper_bound"]),
        },
        index=common_index,
    )

    total_count = len(coverage_df)
    in_range_count = coverage_df["if_in_range"].sum()
    # end

    # Equivalent code
    # covered_data = 0
    # total_rows = len(satori_aligned)

    # for i in range(total_rows):
    #     aligned_value = satori_aligned.iloc[i]
    #     lower_bound = predictions_aligned['lower_bound'].iloc[i]
    #     upper_bound = predictions_aligned['upper_bound'].iloc[i]

    #     if (aligned_value >= lower_bound) and (aligned_value <= upper_bound):
    #         covered_data += 1

    # coverage = covered_data / total_rows
    # end Equivalent code

    # Calculate total area of the interval
    area = (
        predictions_aligned["upper_bound"] - predictions_aligned["lower_bound"]
    ).sum()

    # debug(f"Total data points: {total_count}")
    # debug(f"Data points within range: {in_range_count}")
    debug(
        f"Predicted interval coverage assuming Gaussian distribution: {round(100*coverage, 2)}%"
    )
    # debug(f"Total area of the interval: {round(area, 2)}")
    # debug(coverage_df)

    return coverage, area


def calculate_error(y_true, backtest_prediction, y_train, error_type="mase"):
    """
    Calculate either Mean Absolute Scaled Error (MASE), Mean Squared Error (MSE),
    or Mean Absolute Error (MAE) based on the specified error_type for forecast data.

    Parameters:
    y_true (pandas.Series): True values
    backtest_prediction (pandas.DataFrame): DataFrame containing predictions and intervals
    y_train (pandas.Series): Training data used for scaling in MASE
    error_type (str): Type of error to calculate ('mase', 'mse', or 'mae')

    Returns:
    float: Calculated error value
    """
    # Ensure y_true and y_train are pandas Series with a datetime index
    if not isinstance(y_true, pd.Series):
        raise ValueError("y_true must be a pandas Series with a datetime index")
    if not isinstance(y_train, pd.Series):
        raise ValueError("y_train must be a pandas Series with a datetime index")

    # Align y_true with backtest_prediction
    aligned_true = y_true.loc[backtest_prediction.index]

    # Extract point predictions
    y_pred = backtest_prediction["pred"]

    if error_type.lower() == "mase":
        return mean_absolute_scaled_error(aligned_true, y_pred, y_train=y_train)
    elif error_type.lower() == "mse":
        return mean_squared_error(aligned_true, y_pred)
    elif error_type.lower() == "mae":
        return mean_absolute_error(aligned_true, y_pred)
    else:
        raise ValueError("Invalid error_type. Choose 'mase', 'mse', or 'mae'.")


def model_create_train_test_and_predict(
    model_name: str,
    dataset: pd.DataFrame,
    dataset_train: pd.DataFrame,
    end_validation: pd.Timestamp,
    end_test: pd.Timestamp,
    sampling_freq: str,
    differentiation: int,
    selected_lags: int,
    selected_exog: list,
    dataset_selected_features: pd.DataFrame,
    data_missing: bool,
    weight: Union[Callable[[pd.DatetimeIndex], np.ndarray], None],
    baseline_1: str,
    baseline_2: int,
    baseline_3: int,
    select_hyperparameters: bool = True,
    default_hyperparameters: Union[Dict[str, Any], None] = None,
    random_state_hyper: int = 123,
    backtest_steps: int = 24,
    interval: List[int] = [10, 90],
    metric: str = "mase",
    forecast_calendar_features: Union[List[str], None] = None,
    forecasting_steps: int = 24,
    hour_seasonality: bool = False,
    dayofweek_seasonality: bool = False,
    week_seasonality: bool = False,
    mode: str = "train",
    forecaster: Optional[Any] = None,
):

    if data_missing:
        value = "value_imputed"
    else:
        value = "value"

    y = dataset_selected_features[value].copy()
    y.index = pd.to_datetime(y.index)
    split_index = y.index.get_loc(end_validation)
    y_train = y.iloc[: split_index + 1]
    y_train.index = pd.to_datetime(y_train.index)
    y_test = y.iloc[split_index + 1 :]

    if model_name.lower() == "baseline":
        baseline_forecaster = create_forecaster(
            "baseline",
            time_metric_baseline=baseline_1,
            forecasterequivalentdate=baseline_2,
            forecasterequivalentdate_n_offsets=baseline_3,
        )

        backtest_predictions = None
        backtest_error = None
        coverage = None
        forecast = None

        if mode == "all" or mode == "train":
            baseline_forecaster.fit(
                y=dataset_selected_features.loc[:end_validation, value]
            )
            backtest_predictions = baseline_forecaster.predict(steps=backtest_steps)
            backtest_predictions = backtest_predictions.to_frame(name="pred")
            backtest_error = calculate_error(
                y_test,
                backtest_predictions[: len(y_test)],
                y_train,
                error_type=metric,
            )
            coverage = "NA"

        if mode == "all" or mode == "predict":
            baseline_forecaster.fit(y=dataset_selected_features[value])
            forecast = baseline_forecaster.predict(steps=forecasting_steps)
            forecast = forecast.to_frame(name="pred")

        return ForecastModelResult(
            model_name=model_name,
            sampling_freq=sampling_freq,
            differentiation=differentiation,
            selected_lags=selected_lags,
            selected_exog=selected_exog,
            dataset_selected_features=dataset_selected_features,
            selected_hyperparameters=None,
            backtest_steps=backtest_steps,
            backtest_prediction_interval=interval,
            backtest_predictions=backtest_predictions,
            backtest_error=backtest_error,
            backtest_interval_coverage=coverage,
            model_trained_on_all_data=baseline_forecaster,
            forecasting_steps=forecasting_steps,
            forecast=forecast,
        )

    elif model_name.lower() == "arima":
        sampling_timedelta = pd.Timedelta(sampling_freq)
        day_timedelta = pd.Timedelta(days=1)
        
        seasonal = False
        if hour_seasonality:
            m = forecasting_steps
            seasonal = True
        elif dayofweek_seasonality:
            if sampling_timedelta <= pd.Timedelta(hours=1):
                m = forecasting_steps * 7
            else:
                multiplier = max(1, int(day_timedelta / sampling_timedelta))
                m = 7 * multiplier
            seasonal = True
        else:
            m = 1

        m = int(m)
        # print(m)

        arima_forecaster = create_forecaster(
            model_type="arima",
            y=dataset_selected_features.loc[:end_validation, value],
            start_p=forecasting_steps,
            start_q=0,
            max_p=forecasting_steps,
            max_q=2, # was 1
            seasonal=seasonal,
            test="adf",
            m=m,
            d=None,
            D=None,
        )

        backtest_predictions = None
        backtest_error = None
        coverage = None
        forecast = None

        if mode == "all" or mode == "train":
            arima_forecaster.fit(
                y=dataset_selected_features.loc[:end_validation, value],
                suppress_warnings=True
            )
            backtest_predictions = arima_forecaster.predict_interval(
                steps=backtest_steps, interval=interval
            )
            backtest_error = calculate_error(
                y_test,
                backtest_predictions[: len(y_test)],
                y_train,
                error_type=metric,
            )
            coverage, _ = calculate_interval_coverage(
                dataset_selected_features,
                backtest_predictions,
                end_validation,
                end_test,
                value,
            )

        if mode == "all" or mode == "predict":
            arima_forecaster.fit(y=dataset_selected_features[value], suppress_warnings=True)
            forecast = predict_interval_custom(
                forecaster=arima_forecaster, steps=forecasting_steps, alpha=0.05
            )

        return ForecastModelResult(
            model_name=model_name,
            sampling_freq=sampling_freq,
            differentiation=differentiation,
            selected_lags=selected_lags,
            selected_exog=selected_exog,
            dataset_selected_features=dataset_selected_features,
            selected_hyperparameters=None,  # ARIMA doesn't use hyperparameters in this implementation
            backtest_steps=backtest_steps,
            backtest_prediction_interval=interval,
            backtest_predictions=backtest_predictions,
            backtest_error=backtest_error,
            backtest_interval_coverage=coverage,
            model_trained_on_all_data=arima_forecaster,
            forecasting_steps=forecasting_steps,
            forecast=forecast,
        )

    elif model_name[:3].lower() == "skt":

        if model_name.lower() == "skt_lstm_deeplearning":
            forecaster = NeuralForecastLSTM(freq=sampling_freq, max_steps=200)

        elif model_name.lower() == "skt_prophet_additive":
            forecaster = Prophet(
                seasonality_mode="additive",
                daily_seasonality="auto",
                weekly_seasonality="auto",
                yearly_seasonality="auto",
            )

        elif model_name.lower() == "skt_prophet_hyper":
            param_grid = {
                "growth": optuna.distributions.CategoricalDistribution(
                    ["linear", "logistic"]
                ),
                "n_changepoints": optuna.distributions.IntDistribution(5, 20),
                "changepoint_range": optuna.distributions.FloatDistribution(0.7, 0.9),
                "seasonality_mode": optuna.distributions.CategoricalDistribution(
                    ["additive", "multiplicative"]
                ),
                "seasonality_prior_scale": optuna.distributions.LogUniformDistribution(
                    0.01, 10.0
                ),
                "changepoint_prior_scale": optuna.distributions.LogUniformDistribution(
                    0.001, 0.5
                ),
                "holidays_prior_scale": optuna.distributions.LogUniformDistribution(
                    0.01, 10.0
                ),
                "daily_seasonality": optuna.distributions.CategoricalDistribution(
                    ["auto"]
                ),
                "weekly_seasonality": optuna.distributions.CategoricalDistribution(
                    ["auto"]
                ),
                "yearly_seasonality": optuna.distributions.CategoricalDistribution(
                    ["auto"]
                ),
            }

            forecaster_initial = Prophet()

            # Set up a more efficient time series cross-validation
            cv = SlidingWindowSplitter(
                initial_window=int(
                    len(y_train) * 0.6
                ),  # Use 60% of data for initial training
                step_length=int(len(y_train) * 0.2),  # Move forward by 20% each time
                fh=np.arange(1, forecasting_steps + 1),  # Forecast horizon of 12 steps
            )

            fos = ForecastingOptunaSearchCV(
                forecaster=forecaster_initial,
                param_grid=param_grid,
                cv=cv,
                n_evals=50,
                strategy="refit",
                scoring=MeanAbsoluteScaledError(sp=1),
                verbose=-1,
            )

            fos.fit(y_train)
            forecaster = fos.best_forecaster_

        elif (
            model_name.lower() == "skt_ets"
        ):  # faster implementation available and should be implemented in the future
            # debug("entered")
            forecaster = AutoETS(
                error="add",
                trend=None,
                damped_trend=False,
                seasonal=None,
                sp=forecasting_steps,
                initialization_method="estimated",
                initial_level=None,
                initial_trend=None,
                initial_seasonal=None,
                bounds=None,
                dates=None,
                freq=None,
                missing="none",
                start_params=None,
                maxiter=1000,
                full_output=True,
                disp=False,
                callback=None,
                return_params=False,
                auto=True,
                information_criterion="aic",
                allow_multiplicative_trend=True,
                restrict=True,
                additive_only=False,
                ignore_inf_ic=True,
                # n_jobs=-1,
                random_state=random_state_hyper,
            )

        elif model_name[:9].lower() == "skt_tbats":
            splist = []
            sampling_timedelta = pd.Timedelta(sampling_freq)
            day_timedelta = pd.Timedelta(days=1)
            use_box_cox = None
            if model_name.lower() == "skt_tbats_quick":
                use_box_cox = False
            if differentiation == 0:
                use_trend = False
                use_damped_trend = False
            else:
                use_trend = True
                if model_name.lower() == "skt_tbats_damped":
                    use_damped_trend = True
                else:
                    use_damped_trend = False

            
            if hour_seasonality == True:
                splist.append(forecasting_steps)

            if dayofweek_seasonality == True:
                if sampling_timedelta <= pd.Timedelta(hours=1):
                    multiplier = 7
                    splist.append(forecasting_steps * multiplier)
                else:
                    multiplier = max(1, int(day_timedelta / sampling_timedelta))
                    splist.append(7 * multiplier)

            if (
                week_seasonality == True
            ):  # calculation of week seasonality test to determine seasonal period of a year can be improved in the future by using day_of_year instead of week_of_year
                if sampling_timedelta <= pd.Timedelta(hours=1):
                    multiplier = 365.25
                    splist.append(forecasting_steps * multiplier)
                else:
                    multiplier = max(1, int(day_timedelta / sampling_timedelta))
                    splist.append(365.25 * multiplier)

            forecaster = TBATS(
                use_box_cox=use_box_cox,
                box_cox_bounds=(0, 1),
                use_trend=use_trend,
                use_damped_trend=use_damped_trend,
                sp=splist,
                use_arma_errors=True,
                show_warnings=False,
                # n_jobs=-1,
                multiprocessing_start_method="spawn",
                context=None,
            )

        rounded_index = y_test.index.floor(sampling_freq)
        difference_minutes = (y_test.index - rounded_index).total_seconds() / 60
        y.index = y.index.floor(sampling_freq)
        y_test.index = y_test.index.floor(sampling_freq)
        y_train.index = y_train.index.floor(sampling_freq)
        time_delta = pd.Timedelta(minutes=float(difference_minutes[0]))

        lower, upper = interval
        backtest_prediction = None
        error = None
        coverage = None
        forecast = None

        if mode == "all" or mode == "train":
            if model_name.lower() == "skt_lstm_deeplearning":
                forecaster.fit(y_train, fh=list(range(1, backtest_steps + 1)))
                y_pred_backtest = forecaster.predict(list(range(1, backtest_steps + 1)))
                backtest_prediction = y_pred_backtest.to_frame(name="pred")
                error = calculate_error(
                    y_test, backtest_prediction[: len(y_test)], y_train, error_type=metric
                )
                backtest_prediction.index = backtest_prediction.index + time_delta
                coverage = "NA"
            else:
                forecaster.fit(y_train)
                y_pred_backtest = forecaster.predict(list(range(1, backtest_steps + 1)))
                
                y_pred_backtest_interval = forecaster.predict_interval(
                    fh=list(range(1, backtest_steps + 1)), coverage=[(upper - lower) / 100]
                )
                y_pred_backtest_df = y_pred_backtest.to_frame(name="pred")
                backtest_prediction = pd.concat(
                    [y_pred_backtest_df, y_pred_backtest_interval], axis=1
                )
                backtest_prediction.columns = ["pred", "lower_bound", "upper_bound"]
                error = calculate_error(
                    y_test, backtest_prediction[: len(y_test)], y_train, error_type=metric
                )
                backtest_prediction.index = backtest_prediction.index + time_delta
                coverage, _ = calculate_interval_coverage(
                    dataset_selected_features,
                    backtest_prediction,
                    end_validation,
                    end_test,
                    value,
                )

        if mode == "all" or mode == "predict":
            if model_name.lower() == "skt_lstm_deeplearning":
                forecaster.fit(y, fh=list(range(1, forecasting_steps + 1)))
                y_pred_future = forecaster.predict(list(range(1, forecasting_steps + 1)))
                forecast = y_pred_future.to_frame(name="pred")
                forecast.index = forecast.index + time_delta
            else:
                forecaster.fit(y)
                y_pred_future = forecaster.predict(list(range(1, forecasting_steps + 1)))
                y_pred_future_interval = forecaster.predict_interval(
                    fh=list(range(1, forecasting_steps + 1)),
                    coverage=[(upper - lower) / 100],
                )
                y_pred_df = y_pred_future.to_frame(name="pred")
                forecast = pd.concat([y_pred_df, y_pred_future_interval], axis=1)
                forecast.columns = ["pred", "lower_bound", "upper_bound"]
                forecast.index = forecast.index + time_delta

        return ForecastModelResult(
            model_name=model_name,
            sampling_freq=sampling_freq,
            differentiation=differentiation,
            selected_lags=selected_lags,
            selected_exog=selected_exog,
            dataset_selected_features=dataset_selected_features,
            selected_hyperparameters=None,  # SKT models don't use hyperparameters in this implementation
            backtest_steps=backtest_steps,
            backtest_prediction_interval=interval,
            backtest_predictions=backtest_prediction,
            backtest_error=error,
            backtest_interval_coverage=coverage,
            model_trained_on_all_data=forecaster,
            forecasting_steps=forecasting_steps,
            forecast=forecast,
        )

    else:

        if model_name[:6].lower() == "direct":

            if model_name == "direct_linearregression":
                random_state = None
            else:
                random_state = 123
            verbose = None
            differentiation = None
            steps = forecasting_steps

        else:

            if model_name == "autoreg_linearregression":
                random_state = None
            else:
                random_state = 123
            verbose = -1
            differentiation = differentiation

            if model_name == "autoreg_linearboost":
                differentiation = None
            steps = None

        if selected_exog == []:
            if_exog = None
        else:
            if_exog = StandardScaler()

        backtest_predictions = None
        backtest_error = None
        coverage = None
        forecast = None
        unfitted_forecaster = None
        selected_hyperparameters = None

        if mode == "all" or mode == "train":
        # Create forecaster
            forecaster = create_forecaster(
                model_name,
                random_state=random_state,
                verbose=verbose,
                lags=selected_lags,
                steps=steps,
                weight=weight,
                differentiation=differentiation,
                if_exog=if_exog,
            )

            if model_name.lower() != "direct_linearregression":
                # # Hyperparameter search if required
                if select_hyperparameters:
                    forecaster_search = GeneralizedHyperparameterSearch(
                        forecaster=forecaster,
                        y=dataset_selected_features.loc[:end_validation, value],
                        lags=selected_lags,
                        exog=dataset_selected_features.loc[:end_validation, selected_exog],
                        steps=forecasting_steps,
                        initial_train_size=len(dataset_train),  # dataset with features
                        # metric=metric,
                    )
                    results_search, _ = forecaster_search.bayesian_search(
                        param_ranges=default_hyperparameters,
                        n_trials=20,
                        random_state=random_state_hyper,
                    )
                    selected_hyperparameters = results_search["params"].iat[0]
                else:
                    selected_hyperparameters = default_hyperparameters
            else:
                selected_hyperparameters = None

            # # Create final forecaster with best parameters
            final_forecaster = create_forecaster(
                model_name,
                random_state=random_state,
                verbose=verbose,
                lags=selected_lags,
                steps=steps,
                weight=weight,
                differentiation=differentiation,
                custom_params=selected_hyperparameters,
                if_exog=if_exog,
            )

            unfitted_forecaster = copy.deepcopy(final_forecaster)

            if len(selected_exog) == 0:
                final_forecaster.fit(
                    y=dataset_selected_features.loc[:end_validation, value]
                )
            else:
                final_forecaster.fit(
                    y=dataset_selected_features.loc[:end_validation, value],
                    exog=dataset_selected_features.loc[:end_validation, selected_exog],
                )

            if len(selected_exog) == 0:
                backtest_predictions = final_forecaster.predict_interval(
                    steps=backtest_steps, interval=interval
                )
            else:
                backtest_predictions = final_forecaster.predict_interval(
                    steps=backtest_steps,
                    exog=dataset_selected_features.loc[
                        dataset_selected_features.index > end_validation, selected_exog
                    ],
                    interval=interval,
                )

            backtest_error = calculate_error(
                y_test, backtest_predictions[: len(y_test)], y_train, error_type=metric
            )

            coverage, _ = calculate_interval_coverage(
                dataset_selected_features,
                backtest_predictions,
                end_validation,
                end_test,
                value,
            )

        if mode == "all" or mode == "predict":
            final_forecaster = forecaster
            if len(selected_exog) == 0:
                final_forecaster.fit(y=dataset_selected_features[value])
            else:
                final_forecaster.fit(
                    y=dataset_selected_features[value],
                    exog=dataset_selected_features.loc[:end_test, selected_exog],
                )

            if len(selected_exog) == 0:
                exog = None
            else:
                exog = forecast_calendar_features[selected_exog]

            # Make the forecast
            forecast = predict_interval_custom(
                forecaster=final_forecaster,
                exog=exog,
                steps=forecasting_steps,
                interval=interval,
                n_boot=20,
            )

        return ForecastModelResult(
            model_name=model_name,
            sampling_freq=sampling_freq,
            differentiation=differentiation,
            selected_lags=selected_lags,
            selected_exog=selected_exog,
            dataset_selected_features=dataset_selected_features,
            selected_hyperparameters=selected_hyperparameters,
            backtest_steps=backtest_steps,
            backtest_prediction_interval=interval,
            backtest_predictions=backtest_predictions,
            backtest_error=backtest_error,
            backtest_interval_coverage=coverage,
            model_trained_on_all_data=final_forecaster,
            unfitted_forecaster=unfitted_forecaster,
            forecasting_steps=forecasting_steps,
            forecast=forecast,
        )
