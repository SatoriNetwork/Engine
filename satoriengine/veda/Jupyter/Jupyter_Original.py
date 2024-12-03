# Data processing
# ==============================================================================
import numpy as np
import pandas as pd

# Modelling and Forecasting
# ==============================================================================
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# Supplemental functions related to feature extraction and selection
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

# Statistical tests for stationarity
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

# skforecast wrappers/interfaces that simply the use of a combination of different capabilities
import skforecast
from skforecast.ForecasterBaseline import ForecasterEquivalentDate
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import bayesian_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import select_features
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster

# import optuna

from skforecast.Sarimax import Sarimax
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.model_selection_sarimax import backtesting_sarimax
from skforecast.model_selection_sarimax import grid_search_sarimax

from pmdarima import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Plots and Graphs
# ==============================================================================

import matplotlib.dates as mdates
import plotly.graph_objects as go
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.size'] = 10

# Warnings configuration
# ==============================================================================
import warnings
warnings.filterwarnings('once')

from skforecast.model_selection import grid_search_forecaster, random_search_forecaster, bayesian_search_forecaster
from scipy.stats import kruskal
from statsmodels.tsa.stattools import adfuller, kpss
from pandas.tseries.frequencies import to_offset
from datetime import datetime, timedelta
import random

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
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error, mean_squared_error, mean_absolute_error
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError #check if this is needed
import optuna

def create_forecaster(model_type, if_exog=None, random_state=None, verbose=None, lags=None, differentiation=None, custom_params=None, weight=None, steps=None, time_metric_baseline="days", forecasterequivalentdate=1, forecasterequivalentdate_n_offsets=7, y=None, start_p=24, start_q=0, max_p=24, max_q=1, seasonal=True, test='adf', m=24, d=None, D=None):
    forecaster_params = {
        'lags': lags,
        'differentiation': differentiation if differentiation and differentiation > 0 else None,
        'weight_func': weight,
        'transformer_y': StandardScaler(),
        'transformer_exog': StandardScaler() if if_exog else None
    }
    forecaster_params = {k: v for k, v in forecaster_params.items() if v is not None}

    regressor_params = {'random_state': random_state} if random_state else {}

    if custom_params:
        regressor_params.update(custom_params)

    def create_autoreg(regressor_class, **extra_params):
        params = {**regressor_params, **extra_params}
        return lambda: ForecasterAutoreg(regressor=regressor_class(**params), **forecaster_params)

    def create_autoreg_direct(regressor_class, **extra_params):
        params = {**regressor_params, **extra_params}
        return lambda: ForecasterAutoregDirect(regressor=regressor_class(**params), steps=steps, **forecaster_params)

    model_creators = {
        'baseline': lambda: ForecasterEquivalentDate(
            offset=pd.DateOffset(**{time_metric_baseline: forecasterequivalentdate}),
            n_offsets=forecasterequivalentdate_n_offsets
        ),
        'arima': lambda: ForecasterSarimax(
            regressor=auto_arima(
                y=y, start_p=start_p, start_q=start_q, max_p=max_p, max_q=max_q,
                seasonal=seasonal, test=test or 'adf', m=m, d=d, D=D,
                trace=True, error_action='ignore', suppress_warnings=True, stepwise=True
            ),
            **forecaster_params
        ),
        'direct_linearregression': create_autoreg_direct(LinearRegression),
        'direct_ridge': create_autoreg_direct(Ridge),
        'direct_lasso': create_autoreg_direct(Lasso),
        'direct_linearboost': create_autoreg_direct(LinearBoostRegressor, base_estimator=LinearRegression()),
        'direct_lightgbm': create_autoreg_direct(LGBMRegressor),
        'direct_xgb': create_autoreg_direct(XGBRegressor),
        'direct_catboost': create_autoreg_direct(CatBoostRegressor),
        'direct_histgradient': create_autoreg_direct(HistGradientBoostingRegressor),
        'autoreg_linearregression': create_autoreg(LinearRegression),
        'autoreg_ridge': create_autoreg(Ridge),
        'autoreg_lasso': create_autoreg(Lasso),
        'autoreg_linearboost': create_autoreg(LinearBoostRegressor, base_estimator=LinearRegression()), # test
        'autoreg_lightgbm': create_autoreg(LGBMRegressor, verbose=verbose),
        'autoreg_randomforest': create_autoreg(RandomForestRegressor),
        'autoreg_xgb': create_autoreg(XGBRegressor),
        'autoreg_catboost': create_autoreg(CatBoostRegressor, verbose=False, allow_writing_files=False, boosting_type='Plain', leaf_estimation_iterations=10),
        'autoreg_histgradient': create_autoreg(HistGradientBoostingRegressor, verbose=0 if verbose == -1 else verbose)
    }

    if model_type not in model_creators:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_type == 'arima' and y is None:
        raise ValueError("For ARIMA model, 'y' parameter is required.")

    return model_creators[model_type]()


def perform_backtesting(forecaster, y, end_validation, exog=None, steps=24, metric='mean_absolute_scaled_error', refit=False,
                        interval=None, n_boot=0, in_sample_residuals=True, binned_residuals=False,
                        is_sarimax=False, fixed_train_size=False, suppress_warnings_fit=True):
    print(len(y))
    print(len(y.loc[:end_validation]))
    if is_sarimax:
        backtesting_params = {
            'forecaster': forecaster,
            'y': y,
            'initial_train_size': len(y.loc[:end_validation]),
            'fixed_train_size': fixed_train_size,
            'steps': steps,
            'metric': metric,
            'refit': False,
            'n_jobs': 'auto',
            'suppress_warnings_fit': suppress_warnings_fit,
            'verbose': False,
            'show_progress': False
        }
        backtest_metric, backtest_predictions = backtesting_sarimax(**backtesting_params)
    else:
        backtesting_params = {
            'forecaster': forecaster,
            'y': y,
            'exog': exog,
            'steps': steps,
            'metric': metric,
            'initial_train_size': len(y.loc[:end_validation]),
            'refit': refit,
            'interval': interval,
            'n_boot': n_boot,
            'in_sample_residuals': in_sample_residuals,
            'binned_residuals': binned_residuals,
            'n_jobs': 'auto',
            'verbose': False,
            'show_progress': True
        }
        backtest_metric, backtest_predictions = backtesting_forecaster(**backtesting_params)

    print(f"Backtest error ({metric}): {backtest_metric}")
    return backtest_metric, backtest_predictions


def fractional_hour_generator (datetimeparameter):
    # print('starting function')
    whole_time = datetimeparameter.time()
    fractional_hour = whole_time.hour + whole_time.minute/60.0 + 1
    return fractional_hour

def test_seasonality(differenced_dataset, SF, sampling_frequency):
    print(f"new seasonality test with SF = {SF}")

    if SF == 'month':
        differenced_dataset[SF] = differenced_dataset.index.month
    elif SF == 'week':
        differenced_dataset[SF] = differenced_dataset.index.isocalendar().week
    elif SF == 'day_of_week':
        differenced_dataset[SF] = differenced_dataset.index.dayofweek + 1
    elif SF == 'hour':
        differenced_dataset[SF] = differenced_dataset.index.hour + 1
    elif SF == 'fractional_hour':
        differenced_dataset[SF] = differenced_dataset.index.map(fractional_hour_generator)
    else:
        # For any other potential SF values
        try:
            differenced_dataset[SF] = getattr(differenced_dataset.index, SF)
        except AttributeError:
            print(f"Error: {SF} is not a valid attribute of the DatetimeIndex.")
            return False, None

    unique_seasonal_frequency = differenced_dataset[SF].unique()

    if len(unique_seasonal_frequency) < 2:
        print(f"{SF.capitalize()} has less than 2 unique values. Cannot perform seasonality test.")
        return False, None

    res = []
    for i in unique_seasonal_frequency:
        group_data = differenced_dataset[differenced_dataset[SF] == i]['value']
        if not group_data.empty:
            res.append(group_data)
        else:
            print(f"Seasonal frequency {i} has no data.")

    if len(res) < 2:
        print(f"{SF.capitalize()} has less than 2 non-empty groups. Cannot perform seasonality test!!!!.")
        return False, None
    try:
        H_statistic, p_value = kruskal(*res)
        p_value = round(p_value, 3)
        seasonal = p_value <= 0.05

        print(f"{SF.capitalize()} H_statistic is {H_statistic}")
        print(f"{SF.capitalize()} p_value is {p_value}")
        print(f"Seasonality that is built on {SF} is {seasonal}")

        return seasonal, p_value
    except ValueError as e:
        print(f"Error in seasonality test for {SF}: {str(e)}")
        return False, None

def create_exogenous_features(original_dataset, optimally_differenced_dataset, dataset_start_time, dataset_end_time, include_fractional_hour = False, exogenous_feature_type='AdditiveandMultiplicativeExogenousFeatures', sampling_frequency='h'):
    if exogenous_feature_type == 'NoExogenousFeatures':
        return [], pd.DataFrame(), pd.DataFrame(), False, False, False

    # Initialize new dataset
    new_dataset = pd.DataFrame(index=original_dataset.index)
    # print(new_dataset)
    dataset_start_time = dataset_start_time.strftime('%Y-%m-%d %H:%M:%S')
    dataset_end_time = dataset_end_time.strftime('%Y-%m-%d %H:%M:%S')

    # start_date_time_object = datetime.datetime.strptime(dataset_start_time, '%Y-%m-%d %H:%M:%S')
    # end_date_time_object = datetime.datetime.strptime(dataset_end_time, '%Y-%m-%d %H:%M:%S')

    start_date_time_object = datetime.strptime(dataset_start_time, '%Y-%m-%d %H:%M:%S')
    end_date_time_object = datetime.strptime(dataset_end_time, '%Y-%m-%d %H:%M:%S')

    dataset_delta = end_date_time_object - start_date_time_object
    # print(dataset_delta)

    dataset_offset = to_offset('{td.days}D{td.seconds}s'.format(td=dataset_delta))

    SeasonalFrequency = []

    sampling_frequency_offset = to_offset(sampling_frequency)
    # print(sampling_frequency)
    # print(sampling_frequency_offset)
    hour_test = to_offset('1h')
    day_of_week_test = to_offset('1d')
    week_test = to_offset('7d')
    month_test = to_offset('31d')
    year_test = to_offset('366d')


    if sampling_frequency_offset <= hour_test and dataset_offset >= to_offset('3d') :
        SeasonalFrequency.append('hour')
        # temp
        # print(" features include hour")
        # if sampling_frequency_offset < hour_test:
        #     print(" sampling frequencyt offset is less than hour test")
        # if exogenous_feature_type != 'AdditiveandMultiplicativeExogenousFeatures':
        #     print(" case for AdditiveandMultiplicativeExogenousFeatures")
        # if include_fractional_hour == True:
        #     print(" include_fractional_hour set as True")
        # if (exogenous_feature_type != 'AdditiveandMultiplicativeExogenousFeatures' or include_fractional_hour == True):
        #     print(" right part true")
        #temp
        if sampling_frequency_offset < hour_test and ((exogenous_feature_type in ['ExogenousFeaturesBasedonSeasonalityTest', 'ExogenousFeaturesBasedonSeasonalityTestWithAdditivenMultiplicative']) or include_fractional_hour == True):
            # print(" inside fractional hour")
            SeasonalFrequency.append('fractional_hour')

    if sampling_frequency_offset <= day_of_week_test and dataset_offset >= to_offset('21d') :
        SeasonalFrequency.append('day_of_week')

    if sampling_frequency_offset <= week_test and dataset_offset >= to_offset('1095d') :
        SeasonalFrequency.append('week')

    if sampling_frequency_offset <= month_test and dataset_offset >= to_offset('1095d') :
        SeasonalFrequency.append('month')

    if sampling_frequency_offset <= year_test and dataset_offset >= to_offset('1095d') : # in the future we can add in holidays and yearly_quarters
        SeasonalFrequency.append('year')

    print("Finished creating list of SF")
    print("Here is the issue")
    print(SeasonalFrequency)
    # Create all calendar features
    for SF in SeasonalFrequency:
        if SF == 'hour' :
            new_dataset[SF] = new_dataset.index.hour + 1 # we should consider for odd sampling freq may decide for the value to be a fraction ( need different formula )
        elif SF == 'fractional_hour':
            # print("entered")
            new_dataset[SF] = new_dataset.index.map(fractional_hour_generator) # set the right parameter
        elif SF == 'day_of_week':
            new_dataset[SF] = new_dataset.index.dayofweek + 1
        elif SF == 'week':
            new_dataset[SF] = new_dataset.index.isocalendar().week
        elif SF == 'month':
            new_dataset[SF] = new_dataset.index.month
        elif SF == 'year':
            new_dataset[SF] = new_dataset.index.year
    # print(new_dataset)
    # print(new_dataset['hour'])
    # print(new_dataset['fractional_hour'])
    # print(new_dataset['fractional_hour', 'hour'])
    # print(SeasonalFrequency)
    # Create cyclical features
    for feature in new_dataset.columns:
        new_dataset[f'sin_{feature}'] = np.sin(2 * np.pi * new_dataset[feature] / new_dataset[feature].max())
        new_dataset[f'cos_{feature}'] = np.cos(2 * np.pi * new_dataset[feature] / new_dataset[feature].max())
    # print(new_dataset)
    frac_hour_seasonal = None
    hour_seasonal = None
    day_of_week_seasonal = None
    week_seasonal = None
    if exogenous_feature_type in ['ExogenousFeaturesBasedonSeasonalityTest', 'ExogenousFeaturesBasedonSeasonalityTestWithAdditivenMultiplicative']:
        # Perform seasonality tests
        # run the seasonality test on both hour and fractional_hour to test both
        # if none is seasonal ( seasonality test is false for both )  then both return false for seasonality
        # if one is true then we return for that seasonality as true and the other as false
        # if both are true for seasonality then we want to return to one as true and the other as false ( priority for smaller p_value and return its seasonality as true ) (if p_value same then return hour seasonality as true)
        seasonal_periods = []
        p_values = {}
        chosen_hour_type = None  # This will store either 'hour' or 'fractional_hour'

        # Special handling for hour and fractional_hour
        print("Here")
        if 'hour' in SeasonalFrequency:
            hour_seasonal, hour_p_value = test_seasonality(optimally_differenced_dataset, 'hour', sampling_frequency)
            frac_hour_seasonal, frac_hour_p_value = test_seasonality(optimally_differenced_dataset, 'fractional_hour', sampling_frequency)
        if 'day_of_week' in SeasonalFrequency:
            day_of_week_seasonal, day_of_week_p_value = test_seasonality(optimally_differenced_dataset, 'day_of_week', sampling_frequency)
        if 'week' in SeasonalFrequency:
            week_seasonal, week_p_value = test_seasonality(optimally_differenced_dataset, 'week', sampling_frequency)
            # calculation of week seasonality test to determine seasonal period of a year can be improved in the future by using day_of_year instead of week_of_year
        if not hour_seasonal and not frac_hour_seasonal:
            print("Neither hour nor fractional_hour is seasonal.")
        elif hour_seasonal and frac_hour_seasonal:
            if hour_p_value <= frac_hour_p_value:
                chosen_hour_type = 'hour'
                seasonal_periods.append('hour')
                p_values['hour'] = hour_p_value
                print("Both hour and fractional_hour are seasonal. Choosing hour due to lower or equal p-value.")
            else:
                chosen_hour_type = 'fractional_hour'
                seasonal_periods.append('fractional_hour')
                p_values['fractional_hour'] = frac_hour_p_value
                print("Both hour and fractional_hour are seasonal. Choosing fractional_hour due to lower p-value.")
        elif hour_seasonal:
            chosen_hour_type = 'hour'
            seasonal_periods.append('hour')
            p_values['hour'] = hour_p_value
            print("Only hour is seasonal.")
        elif frac_hour_seasonal:
            chosen_hour_type = 'fractional_hour'
            seasonal_periods.append('fractional_hour')
            p_values['fractional_hour'] = frac_hour_p_value
            print("Only fractional_hour is seasonal.")

        # Test other seasonal frequencies
        for SF in [sf for sf in SeasonalFrequency if sf not in ['hour', 'fractional_hour']]:
            is_seasonal, p_value = test_seasonality(optimally_differenced_dataset, SF, sampling_frequency)
            if is_seasonal:
                seasonal_periods.append(SF)
                p_values[SF] = p_value

        if not seasonal_periods:
            print("No seasonal periods detected. Returning no exogenous calendar related features.")
            new_dataset = pd.DataFrame(index=original_dataset.index)
        else:
            print("Detected seasonal periods:")
            for period in seasonal_periods:
                print(f"{period}: p-value = {p_values[period]}")

            # Keep only features for seasonal periods, excluding the non-chosen hour type
            seasonal_features = [col for col in new_dataset.columns if any(period in col for period in seasonal_periods) and
                                 (chosen_hour_type not in ['hour', 'fractional_hour'] or
                                  (chosen_hour_type == 'hour' and 'fractional_hour' not in col) or
                                  (chosen_hour_type == 'fractional_hour' and 'hour' in col))]
            new_dataset = new_dataset[seasonal_features]

    num_columns = new_dataset.shape[1]
    # print(f"Number of columns: {num_columns}")

    if exogenous_feature_type in ['AdditiveandMultiplicativeExogenousFeatures', 'ExogenousFeaturesBasedonSeasonalityTestWithAdditivenMultiplicative'] and num_columns > 0:
        # Apply polynomial features (multiplicative case)
        polynomialobject2 = PolynomialFeatures(
            degree=2,
            interaction_only=True, # was False
            include_bias=False
        ).set_output(transform="pandas")

        # print(new_dataset)
        # new_dataset.dropna()
        # print(new_dataset)
        num_columns = new_dataset.shape[1]
        print(f"Number of columns: {num_columns}")
        new_dataset = polynomialobject2.fit_transform(new_dataset.dropna())
        # print(new_dataset)

    # Get exogenous feature names
    exog_features = new_dataset.columns.tolist()

    # Create final dataframe of exogenous features
    df_exogenous_features = new_dataset.copy()
    # print(df_exogenous_features)
    return exog_features, new_dataset, df_exogenous_features, hour_seasonal, day_of_week_seasonal, week_seasonal

def generate_exog_data(end_date, freq, steps, date_format):
    end_validation = pd.to_datetime(end_date, format=date_format)

    # Generate date range for the exogenous series
    date_range = pd.date_range(start=end_validation + pd.Timedelta(freq),
                               periods=steps,
                               freq=freq)

    # Create exog_series with 0 values
    exog_series = pd.Series(0, index=date_range)

    # Create exog_timewindow
    exog_timewindow = exog_series.reset_index()
    exog_timewindow.columns = ['date_time', 'value']
    exog_timewindow['date_time'] = pd.to_datetime(exog_timewindow['date_time'], format=date_format)
    exog_timewindow = exog_timewindow.set_index('date_time')
    exog_timewindow = exog_timewindow.asfreq(freq)

    return exog_series, exog_timewindow

# Usage:
# exog_series, exog_timewindow = generate_exog_data('2017-06-30 23:00:00', 'H', 24, '%Y-%m-%d %H:%M:%S')

class GeneralizedHyperparameterSearch:
    def __init__(self, forecaster, y, lags, exog=None, steps=12, metric='mean_absolute_scaled_error',
                 initial_train_size=None, fixed_train_size=False, refit=False,
                 return_best=True, n_jobs='auto', verbose=False, show_progress=True):
        self.forecaster = forecaster
        self.y = y
        self.exog = exog
        self.steps = steps
        self.metric = metric
        self.initial_train_size = initial_train_size
        self.fixed_train_size = fixed_train_size
        self.refit = refit
        self.return_best = return_best
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.show_progress = show_progress
        self.default_param_ranges = {
                'lightgbm': {
                    'n_estimators': ('int', 400, 1200, 100),
                    'max_depth': ('int', 3, 10, 1),
                    'min_data_in_leaf': ('int', 25, 500),
                    'learning_rate': ('float', 0.01, 0.5),
                    'feature_fraction': ('float', 0.5, 1, 0.1),
                    'max_bin': ('int', 50, 250, 25),
                    'reg_alpha': ('float', 0, 1, 0.1),
                    'reg_lambda': ('float', 0, 1, 0.1),
                    'lags': ('categorical', [lags])
                },
                'catboost': {
                    'n_estimators': ('int', 100, 1000, 100),
                    'max_depth': ('int', 3, 10, 1),
                    'learning_rate': ('float', 0.01, 1),
                    'lags': ('categorical', [lags])
                },
                'randomforest': {
                    'n_estimators'    : ('int', 400, 1200, 100),
                    'max_depth'       : ('int', 3, 10, 1),
                    'ccp_alpha'       : ('float', 0, 1, 0.1),
                    'lags'            : ('categorical', [lags])
                },
                'xgboost': {
                    'n_estimators'    : ('int', 30, 5000),
                    'max_depth'       : ('int' , -1, 256),
                    'learning_rate'   : ('float', 0.01, 1),
                    'subsample'       : ('float', 0.01, 1.0),
                    'colsample_bytree': ('float', 0.01, 1.0),
                    'gamma'           : ('float', 0, 1),
                    'reg_alpha'       : ('float', 0, 1),
                    'reg_lambda'      : ('float', 0, 1),
                    'lags'            : ('categorical', [lags])
                },
                'histgradient': {
                    'max_iter'          : ('int', 400, 1200, 100),
                    'max_depth'         : ('int' , 1, 256),
                    'learning_rate'     : ('float', 0.01, 1),
                    'min_samples_leaf'  : ('int', 1, 20, 1),
                    'l2_regularization' : ('float', 0, 1),
                    'lags'              : ('categorical', [lags])
                }
            }
        self.model_type = self._detect_model_type()
        self.current_param_ranges = self.default_param_ranges.get(self.model_type, {}).copy()

    def _detect_model_type(self):
        if 'LGBMRegressor' in str(type(self.forecaster.regressor)):
            return 'lightgbm'
        elif 'CatBoostRegressor' in str(type(self.forecaster.regressor)):
            return 'catboost'
        elif 'RandomForestRegressor' in str(type(self.forecaster.regressor)):
            return 'randomforest'
        elif 'XGBRegressor' in str(type(self.forecaster.regressor)):
            return 'xgboost'
        elif 'HistGradientBoostingRegressor' in str(type(self.forecaster.regressor)):
            return 'histgradient'
        else:
            return 'unknown'

    def exclude_parameters(self, params_to_exclude):
        """
        Exclude specified parameters from the search space.

        :param params_to_exclude: List of parameter names to exclude
        """
        for param in params_to_exclude:
            if param in self.current_param_ranges:
                del self.current_param_ranges[param]
            else:
                print(f"Warning: Parameter '{param}' not found in the current search space.")

    def include_parameters(self, params_to_include):
        """
        Include previously excluded parameters back into the search space.

        :param params_to_include: List of parameter names to include
        """
        default_ranges = self.default_param_ranges.get(self.model_type, {})
        for param in params_to_include:
            if param in default_ranges and param not in self.current_param_ranges:
                self.current_param_ranges[param] = default_ranges[param]
            elif param in self.current_param_ranges:
                print(f"Warning: Parameter '{param}' is already in the current search space.")
            else:
                print(f"Warning: Parameter '{param}' not found in the default search space for {self.model_type}.")

    def update_parameter_range(self, param, new_range):
        """
        Update the range of a specific parameter.

        :param param: Name of the parameter to update
        :param new_range: New range for the parameter (tuple)
        """
        if param in self.current_param_ranges:
            self.current_param_ranges[param] = new_range
        else:
            print(f"Warning: Parameter '{param}' not found in the current search space.")

    def display_available_parameters(self):
        """
        Display the available parameters and their current ranges for the selected model type.
        """
        print(f"Available parameters for {self.model_type.upper()} model:")
        self._display_params(self.current_param_ranges)
        print("\nYou can override these parameters by passing a dictionary to the bayesian_search method.")

    def _display_params(self, param_ranges):
        for param, config in param_ranges.items():
            param_type = config[0]
            if param_type in ['int', 'float']:
                step = config[3] if len(config) > 3 else 'N/A'
                print(f"  {param}: {param_type}, range: {config[1]} to {config[2]}, step: {step}")
            elif param_type == 'categorical':
                print(f"  {param}: {param_type}, choices: {config[1]}")


    def _prepare_lags_grid(self, lags):
        if isinstance(lags, dict):
            return lags
        elif isinstance(lags, (list, np.ndarray)):
            return {'lags': lags}
        else:
            raise ValueError("lags must be either a dict, list, or numpy array")

    def _prepare_param_grid(self, param_ranges):
        param_grid = {}
        for param, config in param_ranges.items():
            param_type = config[0]
            if param_type in ['int', 'float']:
                start, stop = config[1:3]
                step = config[3] if len(config) > 3 else 1
                if param_type == 'int':
                    param_grid[param] = list(range(start, stop + 1, step))
                else:
                    param_grid[param] = list(np.arange(start, stop + step, step))
            elif param_type == 'categorical':
                param_grid[param] = config[1]
        return param_grid

    def _prepare_param_distributions(self, param_ranges):
        param_distributions = {}
        for param, config in param_ranges.items():
            param_type = config[0]
            if param_type in ['int', 'float']:
                start, stop = config[1:3]
                step = config[3] if len(config) > 3 else 1
                if param_type == 'int':
                    param_distributions[param] = np.arange(start, stop + 1, step, dtype=int)
                else:
                    param_distributions[param] = np.arange(start, stop + step, step)
            elif param_type == 'categorical':
                param_distributions[param] = config[1]
        return param_distributions

    def grid_search(self, lags_grid, param_ranges=None):
        if param_ranges is None:
            param_ranges = self.current_param_ranges
        else:
            self.current_param_ranges.update(param_ranges)

        param_grid = self._prepare_param_grid(self.current_param_ranges)
        lags_grid = self._prepare_lags_grid(lags_grid)

        return grid_search_forecaster(
            forecaster=self.forecaster,
            y=self.y,
            exog=self.exog,
            param_grid=param_grid,
            lags_grid=lags_grid,
            steps=self.steps,
            metric=self.metric,
            initial_train_size=self.initial_train_size,
            fixed_train_size=self.fixed_train_size,
            refit=self.refit,
            return_best=self.return_best,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            show_progress=self.show_progress
        )

    # needs to be fixed
    def random_search(self, lags_grid, param_ranges=None, n_iter=10, random_state=123):
        if param_ranges is None:
            param_ranges = self.current_param_ranges
        else:
            self.current_param_ranges.update(param_ranges)

        param_distributions = self._prepare_param_distributions(self.current_param_ranges)

        return random_search_forecaster(
            forecaster=self.forecaster,
            y=self.y,
            exog=self.exog,
            param_distributions=param_distributions,
            lags_grid=lags_grid,
            steps=self.steps,
            n_iter=n_iter,
            metric=self.metric,
            initial_train_size=self.initial_train_size,
            fixed_train_size=self.fixed_train_size,
            refit=self.refit,
            return_best=self.return_best,
            random_state=random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            show_progress=self.show_progress
        )


    def bayesian_search(self, param_ranges=None, n_trials=20, random_state=123):
        if param_ranges is None:
            param_ranges = {}

        # Update current_param_ranges with user-provided param_ranges
        for param, range_value in param_ranges.items():
            if param in self.current_param_ranges:
                self.current_param_ranges[param] = range_value
            else:
                self.current_param_ranges[param] = range_value
                print(f"New parameter '{param}' added to the search space.")

        def create_search_space(trial, param_ranges):
            search_space = {}
            for param, config in param_ranges.items():
                param_type = config[0]

                if param_type == 'int':
                    start, stop = config[1:3]
                    step = config[3] if len(config) > 3 else 1
                    search_space[param] = trial.suggest_int(param, start, stop, step=step)
                elif param_type == 'float':
                    start, stop = config[1:3]
                    step = config[3] if len(config) > 3 else None
                    if step:
                        search_space[param] = trial.suggest_float(param, start, stop, step=step)
                    else:
                        search_space[param] = trial.suggest_float(param, start, stop)
                elif param_type == 'categorical':
                    choices = config[1]
                    search_space[param] = trial.suggest_categorical(param, choices)
                else:
                    raise ValueError(f"Unknown parameter type for {param}: {param_type}")
            return search_space

        def search_space_wrapper(trial):
            return create_search_space(trial, self.current_param_ranges)

        print(random_state)
        return bayesian_search_forecaster(
            forecaster=self.forecaster,
            y=self.y,
            exog=self.exog,
            search_space=search_space_wrapper,
            steps=self.steps,
            metric=self.metric,
            initial_train_size=self.initial_train_size,
            fixed_train_size=self.fixed_train_size,
            refit=self.refit,
            return_best=self.return_best,
            n_trials=n_trials,
            random_state=random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            show_progress=self.show_progress
        )


def predict_interval_custom(forecaster, steps=24, interval=[10, 90], n_boot=20, exog=None, alpha=None):
    if isinstance(forecaster, ForecasterSarimax):
        # Case for ARIMA model
        if alpha is None:
            alpha = 0.05  # Default value if not provided
        return forecaster.predict_interval(steps=steps, alpha=alpha)
    elif exog is not None:
        # Case with exogenous features for other models
        return forecaster.predict_interval(
            exog=exog,
            steps=steps,
            interval=interval,
            n_boot=n_boot
        )
    else:
        # Case without exogenous features for other models
        return forecaster.predict_interval(
            steps=steps,
            interval=interval,
            n_boot=n_boot
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

def calculate_interval_coverage(satoridataset, interval_predictions, end_validation, end_test, value):
    # Ensure both datasets have the same index
    common_index = satoridataset.loc[end_validation:end_test].index.intersection(interval_predictions.index)


    # Align the datasets
    satori_aligned = satoridataset.loc[common_index, value]
    predictions_aligned = interval_predictions.loc[common_index]

    # Calculate coverage
    coverage = np.mean(
        np.logical_and(
            satori_aligned >= predictions_aligned["lower_bound"],
            satori_aligned <= predictions_aligned["upper_bound"]
        )
    )

    #pseudo-code
    # a df which contains a common index [date-time], 3 columns[  predictions_aligned["lower_bound"], satori_aligned["value"], predictions_aligned["upper_bound"] ]
    coverage_df = pd.DataFrame({
        'lower_bound': predictions_aligned['lower_bound'],
         value : satori_aligned,
        'upper_bound': predictions_aligned['upper_bound'],
        'if_in_range': (satori_aligned >= predictions_aligned['lower_bound']) &
                       (satori_aligned <= predictions_aligned['upper_bound'])
    }, index=common_index)

    total_count = len(coverage_df)
    in_range_count = coverage_df['if_in_range'].sum()
    #end

    #Equivalent code
    # covered_data = 0
    # total_rows = len(satori_aligned)

    # for i in range(total_rows):
    #     aligned_value = satori_aligned.iloc[i]
    #     lower_bound = predictions_aligned['lower_bound'].iloc[i]
    #     upper_bound = predictions_aligned['upper_bound'].iloc[i]

    #     if (aligned_value >= lower_bound) and (aligned_value <= upper_bound):
    #         covered_data += 1

    # coverage = covered_data / total_rows
    #end Equivalent code

    # Calculate total area of the interval
    area = (predictions_aligned['upper_bound'] - predictions_aligned['lower_bound']).sum()

    print(f"Total data points: {total_count}")
    print(f"Data points within range: {in_range_count}")
    print(f"Predicted interval coverage assuming Gaussian distribution: {round(100*coverage, 2)}%")
    print(f"Total area of the interval: {round(area, 2)}")
    # print(coverage_df)

    return coverage, area

def perform_RFECV_feature_selection(
    forecaster,
    y,
    exog,
    end_train,
    step=1,
    cv=2,
    min_features_to_select=1,
    subsample=0.5,
    random_state=123,
    verbose=False
):

    selector = RFECV(
        estimator=forecaster.regressor,
        step=step,
        cv=cv,
        min_features_to_select=min_features_to_select,
        n_jobs=-1  # Use all available cores
    )

    # Perform feature selection
    selected_lags, selected_exog = select_features(
        forecaster=forecaster,
        selector=selector,
        y=y,
        exog=exog,
        select_only=None,
        force_inclusion=None,
        subsample=subsample,
        random_state=random_state,
        verbose=verbose
    )

    return selected_lags, selected_exog

def perform_RFE_feature_selection(
    estimator,
    forecaster,
    y,
    exog,
    n_features_to_select=None,
    step=1,
    subsample=0.5,
    random_state=123,
    verbose=False
):
    # Create the RFE selector
    selector = RFE(
        estimator=estimator,
        step=step,
        n_features_to_select=n_features_to_select
    )

    # Perform feature selection
    selected_lags, selected_exog = select_features(
        forecaster=forecaster,
        selector=selector,
        y=y,
        exog=exog,
        select_only=None,
        force_inclusion=None,
        subsample=subsample,
        random_state=random_state,
        verbose=verbose
    )

    return selected_lags, selected_exog

def determine_differentiation(data_train_with_id, max_diff=5):
    differentiation = 0

    if 'id' in data_train_with_id.columns:
        data_train = data_train_with_id.drop('id', axis=1)

    # to test differentiation, we create a new dataset that replaces missing values with interpolated values

    data_diff = data_train.copy()

    # make a function parameter : ( dataset, replace=False, 'imputed_value', method, order=None ) return dataset
    # temporary_series = data_diff['value'].interpolate(method='polynomial', order=2)
    # # we are doing this in a non-efficient way of python handling the above line
    # data_diff = data_diff.drop(columns=['value']) #
    # data_diff['value'] = temporary_series.values

    data_diff = impute_data(data_diff, replace=True, imputed_value='value', method='polynomial', order=2)
    print(data_diff.head(30))
    print("below is the tail")
    print(data_diff.tail(30))
    print("----------------------------------------------------------------")
    print(data_diff['value'].isna().sum())

    for i in range(max_diff):
        # print(i)
        # print(data_diff['value']) # Series([], Freq: 55min, Name: value, dtype: float64)
        # print(data_diff['value'].nunique())
        # Check if data is constant
        if data_diff['value'].nunique() <= 1:
            print(f'The time series (diff order {i}) is constant or nearly constant.')
            break

        try:
            adfuller_result = adfuller(data_diff['value'])
            kpss_result = kpss(data_diff['value'])
        except ValueError as e:
            print(f"Error during statistical test: {e}")
            break

        print(f'adfuller stat and adfuller boolean is: {adfuller_result[1]}, {adfuller_result[1] < 0.05}')
        print(f'kpss stat and kpss boolean is: {kpss_result[1]}, {kpss_result[1] > 0.05}')

        if adfuller_result[1] < 0.05 and kpss_result[1] > 0.05:
            print(f'The time series (diff order {i}) is likely to be stationary.')
            break
        else:
            print(f'The time series (diff order {i}) is likely to be non-stationary.')
            differentiation += 1
            if differentiation < max_diff:
                data_diff = data_diff.diff().dropna()
            else:
                break

    if differentiation == 0:
        data_diff = pd.DataFrame()

    print(f'The differentiation is: {differentiation}')
    return differentiation, data_diff

def baseline_forecasting(satoridataset, end_validation, steps=24):
    baseline_forecaster = create_forecaster("baseline")
    baseline_forecaster.fit(y=satoridataset.loc[:end_validation, 'value'])

    # Perform backtesting
    baseline_metric, baseline_backtesting_predictions = perform_backtesting(
        baseline_forecaster,
        y=satoridataset['value'],
        end_validation=end_validation,
        steps=steps,
        metric='mean_absolute_scaled_error'
    )

    # Fit the forecaster on the entire dataset
    baseline_forecaster.fit(y=satoridataset['value'])

    # Make predictions
    baseline_predictions = baseline_forecaster.predict(steps=steps)

    return baseline_metric, baseline_predictions

# loop over 'df_exogenous_features' columns, and if column name in not in list lightgbm_selected_exog then delete that column
def filter_dataframe_col(df, selected_col):
    col_to_keep = [ col for col in df.columns if col in selected_col ]
    return df[col_to_keep]



def determine_feature_set(dataset, data_train, end_validation, end_train, dataset_start_time, dataset_end_time, dataset_with_features, weight_para=False, metric="mean_absolute_scaled_error", modeltype="tree_or_forest",initial_lags=None, exogenous_feature_type=None, feature_set_reduction=False, feature_set_reduction_method=None, FeatureSetReductionStep=5, FeatureSetReductionSubSample=0.5, RFECV_CV=2, RFECV_min_features_to_select=10, RFE_n_features_to_select=None, bayesian_trial=None, random_state_hyper=123, last_observed_datetime=None, frequency='h', backtest_steps=24, prediction_steps=24):

    # Determining Differencing Steps
    differentiation, data_diff = determine_differentiation(dataset) # Maybe changed later

    if dataset['value'].isna().any():
        if weight_para:
            gaps = find_nan_gaps(dataset, 'value')
            total_nans = dataset.isna().sum().value
            ratio = (total_nans/len(gaps))
            multiplicative_factor = round(ratio*0.75)
            adjustment = pd.Timedelta(frequency * multiplicative_factor)
            weight = create_weight_func(dataset, frequency, adjustment)
        else:
            weight = None
        # dataset = impute_data(dataset, replace=False, imputed_value='value', method='polynomial', order=2)
        dataset = impute_data(dataset, replace=False, imputed_value='value', method='quadratic')
        value = 'value_imputed'
        missing_values = True
    else:
        value = 'value'
        missing_values = False
        weight = None

    if exogenous_feature_type in ['ExogenousFeaturesBasedonSeasonalityTest', 'ExogenousFeaturesBasedonSeasonalityTestWithAdditivenMultiplicative']:
        include_fractional_hour=True
    else:
        include_fractional_hour=False

    if data_diff.empty:
        # case where differentiation is zero

        exog_features, calendar_features , df_exogenous_features, hour_seasonality, dow_seasonality, week_seasonality = create_exogenous_features(original_dataset=dataset,
                                                                                             optimally_differenced_dataset=dataset,
                                                                                             dataset_start_time=dataset_start_time,
                                                                                             dataset_end_time=dataset_end_time,
                                                                                             include_fractional_hour=include_fractional_hour,
                                                                                             exogenous_feature_type=exogenous_feature_type,
                                                                                             sampling_frequency=frequency
                                                                                            )
    else:
        # case where differentiation is greater than or equal to 1
        exog_features, calendar_features , df_exogenous_features, hour_seasonality, dow_seasonality, week_seasonality = create_exogenous_features(original_dataset=dataset,
                                                                                             optimally_differenced_dataset=data_diff,
                                                                                             dataset_start_time=dataset_start_time,
                                                                                             dataset_end_time=dataset_end_time,
                                                                                             include_fractional_hour = include_fractional_hour,
                                                                                             exogenous_feature_type=exogenous_feature_type,
                                                                                             sampling_frequency=frequency
                                                                                            )

    if feature_set_reduction == True:

        if exog_features == []:
            if_exog = None
        else:
            print("Exog exists")
            if_exog = StandardScaler()
            # print(if_exog)
        print("outside here maybe")
        if modeltype == "tree_or_forest":
            hyper_forecaster = create_forecaster(
                'autoreg_lightgbm',
                random_state=123,
                verbose=-1,
                lags=initial_lags,
                weight=weight, # only for missing data
                differentiation=differentiation,  # This will be used only if it's not None and > 0
                if_exog=if_exog
            )
        print("outside here maybe")
        # Hyper-Parameter Search
        model_search = GeneralizedHyperparameterSearch(forecaster=hyper_forecaster, y=dataset.loc[:end_validation, value], lags=initial_lags, steps=backtest_steps, initial_train_size=len(data_train), metric=metric)
        print("outside here maybe")
        results_search, frozen_trial = model_search.bayesian_search(
            n_trials=bayesian_trial,
            random_state=random_state_hyper)
        print("outside here maybe")
        best_params = results_search['params'].iat[0]

        # Feature-Selection Loop
        # lightgbm_hyper_forecaster = create_forecaster('lightgbm', random_state=123, verbose=-1, lags=initial_lags, differentiation=differentiation, custom_params=lightgbm_best_params, weight=weight )
        # lightgbm_hyper_forecaster = create_forecaster('lightgbm', random_state=123, verbose=-1, lags=initial_lags, differentiation=differentiation, custom_params=lightgbm_best_params )

        if modeltype == "tree_or_forest":
            feature_forecaster = create_forecaster('autoreg_lightgbm', random_state=123, verbose=-1, lags=initial_lags, differentiation=differentiation, custom_params=best_params, weight=weight, if_exog=if_exog )

        print("outside here maybe")
        if feature_set_reduction_method=='RFECV':
            # print(lightgbm_hyper_forecaster.get_feature_importances())
            selected_lags, selected_exog = perform_RFECV_feature_selection(
                forecaster=feature_forecaster,
                y=dataset.loc[:end_train, value],
                exog=dataset_with_features.loc[:end_train, exog_features], # replace exog_features if needed
                end_train=end_train,
                step=FeatureSetReductionStep, # set to 1
                cv=RFECV_CV,
                min_features_to_select=RFECV_min_features_to_select, # set to 1
                subsample=FeatureSetReductionSubSample, # set to 0.5
                verbose=True
            )
            # print(lightgbm_selected_exog)
        elif feature_set_reduction_method=='RFE':
            selected_lags, selected_exog = perform_RFE_feature_selection(
                estimator=feature_forecaster.regressor,
                forecaster=feature_forecaster,
                y=dataset.loc[:end_train, value],
                exog=dataset_with_features.loc[:end_train, exog_features],
                step=FeatureSetReductionStep,
                subsample=FeatureSetReductionSubSample,
                n_features_to_select=RFE_n_features_to_select
            )
        # print(f"df_exogenous_features : {df_exogenous_features}")
        # print(f"lightgbm_selected_exog : {lightgbm_selected_exog}")
        df_exogenous_features = filter_dataframe_col(df_exogenous_features, selected_exog)


    else:
        selected_lags = initial_lags
        selected_exog = exog_features


    if exogenous_feature_type != 'NoExogenousFeatures' and len(selected_exog) > 0:
        # make sure this works for the case where the below doesn't break for the noexog features
        _ , exog_timewindow = generate_exog_data( last_observed_datetime, frequency, prediction_steps, '%Y-%m-%d %H:%M:%S')
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        _ , forecast_calendar_features, _, _, _, _  = create_exogenous_features(original_dataset=exog_timewindow,
                                                                      optimally_differenced_dataset=exog_timewindow,
                                                                      dataset_start_time=dataset_start_time,
                                                                      dataset_end_time=dataset_end_time,
                                                                      include_fractional_hour=True,
                                                                      exogenous_feature_type='AdditiveandMultiplicativeExogenousFeatures',
                                                                      sampling_frequency=frequency)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(forecast_calendar_features)
        forecast_calendar_features = filter_dataframe_col(forecast_calendar_features, selected_exog)
        # print(lightgbm_selected_exog)
        # print(forecast_calendar_features)
    else:
        forecast_calendar_features = []

    if missing_values:
        dataset_selected_features = dataset[['value', 'value_imputed']].merge(
            df_exogenous_features,
            left_index=True,
            right_index=True,
            how='left'
        )
    else:
        dataset_selected_features = dataset[['value']].merge(
            df_exogenous_features,
            left_index=True,
            right_index=True,
            how='left'
        )
    # temp modification for reference
    # lightgbm_forecaster.fit(
    #         y=dataset.loc[:end_times['validation'], 'value'],
    #         exog=dataset.loc[:end_times['validation'], lightgbm_selected_exog]
    #     )

    # backtest_error, backtest_predictions = perform_backtesting(
    #     lightgbm_forecaster,
    #     dataset['value'],
    #     end_times['validation'],
    #     exog=dataset[lightgbm_selected_exog],
    #     interval=[10, 90],
    #     n_boot=20
    # )
    # print(f"Lightgbm selected_exog : {lightgbm_selected_exog}")
    # print(len(lightgbm_selected_exog))
    # print(f"Lightgbm Important Features : {lightgbm_forecaster.get_feature_importances()}")

    # seasonal_hour = False
    #end of temp modification

    return selected_lags, selected_exog, differentiation, dataset_selected_features, forecast_calendar_features, hour_seasonality, dow_seasonality, week_seasonality, missing_values, weight
    # return lightgbm_selected_lags, lightgbm_selected_exog, differentiation, forecast_calendar_features

def round_time(first_day, dt, round_to_hours, round_to_minutes, round_to_seconds, offset_hours=0, offset_minutes=0, offset_seconds=0):
    # Apply offset
    dt = dt + timedelta(hours=offset_hours, minutes=offset_minutes, seconds=offset_seconds)

    # Calculate total seconds for the rounding interval
    total_seconds = (round_to_hours * 3600) + (round_to_minutes * 60) + round_to_seconds

    # Calculate seconds since first day
    seconds_since_first = round((dt - first_day).total_seconds())

    # Round to the nearest interval
    rounded_seconds = round(seconds_since_first / total_seconds) * total_seconds

    # Create new datetime with rounded seconds
    rounded_dt = first_day + timedelta(seconds=rounded_seconds)

    # Remove the offset
    rounded_dt -= timedelta(hours=offset_hours, minutes=offset_minutes, seconds=offset_seconds)

    return rounded_dt

def round_to_nearest_minute(dt):
    return dt.replace(second=0, microsecond=0) + timedelta(minutes=1 if dt.second >= 30 else 0)

def process_noisy_dataset(df, round_to_hours, round_to_minutes, round_to_seconds=0, offset_hours=0, offset_minutes=0, offset_seconds=0, datetime_column='date_time'):
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Check if the datetime is in a column or is the index
    if isinstance(df_copy.index, pd.DatetimeIndex):
        datetime_series = df_copy.index
        is_index = True
    elif datetime_column in df_copy.columns:
        datetime_series = pd.to_datetime(df_copy[datetime_column])
        is_index = False
    else:
        raise ValueError(f"Datetime column '{datetime_column}' not found in DataFrame")

    # Get the first day and round it to the nearest minute
    first_day = round_to_nearest_minute(datetime_series.min())

    # Apply round_time function
    rounded_datetimes = datetime_series.map(
        lambda x: round_time(first_day, x, round_to_hours, round_to_minutes, round_to_seconds,
                             offset_hours, offset_minutes, offset_seconds)
    )

    # Update the DataFrame
    if is_index:
        df_copy.index = rounded_datetimes
    else:
        df_copy[datetime_column] = rounded_datetimes

    return df_copy

def roundto_minute(roundingdatetime):
    """
    Round the given timedelta to the nearest minute and return the minute component.
    """
    # Add 30 seconds for rounding
    rounded = roundingdatetime + timedelta(seconds=30)

    # Get total seconds and convert to minutes
    total_minutes = rounded.total_seconds() / 60

    # Extract just the minute component
    minutes = int(total_minutes % 60)

    # print(rounded)
    # print(type(rounded))
    return minutes

def round_time(first_day, dt, round_to_hours, round_to_minutes, round_to_seconds, offset_hours=0, offset_minutes=0, offset_seconds=0):
    # Apply offset
    dt = dt + timedelta(hours=offset_hours, minutes=offset_minutes, seconds=offset_seconds)

    # Calculate total seconds for the rounding interval
    total_seconds = (round_to_hours * 3600) + (round_to_minutes * 60) + round_to_seconds

    # Calculate seconds since first day
    seconds_since_first = round((dt - first_day).total_seconds())

    # Round to the nearest interval
    rounded_seconds = round(seconds_since_first / total_seconds) * total_seconds

    # Create new datetime with rounded seconds
    rounded_dt = first_day + timedelta(seconds=rounded_seconds)

    # Remove the offset
    rounded_dt -= timedelta(hours=offset_hours, minutes=offset_minutes, seconds=offset_seconds)

    return rounded_dt

def round_to_nearest_minute(dt):
    return dt.replace(second=0, microsecond=0) + timedelta(minutes=1 if dt.second >= 30 else 0)

def process_noisy_dataset(df, round_to_hours, round_to_minutes, round_to_seconds=0, offset_hours=0, offset_minutes=0, offset_seconds=0, datetime_column='date_time'):
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Check if the datetime is in a column or is the index
    if isinstance(df_copy.index, pd.DatetimeIndex):
        datetime_series = df_copy.index
        is_index = True
    elif datetime_column in df_copy.columns:
        datetime_series = pd.to_datetime(df_copy[datetime_column])
        is_index = False
    else:
        raise ValueError(f"Datetime column '{datetime_column}' not found in DataFrame")

    # Get the first day and round it to the nearest minute
    first_day = round_to_nearest_minute(datetime_series.min())

    # Apply round_time function
    rounded_datetimes = datetime_series.map(
        lambda x: round_time(first_day, x, round_to_hours, round_to_minutes, round_to_seconds,
                             offset_hours, offset_minutes, offset_seconds)
    )

    # Update the DataFrame
    if is_index:
        df_copy.index = rounded_datetimes
    else:
        df_copy[datetime_column] = rounded_datetimes

    return df_copy

def process_data(filename, sampling_frequency=None, col_names=None, training_percentage=80, validation_percentage=10,
                 test_percentage=10, date_time_format='%Y-%m-%d %H:%M:%S', quick_start=False):

    # Use default column names if not provided
    # if col_names is None:
    #     col_names = ['date_time', 'value', 'id']

    # # Read the CSV file
    # raw_dataset = pd.read_csv(filename, names=col_names, header=None)

    # # Process date_time column
    # raw_dataset['date_time'] = pd.to_datetime(raw_dataset['date_time'], format=date_time_format)
    # raw_dataset = raw_dataset.set_index('date_time')

    if col_names is None:
        col_names = ['date_time', 'value', 'id']

    # Read the CSV file
    raw_dataset = pd.read_csv(filename, names=col_names, header=None)

    # Process date_time column with flexible parsing and standardization
    raw_dataset['date_time'] = pd.to_datetime(raw_dataset['date_time'])

    # Convert to '%Y-%m-%d %H:%M:%S' format
    raw_dataset['date_time'] = raw_dataset['date_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    raw_dataset['date_time'] = pd.to_datetime(raw_dataset['date_time'], format='%Y-%m-%d %H:%M:%S')

    raw_dataset = raw_dataset.set_index('date_time')

    # temp
    raw_diff_dat = raw_dataset.index.to_series().diff()
    value_counts = raw_diff_dat.value_counts().sort_index()
    # result_df = pd.DataFrame({'Value': value_counts.index, 'Occurrences': value_counts.values})
    # result_df = result_df.sort_values('Value').reset_index(drop=True)
    num_distinct_values = len(value_counts)
    if num_distinct_values > (len(raw_dataset) * 0.05):

        median = raw_dataset.index.to_series().diff().median()
        if median < timedelta(hours=1, minutes=0, seconds=29):
            if (median >= timedelta(minutes = 59, seconds = 29)):
                round_to_hour = 1
                round_to_minute = 0
            else:
                round_to_hour = 0
                round_to_minute = roundto_minute(median)
        else:
            round_to_hour = median.total_seconds() // 3600
            round_to_minute = roundto_minute(median - timedelta(hours=round_to_hour, minutes=0, seconds=0))

        dataset = process_noisy_dataset(raw_dataset, round_to_hours=round_to_hour, round_to_minutes=round_to_minute)

    else:
        dataset = raw_dataset

    if sampling_frequency is None:
        sf = dataset.index.to_series().diff().median()

        # Convert to frequency string
        abbr = {'days': 'd', 'hours': 'h', 'minutes': 'min', 'seconds': 's', 'milliseconds': 'ms', 'microseconds': 'us',
            'nanoseconds': 'ns'}

        fmt = lambda sf:"".join(f"{v}{abbr[k]}" for k, v in sf.components._asdict().items() if v != 0)
        if isinstance(sf, pd.Timedelta):
            sampling_frequency = fmt(sf)
        elif isinstance(sf,pd.TimedeltaIndex):
            sampling_frequency = sf.map(fmt)
        else:
            raise ValueError

    # Convert sampling_frequency to timedelta for comparison
    # sf_timedelta = pd.Timedelta(sampling_frequency)

    # # Check if sampling frequency is between 1 day and 1.5 days
    # if pd.Timedelta(days=1) < sf_timedelta < pd.Timedelta(days=1.5):
    #     sampling_frequency = '1d'  # Set to exactly one day

    # Handle duplicates
    duplicates = dataset[dataset.index.duplicated(keep=False)]
    duplicates_sorted = duplicates.sort_index()

    dataset_averaged = dataset.groupby(level=0).agg({
        'value': 'mean',
        'id': 'first'  # Keep the first ID, or you could use 'last' or another method
    })

    # Replace the original dataset with the averaged one
    dataset = dataset_averaged

    print(dataset.tail(5))
    print('***************')
    # Apply the sampling frequency
    print(sampling_frequency)
    # dataset = dataset.asfreq(sampling_frequency)
    dataset = dataset.asfreq(sampling_frequency, method='nearest')
    print(dataset.tail(5))
    print('****************')
    datasetsize = len(dataset)
    #temp
    # print(datasetsize)
    nan_count = dataset['value'].isna().sum()
    # print(f"The number of NaN values in the 'value' column is: {nan_count}")
    #end


    training_index = round(training_percentage / 100 * datasetsize)
    validation_index = min(training_index + round(validation_percentage / 100 * datasetsize), datasetsize - 2 )
    test_index = datasetsize - 1

    # print("start")
    # print(training_index)
    # print(validation_index)
    # print(test_index)
    # print("end")

    print(dataset.tail(5))

    # print(dataset)

    dataset = dataset.reset_index()
    end_train = dataset.iloc[training_index]['date_time']
    end_validation = dataset.iloc[validation_index]['date_time']
    end_test = dataset.iloc[test_index]['date_time']

    dataset_start_time = dataset.iloc[0]['date_time']

    dataset = dataset.set_index('date_time')

    dataset = dataset.asfreq(sampling_frequency)

    print(dataset.tail(5))
    # datasetsize = len(dataset)
    #temp
    # print(dataset)
    # print(len(dataset))
    nan_count = dataset['value'].isna().sum()
    # print(f"The number of NaN values in the 'value' column is: {nan_count}")
    #end

    # data['users_imputed'] = data['users'].interpolate(method='linear')
    # data_train = data.loc[: end_train, :]
    # data_test  = data.loc[end_train:, :]

    # Split data into subsets
    data_train = dataset.loc[:end_train, :]
    data_val = dataset.loc[end_train:end_validation, :].iloc[1:]
    data_trainandval = dataset.loc[:end_validation, :]
    data_test = dataset.loc[end_validation:end_test, :].iloc[1:]

    # Prepare return values
    end_times = {
        'train': end_train,
        'validation': end_validation,
        'test': end_test
    }

    # print(end_train)
    # print(end_validation)
    # print(end_test)

    data_subsets = {
        'train': data_train,
        'validation': data_val,
        'train_and_val': data_trainandval,
        'test': data_test
    }


    dataset_end_time = dataset.index[-1]

    # print(sampling_frequency)
    # print(dataset)
    # print(len(dataset))
    include_fractional_hour = True
    _ , _ , df_exogenous_features, _, _, _ = create_exogenous_features(original_dataset=dataset,
                                                              optimally_differenced_dataset=dataset,
                                                              dataset_start_time=dataset_start_time,
                                                              dataset_end_time=dataset_end_time,
                                                              include_fractional_hour=include_fractional_hour,
                                                              exogenous_feature_type='AdditiveandMultiplicativeExogenousFeatures',
                                                              sampling_frequency=sampling_frequency)


    dataset_withfeatures = dataset[['value']].merge(
        df_exogenous_features,
        left_index=True,
        right_index=True,
        how='left'
    )

    dataset_withfeatures = dataset_withfeatures.astype({col: np.float32 for col in dataset_withfeatures.select_dtypes("number").columns})
    data_train_withfeatures = dataset_withfeatures.loc[: end_train, :].copy()
    data_val_withfeatures   = dataset_withfeatures.loc[end_train:end_validation, :].copy()
    data_test_withfeatures  = dataset_withfeatures.loc[end_validation:, :].copy()

    dataset_with_features_subsets = {
        'train': data_train_withfeatures,
        'validation': data_val_withfeatures,
        'test': data_test_withfeatures
    }

    # random_start
    # make a random model selector excluding random_forest
    # system-clock used to generate the random seed
    # things to be randomized :
        # model
        # feature_set_reduction
        # FeatureReductionType : RFECV or RFE ( default setting for no.of parameters )
        # Exogtype : any one of the 4 ( if dataset size does not allow it be anything else then do accordingly and not randomize them )
        # random_state for bayesian search inside the generalized_hyperparameter search

    all_models = ['baseline', 'direct_linearregression', 'direct_ridge', 'direct_lasso', 'direct_linearboost',
                      'direct_lightgbm', 'autoreg_linearregression', 'autoreg_ridge', 'autoreg_lasso', 'autoreg_linearboost',
                      'autoreg_lightgbm', 'autoreg_histgradient', 'autoreg_xgb', 'autoreg_catboost', 'arima',
                      'skt_prophet_additive', 'skt_prophet_hyper', 'skt_ets', 'skt_tbats_damped', 'skt_tbats_standard', 'skt_tbats_quick',
                      'skt_lstm_deeplearning', 'autoreg_randomforest', 'direct_xgb', 'direct_catboost', 'direct_histgradient']

    allowed_models = ['baseline', 'direct_linearregression', 'direct_ridge', 'direct_lasso', 'direct_linearboost',
                      'direct_lightgbm', 'autoreg_linearregression', 'autoreg_ridge', 'autoreg_lasso', 'autoreg_linearboost',
                      'autoreg_lightgbm', 'autoreg_histgradient', 'autoreg_xgb', 'autoreg_catboost', 'arima',
                      'skt_ets', 'skt_tbats_damped','skt_prophet_additive', 'skt_prophet_hyper', 'skt_tbats_standard', 'skt_tbats_quick',
                      'skt_lstm_deeplearning']

    dataset_duration = dataset_end_time - dataset_start_time
    sampling_timedelta = pd.Timedelta(sampling_frequency)
    week_timedelta = pd.Timedelta(days=7)
    steps_in_week = int(week_timedelta / sampling_timedelta)

    if sampling_timedelta > pd.Timedelta(hours=1):
        lags = round(min(len(data_subsets['test']), steps_in_week))
    else:
        lags = round(min(0.3 * len(dataset), steps_in_week))

    if_small_dataset = False
    time_metric_baseline = 'hours'
    forecasterequivalentdate = 1
    forecasterequivalentdate_n_offsets = 1

    # Remove Prophet models if dataset duration is less than 2 years
    if dataset_duration < pd.Timedelta(days=365 * 2):
        allowed_models = [model for model in allowed_models if model not in ['skt_prophet_additive', 'skt_prophet_hyper']]

    forecasting_steps = None
    sampling_timedelta = pd.Timedelta(sampling_frequency)
    week_timedelta = pd.Timedelta(days=7)
    day_timedelta = pd.Timedelta(days=1)

    if sampling_timedelta < day_timedelta:
        # Calculate steps relative to a day
        forecasting_steps = int(day_timedelta / sampling_timedelta)
    elif sampling_timedelta >= day_timedelta:
        # Calculate steps relative to a week
        forecasting_steps = min(int(week_timedelta / sampling_timedelta), len(data_subsets['test'])) # can override the setting but have to not do backtesting

    if dataset_duration >= pd.Timedelta(days=19) and len(dataset) >= 25:
        print("Hits the >= 25 length dataset case and >= 19 days")
        # quick_start : linear_regression with no_exog, feature_set_reduction = False
        if quick_start:
                allowed_models = ['direct_linearregression']

        # if sampling_timedelta < day_timedelta:
        #     # Calculate steps relative to a day
        #     forecasting_steps = int(day_timedelta / sampling_timedelta)
        # elif sampling_timedelta >= day_timedelta:
        #     # Calculate steps relative to a week
        #     forecasting_steps = min(int(week_timedelta / sampling_timedelta), len(data_subsets['test'])) # can override the setting but have to not do backtesting

        use_weight = True
        time_metric_baseline = "days"
        forecasterequivalentdate = 1
        forecasterequivalentdate_n_offsets = 7
    else:
        # quick_start : linear_regression with no_exog, feature_set_reduction = False
        if dataset_duration >= pd.Timedelta(days=3) and len(dataset) >= 72:
            print("Hits the >= 72 length dataset case and >= 3 days")
            allowed_models = ['baseline', 'direct_linearregression', 'direct_ridge', 'direct_lasso','direct_lightgbm', 'direct_xgb', 'direct_catboost',
                              'direct_histgradient', 'autoreg_linearregression', 'autoreg_ridge', 'autoreg_lasso', 'autoreg_lightgbm', 'autoreg_histgradient',
                              'autoreg_xgb', 'autoreg_catboost', 'arima', 'skt_ets'] # testing
            if quick_start:
                allowed_models = ['autoreg_lightgbm']
            time_metric_baseline = "days"
            forecasterequivalentdate = 1
            forecasterequivalentdate_n_offsets = min(dataset_duration.days - 1, 1)
            # if sampling_timedelta < day_timedelta:
            #     # Calculate steps relative to a day
            #     forecasting_steps = int(day_timedelta / sampling_timedelta)
            # elif sampling_timedelta >= day_timedelta:
            #     # Calculate steps relative to a week
            #     forecasting_steps = min(int(week_timedelta / sampling_timedelta), len(data_subsets['test'])) # can override the setting but have to not do backtesting
        elif (dataset_duration.total_seconds() / 3600) >= 12 and len(dataset) >= 12:
            # quick_start : baseline with no_exog, feature_set_reduction = False
            allowed_models = ['baseline', 'autoreg_lightgbm', 'autoreg_linearregression' ] # testing
            if quick_start:
                allowed_models = ['baseline']
            print("Hits the >= 12 length dataset case and >= 12 hours")

            if sampling_timedelta > pd.Timedelta(hours=1):
                time_metric_baseline = "days"
                forecasterequivalentdate_n_offsets = min(dataset_duration.days - 1, 1)
            else:
                time_metric_baseline = "hours"
                forecasterequivalentdate_n_offsets = int(dataset_duration.total_seconds() / 7200)

            forecasterequivalentdate = 1
            # forecasting_steps = lags
        elif len(dataset) >= 6 :
            print("Hits the >= 6 length dataset case")
            # quick_start : Baseline with no_exog, feature_set_reduction = False
            # print("inside smaller dataset size < 12 hours")
            allowed_models = ['baseline']
            if quick_start:
                allowed_models = ['baseline']
            lags = 1
            forecasting_steps = 1
            if sampling_timedelta > pd.Timedelta(hours=1):
                time_metric_baseline = "days"
            else:
                time_metric_baseline = "hours"
            forecasterequivalentdate = 1
            forecasterequivalentdate_n_offsets = 1
        else:
            print("Hits the invalid dataset case")
            if_small_dataset = True


        use_weight = False


    backtest_steps = forecasting_steps

    nan_percentage = dataset.isna().mean()
    print(nan_percentage)
    if nan_percentage.value > 0.4 :
        use_weight = False



    print(allowed_models)
    return end_times, dataset, data_subsets, dataset_withfeatures, dataset_with_features_subsets, dataset_start_time, dataset_end_time, sampling_frequency, int(lags), backtest_steps, forecasting_steps, use_weight, time_metric_baseline, forecasterequivalentdate, forecasterequivalentdate_n_offsets, if_small_dataset, allowed_models

def impute_data(dataset, replace=False, imputed_value='value', method='polynomial', order=None):
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("Input dataset must be a pandas DataFrame")

    if imputed_value not in dataset.columns:
        raise ValueError(f"Column '{imputed_value}' not found in the dataset")

    # Create a copy of the dataset to avoid modifying the original
    data_copy = dataset.copy()

    # Perform the imputation
    if method == 'polynomial' and order is not None:
        temporary_series = data_copy[imputed_value].interpolate(method=method, order=order)
    else:
        temporary_series = data_copy[imputed_value].interpolate(method=method)

    if replace:
        # Replace the original column with the imputed values
        data_copy[imputed_value] = temporary_series
    else:
        # Create a new column with the imputed values
        data_copy[f'{imputed_value}_imputed'] = temporary_series

    return data_copy

def find_nan_gaps(df, column_name):
    df_copy = df.copy()
    df_copy.index = pd.to_datetime(df_copy.index)
    nan_mask = df_copy[column_name].isna()

    # Use astype(bool) instead of fillna(False)
    gap_starts = df_copy.index[nan_mask & ~nan_mask.shift(1).astype(bool)]
    gap_ends = df_copy.index[nan_mask & ~nan_mask.shift(-1).astype(bool)]

    gaps = [
        [start.strftime('%Y-%m-%d %H:%M:%S'), end.strftime('%Y-%m-%d %H:%M:%S')]
        for start, end in zip(gap_starts, gap_ends)
    ]
    return gaps

def create_weight_func(data, sf, adjustment):
    def custom_weights(index):
        gaps = find_nan_gaps(data, 'value')
        missing_dates = [pd.date_range(
                            start = pd.to_datetime(gap[0]) - adjustment,
                            end   = pd.to_datetime(gap[1]) + adjustment,
                            freq  = sf
                        ) for gap in gaps]
        missing_dates = pd.DatetimeIndex(np.concatenate(missing_dates))
        weights = np.where(index.isin(missing_dates), 0, 1)
        return weights
    return custom_weights

# Then use it like this:
# weight_func = create_weight_func(dataset, samp_freq, adjustment)

def model_create_train_test_and_predict(
    model_name,
    dataset,
    dataset_train,
    end_validation,
    end_test,
    sampling_freq,
    differentiation,
    selected_lags,
    selected_exog,
    dataset_selected_features,
    data_missing,
    weight,
    baseline_1,
    baseline_2,
    baseline_3,
    select_hyperparameters=True,
    default_hyperparameters=None,
    random_state_hyper=123,
    backtest_steps=24,
    interval=[10, 90],
    metric='mase',
    forecast_calendar_features=None,
    forecasting_steps=24,
    hour_seasonality=False,
    dayofweek_seasonality=False,
    week_seasonality=False
):
    print("Hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    print(model_name)
    if data_missing:
        value = 'value_imputed'
    else:
        value = 'value'


    y = dataset_selected_features[value].copy()
    y.index = pd.to_datetime(y.index)
    split_index = y.index.get_loc(end_validation)
    y_train = y.iloc[:split_index + 1]
    # print(y_train.tail())
    y_train.index = pd.to_datetime(y_train.index)
    # print(y_train.tail())
    y_test = y.iloc[split_index + 1:]

    if model_name.lower() == "baseline":

        # Run baseline forecasting

        baseline_forecaster = create_forecaster("baseline",
                                                time_metric_baseline = baseline_1,
                                                forecasterequivalentdate = baseline_2,
                                                forecasterequivalentdate_n_offsets = baseline_3
                                               )
        baseline_forecaster.fit(y=dataset_selected_features.loc[:end_validation, value])
        print(dataset_selected_features.loc[:end_validation, value].tail())
        # print(backtest_steps)
        backtest_predictions = baseline_forecaster.predict(steps=backtest_steps)
        backtest_predictions = backtest_predictions.to_frame(name='pred')
        print('here0')
        print(backtest_predictions.head())
        print(y_train.tail())
        print(y_test.head())
        # print(y_test.tail())
        backtest_error = calculate_error(y_test, backtest_predictions[:len(y_test)], y_train, error_type=metric)
        print(backtest_error)
        print('here1')
        # Calculate interval coverage
        coverage, _ = "NA"

        # Fit the forecaster on the entire dataset
        baseline_forecaster.fit(y=dataset_selected_features[value])
        # Make predictions
        print(dataset_selected_features[value].tail())
        forecast = baseline_forecaster.predict(steps=forecasting_steps)

        return {
            'model_name': model_name,
            'sampling_freq': sampling_freq,
            'differentiation': differentiation,
            'selected_lags': None,
            'selected_exog': None,
            'dataset_selected_features': dataset_selected_features,
            'selected_hyperparameters': None,
            'backtest_steps': backtest_steps,
            'backtest_prediction_interval': None,
            'backtest_error': backtest_error,
            'backtest_interval_coverage': coverage,
            'backtest_predictions':backtest_predictions,
            'model_trained_on_all_data': baseline_forecaster,
            'forecasting_steps': forecasting_steps,
            'forecast': forecast
        }
    #add the linear regressors here
    elif model_name.lower() == "arima":
        sampling_timedelta = pd.Timedelta(sampling_freq)
        day_timedelta = pd.Timedelta(days=1)
        # Run baseline forecasting
        seasonal = False
        if hour_seasonality:
            m = forecasting_steps
            seasonal = True
        elif dayofweek_seasonality:
            if sampling_timedelta <= pd.Timedelta(hours=1):
                m = forecasting_steps * 7
            else:
                multiplier = max( 1, int(day_timedelta / sampling_timedelta))
                m = 7 * multiplier
            seasonal = True
        else:
            m = 1

        m = int(m)
        print(m)
        # seasonal = False

        arima_forecaster = create_forecaster(
            model_type='arima',
            y=dataset_selected_features.loc[:end_validation, value],
            start_p=forecasting_steps, # no.of lags at the beginning of the parameter search
            start_q=0, # no.of moving averages at the beginnning of the parameter search
            max_p=forecasting_steps, # max no.of lags
            max_q=1, # max no.of moving averages # can change to 1 if it takes longer to test
            seasonal=seasonal, # was True
            test='adf', # 'adf' test used to determine trend non-stationarity
            m=m, # was 24
            d=None,
            D=None
        )

        arima_forecaster.fit(y=dataset_selected_features.loc[:end_validation, value], suppress_warnings=True)

        backtest_predictions = arima_forecaster.predict_interval(steps=backtest_steps, interval=interval)
        backtest_error = calculate_error(y_test, backtest_predictions[:len(y_test)], y_train, error_type=metric)

        # Calculate interval coverage
        coverage, _ = calculate_interval_coverage(dataset_selected_features, backtest_predictions, end_validation, end_test, value)

        arima_forecaster.fit(y=dataset_selected_features[value], suppress_warnings=True)

        arima_predictions = predict_interval_custom(
            forecaster=arima_forecaster,
            steps=forecasting_steps,
            alpha=0.05
        )

        return {
            'model_name': model_name,
            'sampling_freq': sampling_freq,
            'differentiation': differentiation,
            'selected_lags': None,
            'selected_exog': None,
            'dataset_selected_features': None,
            'selected_hyperparameters': None,
            'backtest_steps': backtest_steps,
            'backtest_prediction_interval': None,
            'backtest_error': backtest_error,
            'backtest_interval_coverage': None,
            'backtest_predictions':backtest_predictions,
            'model_trained_on_all_data': arima_forecaster,
            'forecasting_steps': forecasting_steps,
            'forecast': arima_predictions
        }

    elif model_name[:3].lower() == 'skt':

        if model_name.lower() == 'skt_lstm_deeplearning':
            forecaster = NeuralForecastLSTM(
                freq=sampling_freq, max_steps=10
            )

        elif model_name.lower() == 'skt_prophet_additive':
            forecaster = Prophet(
                seasonality_mode='additive',
                daily_seasonality='auto',
                weekly_seasonality='auto',
                yearly_seasonality='auto'
                )

        elif model_name.lower() == 'skt_prophet_hyper':
            param_grid = {
                'growth': optuna.distributions.CategoricalDistribution(['linear', 'logistic']),
                'n_changepoints': optuna.distributions.IntDistribution(5, 20),
                'changepoint_range': optuna.distributions.FloatDistribution(0.7, 0.9),
                'seasonality_mode': optuna.distributions.CategoricalDistribution(['additive', 'multiplicative']),
                'seasonality_prior_scale': optuna.distributions.LogUniformDistribution(0.01, 10.0),
                'changepoint_prior_scale': optuna.distributions.LogUniformDistribution(0.001, 0.5),
                'holidays_prior_scale': optuna.distributions.LogUniformDistribution(0.01, 10.0),
                'daily_seasonality': optuna.distributions.CategoricalDistribution(['auto']),
                'weekly_seasonality': optuna.distributions.CategoricalDistribution(['auto']),
                'yearly_seasonality': optuna.distributions.CategoricalDistribution(['auto'])
            }

            forecaster_initial = Prophet()


            # Set up a more efficient time series cross-validation
            cv = SlidingWindowSplitter(
                initial_window=int(len(y_train) * 0.6),  # Use 60% of data for initial training
                step_length=int(len(y_train) * 0.2),     # Move forward by 20% each time
                fh=np.arange(1, forecasting_steps + 1)                      # Forecast horizon of 12 steps
            )

            fos = ForecastingOptunaSearchCV(
                forecaster=forecaster_initial,
                param_grid=param_grid,
                cv=cv,
                n_evals=50,
                strategy="refit",
                scoring=MeanAbsoluteScaledError(sp=1),
                verbose=1
                )

            fos.fit(y_train)
            forecaster = fos.best_forecaster_

        elif model_name.lower() == 'skt_ets': # faster implementation available and should be implemented in the future
            print("entered")
            forecaster = AutoETS(error='add',
                     trend=None,
                     damped_trend=False,
                     seasonal=None,
                     sp=forecasting_steps,
                     initialization_method='estimated',
                     initial_level=None,
                     initial_trend=None,
                     initial_seasonal=None,
                     bounds=None,
                     dates=None,
                     freq=None,
                     missing='none',
                     start_params=None,
                     maxiter=1000,
                     full_output=True,
                     disp=False,
                     callback=None,
                     return_params=False,
                     auto=True,
                     information_criterion='aic',
                     allow_multiplicative_trend=True,
                     restrict=True,
                     additive_only=False,
                     ignore_inf_ic=True,
                     n_jobs=-1,
                     random_state=random_state_hyper)

        elif model_name[:9].lower() == 'skt_tbats':
            splist=[]
            sampling_timedelta = pd.Timedelta(sampling_freq)
            day_timedelta = pd.Timedelta(days=1)
            use_box_cox = None
            if  model_name.lower() == 'skt_tbats_quick':
                use_box_cox = False
            if differentiation == 0:
                use_trend = False
                use_damped_trend = False
            else:
                use_trend = True
                if model_name.lower() == 'skt_tbats_damped':
                    use_damped_trend = True
                else:
                    use_damped_trend = False

            print("testing")
            # print(use_trend)
            # print(use_damped_trend)
            # print(use_box_cox)
            if hour_seasonality == True:
                splist.append(forecasting_steps)

            if dayofweek_seasonality == True:
                print("inside")
                if sampling_timedelta <= pd.Timedelta(hours=1):
                    multiplier = 7
                    splist.append(forecasting_steps*multiplier)
                else:
                    multiplier = max( 1, int(day_timedelta / sampling_timedelta))
                    splist.append(7*multiplier)

            if week_seasonality == True: # calculation of week seasonality test to determine seasonal period of a year can be improved in the future by using day_of_year instead of week_of_year
                print("inside")
                if sampling_timedelta <= pd.Timedelta(hours=1):
                    multiplier = 365.25
                    splist.append(forecasting_steps*multiplier)
                else:
                    multiplier = max( 1, int(day_timedelta / sampling_timedelta))
                    splist.append(365.25*multiplier)

            print(splist)
            forecaster = TBATS(use_box_cox=use_box_cox,
                  box_cox_bounds=(0, 1),
                  use_trend=use_trend,
                  use_damped_trend=use_damped_trend,
                  sp=splist,
                  use_arma_errors=True,
                  show_warnings=True,
                  n_jobs=-1,
                  multiprocessing_start_method='spawn',
                  context=None
            )

        rounded_index = y_test.index.floor(sampling_freq)
        difference_minutes = (y_test.index - rounded_index).total_seconds() / 60
        y.index = y.index.floor(sampling_freq)
        y_test.index = y_test.index.floor(sampling_freq)
        y_train.index = y_train.index.floor(sampling_freq)
        time_delta = pd.Timedelta(minutes=float(difference_minutes[0]))

        if model_name.lower() == 'skt_lstm_deeplearning':
            forecaster.fit(y_train, fh=list(range(1, backtest_steps + 1)))
            y_pred_backtest = forecaster.predict(list(range(1, backtest_steps + 1)))
            backtest_prediction = y_pred_backtest.to_frame(name='pred')
            error = calculate_error(y_test, backtest_prediction[:len(y_test)], y_train, error_type=metric)
            backtest_prediction.index = backtest_prediction.index + time_delta
            coverage = 'NA'
            forecaster.fit(y, fh=list(range(1, forecasting_steps + 1)))
            y_pred_future = forecaster.predict(list(range(1, forecasting_steps + 1)))
            forecast = y_pred_future.to_frame(name='pred')
            forecast.index = forecast.index + time_delta

        else:
            forecaster.fit(y_train)
            y_pred_backtest = forecaster.predict(list(range(1, backtest_steps + 1)))
            lower, upper = interval
            y_pred_backtest_interval = forecaster.predict_interval(fh=list(range(1, backtest_steps + 1)), coverage=[(upper - lower) / 100])
            y_pred_backtest_df = y_pred_backtest.to_frame(name='pred')
            backtest_prediction = pd.concat([y_pred_backtest_df, y_pred_backtest_interval], axis=1)
            backtest_prediction.columns = ['pred', 'lower_bound', 'upper_bound']
            error = calculate_error(y_test, backtest_prediction[:len(y_test)], y_train, error_type=metric)
            # dataset_feature_selection.index = dataset_feature_selection.index - time_delta
            backtest_prediction.index = backtest_prediction.index + time_delta
            coverage, _ = calculate_interval_coverage(dataset_selected_features, backtest_prediction, end_validation, end_test, value)
            forecaster.fit(y)
            y_pred_future = forecaster.predict(list(range(1, forecasting_steps + 1)))
            y_pred_future_interval = forecaster.predict_interval(fh=list(range(1, forecasting_steps + 1)), coverage=[(upper - lower) / 100])
            # y_pred_future.index = y_pred_future.index + time_delta
            y_pred_df = y_pred_future.to_frame(name='pred')
            forecast = pd.concat([y_pred_df, y_pred_future_interval], axis=1)
            forecast.columns = ['pred', 'lower_bound', 'upper_bound']
            forecast.index = forecast.index + time_delta

        return {
            'model_name': model_name,
            'sampling_freq': sampling_freq,
            'differentiation': differentiation,
            'selected_lags': None,
            'selected_exog': None,
            'dataset_selected_features': None,
            'selected_hyperparameters': None,
            'backtest_steps': backtest_steps,
            'backtest_prediction_interval': interval,
            'backtest_error': error,
            'backtest_interval_coverage': coverage,
            'backtest_predictions':backtest_prediction,
            'model_trained_on_all_data': forecaster,
            'forecasting_steps': forecasting_steps,
            'forecast': forecast
        }

    else:

        if model_name[:6].lower() == 'direct':

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

        # Create forecaster
        forecaster = create_forecaster(
            model_name,
            random_state=random_state,
            verbose=verbose,
            lags=selected_lags,
            steps=steps,
            weight=weight,
            differentiation=differentiation,
            if_exog=if_exog
        )

        print(random_state_hyper)
        if model_name.lower() != 'direct_linearregression':
            # # Hyperparameter search if required
            if select_hyperparameters:
                forecaster_search = GeneralizedHyperparameterSearch(
                    forecaster=forecaster,
                    y=dataset_selected_features.loc[:end_validation, value],
                    lags=selected_lags,
                    exog=dataset_selected_features.loc[:end_validation, selected_exog],
                    steps=forecasting_steps,
                    initial_train_size=len(dataset_train), #dataset with features
                    # metric=metric,
                )
                results_search, _ = forecaster_search.bayesian_search(
                    param_ranges=default_hyperparameters,
                    n_trials=20,
                    random_state=random_state_hyper
                )
                selected_hyperparameters = results_search['params'].iat[0]
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
            if_exog=if_exog
        )

        if len(selected_exog) == 0:
            final_forecaster.fit(
                y=dataset_selected_features.loc[:end_validation, value]
            )
        else:
            final_forecaster.fit(
                y=dataset_selected_features.loc[:end_validation, value],
                exog=dataset_selected_features.loc[:end_validation, selected_exog]
            )

        if len(selected_exog) == 0:
            backtest_predictions = final_forecaster.predict_interval(steps=backtest_steps, interval=interval)
        else:
            backtest_predictions = final_forecaster.predict_interval(steps=backtest_steps, exog=dataset_selected_features.loc[dataset_selected_features.index > end_validation, selected_exog], interval=interval)



        backtest_error = calculate_error(y_test, backtest_predictions[:len(y_test)], y_train, error_type=metric)

        coverage, _ = calculate_interval_coverage(dataset_selected_features, backtest_predictions, end_validation, end_test, value)

        if len(selected_exog) == 0:
            final_forecaster.fit(
                y=dataset_selected_features[value]
            )
        else:
            final_forecaster.fit(
                y=dataset_selected_features[value],
                exog=dataset_selected_features.loc[:end_test, selected_exog]
            )

        if len(selected_exog) == 0:
            exog=None
        else:
            exog=forecast_calendar_features[selected_exog]

        # Make the forecast
        forecast = predict_interval_custom(
            forecaster=final_forecaster,
            exog=exog,
            steps=forecasting_steps,
            interval=interval,
            n_boot=20
        )

        return {
            'model_name': model_name,
            'sampling_freq': sampling_freq,
            'differentiation': differentiation,
            'selected_lags': selected_lags,
            'selected_exog': selected_exog,
            'dataset_selected_features': dataset_selected_features,
            'selected_hyperparameters': selected_hyperparameters,
            'backtest_steps': backtest_steps,
            'backtest_prediction_interval': interval,
            'backtest_predictions':backtest_predictions,
            'backtest_error': backtest_error,
            'backtest_interval_coverage': coverage,
            'model_trained_on_all_data': final_forecaster,
            'forecasting_steps': forecasting_steps,
            'forecast': forecast
        }

def demonstration(
    filename,
    col_names,
    list_of_models,
    interval=[10,90],
    feature_set_reduction=False,
    exogenous_feature_type='ExogenousFeaturesBasedonSeasonalityTestWithAdditivenMultiplicative',
    feature_set_reduction_method='RFECV',
    random_state_hyperr = 123,
    metric='mase'
):
    ''' Demonstration function for the Satori Engine '''

    list_of_models = [model.lower() for model in list_of_models]

    quick_start_present = "quick_start" in list_of_models
    random_model_present = "random_model" in list_of_models
    random_state_hyper = random_state_hyperr

    # Process data first to get allowed_models
    end_times, dataset, data_subsets, dataset_withfeatures, dataset_with_features_subsets, start_date_time, last_date_time, sf, lags, backtest_steps, forecasting_steps, if_use_weight, baseline_time, baseline_offsets, baseline_n_offsets, if_small_dataset, allowed_models = process_data(
        filename,
        col_names=col_names,
        training_percentage=80,
        validation_percentage=10,
        date_time_format='%Y-%m-%d %H:%M:%S',
        quick_start=quick_start_present
    )

    if quick_start_present and random_model_present:
        warnings.warn("Both 'quick_start' and 'random_model' are present. 'quick_start' will take precedence.")

    if random_model_present and not quick_start_present:
        # Generate a random seed based on the current date and time
        # current_time = datetime.datetime.now()
        current_time = datetime.now()
        seed = int(current_time.strftime("%Y%m%d%H%M%S%f"))
        random.seed(seed)
        print(f"Using random seed: {seed}")

        # Randomly select options
        feature_set_reduction = random.choice([True, False])
        exogenous_feature_type = random.choice(["NoExogenousFeatures", "Additive", "AdditiveandMultiplicativeExogenousFeatures", "ExogenousFeaturesBasedonSeasonalityTestWithAdditivenMultiplicative"])
        feature_set_reduction_method = random.choice(["RFECV", "RFE"])
        random_state_hyper = random.randint(0, 2**32 - 1)

        # Replace 'random_model' with a randomly selected model from allowed_models
        list_of_models = [random.choice(allowed_models) if model == "random_model" else model for model in list_of_models]
        print(f"Randomly selected models: {list_of_models}")
        print(f"feature_set_reduction: {feature_set_reduction}")
        print(f"exogenous_feature_type: {exogenous_feature_type}")
        print(f"feature_set_reduction_method: {feature_set_reduction_method}")
        print(f"random_state_hyper: {random_state_hyper}")

    if quick_start_present:
        feature_set_reduction = False
        exogenous_feature_type = "NoExogenousFeatures"
        list_of_models = allowed_models

    if if_small_dataset:
        return 2, "Status = 2 (insufficient amount of data)"

    # Check if the requested models are suitable based on the allowed_models
    suitable_models, unsuitable_models = check_model_suitability(list_of_models, allowed_models, len(dataset))

    if unsuitable_models:
        print("The following models are not allowed due to insufficient data:")
        for model, reason in unsuitable_models:
            print(f"- {model}: {reason}")

    if not any(suitable_models):
        return 3, "Status = 3 (none of the requested models are suitable for the available data)"

    # Filter the list_of_models to include only suitable models
    list_of_models = [model for model, is_suitable in zip(list_of_models, suitable_models) if is_suitable]

    try:
        sel_lags, sel_exogs, differentiation, dataset_feature_selection, f_c_f, hr_s, dow_s, w_s,  missing, weight = determine_feature_set(
            dataset=dataset,
            data_train=data_subsets['train'],
            end_validation=end_times['validation'],
            end_train=end_times['train'],
            dataset_with_features=dataset_withfeatures,
            dataset_start_time=start_date_time,
            dataset_end_time=last_date_time,
            initial_lags=lags,
            weight_para=if_use_weight,
            exogenous_feature_type=exogenous_feature_type,
            feature_set_reduction=feature_set_reduction,
            feature_set_reduction_method=feature_set_reduction_method,
            bayesian_trial=20,
            random_state_hyper=random_state_hyper,
            frequency=sf,
            backtest_steps=backtest_steps,
            prediction_steps=forecasting_steps
        )

        list_of_results = []
        for model_name in list_of_models:
            result = model_create_train_test_and_predict(
                model_name=model_name,
                dataset=dataset,
                dataset_train=data_subsets['train'],
                end_validation=end_times['validation'],
                end_test=end_times['test'],
                sampling_freq=sf,
                differentiation=differentiation,
                selected_lags=sel_lags,
                selected_exog=sel_exogs,
                dataset_selected_features=dataset_feature_selection,
                data_missing=missing,
                weight=weight,
                select_hyperparameters=True,
                default_hyperparameters=None,
                random_state_hyper=random_state_hyper,
                backtest_steps=backtest_steps,
                interval=interval,
                metric=metric,
                forecast_calendar_features=f_c_f,
                forecasting_steps=forecasting_steps,
                hour_seasonality=hr_s,
                dayofweek_seasonality=dow_s,
                week_seasonality=w_s,
                baseline_1=baseline_time,
                baseline_2=baseline_offsets,
                baseline_3=baseline_n_offsets
            )
            list_of_results.append(result)

        return 1, list_of_results  # Status = 1 (ran correctly)

    except Exception as e:
        return 4, f"An error occurred: {str(e)}"  # Additional status code for unexpected errors

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

def create_model_comparison_table(model_results, metric="mean_absolute_scaled_error"):
    if isinstance(model_results, dict):
        model_results = [model_results]

    data = []
    for result in model_results:
        if result['model_name']=='baseline' or result['model_name']=='arima' or result['model_name']=='skt_lstm_deeplearning':
            data.append({
                'Model': (result['model_name']).upper(),
                metric: result['backtest_error'],
                'Specified Interval Coverage': "NA",
                'Actual Interval Coverage ': "NA"
            })
        else:
            data.append({
                'Model': (result['model_name']).upper(),
                metric: result['backtest_error'],
                'Specified Interval Coverage': f"{calculate_theoretical_coverage(result['backtest_prediction_interval']):.2f}%",
                'Actual Interval Coverage ': f"{(result['backtest_interval_coverage'])*100:.2f}%"
            })

    df = pd.DataFrame(data)
    df_sorted = df.sort_values(metric).reset_index(drop=True)

    # Set display options for better formatting
    pd.set_option('display.float_format', '{:.6f}'.format)

    return df_sorted

def create_ranked_model_comparison(models):
    all_data = []

    for model_results in models:
        # Get the variable name as a string
        user = [name for name, value in globals().items() if value is model_results][0]

        if isinstance(model_results, dict):
            model_results = [model_results]

        for result in model_results:
            if result['model_name'] in ['baseline', 'arima']:
                all_data.append({
                    'User': user,
                    'Model': result['model_name'].upper(),
                    'MASE': result['backtest_error'],
                    'Specified Interval Coverage': "NA",
                    'Interval_Coverage': "NA"
                })
            else:
                all_data.append({
                    'User': user,
                    'Model': result['model_name'].upper(),
                    'MASE': result['backtest_error'],
                    'Specified Interval Coverage': f"{calculate_theoretical_coverage(result['backtest_prediction_interval']):.2f}%",
                    'Interval_Coverage': f"{(result['backtest_interval_coverage'])*100:.2f}%"
                })

    df = pd.DataFrame(all_data)
    df_sorted = df.sort_values('MASE').reset_index(drop=True)
    df_sorted.index = df_sorted.index + 1  # Start ranking from 1 instead of 0
    df_sorted.index.name = 'Rank'

    # Set display options for better formatting
    pd.set_option('display.float_format', '{:.2f}'.format)

    return df_sorted

# Example usage:
# models = [krishna1, krishna2, krishna3]  # List of model result objects
# ranked_df = create_ranked_model_comparison(models)
# print(ranked_df)

def run_multiple_demonstrations(filename, num_calls=5):
    results = []
    global_vars = globals()

    # Initial call with 'quick_start'
    status, user = demonstration(
        filename,
        ['date_time', 'value', 'id'],
        ['quick_start'],
        interval=[10, 90],
        metric='mean_absolute_scaled_error'
    )
    if status == 1:
        global_vars['user0'] = user  # Store quick_start result as user0
        results.append(user)
    else:
        print(f"Quick start demonstration failed with status: {status}")

    # Subsequent calls with 'random_model'
    for i in range(1, num_calls + 1):
        status, user = demonstration(
            filename,
            ['date_time', 'value', 'id'],
            ['random_model'],
            interval=[10, 90],
            metric='mean_absolute_scaled_error'
        )

        if status == 1:  # Assuming status 1 means success
            user_var_name = f"user{i}"
            global_vars[user_var_name] = user
            results.append(user)
        else:
            print(f"demonstration call {i} failed with status: {status}")

    return results

# Example usage:
# models = run_multiple_demonstrations('modifiedkaggletraffic2_small.csv', 5)
# print(models)
# print(user0)  # This will print the result of the quick_start call
# print(user1)  # This will work if the function was called with at least 1 successful random_model demonstration
# print(user2)  # This will work if the function was called with at least 2 successful random_model demonstrations

def calculate_error(y_true, backtest_prediction, y_train, error_type='mase'):
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
    y_pred = backtest_prediction['pred']

    if error_type.lower() == 'mase':
        return mean_absolute_scaled_error(aligned_true, y_pred, y_train=y_train)
    elif error_type.lower() == 'mse':
        return mean_squared_error(aligned_true, y_pred)
    elif error_type.lower() == 'mae':
        return mean_absolute_error(aligned_true, y_pred)
    else:
        raise ValueError("Invalid error_type. Choose 'mase', 'mse', or 'mae'.")
