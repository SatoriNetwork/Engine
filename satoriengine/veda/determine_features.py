import warnings

warnings.filterwarnings('ignore')


from typing import Callable, Union

import numpy as np
import pandas as pd
import datetime
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from scipy.stats import kruskal
from statsmodels.tsa.stattools import adfuller, kpss
from skforecast.ForecasterBaseline import ForecasterEquivalentDate
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster, random_search_forecaster, bayesian_search_forecaster
from skforecast.model_selection import select_features
from skforecast.ForecasterSarimax import ForecasterSarimax
from pmdarima import auto_arima
from pandas.tseries.frequencies import to_offset
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from lineartree import LinearBoostRegressor
from sklearn.preprocessing import StandardScaler

from .process import fractional_hour_generator, test_seasonality, create_exogenous_features, generate_exog_data, filter_dataframe_col

from satorilib.logging import debug, info, error




class Features:
    def __init__(
        self,
        selected_lags: int, 
        selected_exog: list,
        differentiation: int, 
        dataset_selected_features: pd.DataFrame, 
        forecast_calendar_features: list, 
        hour_seasonality: Union[bool, None], 
        dow_seasonality: Union[bool, None], 
        week_seasonality: Union[bool, None], 
        missing_values: bool, 
        weight: Union[Callable[[pd.DatetimeIndex], np.ndarray], None]
    ):
        self.selected_lags = selected_lags
        self.selected_exog = selected_exog
        self.differentiation = differentiation
        self.dataset_selected_features = dataset_selected_features
        self.forecast_calendar_features = forecast_calendar_features
        self.hour_seasonality = hour_seasonality
        self.dow_seasonality = dow_seasonality
        self.week_seasonality = week_seasonality
        self.missing_values = missing_values
        self.weight = weight
        

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


class GeneralizedHyperparameterSearch:
    def __init__(self, forecaster, y, lags, exog=None, steps=12, metric='mean_absolute_scaled_error',
                 initial_train_size=None, fixed_train_size=False, refit=False,
                 return_best=True, n_jobs='auto', verbose=False, show_progress=False):
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
                debug(f"Warning: Parameter '{param}' not found in the current search space.")

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
                debug(f"Warning: Parameter '{param}' is already in the current search space.")
            else:
                debug(f"Warning: Parameter '{param}' not found in the default search space for {self.model_type}.")

    def update_parameter_range(self, param, new_range):
        """
        Update the range of a specific parameter.

        :param param: Name of the parameter to update
        :param new_range: New range for the parameter (tuple)
        """
        if param in self.current_param_ranges:
            self.current_param_ranges[param] = new_range
        else:
            debug(f"Warning: Parameter '{param}' not found in the current search space.")

    def display_available_parameters(self):
        """
        Display the available parameters and their current ranges for the selected model type.
        """
        debug(f"Available parameters for {self.model_type.upper()} model:")
        self._display_params(self.current_param_ranges)
        debug("\nYou can override these parameters by passing a dictionary to the bayesian_search method.")

    def _display_params(self, param_ranges):
        for param, config in param_ranges.items():
            param_type = config[0]
            if param_type in ['int', 'float']:
                step = config[3] if len(config) > 3 else 'N/A'
                debug(f"  {param}: {param_type}, range: {config[1]} to {config[2]}, step: {step}")
            elif param_type == 'categorical':
                debug(f"  {param}: {param_type}, choices: {config[1]}")


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
                debug(f"New parameter '{param}' added to the search space.")

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

    for i in range(max_diff):
        # print(i)
        # print(data_diff['value']) # Series([], Freq: 55min, Name: value, dtype: float64)
        # print(data_diff['value'].nunique())
        # Check if data is constant
        if data_diff['value'].nunique() <= 1:
            debug(f'The time series (diff order {i}) is constant or nearly constant.')
            break

        try:
            adfuller_result = adfuller(data_diff['value'])
            kpss_result = kpss(data_diff['value'])
        except ValueError as e:
            debug(f"Error during statistical test: {e}")
            break

        debug(f'adfuller stat and adfuller boolean is: {adfuller_result[1]}, {adfuller_result[1] < 0.05}')
        debug(f'kpss stat and kpss boolean is: {kpss_result[1]}, {kpss_result[1] > 0.05}')

        if adfuller_result[1] < 0.05 and kpss_result[1] > 0.05:
            debug(f'The time series (diff order {i}) is likely to be stationary.')
            break
        else:
            debug(f'The time series (diff order {i}) is likely to be non-stationary.')
            differentiation += 1
            if differentiation < max_diff:
                data_diff = data_diff.diff().dropna()
            else:
                break

    if differentiation == 0:
        data_diff = pd.DataFrame()

    debug(f'The differentiation is: {differentiation}')
    return differentiation, data_diff

# loop over 'df_exogenous_features' columns, and if column name in not in list lightgbm_selected_exog then delete that column
# def filter_dataframe_col(df, selected_col):
    # col_to_keep = [ col for col in df.columns if col in selected_col ]
    # return df[col_to_keep]

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

def determine_feature_set(
    dataset: pd.DataFrame,
    data_train: pd.DataFrame,
    end_validation: pd.Timestamp,
    end_train: pd.Timestamp,
    dataset_start_time: pd.Timestamp,
    dataset_end_time: pd.Timestamp,
    dataset_with_features: pd.DataFrame, 
    weight_para: bool = False,
    modeltype: str = "tree_or_forest",
    initial_lags : Union[int, None] = None, 
    exogenous_feature_type : Union[str, None] = None, 
    feature_set_reduction : bool = False, 
    feature_set_reduction_method : Union[str, None] = None, 
    FeatureSetReductionStep : int = 5, 
    FeatureSetReductionSubSample : float = 0.5, 
    RFECV_CV: int = 2,
    RFECV_min_features_to_select: int = 10,
    RFE_n_features_to_select: Union[int, None] = None,
    bayesian_trial: Union[int, None] = None, 
    random_state_hyper: int = 123, 
    frequency: str = '1h',
    backtest_steps: int = 24,
    prediction_steps: int = 24,
    hyper_flag: bool = True
) -> Features:

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
        exog_features, _, df_exogenous_features, hour_seasonality, dow_seasonality, week_seasonality = create_exogenous_features(original_dataset=dataset,
                                                                                             optimally_differenced_dataset=dataset,
                                                                                             dataset_start_time=dataset_start_time,
                                                                                             dataset_end_time=dataset_end_time,
                                                                                             include_fractional_hour=include_fractional_hour,
                                                                                             exogenous_feature_type=exogenous_feature_type,
                                                                                             sampling_frequency=frequency
                                                                                            )
    else:
        exog_features, _, df_exogenous_features, hour_seasonality, dow_seasonality, week_seasonality = create_exogenous_features(original_dataset=dataset,
                                                                                             optimally_differenced_dataset=data_diff,
                                                                                             dataset_start_time=dataset_start_time,
                                                                                             dataset_end_time=dataset_end_time,
                                                                                             include_fractional_hour = include_fractional_hour,
                                                                                             exogenous_feature_type=exogenous_feature_type,
                                                                                             sampling_frequency=frequency
                                                                                            )

    if not hyper_flag:
        feature_set_reduction = False
        
    if feature_set_reduction == True:

        if exog_features == []:
            if_exog = None
        else:
            if_exog = StandardScaler()

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

        # Hyper-Parameter Search
        model_search = GeneralizedHyperparameterSearch(forecaster=hyper_forecaster, y=dataset.loc[:end_validation, value], lags=initial_lags, steps=backtest_steps, initial_train_size=len(data_train), metric="mean_absolute_scaled_error")
        
        results_search, _ = model_search.bayesian_search(
            n_trials=bayesian_trial,
            random_state=random_state_hyper)
        best_params = results_search['params'].iat[0]

        if modeltype == "tree_or_forest":
            feature_forecaster = create_forecaster('autoreg_lightgbm', random_state=123, verbose=-1, lags=initial_lags, differentiation=differentiation, custom_params=best_params, weight=weight, if_exog=if_exog )

        if feature_set_reduction_method=='RFECV':
            selected_lags, selected_exog = perform_RFECV_feature_selection(
                forecaster=feature_forecaster,
                y=dataset.loc[:end_train, value],
                exog=dataset_with_features.loc[:end_train, exog_features], # replace exog_features if needed
                end_train=end_train,
                step=FeatureSetReductionStep, 
                cv=RFECV_CV,
                min_features_to_select=RFECV_min_features_to_select,
                subsample=FeatureSetReductionSubSample, 
                verbose=False
            )
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
        df_exogenous_features = filter_dataframe_col(df_exogenous_features, selected_exog)
    else:
        selected_lags = initial_lags
        selected_exog = exog_features


    if exogenous_feature_type != 'NoExogenousFeatures' and len(selected_exog) > 0:
        _ , exog_timewindow = generate_exog_data( dataset_end_time, frequency, prediction_steps, '%Y-%m-%d %H:%M:%S')
        _ , forecast_calendar_features, _, _, _, _  = create_exogenous_features(original_dataset=exog_timewindow,
                                                                      optimally_differenced_dataset=exog_timewindow,
                                                                      dataset_start_time=dataset_start_time,
                                                                      dataset_end_time=dataset_end_time,
                                                                      include_fractional_hour=True,
                                                                      exogenous_feature_type='AdditiveandMultiplicativeExogenousFeatures',
                                                                      sampling_frequency=frequency)
        forecast_calendar_features = filter_dataframe_col(forecast_calendar_features, selected_exog)
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

    return Features(
        selected_lags, selected_exog, differentiation, dataset_selected_features, forecast_calendar_features, hour_seasonality, dow_seasonality, week_seasonality, missing_values, weight)




