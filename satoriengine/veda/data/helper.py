# file for data related functions
import pandas as pd
from typing import Any

def validate_dataframe(df: pd.DataFrame) -> bool:
    try:
        if 'date_time' not in df.columns or 'value' not in df.columns:
            return False
        datetime_check = pd.to_datetime(df['date_time'], errors='coerce')
        if datetime_check.isna().any():
            return False
        value_check = pd.to_numeric(df['value'], errors='coerce')
        if value_check.isna().any():
            return False
        return True
        
    except Exception as e:
        return False
    

def validate_single_entry(date_time: Any, value: Any) -> bool:
    try:
        datetime_valid = not pd.isna(pd.to_datetime(date_time, errors='coerce'))
        value_valid = not pd.isna(pd.to_numeric(value, errors='coerce'))
        return datetime_valid and value_valid
    except Exception as e:
        return False