import warnings
import pandas as pd
from sklearn.neighbors import KDTree
import tsfresh as ts
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import numpy as np
import pymc

from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from sklearn.ensemble import AdaBoostRegressor
from tsfresh.utilities.dataframe_functions import impute
import tqdm
from tsfresh.feature_extraction import extract_features, MinimalFCParameters, EfficientFCParameters
import logging
logging.getLogger('tsfresh').setLevel(logging.ERROR)


def main():
    files = pd.read_excel('/home/velaraptor/Downloads/Raw Data 10yrs (2018).xlsx', header=1)
    files = files.fillna(0)
    groups = files.groupby('Name')
    forecast_df = []
    for name, group in tqdm.tqdm(groups):
        if len(group) > 1:
            group.index = group.Year
            df_shift, y = make_forecasting_frame(group["FantPt"], kind=name, max_timeshift=10, rolling_direction=1)
            forecast_df.append(df_shift)

    features_df = []
    for sample in tqdm.tqdm(forecast_df):
        X = extract_features(sample, column_id="id", column_sort="time", column_value="value", impute_function=impute,
                             show_warnings=False, disable_progressbar=True, default_fc_parameters=EfficientFCParameters())
        X = X.reset_index()
        X.loc[:, 'Name'] = sample['kind']
        features_df.append(X)
    features_time_series = pd.concat(features_df)
    features_time_series.to_csv('features_time_series.csv', index=False)


if __name__ == '__main__':
    main()
