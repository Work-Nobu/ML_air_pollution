"""
Written by Nobuhiro Suzuki


This method 
- reads the original csv data and returns pandas.DataFrame
- converts the date to Timestamp
- sorts by 'Place_ID' and 'Date'
- adds columns for wind speed and wind angle
- removes significantly negative concentration values
- replaces outliers with the values from the closest day at the same location.
- gets rid of unwanted columns

------------
Assumptions:
------------
Test.csv and Train.csv must be the original files downloaded from:
https://zindi.africa/competitions/zindiweekendz-learning-urban-air-pollution-challenge/data

-------
INPUTS:
-------
path_train_data: the path to Train.csv
path_test_data: the path to Test.csv
feature_columns_in: the names of the features you want to keep
target_columns: the names of the targets you want to keep
method: the cleaning method. 
    method = 1: 
    For the train set, fills NaN's from the past and future for up to 4 days then drops the remaining NaN.
    For the test set, fills NaN's from the past and future for up to 6 days.

--------
OUTPUTS:
--------
X_train: Cleaned but unscaled train data. pandas.DataFrame.
y_train: Cleaned but unscaled train target data. pandas.DataFrame.
X_test: Cleaned but unscaled test data. pandas.DataFrame.

If it is included, the 'Date' column is converted to class 'pandas._libs.tslibs.timestamps.Timestamp'.

--------------------
How-to-call example:
--------------------
from Get_data import Get_data

path_train_data = '../data/Train.csv'
path_test_data  = '../data/Test.csv'
    
target_columns = ['target', 'target_min', 'target_max', 'target_variance', 'target_count']
    
feature_columns = [ \
    'Place_ID X Date', \
    'Date', \
    'Place_ID', \
    'precipitable_water_entire_atmosphere', \
    'relative_humidity_2m_above_ground', \
    'specific_humidity_2m_above_ground', \
    'temperature_2m_above_ground', \
    'u_component_of_wind_10m_above_ground', \
    'v_component_of_wind_10m_above_ground', \
    \
    'wind_speed', 'wind_angle', \
    \
    'L3_NO2_NO2_column_number_density', \
    'L3_NO2_NO2_slant_column_number_density', \
    'L3_NO2_absorbing_aerosol_index', \
    'L3_NO2_cloud_fraction', \
    'L3_NO2_sensor_altitude', \
    'L3_NO2_sensor_azimuth_angle', \
    'L3_NO2_sensor_zenith_angle', \
    'L3_NO2_solar_azimuth_angle', \
    'L3_NO2_solar_zenith_angle', \
    'L3_NO2_stratospheric_NO2_column_number_density', \
    'L3_NO2_tropopause_pressure', \
    'L3_NO2_tropospheric_NO2_column_number_density', \
    'L3_O3_O3_column_number_density', \
    'L3_O3_O3_effective_temperature', \
    'L3_O3_cloud_fraction', \
    'L3_O3_sensor_azimuth_angle', \
    'L3_O3_sensor_zenith_angle', \
    'L3_O3_solar_azimuth_angle', \
    'L3_O3_solar_zenith_angle', \
    'L3_CO_CO_column_number_density', \
    'L3_CO_H2O_column_number_density', \
    'L3_CO_cloud_height', \
    'L3_CO_sensor_altitude', \
    'L3_CO_sensor_azimuth_angle', \
    'L3_CO_sensor_zenith_angle', \
    'L3_CO_solar_azimuth_angle', \
    'L3_CO_solar_zenith_angle', \
    'L3_HCHO_HCHO_slant_column_number_density', \
    'L3_HCHO_cloud_fraction', \
    'L3_HCHO_sensor_azimuth_angle', \
    'L3_HCHO_sensor_zenith_angle', \
    'L3_HCHO_solar_azimuth_angle', \
    'L3_HCHO_solar_zenith_angle', \
    'L3_HCHO_tropospheric_HCHO_column_number_density', \
    'L3_HCHO_tropospheric_HCHO_column_number_density_amf', \
    'L3_CLOUD_cloud_base_height', \
    'L3_CLOUD_cloud_base_pressure', \
    'L3_CLOUD_cloud_fraction', \
    'L3_CLOUD_cloud_optical_depth', \
    'L3_CLOUD_cloud_top_height', \
    'L3_CLOUD_cloud_top_pressure', \
    'L3_CLOUD_sensor_azimuth_angle', \
    'L3_CLOUD_sensor_zenith_angle', \
    'L3_CLOUD_solar_azimuth_angle', \
    'L3_CLOUD_solar_zenith_angle', \
    'L3_CLOUD_surface_albedo', \
    'L3_AER_AI_absorbing_aerosol_index', \
    'L3_AER_AI_sensor_altitude', \
    'L3_AER_AI_sensor_azimuth_angle', \
    'L3_AER_AI_sensor_zenith_angle', \
    'L3_AER_AI_solar_azimuth_angle', \
    'L3_AER_AI_solar_zenith_angle', \
    'L3_SO2_SO2_column_number_density', \
    'L3_SO2_SO2_column_number_density_amf', \
    'L3_SO2_SO2_slant_column_number_density', \
    'L3_SO2_absorbing_aerosol_index', \
    'L3_SO2_cloud_fraction', \
    'L3_SO2_sensor_azimuth_angle', \
    'L3_SO2_sensor_zenith_angle', \
    'L3_SO2_solar_azimuth_angle', \
    'L3_SO2_solar_zenith_angle', \
    'L3_CH4_CH4_column_volume_mixing_ratio_dry_air', \
    'L3_CH4_aerosol_height', \
    'L3_CH4_aerosol_optical_depth', \
    'L3_CH4_sensor_azimuth_angle', \
    'L3_CH4_sensor_zenith_angle', \
    'L3_CH4_solar_azimuth_angle', \
    'L3_CH4_solar_zenith_angle']
    
X_train, y_train, X_test = Get_data(path_train_data, path_test_data, feature_columns, target_columns, method = 1)
"""

# %%
import numpy as np
import pandas as pd
import copy
from scipy import stats

# %%
def Get_data(path_train_data, path_test_data, feature_columns_in, target_columns, method = 1):
    feature_columns = copy.deepcopy(feature_columns_in)
    
    # Read the data
    df_train = pd.read_csv(path_train_data) # the train data
    df_test  = pd.read_csv(path_test_data)   # the test data

    # Convert the date str to Timestamp
    df_train.Date = pd.to_datetime(df_train.Date, format='%Y-%m-%d')
    df_test.Date  = pd.to_datetime(df_test.Date,  format='%Y-%m-%d')

    # Sort by 'Place_ID' and 'Date'
    df_train.sort_values(by=['Place_ID', 'Date'], inplace=True)
    df_test.sort_values(by=['Place_ID', 'Date'], inplace=True)
    
    # Add a column for the wind speed
    df_train['wind_speed'] = np.sqrt( df_train['u_component_of_wind_10m_above_ground']**2 + \
                                      df_train['v_component_of_wind_10m_above_ground']**2 )
    df_test['wind_speed']  = np.sqrt(  df_test['u_component_of_wind_10m_above_ground']**2 + \
                                       df_test['v_component_of_wind_10m_above_ground']**2 )
    
    # Add a column for the wind angle
    df_train['wind_angle'] = np.arctan2(df_train['v_component_of_wind_10m_above_ground'], df_train['u_component_of_wind_10m_above_ground'])
    df_test['wind_angle']  = np.arctan2( df_test['v_component_of_wind_10m_above_ground'],  df_test['u_component_of_wind_10m_above_ground'])    
    
    # Get rid of the unwanted columns
    features_and_targets = feature_columns + target_columns
    df_train = df_train[features_and_targets]
    df_test = df_test[feature_columns]
    
    # Take care of negative concentrations (-0.001 mol/m2, according to https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2#description)
    if True:
        # Take care only of these columns
        correct_these_columns = [ \
            'L3_NO2_NO2_column_number_density', \
            'L3_NO2_tropospheric_NO2_column_number_density', \
            'L3_NO2_stratospheric_NO2_column_number_density', \
            'L3_NO2_NO2_slant_column_number_density', \
            'L3_O3_O3_column_number_density', \
            'L3_CO_CO_column_number_density', \
            'L3_CO_H2O_column_number_density', \
            'L3_HCHO_HCHO_slant_column_number_density', \
            'L3_HCHO_tropospheric_HCHO_column_number_density', \
            'L3_HCHO_tropospheric_HCHO_column_number_density_amf', \
            'L3_SO2_SO2_column_number_density', \
            'L3_SO2_SO2_column_number_density_amf', \
            'L3_SO2_SO2_slant_column_number_density']
        correct_these_columns = list( set(correct_these_columns) & set(feature_columns) )
        
        if True:
            # Set the negative outlier value to NaN so that a new value will be assigned later by ffill or bfill
            df_train[correct_these_columns] = df_train[correct_these_columns].where(df_train[correct_these_columns] > -0.001, other=np.nan)
            df_test[correct_these_columns] = df_test[correct_these_columns].where(df_test[correct_these_columns] > -0.001, other=np.nan)
            # Set the remaining negative concentration values to 0
            df_train[correct_these_columns] = df_train[correct_these_columns].where(np.logical_or(df_train[correct_these_columns] >= 0.0, df_train[correct_these_columns].isna()), other=0.0)
            df_test[correct_these_columns] = df_test[correct_these_columns].where(np.logical_or(df_test[correct_these_columns] >= 0.0, df_test[correct_these_columns].isna()), other=0.0)
        else:
            # Drop the rows having an outlier
            df_train = df_train[ ~((df_train[correct_these_columns] < -0.001).any(axis=1)) ]
            
    # Take care of the outliers based on the z-score.
    if True:
        # Take care only of these columns
        correct_these_columns = [ \
            'L3_NO2_NO2_column_number_density', \
            'L3_NO2_tropospheric_NO2_column_number_density', \
            'L3_NO2_stratospheric_NO2_column_number_density', \
            'L3_NO2_NO2_slant_column_number_density', \
            'L3_O3_O3_column_number_density', \
            'L3_CO_CO_column_number_density', \
            'L3_CO_H2O_column_number_density', \
            'L3_HCHO_HCHO_slant_column_number_density', \
            'L3_HCHO_tropospheric_HCHO_column_number_density', \
            'L3_HCHO_tropospheric_HCHO_column_number_density_amf', \
            'L3_SO2_SO2_column_number_density', \
            'L3_SO2_SO2_column_number_density_amf', \
            'L3_SO2_SO2_slant_column_number_density']
        correct_these_columns = list( set(correct_these_columns) & set(feature_columns) )
        # Set the outlier [abs(z-score) > 3] value to NaN so that a new value will be assigned later by ffill or bfill.
        # the stats & z-score are computed at each location
        #-- Train set 
        place_IDs = df_train.Place_ID.unique()
        for place in place_IDs:
            tmp = df_train.loc[df_train.Place_ID == place, correct_these_columns]
            tmp[np.abs(stats.zscore(tmp, axis=0, nan_policy='omit')) > 3] = np.nan
            df_train.loc[df_train.Place_ID == place, correct_these_columns] = tmp
        #-- Test set
        place_IDs = df_test.Place_ID.unique()
        for place in place_IDs:
            tmp = df_test.loc[df_test.Place_ID == place, correct_these_columns]
            tmp[np.abs(stats.zscore(tmp, axis=0, nan_policy='omit')) > 3] = np.nan
            df_test.loc[df_test.Place_ID == place, correct_these_columns] = tmp
    

    # Fill or drop NaN's
    if method == 1:
        #----- Train set:
        # Fill NaN with the forward & backward-fill for up to 2 days from the past and future each
        for i in range(0,2):
            df_train.fillna(method='ffill', axis=0, inplace=True, limit=1)
            df_train.fillna(method='bfill', axis=0, inplace=True, limit=1)

        if True:
            # Drop the remaining NaN's
            df_train.dropna(axis=0, how='any', inplace=True)
        else:
            # Fill with the mean value at each location
            place_IDs = df_train.Place_ID.unique()
            for place in place_IDs:
                df_train[df_train.Place_ID == place].fillna(value=df_train[df_train.Place_ID == place].mean(numeric_only=True), axis=0, inplace=True)
        
        #----- Test set
        if True:
            # Fill NaN with the forward & backward-fill for up to 3 days from the past and future each
            for i in range(0,3):
                df_test.fillna(method='ffill', axis=0, inplace=True, limit=1)
                df_test.fillna(method='bfill', axis=0, inplace=True, limit=1)
        if False:
            # Drop the remaining NaN's
            df_test.dropna(axis=0, how='any', inplace=True)
        if False:
            # Fill with the mean value at each location
            place_IDs = df_test.Place_ID.unique()
            for place in place_IDs:
                df_test[df_test.Place_ID == place].fillna(value=df_test[df_test.Place_ID == place].mean(numeric_only=True), axis=0, inplace=True)
                        
    # Separate the train features and train targets
    y_train = df_train[target_columns]
    df_train.drop(columns=target_columns, inplace=True)
        
    return df_train, y_train, df_test
    

# %%
# The code below is just for debugging.
if __name__ == "__main__":
    # %%
    path_train_data = '../data/Train.csv'
    path_test_data  = '../data/Test.csv'
    # %%
    target_columns = ['target']
    
    feature_columns = [ \
    'Date', \
    'Place_ID',\
    'L3_NO2_NO2_column_number_density', \
    'L3_O3_O3_column_number_density', \
    'L3_CO_CO_column_number_density', \
    'L3_SO2_SO2_column_number_density_amf', \
    'L3_SO2_SO2_column_number_density', \
    'L3_CO_H2O_column_number_density', \
    \
    'L3_SO2_absorbing_aerosol_index', \
    \
    'wind_speed', 'wind_angle', \
    'L3_CLOUD_cloud_fraction', \
    'L3_CLOUD_surface_albedo', \
    \
    'L3_AER_AI_absorbing_aerosol_index' ]
        
    X_train, y_train, X_test = Get_data(path_train_data, path_test_data, feature_columns, target_columns, method = 1)
    
    # %%
    print( X_train.shape )
    print( y_train.shape )
    print( X_test.shape  )
