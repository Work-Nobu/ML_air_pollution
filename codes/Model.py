# %% Import
import numpy as np
import pandas as pd
from Get_data import Get_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import pickle

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import seaborn as sns

#plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 20})
plt.ion()

# %%
if __name__ == "__main__":

    # %% Set the data path
    path_train_data = '../data/Train.csv'
    path_test_data  = '../data/Test.csv'
    
    # %% Define targets and features
    target_columns = ['target'] #, 'target_min', 'target_max', 'target_variance', 'target_count']

    use_these_columns = 'medium' #'small', 'medium', 'large'
    
    # Define the list of features
    if use_these_columns == 'small':
        """
        Extra tree
                
        X_train size =  24103
        X_test size =  4879

        RF R2 for the train set =  1.0
        RF R2 for the test set =  0.39735559631795925

        RF MSE for the train set =  1.2592318614362272e-29
        RF MSE for the test set =  1292.698486780385

            index       val                                               name
        0       1  0.144020                     L3_CO_CO_column_number_density
        1      15  0.126811    L3_HCHO_tropospheric_HCHO_column_number_density
        2       8  0.123669                                         wind_speed
        3       0  0.074996                   L3_NO2_NO2_column_number_density
        4       2  0.065500                        temperature_2m_above_ground
        5      16  0.045878  L3_HCHO_tropospheric_HCHO_column_number_densit...
        6       4  0.045828                  specific_humidity_2m_above_ground
        7      14  0.045759                     L3_O3_O3_column_number_density
        8       3  0.041355                  relative_humidity_2m_above_ground
        9       6  0.041222                            L3_CLOUD_cloud_fraction
        10      5  0.039586               precipitable_water_entire_atmosphere
        11     11  0.036372                     L3_SO2_absorbing_aerosol_index
        12      7  0.035324                    L3_CO_H2O_column_number_density
        13     13  0.034807               L3_SO2_SO2_column_number_density_amf
        14     12  0.033713                   L3_SO2_SO2_column_number_density
        15      9  0.032915                  L3_AER_AI_absorbing_aerosol_index
        16     10  0.032244                     L3_NO2_absorbing_aerosol_index
        """
        feature_columns = [ \
            'Date', \
            'Place_ID', \
            'L3_NO2_NO2_column_number_density', \
            'L3_CO_CO_column_number_density', \
            'temperature_2m_above_ground', \
            \
            'relative_humidity_2m_above_ground', \
            'specific_humidity_2m_above_ground', \
            'precipitable_water_entire_atmosphere', \
            'L3_CLOUD_cloud_fraction', \
            'L3_CO_H2O_column_number_density', \
            \
            'wind_speed', \
            'L3_AER_AI_absorbing_aerosol_index', \
            'L3_NO2_absorbing_aerosol_index', \
            'L3_SO2_absorbing_aerosol_index', \
            \
            'L3_SO2_SO2_column_number_density', \
            'L3_SO2_SO2_column_number_density_amf', \
            \
            'L3_O3_O3_column_number_density', \
            'L3_HCHO_tropospheric_HCHO_column_number_density', \
            'L3_HCHO_tropospheric_HCHO_column_number_density_amf']
    elif use_these_columns == 'medium':
        """
        Extra tree
        
        X_train size =  24087
        X_test size =  4876

        RF R2 for the train set =  1.0
        RF R2 for the test set =  0.4355897294325175

        RF MSE for the train set =  1.2600683171917377e-29
        RF MSE for the test set =  1211.2274455556808

            index       val                                               name
        0       1  0.102634                     L3_CO_CO_column_number_density
        1       8  0.100099                                         wind_speed
        2      19  0.090403                          L3_AER_AI_sensor_altitude
        3      15  0.088421    L3_HCHO_tropospheric_HCHO_column_number_density
        4      18  0.075048             L3_NO2_NO2_slant_column_number_density
        5      17  0.057928           L3_HCHO_HCHO_slant_column_number_density
        6       2  0.049370                        temperature_2m_above_ground
        7       0  0.048973                   L3_NO2_NO2_column_number_density
        8      20  0.039185                       L3_AER_AI_solar_zenith_angle
        9      16  0.038419  L3_HCHO_tropospheric_HCHO_column_number_densit...
        10      4  0.036973                  specific_humidity_2m_above_ground
        11      3  0.031908                  relative_humidity_2m_above_ground
        12      5  0.030962               precipitable_water_entire_atmosphere
        13     14  0.029925                     L3_O3_O3_column_number_density
        14      6  0.028432                            L3_CLOUD_cloud_fraction
        15     11  0.027992                     L3_SO2_absorbing_aerosol_index
        16      7  0.026117                    L3_CO_H2O_column_number_density
        17     12  0.024649                   L3_SO2_SO2_column_number_density
        18     13  0.024604               L3_SO2_SO2_column_number_density_amf
        19      9  0.024232                  L3_AER_AI_absorbing_aerosol_index
        20     10  0.023726                     L3_NO2_absorbing_aerosol_index
        """
        feature_columns = [ \
            'Date', \
            'Place_ID', \
            'L3_NO2_NO2_column_number_density', \
            'L3_CO_CO_column_number_density', \
            'temperature_2m_above_ground', \
            \
            'relative_humidity_2m_above_ground', \
            'specific_humidity_2m_above_ground', \
            'precipitable_water_entire_atmosphere', \
            'L3_CLOUD_cloud_fraction', \
            'L3_CO_H2O_column_number_density', \
            \
            'wind_speed', \
            'L3_AER_AI_absorbing_aerosol_index', \
            'L3_NO2_absorbing_aerosol_index', \
            'L3_SO2_absorbing_aerosol_index', \
            \
            'L3_SO2_SO2_column_number_density', \
            'L3_SO2_SO2_column_number_density_amf', \
            \
            'L3_O3_O3_column_number_density', \
            'L3_HCHO_tropospheric_HCHO_column_number_density', \
            'L3_HCHO_tropospheric_HCHO_column_number_density_amf', \
            \
            'L3_HCHO_HCHO_slant_column_number_density', \
            'L3_NO2_NO2_slant_column_number_density', \
            'L3_AER_AI_sensor_altitude', \
            'L3_AER_AI_solar_zenith_angle']
    elif use_these_columns == 'large':
        """
        Extra tree
        
        X_train size =  23372
        X_test size =  4754
        
        RF R2 for the train set =  1.0
        RF R2 for the test set =  0.44142685340816745
        
        RF MSE for the train set =  1.2697514851126295e-29
        RF MSE for the test set =  1180.3848615498528

        Extra tree's feature importance
        0      19  0.077472                     L3_CO_CO_column_number_density
        1       6  0.070794                                         wind_speed
        2      29  0.069456    L3_HCHO_tropospheric_HCHO_column_number_density
        3       9  0.053628             L3_NO2_NO2_slant_column_number_density
        4      39  0.051749                          L3_AER_AI_sensor_altitude
        5      27  0.045684           L3_HCHO_HCHO_slant_column_number_density
        6      15  0.033738      L3_NO2_tropospheric_NO2_column_number_density
        7       3  0.033401                        temperature_2m_above_ground
        8      22  0.024309                              L3_CO_sensor_altitude
        9       8  0.024290                   L3_NO2_NO2_column_number_density
        10     30  0.023740  L3_HCHO_tropospheric_HCHO_column_number_densit...
        11      2  0.021655                  specific_humidity_2m_above_ground
        12      4  0.018299               u_component_of_wind_10m_above_ground
        13     37  0.018208                            L3_CLOUD_surface_albedo
        14     43  0.017705                       L3_AER_AI_solar_zenith_angle
        15     26  0.017510                           L3_CO_solar_zenith_angle
        16      0  0.016906               precipitable_water_entire_atmosphere
        17      1  0.016525                  relative_humidity_2m_above_ground
        18      5  0.016379               v_component_of_wind_10m_above_ground
        19     12  0.016294                             L3_NO2_sensor_altitude
        20     42  0.015435                      L3_AER_AI_solar_azimuth_angle
        21     14  0.014737                         L3_NO2_tropopause_pressure
        22     47  0.014072                     L3_SO2_absorbing_aerosol_index
        23     16  0.013703                     L3_O3_O3_column_number_density
        24     38  0.013085                  L3_AER_AI_absorbing_aerosol_index
        25     10  0.012689                     L3_NO2_absorbing_aerosol_index
        26     28  0.012684                             L3_HCHO_cloud_fraction
        27     17  0.012508                     L3_O3_O3_effective_temperature
        28     20  0.012397                    L3_CO_H2O_column_number_density
        29      7  0.012314                                         wind_angle
        30     25  0.012225                          L3_CO_solar_azimuth_angle
        31     48  0.012182                              L3_SO2_cloud_fraction
        32     13  0.011936     L3_NO2_stratospheric_NO2_column_number_density
        33     45  0.011712               L3_SO2_SO2_column_number_density_amf
        34     11  0.011686                              L3_NO2_cloud_fraction
        35     36  0.011474                        L3_CLOUD_cloud_top_pressure
        36     44  0.011271                   L3_SO2_SO2_column_number_density
        37     34  0.011040                       L3_CLOUD_cloud_optical_depth
        38     33  0.010670                            L3_CLOUD_cloud_fraction
        39     21  0.010605                                 L3_CO_cloud_height
        40     32  0.010471                       L3_CLOUD_cloud_base_pressure
        41     35  0.010433                          L3_CLOUD_cloud_top_height
        42     18  0.009808                               L3_O3_cloud_fraction
        43     46  0.009806             L3_SO2_SO2_slant_column_number_density
        44     31  0.009790                         L3_CLOUD_cloud_base_height
        45     41  0.008934                      L3_AER_AI_sensor_zenith_angle
        46     24  0.008672                          L3_CO_sensor_zenith_angle
        47     40  0.008005                     L3_AER_AI_sensor_azimuth_angle
        48     23  0.007914                         L3_CO_sensor_azimuth_angle
        """
        feature_columns = [ \
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
            'L3_NO2_stratospheric_NO2_column_number_density', \
            'L3_NO2_tropopause_pressure', \
            'L3_NO2_tropospheric_NO2_column_number_density', \
            'L3_O3_O3_column_number_density', \
            'L3_O3_O3_effective_temperature', \
            'L3_O3_cloud_fraction', \
            'L3_CO_CO_column_number_density', \
            'L3_CO_H2O_column_number_density', \
            'L3_CO_cloud_height', \
            'L3_HCHO_HCHO_slant_column_number_density', \
            'L3_HCHO_cloud_fraction', \
            'L3_HCHO_tropospheric_HCHO_column_number_density', \
            'L3_HCHO_tropospheric_HCHO_column_number_density_amf', \
            'L3_CLOUD_cloud_base_height', \
            'L3_CLOUD_cloud_base_pressure', \
            'L3_CLOUD_cloud_fraction', \
            'L3_CLOUD_cloud_optical_depth', \
            'L3_CLOUD_cloud_top_height', \
            'L3_CLOUD_cloud_top_pressure', \
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
            'L3_SO2_cloud_fraction']
   
    # %% Read and clean the data
    X_train, y_train, _ = Get_data(path_train_data, path_test_data, feature_columns, target_columns, method = 1)
    del _
    target_col = y_train.columns
    
    # %% Train-Test split
    
    # Concat the feature columns and the target columns because y_train does not have the 'Place_ID' column.
    X_train = pd.concat([X_train, y_train], axis=1)
    
    # Separate the data (rows in X_train) belonging to a "Place_Id" and 
    # put it in X_test until the test set is about 20% of the train set.    
    place_IDs = X_train.Place_ID.unique()
    flag = 0
    i = 0
    while flag == 0:
        # Copy the first location to X_test
        if i == 0:
            X_test = X_train[X_train.Place_ID == place_IDs[0]]
        else:
            X_test = pd.concat([X_test, X_train[X_train.Place_ID == place_IDs[0]]], axis=0)
            
        X_train = X_train[X_train.Place_ID != place_IDs[0]] # Drop the rows copied to X_test
        
        i = i + 1
        place_IDs = X_train.Place_ID.unique()
        if len(X_test) / len(X_train) >= 0.2: flag = 1

    # Drop unnecessary columns and separate the target and features for the train set
    X_train.drop(columns=['Date'], inplace=True)
    X_train.drop(columns=['Place_ID'], inplace=True)
    y_train = X_train['target']
    X_train.drop(columns=target_col, inplace=True)
    
    # Drop unnecessary columns and separate the target and features for the test set
    date_and_place_test = X_test[['Date', 'Place_ID' ]].copy() # Save the Date and Place_ID of the test set 
    X_test.drop(columns=['Date'], inplace=True)
    X_test.drop(columns=['Place_ID'], inplace=True)
    y_test = X_test['target']
    X_test.drop(columns=target_col, inplace=True)
    
    # Convert y_test from pandas.core.series.Series to numpy.ndarray
    y_test = y_test.values
    
    # Print the length
    print( "X_train size = ", len(X_train))
    print( "X_test size = ",  len(X_test))

    # %% Lightgbm
    if False:
        import lightgbm as lgb
        
        if False:
            # Gridsearch & CV
            regressor=lgb.LGBMRegressor()
            #(boosting_type='gbdt', num_leaves=31, max_depth=-1, 
            # learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, 
            # objective=None, class_weight=None, min_split_gain=0.0, 
            # min_child_weight=0.001, min_child_samples=20, subsample=1.0, 
            # subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, 
            # reg_lambda=0.0, random_state=None, n_jobs=None, 
            # importance_type='split', **kwargs)

            param_grid = {"boosting_type": ["gbdt", "dart"],
                          "n_estimators": [100, 500],
                          "max_depth": [-1, 30],
                          "num_leaves": [31, 201],
                          "learning_rate": [0.1, 0.5],
                          "reg_alpha": [0.0, 0.5, 1.0],
                          "reg_lambda": [0.0, 0.5, 1.0]} 
                          
            search = GridSearchCV(regressor, param_grid, scoring='neg_mean_squared_error')
            #(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, 
            # cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, 
            # return_train_score=False)

            if False:
                # No scaling of the features
                pipe = Pipeline([('search', search)])
            else:
                # With scaled features
                scl = StandardScaler()
                pipe = Pipeline([('scaler', scl), ('search', search)])
            
            pipe.fit(X_train, y_train)
            y_train_pred = pipe.predict(X_train)
            y_test_pred  = pipe.predict(X_test)
            
            print("The best hyperparameters are ", search.best_params_)
            print(search.cv_results_)
        else:
            regressor=lgb.LGBMRegressor(boosting_type="dart", n_estimators=500, num_leaves=201, learning_rate=0.1, reg_alpha=1.0, reg_lambda=0.5)
            regressor.fit(X_train, y_train)
            y_train_pred = regressor.predict(X_train)
            y_test_pred  = regressor.predict(X_test)
        
        print( 'RF R2 for the train set = ', r2_score(y_train, y_train_pred))
        print( 'RF R2 for the test set = ',  r2_score(y_test,  y_test_pred))
        
        print( 'RF MSE for the train set = ', mean_squared_error(y_train, y_train_pred))
        print( 'RF MSE for the test set = ',  mean_squared_error(y_test,  y_test_pred))
        
        #tmp = pd.DataFrame({'val':regressor.feature_importances_, 'name':regressor.feature_names_in_})
        #print( tmp.sort_values(by='val', ascending=False).reset_index() )
        
        """        
        The best hyperparameters are  {'boosting_type': 'dart', 'learning_rate': 0.1, 'max_depth': -1, 'n_estimators': 500, 'num_leaves': 201, 'reg_alpha': 1.0, 'reg_lambda': 0.5}

        RF R2 for the train set =  0.9601198538934005
        RF R2 for the test set =  0.4353689758353302

        RF MSE for the train set =  89.65598276233096
        RF MSE for the test set =  1211.7011839505362        
        """

    # %% KNeighbors
    if True:
        from sklearn.neighbors import KNeighborsRegressor
        
        regressor=KNeighborsRegressor()
        #(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, 
        # p=2, metric='minkowski', metric_params=None, n_jobs=None)

        # Define which parameters to try
        if False:
            param_grid = {"n_neighbors": [5, 10, 15, 20, 25],
                          "p": [1,2,3],
                          "weights": ['uniform', 'distance']}
        else:
            param_grid = {"n_neighbors": [15],
                          "p": [1],
                          "weights": ['distance']}
        
        search = GridSearchCV(regressor, param_grid, scoring='neg_mean_squared_error')
        #(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, 
        # verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)
        
        scl = StandardScaler()
        pipe = Pipeline([('scaler', scl), ('search', search)])
        pipe.fit(X_train, y_train)
        y_train_pred = pipe.predict(X_train)
        y_test_pred  = pipe.predict(X_test)
        
        print("The best hyperparameters are ", search.best_params_)
        print( 'RF R2 for the train set = ', r2_score(y_train, y_train_pred))
        print( 'RF R2 for the test set = ', r2_score(y_test, y_test_pred))
        print( 'RF MSE for the train set = ', mean_squared_error(y_train, y_train_pred))
        print( 'RF MSE for the test set = ', mean_squared_error(y_test, y_test_pred))
        
        """
        X_train size =  24087
        X_test size =  4876
        The best hyperparameters are  {'n_neighbors': 15, 'p': 1, 'weights': 'distance'}
        RF R2 for the train set =  1.0
        RF R2 for the test set =  0.4200393618161633
        RF MSE for the train set =  0.0
        RF MSE for the test set =  1244.5986172504677
        """
    
    # %% XGboost
    if False:
        import xgboost as xgb
        #=========================================================================
        # XGBoost regression: 
        # Parameters: 
        # n_estimators  "Number of gradient boosted trees. Equivalent to number 
        #                of boosting rounds."
        # learning_rate "Boosting learning rate (also known as “eta”)". 0-1.
        # max_depth     "Maximum depth of a tree. Increasing this value will make 
        #                the model more complex and more likely to overfit." 
        #=========================================================================
        regressor=xgb.XGBRegressor() #eval_metric='rmsle'

        param_grid = {"max_depth":    [7, 8],
                    "n_estimators": [700],
                    "learning_rate": [0.025, 0.03]}
        search = GridSearchCV(regressor, param_grid, cv=5).fit(X_train, y_train)
        print("The best hyperparameters are ",search.best_params_)
        regressor=xgb.XGBRegressor(learning_rate = search.best_params_["learning_rate"], \
                                n_estimators  = search.best_params_["n_estimators"], \
                                max_depth     = search.best_params_["max_depth"])
        regressor.fit(X_train, y_train)
        predictions = regressor.predict(X_test)

        y_train_pred = regressor.predict(X_train)
        y_test_pred = regressor.predict(X_test)
        print( 'RF R2 for the train set = ', r2_score(y_train, y_train_pred))
        print( 'RF R2 for the test set = ', r2_score(y_test, y_test_pred))
        print( 'RF MSE for the train set = ', mean_squared_error(y_train, y_train_pred))
        print( 'RF MSE for the test set = ', mean_squared_error(y_test, y_test_pred))
        
        tmp = pd.DataFrame({'val':regressor.feature_importances_, 'name':regressor.feature_names_in_})
        print( tmp.sort_values(by='val', ascending=False).reset_index() )


        """
        X_train size =  24087
        X_test size =  4876
        The best hyperparameters are  {'learning_rate': 0.03, 'max_depth': 8, 'n_estimators': 700}
        RF R2 for the train set =  0.9255452400631007
        RF R2 for the test set =  0.42173840808732965
        RF MSE for the train set =  167.3844086637256
        RF MSE for the test set =  1240.9524549068294
            index       val                                               name
        0      18  0.139116             L3_NO2_NO2_slant_column_number_density
        1       1  0.126539                     L3_CO_CO_column_number_density
        2       8  0.095350                                         wind_speed
        3      15  0.088557    L3_HCHO_tropospheric_HCHO_column_number_density
        4      19  0.074508                          L3_AER_AI_sensor_altitude
        5      17  0.053889           L3_HCHO_HCHO_slant_column_number_density
        6      16  0.045455  L3_HCHO_tropospheric_HCHO_column_number_densit...
        7      20  0.041723                       L3_AER_AI_solar_zenith_angle
        8       4  0.037350                  specific_humidity_2m_above_ground
        9       2  0.037211                        temperature_2m_above_ground
        10      5  0.032390               precipitable_water_entire_atmosphere
        11     11  0.031989                     L3_SO2_absorbing_aerosol_index
        12     10  0.026910                     L3_NO2_absorbing_aerosol_index
        13      6  0.026371                            L3_CLOUD_cloud_fraction
        14      7  0.023777                    L3_CO_H2O_column_number_density
        15     13  0.023257               L3_SO2_SO2_column_number_density_amf
        16     14  0.023090                     L3_O3_O3_column_number_density
        17      3  0.022697                  relative_humidity_2m_above_ground
        18     12  0.018857                   L3_SO2_SO2_column_number_density
        19      9  0.018744                  L3_AER_AI_absorbing_aerosol_index
        20      0  0.012219                   L3_NO2_NO2_column_number_density
        """
        """
        X_train size =  23372
        X_test size =  4754
        
        The best hyperparameters are  {'learning_rate': 0.015, 'max_depth': 4, 'n_estimators': 500}
        param_grid = {"max_depth":    [4, 30],
                    "n_estimators": [100, 500],
                    "learning_rate": [0.01, 0.015]}
                    
        RF R2 for the train set =  0.6339602592977092
        RF R2 for the test set =  0.414633627368584
        
        RF MSE for the train set =  811.5717412047818
        RF MSE for the test set =  1237.0046947841881
        
            index       val                                               name
        0       9  0.114285             L3_NO2_NO2_slant_column_number_density
        1       6  0.108554                                         wind_speed
        2      23  0.095487    L3_HCHO_tropospheric_HCHO_column_number_density
        3      18  0.087859                     L3_CO_CO_column_number_density
        4      21  0.051964           L3_HCHO_HCHO_slant_column_number_density
        5      42  0.045014                              L3_SO2_cloud_fraction
        6      14  0.038926      L3_NO2_tropospheric_NO2_column_number_density
        7      33  0.034675                          L3_AER_AI_sensor_altitude
        8       2  0.025487                  specific_humidity_2m_above_ground
        9      22  0.025160                             L3_HCHO_cloud_fraction
        10      3  0.024633                        temperature_2m_above_ground
        11     37  0.022303                       L3_AER_AI_solar_zenith_angle
        12     24  0.020913  L3_HCHO_tropospheric_HCHO_column_number_densit...
        13     28  0.020647                       L3_CLOUD_cloud_optical_depth
        14      0  0.020312               precipitable_water_entire_atmosphere
        15     41  0.017262                     L3_SO2_absorbing_aerosol_index
        16     29  0.016751                          L3_CLOUD_cloud_top_height
        17     31  0.015092                            L3_CLOUD_surface_albedo
        18     30  0.014577                        L3_CLOUD_cloud_top_pressure
        19     10  0.013884                     L3_NO2_absorbing_aerosol_index
        20      8  0.013683                   L3_NO2_NO2_column_number_density
        21     38  0.012288                   L3_SO2_SO2_column_number_density
        22     13  0.011570                         L3_NO2_tropopause_pressure
        23     26  0.011312                       L3_CLOUD_cloud_base_pressure
        24      5  0.010350               v_component_of_wind_10m_above_ground
        25      7  0.010067                                         wind_angle
        26     32  0.009763                  L3_AER_AI_absorbing_aerosol_index
        27      1  0.009440                  relative_humidity_2m_above_ground
        28     25  0.009150                         L3_CLOUD_cloud_base_height
        29     34  0.008323                     L3_AER_AI_sensor_azimuth_angle
        30     35  0.008241                      L3_AER_AI_sensor_zenith_angle
        31     11  0.007789                              L3_NO2_cloud_fraction
        32      4  0.007750               u_component_of_wind_10m_above_ground
        33     36  0.007374                      L3_AER_AI_solar_azimuth_angle
        34     27  0.007000                            L3_CLOUD_cloud_fraction
        35     39  0.006774               L3_SO2_SO2_column_number_density_amf
        36     20  0.005381                                 L3_CO_cloud_height
        37     12  0.005343     L3_NO2_stratospheric_NO2_column_number_density
        38     40  0.005275             L3_SO2_SO2_slant_column_number_density
        39     19  0.005085                    L3_CO_H2O_column_number_density
        40     16  0.005059                     L3_O3_O3_effective_temperature
        41     15  0.004980                     L3_O3_O3_column_number_density
        42     17  0.004218                               L3_O3_cloud_fraction
        """

    # %% Extra tree
    if False:
        from sklearn.ensemble import ExtraTreesRegressor
        
        scl = StandardScaler()
        
        reg = ExtraTreesRegressor()
        param_grid = {'n_estimators':[700], 'max_depth':[5, 30], 'max_features':[0.3, 0.5], 'min_samples_split': [5] }
        #(n_estimators=100, *, criterion='squared_error', max_depth=None, 
        # min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
        # max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, 
        # oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, 
        # ccp_alpha=0.0, max_samples=None, monotonic_cst=None)
        
        search = GridSearchCV(reg, param_grid, scoring='neg_mean_squared_error')
        #(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, 
        # cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, 
        # return_train_score=False)
        
        pipe = Pipeline([('scaler', scl), ('search', search)])
        pipe.fit(X_train, y_train)
        y_train_pred = pipe.predict(X_train)
        y_test_pred  = pipe.predict(X_test)
        
        print("The best hyperparameters are ", search.best_params_)
        print(search.cv_results_)

        print( 'RF R2 for the train set = ', r2_score(y_train, y_train_pred))
        print( 'RF R2 for the test set = ',  r2_score(y_test,  y_test_pred))
        
        print( 'RF MSE for the train set = ', mean_squared_error(y_train, y_train_pred))
        print( 'RF MSE for the test set = ',  mean_squared_error(y_test,  y_test_pred))
                
        #tmp = pd.DataFrame({'val':search.best_estimator_.feature_importances_, \
        #    'name':search.best_estimator_.feature_names_in_})
        #print( tmp.sort_values(by='val', ascending=False).reset_index() )
        
        
        """
        X_train size =  24087
        X_test size =  4876
        The best hyperparameters are  {'max_depth': 30, 'max_features': 0.5, 'min_samples_split': 5, 'n_estimators': 700}
        'params': 
        [{'max_depth': 5, 'max_features': 0.3, 'min_samples_split': 5, 'n_estimators': 700}, 
        {'max_depth': 5, 'max_features': 0.5, 'min_samples_split': 5, 'n_estimators': 700}, 
        {'max_depth': 30, 'max_features': 0.3, 'min_samples_split': 5, 'n_estimators': 700}, 
        {'max_depth': 30, 'max_features': 0.5, 'min_samples_split': 5, 'n_estimators': 700}], 
        'split0_test_score': array([-1549.22082887, -1463.51068733, -1167.21804885, -1144.13872851]), 
        'split1_test_score': array([-1264.98152711, -1188.39648743,  -845.93547243,  -837.08854365]), 
        'split2_test_score': array([-1613.49967282, -1505.38370751, -1125.10043303, -1110.09597613]), 
        'split3_test_score': array([-2044.80502125, -1905.13660403, -1338.82457624, -1311.91175991]), 
        'split4_test_score': array([-1645.33797948, -1537.76370528, -1054.86530333, -1028.82553041]), 
        'mean_test_score': array([-1623.5690059 , -1520.03823831, -1106.38876678, -1086.41210772]), 
        'std_test_score': array([249.85103229, 228.91540519, 160.35418258, 155.0624616 ]), 
        'rank_test_score': array([4, 3, 2, 1], dtype=int32)}
        
        RF R2 for the train set =  0.9807645145739189
        RF R2 for the test set =  0.4378534921348082
        
        RF MSE for the train set =  43.243982737074404
        RF MSE for the test set =  1206.3693987443023
        """
        
    # %% Random forest
    if False:
        from sklearn.ensemble import RandomForestRegressor
        
        scl = StandardScaler()
        reg=RandomForestRegressor()
        param_grid = {'n_estimators':[100, 500], 'max_depth':[6, 30], 'max_features':[0.3], 'min_samples_split': [5] }
        #(n_estimators=100, *, criterion='squared_error', max_depth=None, 
        # min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
        # max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, 
        # bootstrap=True, oob_score=False, n_jobs=None, random_state=None, 
        # verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)

        search = GridSearchCV(reg, param_grid, scoring='neg_mean_squared_error')
        #(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, 
        # cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, 
        # return_train_score=False)
        
        pipe = Pipeline([('scaler', scl), ('search', search)])
        pipe.fit(X_train, y_train)
        y_train_pred = pipe.predict(X_train)
        y_test_pred  = pipe.predict(X_test)
        
        print("The best hyperparameters are ", search.best_params_)
        print(search.cv_results_)

        print( 'RF R2 for the train set = ', r2_score(y_train, y_train_pred))
        print( 'RF R2 for the test set = ',  r2_score(y_test,  y_test_pred))
        
        print( 'RF MSE for the train set = ', mean_squared_error(y_train, y_train_pred))
        print( 'RF MSE for the test set = ',  mean_squared_error(y_test,  y_test_pred))
                
        #tmp = pd.DataFrame({'val':search.best_estimator_.feature_importances_, 'name':search.best_estimator_.feature_names_in_})
        #print( tmp.sort_values(by='val', ascending=False).reset_index() )
        
        #pickle.dump(search.best_estimator_, open('../models/random_forest_model.sav', 'wb'))

        
        """
        max_depth: 20 or more is sufficient
        max_features: 0.3 is good.
        'min_samples_split': not sensitive <= 10
        'n_estimators': 500 is only slightly better than 100
        'bootstrap' and 'oob_score' don't make difference.
        
        X_train size =  24087
        X_test size =  4876
        The best hyperparameters are  {'max_depth': 30, 'max_features': 0.3, 'min_samples_split': 5, 'n_estimators': 500}
        'params': [
        {'max_depth': 6, 'max_features': 0.3, 'min_samples_split': 5, 'n_estimators': 100}, 
        {'max_depth': 6, 'max_features': 0.3, 'min_samples_split': 5, 'n_estimators': 500}, 
        {'max_depth': 30, 'max_features': 0.3, 'min_samples_split': 5, 'n_estimators': 100}, 
        {'max_depth': 30, 'max_features': 0.3, 'min_samples_split': 5, 'n_estimators': 500}], 
        
        'split0_test_score': array([-1254.97400035, -1253.2115096 , -1096.85563029, -1093.28887519]), 
        'split1_test_score': array([-1020.21672367, -1023.25382889,  -853.48446329,  -846.82315461]), 
        'split2_test_score': array([-1266.3621336 , -1263.38828136, -1131.48277196, -1116.56016855]), 
        'split3_test_score': array([-1647.82888026, -1639.33596381, -1344.17777175, -1335.05195918]), 
        'split4_test_score': array([-1256.24146129, -1239.68818905, -1030.65751862, -1023.04937813]), 
        'mean_test_score': array([-1289.12463983, -1283.77555454, -1091.33163118, -1082.95470713]), 
        'std_test_score': array([201.86426171, 198.79047851, 158.63569755, 157.56643367]), 
        'rank_test_score': array([4, 3, 2, 1], dtype=int32)}
        
        RF R2 for the train set =  0.9316959384314718
        RF R2 for the test set =  0.42354381489345627
        
        RF MSE for the train set =  153.55680368412106
        RF MSE for the test set =  1237.0780422889036        
        """
            
    # %% Lasso
    if False:
        """
        X_train size =  23996
        X_test size =  4867
        RF R2 for the train set =  0.39840092685813033
        RF R2 for the test set =  0.4501466547944587
        RF MSE for the train set =  1219.5037178688158
        RF MSE for the test set =  894.1330893762533
        lassoGSCV.best_params_ =  {'alpha': 0.5}
        """
        from sklearn.linear_model import Lasso
        scl = StandardScaler()
        lasso = Lasso(max_iter=1000, tol=0.0001) #(alpha=1.0, *, fit_intercept=True, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
        parameters = {'alpha':[0.01, 0.1, 0.28, 0.29, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 10.0]}
        lassoGSCV = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error') #(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)
        pipe = Pipeline([('scaler', scl), ('lassoGSCV', lassoGSCV)])
        pipe.fit(X_train, y_train)
        
        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)
        print( 'RF R2 for the train set = ', r2_score(y_train, y_train_pred))
        print( 'RF R2 for the test set = ', r2_score(y_test, y_test_pred))
        print( 'RF MSE for the train set = ', mean_squared_error(y_train, y_train_pred))
        print( 'RF MSE for the test set = ', mean_squared_error(y_test, y_test_pred))
        print( 'lassoGSCV.best_params_ = ', lassoGSCV.best_params_ )
        pickle.dump(pipe, open('../models/lasso_model.sav', 'wb'))
        
    # %% Ridge
    if False:
        """
        X_train size =  24087
        X_test size =  4876
        RF R2 for the train set =  0.4027270835617095
        RF R2 for the test set =  0.3529226928674044
        RF MSE for the train set =  1342.7506047109753
        RF MSE for the test set =  1388.6313461433624
        ridgeGSCV =  {'alpha': 2000}

        'params': [{'alpha': 0.1},
        {'alpha': 1},
        {'alpha': 10},
        {'alpha': 100},
        {'alpha': 1000},
        {'alpha': 1500},
        {'alpha': 2000},
        {'alpha': 2500},
        {'alpha': 3000}],
        'split0_test_score': array([-1670.98256287, -1670.92701103, -1670.37411778, -1665.09499931,
                -1628.84783356, -1616.10636544, -1606.1310744 , -1598.02165582,
                -1591.24860483]),
        'split1_test_score': array([-1164.89290899, -1164.880118  , -1164.75296074, -1163.55134852,
                -1155.84026161, -1153.48011037, -1151.88967496, -1150.84511942,
                -1150.2074949 ]),
        'split2_test_score': array([-1478.37879431, -1478.35240462, -1478.08986557, -1475.59382017,
                -1459.25926583, -1454.09348928, -1450.42835992, -1447.79783051,
                -1445.92461841]),
        'split3_test_score': array([-1577.54186456, -1577.56208422, -1577.76447169, -1579.80507791,
                -1600.54536962, -1611.67106385, -1622.35829874, -1632.61818446,
                -1642.48585937]),
        'split4_test_score': array([-1326.39072983, -1326.37397037, -1326.20807093, -1324.70406783,
                -1318.22596289, -1317.955919  , -1318.83793785, -1320.47998632,
                -1322.65564711]),
        'mean_test_score': array([-1443.63737211, -1443.61911765, -1443.43789734, -1441.74986275,
                -1432.5437387 , -1430.66138959, -1429.92906917, -1429.95255531,
                -1430.50444492]),
        'std_test_score': array([180.14908611, 180.14319438, 180.08473656, 179.54488984,
                177.26078706, 177.33323952, 177.84460641, 178.61928102,
                179.5607586 ]),
        'rank_test_score': array([9, 8, 7, 6, 5, 4, 1, 2, 3], dtype=int32)}
        """
        from sklearn.linear_model import Ridge
        scl = StandardScaler()
        ridge = Ridge() #(alpha=1.0, *, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver='auto', positive=False, random_state=None)
        parameters = {'alpha':[0.1, 1, 10, 100, 1000, 1500, 2000, 2500, 3000]}
        ridgeGSCV = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error') #(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)
        pipe = Pipeline([('scaler', scl), ('ridgeGSCV', ridgeGSCV)])
        pipe.fit(X_train, y_train)

        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)
        print( 'RF R2 for the train set = ', r2_score(y_train, y_train_pred))
        print( 'RF R2 for the test set = ', r2_score(y_test, y_test_pred))
        print( 'RF MSE for the train set = ', mean_squared_error(y_train, y_train_pred))
        print( 'RF MSE for the test set = ', mean_squared_error(y_test, y_test_pred))
        print('ridgeGSCV = ', ridgeGSCV.best_params_)
        pickle.dump(pipe, open('../models/ridge_model.sav', 'wb'))

    # %% Plot the error at the level 1 ML process
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    sns.kdeplot(x=y_test, y=(y_test-y_test_pred), fill=True)
    plt.plot([-10, 260], [0, 0], ':k')
    plt.xlim(-10, 260)
    plt.xlabel('True particle concentration')
    plt.ylabel('Prediction error in particle concentration (Truth - Prediction)')

    # Sample series (Time series at each location and different locations are appended to one after the other)
    fig, ax = plt.subplots(1, 1, figsize=(13, 7))
    plt.plot(y_test, label="truth")
    plt.plot(y_test_pred, label="prediction")
    plt.xlabel('sampling series')
    plt.ylabel('particle concentration')
    
    # Time series
    tmp = pd.DataFrame({'y_test': y_test, 'y_test_pred': y_test_pred})
    tmp = pd.concat( [date_and_place_test, tmp], axis=1 )
    places = tmp.Place_ID.unique()
    fig, ax = plt.subplots(5, 2, figsize=(15,10))
    for j in range(0,10):
        i = int(len(places)/10)*j
        sns.scatterplot(data=tmp[tmp.Place_ID == places[i]], x='Date', y='y_test',      ax=ax[j%5,j%2], legend=False)
        sns.scatterplot(data=tmp[tmp.Place_ID == places[i]], x='Date', y='y_test_pred', ax=ax[j%5,j%2], legend=False)
        plt.legend(labels=['truth', 'Lev-1 prediction'])
        ax[j%5,j%2].set_ylabel('concentration')


    # %% A trial: Can we predict the errors in the prediction? Version 1
    if False:
        from sklearn.neighbors import KNeighborsRegressor
        import lightgbm as lgb
        from sklearn.linear_model import Lasso
        
        if False:
            # KNN
            regressor_lev2=KNeighborsRegressor()
            #(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, 
            # p=2, metric='minkowski', metric_params=None, n_jobs=None)

            # Define which parameters to try
            if True:
                param_grid = {"n_neighbors": [5, 10, 15, 20, 25],
                            "p": [1,2,3],
                            "weights": ['uniform', 'distance']}
            else:
                param_grid = {"n_neighbors": [15],
                            "p": [1],
                            "weights": ['distance']}
        elif False:
            # Lightgbm
            regressor_lev2=lgb.LGBMRegressor()
            #(boosting_type='gbdt', num_leaves=31, max_depth=-1, 
            # learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, 
            # objective=None, class_weight=None, min_split_gain=0.0, 
            # min_child_weight=0.001, min_child_samples=20, subsample=1.0, 
            # subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, 
            # reg_lambda=0.0, random_state=None, n_jobs=None, 
            # importance_type='split', **kwargs)

            param_grid = {"boosting_type": ["gbdt"],
                          "n_estimators": [100],
                          "max_depth": [-1],
                          "num_leaves": [31],
                          "learning_rate": [0.1, 0.5],
                          "reg_alpha": [0.0, 0.5, 1.0],
                          "reg_lambda": [0.0, 0.5, 1.0]}
        elif True:
            # Lasso
            regressor_lev2 = Lasso(max_iter=1000, tol=0.0001) #(alpha=1.0, *, fit_intercept=True, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
            param_grid = {'alpha':[1e-2, 1e-1, 1, 1e+1, 1e+2]}

        search_lev2 = GridSearchCV(regressor_lev2, param_grid, scoring='neg_mean_squared_error')
        #(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, 
        # verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)
        
        scl_lev2 = StandardScaler()
        pipe_lev2 = Pipeline([('scaler', scl_lev2), ('search', search_lev2)])
        
        # Train the model to predict the error in the first prediction from the features
        error_train = y_train - y_train_pred # Error in the first prediction (from the train set)
        pipe_lev2.fit(X_train, error_train) 
        
        # Predict the error from the features
        error_train_pred = pipe_lev2.predict(X_train)
        error_test_pred  = pipe_lev2.predict(X_test)
        
        # Take the predicted error from the first prediction
        y_train_pred = y_train_pred - error_train_pred
        y_test_pred  = y_test_pred - error_test_pred
        
        # Check the scores for the final prediction
        print("The best hyperparameters are ", search_lev2.best_params_)
        print( 'RF R2 for the train set = ', r2_score(y_train, y_train_pred))
        print( 'RF R2 for the test set = ', r2_score(y_test, y_test_pred))
        print( 'RF MSE for the train set = ', mean_squared_error(y_train, y_train_pred))
        print( 'RF MSE for the test set = ', mean_squared_error(y_test, y_test_pred))
    
    # %% A trial: Can we predict the errors in the prediction? Verson 2
    if False:
        # %% Apply another layer of model
        # This process will remove the bias in the level1-prediction.
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.linear_model import LinearRegression

        Z = pd.DataFrame({'y_test': y_test, 'y_test_pred': y_test_pred})
        Z_train, Z_test, date_and_place_test_train, date_and_place_test_test = train_test_split(Z, date_and_place_test, test_size=0.7, shuffle=True)

        lev1_prediction_train = Z_train['y_test_pred'].values #Level1-prediction is now the feature of the level2 regression.
        error_in_lev1_prediction_train = Z_train['y_test'].values - lev1_prediction_train # the error in the level1-prediction of the true target value.    

        lev1_prediction_test = Z_test['y_test_pred'].values # This is the level1-pridiction for the final test set.
        y_test_lev2 = Z_test['y_test'].values # This is the true target value for final test set. 

        if True:
            regressor_lev2 = KNeighborsRegressor()
            param_grid = {"n_neighbors": [5, 10, 15, 20, 25, 30]}
            #(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, 
            # p=2, metric='minkowski', metric_params=None, n_jobs=None)        
        else:
            regressor_lev2 = LinearRegression()
            param_grid = {'fit_intercept': [True]}
            #(*, fit_intercept=True, copy_X=True, n_jobs=None, positive=False)

        search_lev2 = GridSearchCV(regressor_lev2, param_grid, scoring='neg_mean_squared_error')
        #(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, 
        # verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)

        if True:
            # with scaling
            scl_lev2 = StandardScaler()
            pipe_lev2 = Pipeline([('scaler_lev2', scl_lev2), ('search_lev2', search_lev2)])
        else:
            # without scaling
            pipe_lev2 = Pipeline([('search_lev2', search_lev2)])

        pipe_lev2.fit(lev1_prediction_train.reshape(-1, 1), error_in_lev1_prediction_train)
        print("Lev 2: The best hyperparameters are ", search_lev2.best_params_)

        predicted_error_train = pipe_lev2.predict(lev1_prediction_train.reshape(-1, 1))
        predicted_error_test  = pipe_lev2.predict(lev1_prediction_test.reshape(-1, 1))
        
        # Subtract the predicted error from the level-1 prediction
        lev2_prediction_train = lev1_prediction_train - predicted_error_train
        lev2_prediction_test  = lev1_prediction_test  - predicted_error_test
        
        print( 'Lev 2: RF R2 for the train set = ', r2_score(Z_train['y_test'].values, lev2_prediction_train))
        print( 'Lev 2: RF R2 for the test set = ',  r2_score(y_test_lev2, lev2_prediction_test))
        print( 'Lev 2: RF MSE for the train set = ', mean_squared_error(Z_train['y_test'].values, lev2_prediction_train))
        print( 'Lev 2: RF MSE for the test set = ',  mean_squared_error(y_test_lev2, lev2_prediction_test))
        
        
        # Plot error vs predicted error    
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        plt.scatter(error_in_lev1_prediction_train, predicted_error_train)
        plt.xlabel('error at level 1 in train')
        plt.ylabel('predicted error in train')

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        sns.kdeplot(x=lev1_prediction_train, y=error_in_lev1_prediction_train, fill=True)
        plt.plot([0, 200], [0, 0])
        plt.xlabel('lev1-prediction for train set')
        plt.ylabel('error in lev1-prediction')

        # %% Define the final prediction for the final test set
        final_target_val = y_test_lev2
        final_predic_val = lev2_prediction_test

        # %% Plot the final error

        # Traget value vs error from the test 
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        sns.kdeplot(x=final_target_val, y=(final_target_val-final_predic_val), fill=True)
        plt.plot([-10, 260], [0, 0], ':k')
        plt.xlim(-10, 260)
        plt.xlabel('actual value')
        plt.ylabel('error (truth - prediction)')
        plt.title("the accuracy of the final prediction")

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        sns.kdeplot(x=final_predic_val, y=(final_target_val-final_predic_val), fill=True)
        plt.plot([-10, 260], [0, 0], ':k')
        plt.xlim(-10, 260)
        plt.xlabel('predicted value')
        plt.ylabel('error (truth - prediction)')
        plt.title("the accuracy of the final prediction")

        # Time series of target values and prediction, different Place_ID's are consecutively appended to make the series 
        fig, ax = plt.subplots(1, 1, figsize=(13, 7))
        plt.plot(final_target_val, label="truth")
        plt.plot(final_predic_val, label="prediction")
        plt.xlabel('Sampling series')
        plt.ylabel('particle concentration')
        
        # Time series
        tmp = pd.DataFrame({'y_test': final_target_val, 'y_test_pred': final_predic_val})
        tmp = pd.concat( [date_and_place_test_test, tmp], axis=1 )
        places = tmp.Place_ID.unique()
        fig, ax = plt.subplots(5, 2, figsize=(15,10))
        for j in range(0,10):
            i = int(len(places)/10)*j
            sns.scatterplot(data=tmp[tmp.Place_ID == places[i]], x='Date', y='y_test',      ax=ax[j%5,j%2], legend=False)
            sns.scatterplot(data=tmp[tmp.Place_ID == places[i]], x='Date', y='y_test_pred', ax=ax[j%5,j%2], legend=False)
            plt.legend(labels=['truth', 'Lev-1 prediction'])
            ax[j%5,j%2].set_ylabel('concentration')
        