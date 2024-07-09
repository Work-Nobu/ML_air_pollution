# %%
import numpy as np
import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import seaborn as sns
import os
import missingno as msno
from Get_data import Get_data
from sklearn.preprocessing import StandardScaler
plt.ion()


# %%    
def Plot_time_series_of_train_set(plot_at_these_places):
    """
    This function plots time series of the train data
    
    INPUTS: plot_at_these_places = a list of numbers between 0 and 340, specifying the city ID.
    OUTPUTS: Figures.
    """
    
    df = pd.concat([X_train, y_train], axis=1)
    place_IDs = df.Place_ID.unique()

    for use_this_place in plot_at_these_places:
        data = df[df.Place_ID == place_IDs[use_this_place]].drop(columns=['Place_ID', 'target_count', 'target_variance', 'target_min', 'target_max'])
        col = data.drop(columns='Date').columns
        data[col]=StandardScaler().fit_transform(data[col])
        
        # Find the index necessary for plotting
        for i, x in enumerate(data.columns):
            if x == 'Date':
                idate = i
            if x == 'target':
                itarget = i
            if x == 'wind_speed':
                iwind_speed = i
            if x == 'L3_AER_AI_absorbing_aerosol_index':
                iAER = i
            if x == 'L3_NO2_absorbing_aerosol_index':
                iNO2_aero = i
            if x == 'L3_SO2_absorbing_aerosol_index':
                iSO2_aero = i
            if x == 'L3_CLOUD_cloud_fraction':
                icloud = i
            if x == 'L3_CO_H2O_column_number_density':
                iCO_H2O = i
            if x == 'precipitable_water_entire_atmosphere':
                iprec = i
            if x == 'specific_humidity_2m_above_ground':
                ispe_hum = i
            if x == 'relative_humidity_2m_above_ground':
                irel_hum = i
                
        # Plot the time series comparison between the target and each feature 
        for ifig in range(1, len(col)):
            fig, ax1 = plt.subplots(1,1)
            plt.plot(data[data.columns[idate]], data[data.columns[itarget]], label=data.columns[itarget] )
            plt.plot(data[data.columns[idate]], data[data.columns[ifig]], label=data.columns[ifig] )
            plt.xlabel('days')
            plt.ylabel('z-score')
            plt.title('Place ID = ' + place_IDs[use_this_place])
            plt.legend()
            

        # Plot for checking multicolinearity
        fig, ax1 = plt.subplots(1,1)
        sns.lineplot(data=data.iloc[:,[idate, icloud, iCO_H2O, iprec, ispe_hum, irel_hum]])
        plt.xlabel('days')
        plt.ylabel('z-score')
        plt.title('Place ID = ' + place_IDs[use_this_place])
        
        # Plot for checking multicolinearity
        fig, ax1 = plt.subplots(1,1)
        sns.lineplot(data=data.iloc[:,[idate, iwind_speed, iAER, iNO2_aero, iSO2_aero]])
        plt.xlabel('days')
        plt.ylabel('z-score')
        plt.title('Place ID = ' + place_IDs[use_this_place])

        # Plot for checking multicolinearity
        fig, ax1 = plt.subplots(1,1)
        sns.lineplot(data=data.iloc[:,[idate, ispe_hum, irel_hum]])
        plt.xlabel('days')
        plt.ylabel('z-score')
        plt.title('Place ID = ' + place_IDs[use_this_place])

        # Plot for checking multicolinearity
        fig, ax1 = plt.subplots(1,1)
        sns.lineplot(data=data.iloc[:,[idate, icloud, iCO_H2O, iprec]])
        plt.xlabel('days')
        plt.ylabel('z-score')
        plt.title('Place ID = ' + place_IDs[use_this_place])

# %%
if __name__ == "__main__":

    # Get the data
    path_train_data = '../data/Train.csv'
    path_test_data  = '../data/Test.csv'
        
    target_columns = ['target', 'target_min', 'target_max', 'target_variance', 'target_count']

    feature_columns = [ \
        'Date', \
        'Place_ID', \
        'L3_NO2_NO2_column_number_density', \
        'relative_humidity_2m_above_ground', \
        'specific_humidity_2m_above_ground', \
        'precipitable_water_entire_atmosphere', \
        'temperature_2m_above_ground', \
        \
        'wind_speed', \
        'L3_AER_AI_absorbing_aerosol_index', \
        'L3_CLOUD_cloud_fraction', \
        'L3_CO_CO_column_number_density', \
        \
        'L3_SO2_SO2_column_number_density', \
        'L3_SO2_SO2_column_number_density_amf', \
        \
        'L3_NO2_absorbing_aerosol_index', \
        'L3_SO2_absorbing_aerosol_index', \
        \
        'L3_O3_O3_column_number_density', \
        'L3_CO_H2O_column_number_density' ]
        
    X_train, y_train, X_test = Get_data(path_train_data, path_test_data, feature_columns, target_columns, method = 1)

    # Plot
    plot_at_these_places = [0] #300 is a strange data. [0, 100, 340]
    Plot_time_series_of_train_set(plot_at_these_places)
