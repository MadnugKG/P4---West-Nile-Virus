# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from metpy.units import units
import metpy.calc as mpcalc

def create_date_predictors(data, date_col_name):
    data['Year'] = data[date_col_name].dt.year
    data['Month'] = data[date_col_name].dt.month
    data['Week'] = data[date_col_name].dt.isocalendar().week
    data['Year-Month'] = data['Year'].astype('str') + "-" + data['Month'].astype('str')
    data['Year-Week'] = data['Year'].astype('str') + "-" + data['Week'].astype('str')

    return data


def transform_weather_to_weekly(weather):
    # Set all data types correctly
    weather['Date'] = pd.to_datetime(weather['Date'])
    weather['Sunrise_1'] = pd.to_datetime(weather['Sunrise_1'])
    weather['Sunset_1'] = pd.to_datetime(weather['Sunset_1'])

    # Create date related predictors
    weather = create_date_predictors(weather, 'Date')

    # Since mosquitoes dies/greatly inhibited at temperature below 50F, to create 4 additional predictors to indicate this for Tavg and Tmin
    weather['Tavg_1_Below_50F'] = weather['Tavg_1'].apply(lambda x: 1 if x < 50 else 0)
    weather['Tmin_1_Below_50F'] = weather['Tmin_1'].apply(lambda x: 1 if x < 50 else 0)
    weather['Tavg_2_Below_50F'] = weather['Tavg_2'].apply(lambda x: 1 if x < 50 else 0)
    weather['Tmin_2_Below_50F'] = weather['Tmin_2'].apply(lambda x: 1 if x < 50 else 0)

    # Calculate duration of daylight
    weather['Daylight_Duration'] = (weather['Sunset_1'] - weather['Sunrise_1']).dt.total_seconds()/60

    # Calculate relative humidity
    weather['RH_1'] = mpcalc.relative_humidity_from_dewpoint(weather['Tavg_1'].values * units.degF, weather['DewPoint_1'].values * units.degF)
    weather['RH_2'] = mpcalc.relative_humidity_from_dewpoint(weather['Tavg_2'].values * units.degF, weather['DewPoint_2'].values * units.degF)

    # Define wet weather
    wet_weather = ['GR', 'TS', 'RA', 'DZ', 'GS', 'UP', 'SQ', 'SH', 'PY', 'SN', 'SG', 'PL', 'IC']

    # Check if the day is a wet weather
    weather['Wet_Weather_1'] = weather['CodeSum_1'].apply(lambda x: int(any([1 if weather in x else 0 for weather in wet_weather])))
    weather['Wet_Weather_2'] = weather['CodeSum_2'].apply(lambda x: int(any([1 if weather in x else 0 for weather in wet_weather])))

    # Set Date as index
    weather.set_index('Date', inplace=True)


    # Shift Wet_Weather and PrecipTotal by 7 days
    lag = 7
    weather['Wet_Weather_1_shift7'] = weather['Wet_Weather_1'].shift(lag)
    weather['Wet_Weather_2_shift7'] = weather['Wet_Weather_2'].shift(lag)
    weather['PrecipTotal_1_shift7'] = weather['PrecipTotal_1'].shift(lag)
    weather['PrecipTotal_2_shift7'] = weather['PrecipTotal_2'].shift(lag)

    # Shift Wet_Weather and PrecipTotal by 14 days
    lag = 14
    weather['Wet_Weather_1_shift14'] = weather['Wet_Weather_1'].shift(lag)
    weather['Wet_Weather_2_shift14'] = weather['Wet_Weather_2'].shift(lag)
    weather['PrecipTotal_1_shift14'] = weather['PrecipTotal_1'].shift(lag)
    weather['PrecipTotal_2_shift14'] = weather['PrecipTotal_2'].shift(lag)

    # Shift Wet_Weather and PrecipTotal by 21 days
    lag = 21
    weather['Wet_Weather_1_shift21'] = weather['Wet_Weather_1'].shift(lag)
    weather['Wet_Weather_2_shift21'] = weather['Wet_Weather_2'].shift(lag)
    weather['PrecipTotal_1_shift21'] = weather['PrecipTotal_1'].shift(lag)
    weather['PrecipTotal_2_shift21'] = weather['PrecipTotal_2'].shift(lag)

    # Shift Wet_Weather and PrecipTotal by 28 days
    lag = 28
    weather['Wet_Weather_1_shift28'] = weather['Wet_Weather_1'].shift(lag)
    weather['Wet_Weather_2_shift28'] = weather['Wet_Weather_2'].shift(lag)
    weather['PrecipTotal_1_shift28'] = weather['PrecipTotal_1'].shift(lag)
    weather['PrecipTotal_2_shift28'] = weather['PrecipTotal_2'].shift(lag)

    #create dataframe for weekly, monthly and yearly
    weather_weekly = weather.resample('W').mean()

    return pd.DataFrame(weather_weekly)


def transform_test_weather_weekly(data, weather_weekly):
    """
    This function helps to transform the test data into weekly data and merge with the weather_weekly dataset.
    Additonal features are feature engineered to be used for modeling.
    """
    # Drop columns not in the list of column names
    data.drop(columns=[col for col in data.columns if col not in ['Id', 'Date', 'Species', 'Latitude', 'Longitude', 'AddressAccuracy', 'NumMosquitos', 'WnvPresent']], inplace=True)

    # Convert Date to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Create date additional date features
    data = create_date_predictors(data, 'Date')

    # Set Date as index
    data.set_index('Date', inplace=True)

    # Get Weekly Date by resampling and replace date in data through merging
    # Get Weekly Date by resampling
    temp = data['AddressAccuracy'].resample('W').mean()
    temp = pd.DataFrame(temp)
    temp['Year-Week'] = temp.index.year.astype('str') + "-" + temp.index.isocalendar().week.astype('str')
    temp.drop(columns=['AddressAccuracy'], inplace=True)
    temp.reset_index(inplace=True)
    weekly_date_df = temp.drop_duplicates()

    # Replace date in data through merging
    data_weekly = pd.merge(left=data,
                    right=weekly_date_df,
                    on='Year-Week',
                    how='left', )
    data_weekly.set_index('Date', inplace=True)

   # Merge test
    data_weather_weekly = pd.merge(left=data_weekly,
                                    right=weather_weekly,
                                    left_index=True,
                                    right_index=True,
                                    how='left'
                                   )

    # Drop unneeded columns
    data_weather_weekly.drop(columns=['Year_y', 'Month_y', 'Week_y'], inplace=True)
    data_weather_weekly.rename(columns={'Year_x': 'Year', 'Month_x': 'Month', 'Week_x': 'Week'}, inplace=True)

    # Check if there are any null rows after merge
    print(f"There are {data_weather_weekly.isnull().sum().sum()} rows with N.A.")

    #Check that there are no duplicates
    print(f"There are {data_weather_weekly.shape[0] - data_weather_weekly.drop_duplicates().shape[0]} duplicated rows.")

    # Creating new predictors if wet weather or temperature fell below 50F at least once per week
    data_weather_weekly['At_Least_One_Wet_Weather_1'] = data_weather_weekly['Wet_Weather_1'].apply(lambda x: 1 if x > 0 else 0)
    data_weather_weekly['At_Least_One_Wet_Weather_2'] = data_weather_weekly['Wet_Weather_2'].apply(lambda x: 1 if x > 0 else 0)
    data_weather_weekly['At_Least_One_Wet_Weather_1_shift7'] = data_weather_weekly['Wet_Weather_1_shift7'].apply(lambda x: 1 if x > 0 else 0)
    data_weather_weekly['At_Least_One_Wet_Weather_2_shift7'] = data_weather_weekly['Wet_Weather_2_shift7'].apply(lambda x: 1 if x > 0 else 0)
    data_weather_weekly['At_Least_One_Wet_Weather_1_shift14'] = data_weather_weekly['Wet_Weather_1_shift14'].apply(lambda x: 1 if x > 0 else 0)
    data_weather_weekly['At_Least_One_Wet_Weather_2_shift14'] = data_weather_weekly['Wet_Weather_2_shift14'].apply(lambda x: 1 if x > 0 else 0)
    data_weather_weekly['At_Least_One_Wet_Weather_1_shift21'] = data_weather_weekly['Wet_Weather_1_shift21'].apply(lambda x: 1 if x > 0 else 0)
    data_weather_weekly['At_Least_One_Wet_Weather_2_shift21'] = data_weather_weekly['Wet_Weather_2_shift21'].apply(lambda x: 1 if x > 0 else 0)
    data_weather_weekly['At_Least_One_Wet_Weather_1_shift28'] = data_weather_weekly['Wet_Weather_1_shift28'].apply(lambda x: 1 if x > 0 else 0)
    data_weather_weekly['At_Least_One_Wet_Weather_2_shift28'] = data_weather_weekly['Wet_Weather_2_shift28'].apply(lambda x: 1 if x > 0 else 0)
    data_weather_weekly['At_Least_One_Tavg_1_Below_50F'] = data_weather_weekly['Tavg_1_Below_50F'].apply(lambda x: 1 if x > 0 else 0)
    data_weather_weekly['At_Least_One_Tmin_1_Below_50F'] = data_weather_weekly['Tmin_1_Below_50F'].apply(lambda x: 1 if x > 0 else 0)
    data_weather_weekly['At_Least_One_Tavg_2_Below_50F'] = data_weather_weekly['Tavg_2_Below_50F'].apply(lambda x: 1 if x > 0 else 0)
    data_weather_weekly['At_Least_One_Tmin_2_Below_50F'] = data_weather_weekly['Tmin_2_Below_50F'].apply(lambda x: 1 if x > 0 else 0)

    # Create distance of trap from weather stations
    traps = np.array(list(zip(data_weather_weekly['Latitude'].values, data_weather_weekly['Longitude'].values)))
    station1 = np.array([41.786, -87.752])
    station2 = np.array([41.995, -87.933])
    data_weather_weekly['dist_from_s1'] = [np.linalg.norm(i-station1) for i in traps]
    data_weather_weekly['dist_from_s2'] = [np.linalg.norm(i-station2) for i in traps]


    # Drop unneeded columns
    drop_columns = ['Tavg_1_Below_50F',
                    'Tmin_1_Below_50F',
                    'Tavg_2_Below_50F',
                    'Tmin_2_Below_50F',
                    'Wet_Weather_1',
                    'Wet_Weather_2',
                    'Wet_Weather_1_shift7',
                    'Wet_Weather_2_shift7',
                    'Wet_Weather_1_shift14',
                    'Wet_Weather_2_shift14',
                    'Wet_Weather_1_shift21',
                    'Wet_Weather_2_shift21',
                    'Wet_Weather_1_shift28',
                    'Wet_Weather_2_shift28',]
    data_weather_weekly.drop(columns=drop_columns, inplace=True)

    # Convert to object
    convert_to_object=['At_Least_One_Wet_Weather_1',
                       'At_Least_One_Wet_Weather_2',
                       'At_Least_One_Wet_Weather_1_shift7',
                       'At_Least_One_Wet_Weather_2_shift7',
                       'At_Least_One_Wet_Weather_1_shift14',
                       'At_Least_One_Wet_Weather_2_shift14',
                       'At_Least_One_Wet_Weather_1_shift21',
                       'At_Least_One_Wet_Weather_2_shift21',
                       'At_Least_One_Wet_Weather_1_shift28',
                       'At_Least_One_Wet_Weather_2_shift28',
                       'At_Least_One_Tavg_1_Below_50F',
                       'At_Least_One_Tmin_1_Below_50F',
                       'At_Least_One_Tavg_2_Below_50F',
                       'At_Least_One_Tmin_2_Below_50F']
    data_weather_weekly[convert_to_object] = data_weather_weekly[convert_to_object].astype('str')

    # Return dataframe
    return data_weather_weekly
