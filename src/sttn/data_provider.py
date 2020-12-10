import pandas as pd
import geopandas as gpd
import numpy as np
from . import network

class NycTaxiDataProvider:

    @staticmethod
    def get_data(taxi_type, from_date, to_date):
        url = 'https://s3.amazonaws.com/nyc-tlc/trip+data/' + taxi_type + '_tripdata_2020-06.csv'
        column_names = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'passenger_count']
        types = {'PULocationID': np.int32, 'DOLocationID': np.int32}
        df = pd.read_csv(url, usecols=column_names, parse_dates=['tpep_pickup_datetime'], dtype=types)
        df = df.dropna()
        df['passenger_count'] = df['passenger_count'].astype(int)

        labels = gpd.read_file('https://s3.amazonaws.com/nyc-tlc/misc/taxi_zones.zip')
        labels = labels.rename(columns={'OBJECTID': '_id'})
        labels.set_index('_id')
        return network.SpatioTemporalNetwork(df, column_from='PULocationID', column_to='DOLocationID',
                                             time_column='tpep_pickup_datetime', node_labels=labels)
