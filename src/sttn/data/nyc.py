import pandas as pd
import geopandas as gpd
import numpy as np

from sttn import network
from .data_provider import DataProvider


class NycTaxiDataProvider(DataProvider):

    def get_data(self, taxi_type, from_date, to_date):
        taxi_data = self.cache_file('https://s3.amazonaws.com/nyc-tlc/trip+data/' + taxi_type + '_tripdata_2020-06.csv')
        column_names = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'passenger_count']
        types = {'PULocationID': np.int32, 'DOLocationID': np.int32}
        df = pd.read_csv(taxi_data, usecols=column_names, parse_dates=['tpep_pickup_datetime'], dtype=types)
        df = df[(df['PULocationID'] > 0) & (df['PULocationID'] < 264)]
        df = df[(df['DOLocationID'] > 0) & (df['DOLocationID'] < 264)]
        df = df.dropna()
        df = df[(df['tpep_pickup_datetime'] >= from_date) & (df['tpep_pickup_datetime'] <= to_date)]
        df['passenger_count'] = df['passenger_count'].astype(int)

        labels = gpd.read_file('https://s3.amazonaws.com/nyc-tlc/misc/taxi_zones.zip')
        labels = labels.rename(columns={'OBJECTID': 'id'})
        labels = labels.set_index('id')
        edges = df.rename(columns={'PULocationID': 'from', 'DOLocationID': 'to', 'tpep_pickup_datetime': 'time'})
        return network.SpatioTemporalNetwork(edges, node_labels=labels)
