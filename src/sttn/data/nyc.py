from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet
import os

from sttn import network
from .data_provider import DataProvider

TAXI_ZONE_SHAPE_URL = 'https://data.cityofnewyork.us/api/geospatial/d3c5-ddgc?method=export&format=Shapefile'


class NycTaxiDataProvider(DataProvider):

    @staticmethod
    def build_network(taxi_trips, taxi_zones) -> network.SpatioTemporalNetwork:
        edges = taxi_trips.rename(
            columns={'PULocationID': 'origin', 'DOLocationID': 'destination', 'tpep_pickup_datetime': 'time'})
        edges_casted = edges.astype({'origin': 'int64', 'destination': 'int64'})
        taxi_zones = taxi_zones.rename(columns={'objectid': 'id'}).astype({'id': 'int32'})
        taxi_zones = taxi_zones.set_index('id')
        return network.SpatioTemporalNetwork(nodes=taxi_zones, edges=edges_casted)

    def get_data(self, taxi_type: str, month: str) -> network.SpatioTemporalNetwork:
        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{month}.parquet'
        taxi_data = self.cache_file(url)
        column_names = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'passenger_count', 'fare_amount']
        df = pd.read_parquet(taxi_data, columns=column_names)
        df = df[(df['PULocationID'] > 0) & (df['PULocationID'] < 264)]
        df = df[(df['DOLocationID'] > 0) & (df['DOLocationID'] < 264)]
        df = df.dropna()

        from_date = datetime.strptime(month, '%Y-%m')
        to_date = from_date + relativedelta(months=1)
        df = df[(df['tpep_pickup_datetime'] >= from_date) & (df['tpep_pickup_datetime'] <= to_date)]
        df['passenger_count'] = df['passenger_count'].astype(int)
        labels = gpd.read_file(TAXI_ZONE_SHAPE_URL)
        return self.build_network(df, labels)


class Service311RequestsDataProvider(DataProvider):

    def build_network(self, requests, nyc_zip_shape):
        requests['from'] = requests['Incident Zip']
        requests['to'] = requests['Incident Zip']
        column_map = {'Latitude': 'latitude', 'Longitude': 'longitude', 'Complaint Type': 'complaint_type',
                      'Created Date': 'time', 'City': 'city'}
        requests = requests.rename(columns=column_map)
        requests = requests.drop('Incident Zip', 1)

        node_labels = nyc_zip_shape[['ZIPCODE', 'COUNTY', 'PO_NAME', 'geometry']]
        node_labels.columns = node_labels.columns.str.lower()
        node_labels = node_labels.rename(columns={'zipcode': 'id'})
        node_labels = node_labels.set_index('id')
        return network.SpatioTemporalNetwork(requests, node_labels=node_labels)

    def get_data(self, from_date, to_date):
        data = self.cache_file('https://data.cityofnewyork.us/api/views/erm2-nwe9/rows.csv')
        column_names = ['Incident Zip', 'City', 'Latitude', 'Longitude', 'Complaint Type', 'Created Date']
        filtered_file = self.filter_requests(data, from_date, to_date, column_names)

        nyc_shape = gpd.read_file(
            'https://data.cityofnewyork.us/api/views/i8iw-xf4u/files/YObIR0MbpUVA0EpQzZSq5x55FzKGM2ejSeahdvjqR20?filename=ZIP_CODE_040114.zip')
        nyc_shape['ZIPCODE'] = nyc_shape['ZIPCODE'].astype(int)
        requests = pd.read_parquet(filtered_file)
        requests['Incident Zip'] = requests['Incident Zip'].astype(int)

        return self.build_network(requests, nyc_shape)

    def filter_requests(self, requests_file, from_date, to_date, column_names):
        arg_hash = self.hash_args(from_date=from_date.timestamp(), to_date=to_date.timestamp(),
                                  column_names=column_names)
        local_filename = arg_hash + '.parquet'
        file_path = os.path.join(self.cache_dir(), local_filename)

        if not os.path.exists(file_path):
            pqwriter = None
            # write to a temporary file in order to avoid incomplete results
            tmp_file_name = arg_hash + '_tmp.parquet'
            tmp_file_path = os.path.join(self.cache_dir(), tmp_file_name)
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            types = {'Incident Zip': pd.StringDtype(), 'Complaint Type': pd.StringDtype(), 'City': pd.StringDtype()}
            chunks_iter = pd.read_csv(requests_file, parse_dates=['Created Date'], usecols=column_names, dtype=types,
                                      index_col=False, iterator=True, chunksize=1 << 18)
            for chunk in chunks_iter:
                filtered_chunk = chunk[(chunk['Created Date'] >= from_date) & (chunk['Created Date'] <= to_date)]
                filtered_chunk = filtered_chunk.dropna(subset=['Incident Zip'])
                table = pa.Table.from_pandas(df=filtered_chunk)
                if not pqwriter:
                    pqwriter = pa.parquet.ParquetWriter(tmp_file_path, table.schema)
                pqwriter.write_table(table)

            # close the parquet writer
            if pqwriter:
                pqwriter.close()
            os.rename(tmp_file_path, file_path)

        return file_path
