import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet
from dateutil.relativedelta import relativedelta

from sttn import network
from .data_provider import DataProvider

TAXI_ZONE_SHAPE_URL = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip'


class NycTaxiDataProvider(DataProvider):
    """New York Taxi data provider, builds a directed graph (network) where nodes represent taxi zones and edges
    represent taxi trips for a given month. Yellow and green taxi trip records include fields capturing
    pick-up dates/times, pick-up and drop-off locations, itemized fares, and driver-reported passenger counts.
    The data covers trips within New York City and Newark Airport"""

    @staticmethod
    def build_network(taxi_trips, taxi_zones) -> network.SpatioTemporalNetwork:
        edges = taxi_trips.rename(
            columns={'PULocationID': 'origin', 'DOLocationID': 'destination', 'tpep_pickup_datetime': 'time'})
        edges_casted = edges.astype({'origin': 'int64', 'destination': 'int64'})
        taxi_zones = taxi_zones.rename(columns={'OBJECTID': 'id'}).astype({'id': 'int64'})
        taxi_zones = taxi_zones.set_index('id')
        return network.SpatioTemporalNetwork(nodes=taxi_zones, edges=edges_casted)

    def get_data(self, taxi_type: str, month: str) -> network.SpatioTemporalNetwork:
        """
        Retrieves New York City taxi data

        Args:
            taxi_type (str): String taxi type one of the following values:
                'yellow' - Yellow taxi
                'green' - Green taxi
                'fhv' - For-Hire vehicles
                'fhvhv' - High-volume for-hire vehicles
            month (str): A string with year and month in the "YYYY-MM" format.
                The dataset is available from 2011 to 2023.

        Returns:
            SpatioTemporalNetwork: An STTN network where nodes represent New York City taxi zones
                and edges represent individual trips.

            The nodes dataframe contains the following:
                index:
                    'id' (int64) - index column, represents taxi zone id
                columns:
                    'borough' (str) - taxi zone borough
                    'zone' (str) - taxi zone name
                    'geometry' (shape) - shape object for the zone

            The edges dataframe contains the following columns:
                'origin' (int64) - trip origin taxi zone id
                'destination' (int64) - trip destination taxi zone id
                'time' (datetime64[ns]) - trip start time
                'passenger_count' (int64) - number of passengers
                'fare_amount' (float64) - trip fare in USD (can be negative, filter out if not stated otherwise)
        """
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
    """New York City 311 request data provider, builds a network where nodes represent zip codes
     and every edge represents a 311 incident where origin and destination point to the node
     where the incident happened."""

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
        return network.SpatioTemporalNetwork(nodes=node_labels, edges=requests)

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


class RestaurantInspectionDataProvider(DataProvider):
    """New York City Restaurant Inspections data provider builds a bipartite network where the 
     first type of nodes represents restaurants (establishments), the second type of nodes 
     represents types of inspection, and every edge represents an inspection details where the
     origin is the inspection type, and destination is the restaurant where the inspection happened."""

    @staticmethod
    def get_data(self, from_date: str, to_date: str) -> network.SpatioTemporalNetwork:
        """
        Retrieves New York City Restaurant Inspections data

        Args:
            from_date (str): A string with year and month and in the "YYYY-MM-DD" format.
            Specifies the start date to get data from
            
            to_date (str): A string with year, month, and day in the "YYYY-MM-DD" format.
            Specifies the end date to get data from
                

        Returns:
            SpatioTemporalNetwork: An STTN network where the first type of nodes represents restaurants (establishments),
                second type of nodes represents types of inspection, and edges represent inspections.
            
            The nodes dataframe contains the following columns:
                'ID' (str) - represents either restaurant id, also referred as 'camis', or an inspection type which 
                             is a combination of the inspection program and the type of inspection performed
                'IS_RESTAURANT' (bool) - boolean column representing whether the specific row is 
                                         referring to a restaurant (True) or an inspection type (False) 
                'NAME' (str) - restaurant name, also referred as DBA ('doing business as')
                'BORO' (str) - restaurant location borough
                'BUILDING' (str) - restaurant location building code
                'STREET' (str) - restaurant location street name
                'ZIPCODE' (float32) - restaurant location zip code
                'PHONE' (float32) - restaurant phone number
                'CUISINE_DESCRIPTION' (str) - restaurant cuisine description
                'COUNCIL_DISTRICT' (float32) - restaurant council district number
                'CENSUS_TRACT' (float32) - restaurant census tract number
                'COMMUNITY_BOARD' (float32) - restaurant community board number
                'BIN' (float32) - restaurant Building Identification Number
                'BBL' (float32) - restaurant Borough-Block-and-Lot number
                'NTA' (str) - restaurant Neighborhood Tabulation Area code
                'geometry' (geometry) - geometry object for the restaurant location
                

            The edges dataframe contains the following columns:
                'ORIGIN' (str) - the inspection type executed
                'DESTINATION' (str) - inspected restaurant id
                'INSPECTION_DATE' (datetime64[ns]) - inspection date
                'ACTION' (str) - action associated with each restaurant inspection
                'VIOLATION_CODE' (str) - violation code associated with a restaurant inspection
                'VIOLATION_DESCRIPTION' (str) - violation description associated with an
                                                establishment (restaurant) inspection
                'CRITICAL_FLAG' (str) - indicator of critical violation
                'SCORE' (float32) - total score for a particular inspection
                'GRADE' (str) - grade associated with the inspection
                'GRADE_DATE' (datetime64[ns]) - the date when the grade was issued to the establishment (restaurant)
                'RECORD_DATE' (datetime64[ns]) - the date when the record was added to the dataset
       
        """
        url = 'https://data.cityofnewyork.us/api/views/43nn-pn8j/rows.csv?accessType=DOWNLOAD'
        rest_ins_data = self.cache_file(url)
        df = pd.read_csv(rest_ins_data,
                         parse_dates=['INSPECTION DATE', 'RECORD DATE', 'GRADE DATE'],
                         infer_datetime_format=True)

        df = format_data(df, from_date, to_date)

        return build_network(df)

    def format_data(self, df: pd.DataFrame, from_date: str, to_date: str):
        """Format and clean the data."""

        # Drop values with unknown latitude and longitude
        df = df.loc[~((df['Latitude'].isna()) | (df['Longitude'].isna())
                      | (df['Latitude'].round() == 0) | (df['Longitude'].round() == 0))]
        df = df.drop(columns=['Location Point1'])

        # Drop values where INSPECTION DATE == 01.01.1900 (which means they didn't have an inspection yet)
        df = df.loc[df['INSPECTION DATE'] != '01.01.1900']

        # Filter by INSPECTION DATE
        df = df.loc[(df['INSPECTION DATE'] >= from_date) & (df['INSPECTION DATE'] <= to_date)]

        # Change 0's and empty phone numbers to NaNs in BORO and PHONE
        df.loc[:, 'BORO'] = df['BORO'].map({0: np.nan})
        df.loc[:, 'PHONE'] = df['PHONE'].map({'__________': np.nan}).astype(np.float32)

        # Uppercase column names and replace spaces with underscore
        df.columns = df.columns.str.upper().str.strip().str.replace(' ', '_')

        # Change the names of some columns to more straightforward ones
        df = df.rename(columns={'CAMIS': 'ID',  # 'id',
                                'DBA': 'NAME'})
        return df

    def build_network(self, df: pd.DataFrame) -> network.SpatioTemporalNetwork:
        """Build an STTN network from filtered data."""

        # Separate features for edges and two types of nodes
        rest_node_features = ['ID', 'NAME', 'BORO', 'BUILDING', 'STREET', 'ZIPCODE', 'PHONE',
                              'CUISINE_DESCRIPTION', 'LATITUDE', 'LONGITUDE', 'COMMUNITY_BOARD',
                              'COUNCIL_DISTRICT', 'CENSUS_TRACT', 'BIN', 'BBL', 'NTA']
        insp_node_features = ['INSPECTION_TYPE']
        edge_features = ['ID', 'INSPECTION_TYPE', 'INSPECTION_DATE', 'ACTION', 'VIOLATION_CODE',
                         'VIOLATION_DESCRIPTION',
                         'CRITICAL_FLAG', 'SCORE', 'GRADE', 'GRADE_DATE', 'RECORD_DATE',
                         ]

        # Geodf with restaurants
        gdf_rest_nodes = df.loc[:, rest_node_features]
        gdf_rest_nodes.loc[:, 'IS_REST'] = True
        gdf_rest_nodes = gpd.GeoDataFrame(
            gdf_rest_nodes.drop(columns=['LATITUDE', 'LONGITUDE']),
            geometry=gpd.points_from_xy(x=gdf_rest_nodes['LATITUDE'], y=gdf_rest_nodes['LONGITUDE'])
        )
        gdf_rest_nodes.loc[:, 'ID'] = gdf_rest_nodes['ID'].astype(str)

        # df with inspection types
        df_insp_nodes = df.loc[:, insp_node_features]
        df_insp_nodes = df_insp_nodes.rename(columns={'INSPECTION_TYPE': 'ID'})

        # Geodf with all nodes
        gdf_nodes = pd.concat([gdf_rest_nodes, df_insp_nodes], join='outer', ignore_index=True)

        # Fill NaNs in IS_REST with False
        gdf_nodes.loc[:, 'IS_REST'] = gdf_nodes['IS_REST'].fillna(False)

        # Set ID as index
        gdf_nodes = gdf_nodes.set_index('ID')

        # df with edges
        df_edges = df.loc[:, edge_features]
        df_edges.loc[:, 'ORIGIN'] = df_edges['INSPECTION_TYPE']
        df_edges.loc[:, 'DESTINATION'] = df_edges['ID'].astype(str)
        df_edges = df_edges.drop(columns=['ID', 'INSPECTION_TYPE'])

        # Downcast float64 to float32 to reduce memory footprint
        for col in gdf_nodes.select_dtypes(include=['float64']).columns:
            gdf_nodes[col] = gdf_nodes[col].astype('float32')
        for col in df_edges.select_dtypes(include=['float64']).columns:
            df_edges[col] = df_edges[col].astype('float32')

        return network.SpatioTemporalNetwork(nodes=gdf_nodes,
                                             edges=df_edges,
                                             origin='ORIGIN',
                                             destination='DESTINATION',
                                             node_id='ID')
