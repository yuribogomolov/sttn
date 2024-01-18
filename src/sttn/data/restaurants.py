import pandas as pd
import geopandas as gpd
import numpy as np

from sttn import network
from .data_provider import DataProvider

class RestaurantInspectionDataProvider(DataProvider):
    """New York City Restaurant Inspections data provider, builds a network where nodes represent restaurants
     and every edge represents an inspection where the origin and destination point to the node
     where the inspection happened."""
  
    @staticmethod
    def get_data(self, from_date, to_date) -> network.SpatioTemporalNetwork:
        """
        Retrieves New York City Restaurant Inspections data

        Args:
            from_date (str): A string with year and month and in the "YYYY-MM-DD" format.
            Specifies the start date to get data from
            
            to_date (str): A string with year, month, and day in the "YYYY-MM-DD" format.
            Specifies the end date to get data from
                

        Returns:
            SpatioTemporalNetwork: An STTN network where nodes represent New York City restaurants
                and edges represent inspections.
        """
        url = 'https://data.cityofnewyork.us/api/views/43nn-pn8j/rows.csv?accessType=DOWNLOAD'
        rest_ins_data = self.cache_file(url)
        df = pd.read_csv(rest_ins_data,
                        parse_dates=['INSPECTION DATE', 'RECORD DATE'],
                        infer_datetime_format=True)

        df = format_data(df, from_date, to_date)

        return build_network(df)


    def format_data(self, df:pd.DataFrame, from_date, to_date):
        """Format and clean the data."""

        # Drop values with unknown latitude and longitude
        df = df[~((df['Latitude'].isna()) | (df['Longitude'].isna()) 
                |(df['Latitude'].round() == 0) | (df['Longitude'].round() == 0))]
        df = df.drop(columns=['Location Point1'])

        # Drop values where INSPECTION DATE == 01.01.1900 (which means they didn't have an inspection yet)
        df = df[df['INSPECTION DATE'] != '01.01.1900']
        # Filter by INSPECTION DATE
        df = df[(df['INSPECTION DATE'] >= from_date) & (df['INSPECTION DATE'] <= to_date)]

        # Change 0's to NaNs in BORO
        df['BORO'] = df['BORO'].map({0:np.nan})

        # Uppercase column names and replace spaces with underscore
        df.columns = df.columns.str.upper().str.strip().str.replace(' ', '_')

        # Change the names of some columns to more straightforward ones
        df = df.rename(columns={'CAMIS': 'id',
                                'DBA': 'NAME'})

        return df


    def build_network(self, df:pd.DataFrame) -> network.SpatioTemporalNetwork:
        """Build an STTN network from filtered data."""

        # Separate features for edges and nodes
        node_features = ['id', 'NAME', 'BORO', 'BUILDING', 'STREET', 'ZIPCODE', 'PHONE',
            'CUISINE_DESCRIPTION', 'LATITUDE', 'LONGITUDE', 'COMMUNITY_BOARD',
            'COUNCIL_DISTRICT', 'CENSUS_TRACT', 'BIN', 'BBL', 'NTA']
        edge_features = ['id', 'INSPECTION_DATE', 'ACTION', 'VIOLATION_CODE', 'VIOLATION_DESCRIPTION',
            'CRITICAL_FLAG', 'SCORE', 'GRADE', 'GRADE_DATE', 'RECORD_DATE',
            'INSPECTION_TYPE']

        # geodf with nodes
        df_nodes = df[node_features]
        gdf_nodes = gpd.GeoDataFrame(
          df_nodes.drop(columns=['LATITUDE','LONGITUDE']),
          geometry=gpd.points_from_xy(x=df_nodes['LATITUDE'], y=df_nodes['LONGITUDE'])
        )
        gdf_nodes = gdf_nodes.set_index('id')  # Set `id` column as index

        # df with edges
        df_edges = df[edge_features]
        df_edges['origin'] = df_edges['destination'] = df['id']

        return network.SpatioTemporalNetwork(nodes=gdf_nodes, edges=df_edges)
