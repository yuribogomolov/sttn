import pandas as pd
import geopandas as gpd
import os

from sttn import network
from sttn import constants
from .data_provider import DataProvider


class JourneyDataProvider(DataProvider):
    """Brno journey data provider. Builds a network where every node represents a city district
    and every edge contains the number of people who commuted from the origin to the destination
    district within a given hour. The dataset covers one week of data from 7th to 13th October 2019.
    The data is based on the passive communication between mobile phones and cell towers,
    and it covers only one mobile phone company.
    """

    @staticmethod
    def read_trip_file(file_path: str, start_level: int, end_level: int) -> pd.DataFrame:
        trips = pd.read_csv(file_path)
        # keep only trips with known origin and destination
        trips_filtered = trips[~trips['start_kod'].isnull() & ~trips['cil_kod'].isnull()]
        # filter Brno trip levels (urban/suburban)
        trips_filtered = trips_filtered[(trips_filtered['start_level'] == start_level) & (trips_filtered['cil_level'] == end_level)]
        trips_filtered = trips_filtered.astype({"start_kod": int, "cil_kod": int})
        columns_to_rename = {"start_cas": "start_hr", "cil_cas": "end_hr", "start_kod": constants.ORIGIN,
                             "cil_kod": constants.DESTINATION, "pocet": "count", "cz": "is_czech",
                             "pocet_kalibrovano": "count_adjusted"}
        renamed = trips_filtered.rename(columns=columns_to_rename)

        if len(renamed.day.value_counts()) != 1:
            raise ValueError(f"The day file contains data from multiple days: {renamed.day_.value_counts()}")

        day_delta = renamed.day.iloc[0]
        # observation start date is 7.10.2019
        renamed['start_ts'] = renamed.start_hr.apply(
            lambda hr: pd.Timestamp(year=2019, month=10, day=(7 + day_delta), hour=hr))
        renamed['end_ts'] = renamed.end_hr.apply(
            lambda hr: pd.Timestamp(year=2019, month=10, day=(7 + day_delta), hour=hr))
        columns_to_keep = [constants.ORIGIN, constants.DESTINATION, "is_czech", "count", "count_adjusted", "start_ts",
                           "end_ts"]
        return renamed[columns_to_keep]

    @staticmethod
    def read_edges(journey_csv_folder: str, start_level: int, end_level: int) -> pd.DataFrame:
        edge_files = os.listdir(journey_csv_folder)
        edge_dfs = [JourneyDataProvider.read_trip_file(f"{journey_csv_folder}/{fname}", start_level, end_level) for fname in edge_files if
                    fname.endswith(".csv")]
        return pd.concat(edge_dfs)

    @staticmethod
    def read_nodes(shapefile_file_name: str) -> gpd.GeoDataFrame:
        nodes = gpd.read_file(shapefile_file_name)
        columns_to_keep = ["KOD_KU", "NAZ_KU", "KOD_ZUJ", "NAZ_ZUJ", "geometry"]
        nodes = nodes[columns_to_keep]
        column_names = {"KOD_KU": constants.NODE_ID, "NAZ_KU": "block", "KOD_ZUJ": "district_code",
                        "NAZ_ZUJ": "district"}
        renamed = nodes.rename(columns=column_names)
        types = {constants.NODE_ID: int, 'district_code': int}
        converted = renamed.astype(types).set_index(constants.NODE_ID).to_crs(epsg=4326)
        return converted

    @staticmethod
    def get_data(journey_csv_folder: str, shapefile_name: str, start_level: int = 1, end_level: int = 1) -> network.SpatioTemporalNetwork:
        nodes = JourneyDataProvider.read_nodes(shapefile_name)
        edges = JourneyDataProvider.read_edges(journey_csv_folder, start_level=start_level, end_level=end_level)
        sttn_network = network.SpatioTemporalNetwork(nodes=nodes, edges=edges)
        return sttn_network
