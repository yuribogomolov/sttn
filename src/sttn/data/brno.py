import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from sttn import constants
from sttn import network
from typing import Optional
from .data_provider import DataProvider

PRAGUE_DISTRICT_CODES = ['CZ0101', 'CZ0102', 'CZ0103', 'CZ0104', 'CZ0105', 'CZ0106',
                         'CZ0107', 'CZ0108', 'CZ0109', 'CZ010A']
PRAGUE_CODE = 'CZ0100'
HEALTHCARE_DATA_VAR = "STTN_CZ_HEALTH_DATA"


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
        trips_filtered = trips_filtered[
            (trips_filtered['start_level'] == start_level) & (trips_filtered['cil_level'] == end_level)]
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
        edge_dfs = [JourneyDataProvider.read_trip_file(f"{journey_csv_folder}/{fname}", start_level, end_level) for
                    fname in edge_files if
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
    def get_data(journey_csv_folder: str, shapefile_name: str, start_level: int = 1,
                 end_level: int = 1) -> network.SpatioTemporalNetwork:
        nodes = JourneyDataProvider.read_nodes(shapefile_name)
        edges = JourneyDataProvider.read_edges(journey_csv_folder, start_level=start_level, end_level=end_level)
        sttn_network = network.SpatioTemporalNetwork(nodes=nodes, edges=edges)
        return sttn_network


class HealthcareDataProvider(DataProvider):
    """
    Czech Republic healthcare data provider. Builds a network where every node represents a Czech Republic district
    and every edge contains aggregated patients information where the origin is the residence of patients, and 
    the destination is a healthcare facility district they commuted to. The provider gives information on how many visits
    happened from origin to destination districts in a specific month of the year for different specializations and ICD-10
    disease categories. The dataset covers 13 years of monthly data from 2010-01 to 2022-12. 
    """
    @staticmethod
    def get_data(year: Optional[str], data_folder: Optional[str] = None) -> network.SpatioTemporalNetwork:
        """
        Retrieves Czech Republic Healthcare data. 
        Args:
            year (year): 4-digit year of healthcare data

        Returns:
            SpatioTemporalNetwork: An STTN network where nodes represent the Czech Republic districts 
                (also called 'okres'), and edges represent the patients' treatment mobility and details.

            The nodes dataframe contains the following columns:
                'id' (str) - district LAU code, Pandas index column
                'name' (str) - district names
                'geometry' (shape) - shape object for the district

            The edges dataframe contains the following columns:
                'origin' (str) - patients' residence district id
                'destination' (str) - patients' medical facility district id
                'year_visit' (int) - patients' year of visit
                'month_visit' (int) - patients' month of visit
                'specialization' (int) - patients' treatment medical branch specialization number
                'icd10_category' (int) -patients' disease ICD10 category number
                'number_of_visits' (int) - total number of patients who visited from a residence district to a healthcare facility,
                                           grouped by medical specialization and disease category

        
        """
        folder_env = os.getenv(HEALTHCARE_DATA_VAR)
        folder = data_folder or folder_env
        if folder is None:
            raise ValueError(f"data_folder provider argument and {HEALTHCARE_DATA_VAR} variable are not set")
      
        nodes = HealthcareDataProvider.read_nodes(f"{folder}/lau1.geojson")
        edges = HealthcareDataProvider.read_edges(year=int(year), data_file=f"{folder}/healthcare.parquet")

        ids_to_keep = nodes.index
        edges = edges[
            edges.origin.isin(ids_to_keep) & edges.destination.isin(ids_to_keep)]

        nodes_to_keep = np.union1d(edges.origin.unique(), edges.destination.unique())
        nodes = nodes[nodes.index.isin(nodes_to_keep)]
        sttn_network = network.SpatioTemporalNetwork(nodes=nodes, edges=edges)
        return sttn_network

    @staticmethod
    def read_edges(year: Optional[int], data_file: str) -> pd.DataFrame:
        dataset = ds.dataset(data_file, format='parquet')
        if not year is None:
            table = dataset.to_table(filter=ds.field('year_visit') == year)
        else:
            table = dataset.to_table()

        edges = table.to_pandas()
        columns_to_rename = {"okres_residence": constants.ORIGIN,
                             "okres_servise": constants.DESTINATION, }
        renamed = edges.rename(columns=columns_to_rename)
        # map Prague district codes to LAU1
        prague_dict = {key: PRAGUE_CODE for key in PRAGUE_DISTRICT_CODES}
        renamed[constants.ORIGIN] = renamed[constants.ORIGIN].replace(prague_dict)
        renamed[constants.DESTINATION] = renamed[constants.DESTINATION].replace(prague_dict)
        return renamed

    @staticmethod
    def read_nodes(geojson_file: str) -> gpd.GeoDataFrame:
        nodes = gpd.read_file(geojson_file)
        columns_to_keep = ["lau", "name", "geometry"]
        nodes = nodes[columns_to_keep]
        column_names = {"lau": constants.NODE_ID}
        renamed = nodes.rename(columns=column_names)
        converted = renamed.set_index(constants.NODE_ID).to_crs(epsg=4326)
        return converted
