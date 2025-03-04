import pandas as pd

from sttn import network
from . import census
from .data_provider import DataProvider


class OriginDestinationEmploymentDataProvider(DataProvider):
    """Longitudinal Employer-Household Dynamics Origin-Destination Employment Statistics
    data provider. Every node represents a census tract and every edge contains the number
    of people living in the origin and commuting to the destination tract.
    The data covers all US states.

    Data spec: https://lehd.ces.census.gov/data/lodes/LODES7/LODESTechDoc7.5.pdf
    """

    def build_network(self, state: str, year: int) -> network.SpatioTemporalNetwork:
        od_columns = ['w_geocode', 'h_geocode', 'S000', 'SA01', 'SA02', 'SA03', 'SE01', 'SE02', 'SE03', 'SI01', 'SI02',
                      'SI03']
        xwalk_columns = ['tabblk2010', 'trct', 'ctyname', 'zcta']
        od_data = pd.read_csv(self.od_fname, compression='gzip', usecols=od_columns)
        xwalk_data = pd.read_csv(self.xwalk_fname, compression='gzip', index_col='tabblk2010', usecols=xwalk_columns)
        # map census Block Codes to Census Tract codes
        home_joined = xwalk_data[['trct']].merge(od_data, left_index=True, right_on='h_geocode', how='inner').rename(
            columns={'trct': 'origin'})
        joined = xwalk_data[['trct']].merge(home_joined, left_index=True, right_on='w_geocode', how='inner').rename(
            columns={'trct': 'destination'})
        cleaned = joined.drop(['w_geocode', 'h_geocode'], axis=1)
        aggregated_edges = cleaned.groupby(['origin', 'destination']).sum().reset_index()

        rename_map = {'trct': 'id', 'ctyname': 'county', 'zcta': 'zip'}
        renamed = xwalk_data.reset_index()[['trct', 'ctyname', 'zcta']].rename(columns=rename_map)
        # 99999 is used for unknown zip codes
        tract_to_zip = renamed[renamed.zip != 99999].groupby('id').first()
        tract_to_zip['zip'] = tract_to_zip['zip'].astype(str)
        tract_geo_columns = ['GEOID', 'geometry']
        tract_shapes = census.get_tract_geo(state=state, year=year)
        # filter out water-only tracts:
        filtered_tracts = tract_shapes[tract_shapes.ALAND > 0]
        indexed_tracts = filtered_tracts[tract_geo_columns].set_index('GEOID')
        tracts_with_zip = indexed_tracts.merge(tract_to_zip, left_index=True, right_on='id', how='inner')

        # filter out edges for filtered nodes
        ids_to_keep = tracts_with_zip.index
        filtered_edges = aggregated_edges[
            aggregated_edges.origin.isin(ids_to_keep) & aggregated_edges.destination.isin(ids_to_keep)]
        return network.SpatioTemporalNetwork(nodes=tracts_with_zip, edges=filtered_edges)

    def get_data(self, state: str, year: int, part: str = 'main', job_type: int = 0) -> network.SpatioTemporalNetwork:
        """
        Retrieves LEHD Origin-Destination Employment Statistics for a given state and year

        Args:
            state (str): lowercase, 2-letter postal code for a chosen state
            year (str): Year of job data, starting from 2002 to 2019

        Returns:
            SpatioTemporalNetwork: An STTN network where node represent census tracts and edges represent employment
                statistics for people who live in the origin tract and work in the destination tract area.

            The nodes dataframe contains the following:
                index:
                    'id' (int64) - index column, represents census tract ids
                columns:
                    'county' (str) - county of the tract (e.g. "Queens County, NY")
                    'zip' (int) - zip code of the tract
                    'geometry' (shape) - shape object for the tract

            The edges dataframe contains the following columns:
                'origin' (int64) - origin Census tract id
                'destination' (int64) - destination Census tract id
                'S000' (int32) - Num Total number of jobs
                'SA01' (int32) - Num Number of jobs of workers age 29 or younger
                'SA02' (int32) - Num Number of jobs for workers age 30 to 54
                'SA03' (int32) - Num Number of jobs for workers age 55 or older
                'SE01' (int32) - Num Number of jobs with earnings $1250/month or less
                'SE02' (int32) - Num Number of jobs with earnings $1251/month to $3333/month
                'SE03' (int32) - Num Number of jobs with earnings greater than $3333/month
                'SI01' (int32) - Num Number of jobs in Goods Producing industry sectors
                'SI02' (int32) - Num Number of jobs in Trade, Transportation, and Utilities industry sectors
                'SI03' (int32) - Num Number of jobs in All Other Services industry sectors
        """
        self._cache(state=state.lower(), year=year, part=part, job_type=job_type)
        return self.build_network(state=state, year=year)

    def _cache(self, state: str, year: int, part: str = 'main', job_type: int = 0) -> None:
        od_fname = '{state}_od_{part}_JT0{job_type}_{year}.csv.gz'.format(
            state=state, part=part, job_type=job_type, year=year)
        xwalk_fname = '{state}_xwalk.csv.gz'.format(state=state)

        state_url = 'https://lehd.ces.census.gov/data/lodes/LODES7/' + state
        od_url = state_url + '/od/' + od_fname
        self.od_fname = self.cache_file(od_url)

        xwalk_url = state_url + '/' + xwalk_fname
        self.xwalk_fname = self.cache_file(xwalk_url)
