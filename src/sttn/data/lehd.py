import pandas as pd

from sttn import network
from .data_provider import DataProvider
from . import census


class OriginDestinationEmploymentDataProvider(DataProvider):
    """
    Longitudinal Employer-Household Dynamics Origin-Destination Employment Statistics
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
        filtered_edges = aggregated_edges[aggregated_edges.origin.isin(ids_to_keep) & aggregated_edges.destination.isin(ids_to_keep)]
        return network.SpatioTemporalNetwork(nodes=tracts_with_zip, edges=filtered_edges)

    def get_data(self, state: str, year: int, part: str = 'main', job_type: int = 0) -> network.SpatioTemporalNetwork:
        self._cache(state=state, year=year, part=part, job_type=job_type)
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
