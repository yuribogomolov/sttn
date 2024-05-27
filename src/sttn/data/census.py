import pandas as pd
import geopandas as gpd
import numpy as np

def get_state_codes() -> pd.DataFrame:
    url = 'https://www2.census.gov/geo/docs/reference/state.txt'
    columns_to_rename = {'STATE': 'fips_code', 'STUSAB': 'usps_code', 'STATE_NAME': 'state', 'STATENS': 'gnisid'}
    data = pd.read_csv(url, delimiter='|').rename(columns=columns_to_rename)
    return data


def get_tract_geo(state: str, year: int) -> gpd.GeoDataFrame:
    state_codes = get_state_codes()
    fips_code = state_codes[state_codes.usps_code == state.upper()].iloc[0].fips_code
    url = 'https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips:02d}_tract.zip'.format(
        year=year, state_fips=fips_code)
    df = gpd.read_file(url)
    df.GEOID = df.GEOID.astype(np.int64)  # np.int64 to fix windows C long issue
    return df
