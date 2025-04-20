from io import StringIO

import pandas as pd
import requests

HEADERS = {'User-Agent': 'Mozilla/5.0'}


def get_state_codes() -> pd.DataFrame:
    url = 'https://www2.census.gov/geo/docs/reference/state.txt'
    response = requests.get(url, headers=HEADERS)
    columns_to_rename = {'STATE': 'fips_code', 'STUSAB': 'usps_code', 'STATE_NAME': 'state', 'STATENS': 'gnisid'}
    data = pd.read_csv(StringIO(response.text), delimiter='|').rename(columns=columns_to_rename)
    return data


def get_tract_geo_url(state: str, year: int) -> str:
    state_codes = get_state_codes()
    fips_code = state_codes[state_codes.usps_code == state.upper()].iloc[0].fips_code
    url = 'https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips:02d}_tract.zip'.format(
        year=year, state_fips=fips_code)
    return url
