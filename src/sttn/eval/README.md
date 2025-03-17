Below, we provide a dataset supplementary text file with the description of data providers, awareness metrics and how to treat certain categories and situations represented in the queries.

# Used Data Providers
## Origin-Destination Employment Data Provider
### Description
Longitudinal Employer-Household Dynamics Origin-Destination Employment Statistics data provider. Every node represents a census tract, and every edge contains the number of people living in the origin and commuting to the destination tract.

[Data Specification](https://lehd.ces.census.gov/data/lodes/LODES7/LODESTechDoc7.5.pdf)

### Data Retrieval
Retrieves LEHD Origin-Destination Employment Statistics for a given state and year.

**Arguments**
- **state (str)**: Lowercase, 2-letter postal code for a chosen state (e.g. 'md').
- **year (str)**: Year of job data, starting mostly from 2002 and ending in 2019 (different states have different available data horizons).

**Returns**
- **SpatioTemporalNetwork**: An STTN network where nodes represent census tracts and edges represent yearly aggregated employment statistics for people who live in the origin tract and work in the destination tract area.

### Data

#### Nodes
The nodes dataframe contains the following:
- **index**:
    - `id` (int64): Index column, represents census tract ids.
- **columns**:
    - `county` (str): Official county name of the tract or its official equivalent (e.g. "Richmond County, NY", "Lafayette Parish, LA").
    - `zip` (int): Zip code of the tract.
    - `geometry` (shape): Shape object for the tract.

#### Edges
The edges dataframe contains the following columns:
- `origin` (int64): Origin Census tract id.
- `destination` (int64): Destination Census tract id.
- `S000` (int32): Total number of jobs.
- `SA01` (int32): Number of jobs of workers age 29 or younger.
- `SA02` (int32): Number of jobs for workers age 30 to 54.
- `SA03` (int32): Number of jobs for workers age 55 or older.
- `SE01` (int32): Number of jobs with earnings $1250/month or less.
- `SE02` (int32): Number of jobs with earnings $1251/month to $3333/month.
- `SE03` (int32): Number of jobs with earnings greater than $3333/month.
- `SI01` (int32): Number of jobs in Goods Producing industry sectors.
- `SI02` (int32): Number of jobs in Trade, Transportation, and Utilities industry sectors.
- `SI03` (int32): Number of jobs in All Other Services industry sectors.

## New York City Taxi Data Provider

### Description
New York Taxi data provider builds a network where nodes represent taxi zones and edges represent taxi trips for a given month. Yellow taxi trip records include fields capturing pick-up dates/times, pick-up and drop-off locations, itemized fares, and driver-reported passenger counts.

### Data Retrieval
Retrieves New York City taxi data.

**Arguments**
- **taxi_type (str)**: String taxi type, one of the following values:
    - 'yellow': Yellow taxi.
- **month (str)**: A string with year and month in the "YYYY-MM" format. The dataset is available from **2009** to **2023**.

**Returns**
- **SpatioTemporalNetwork**: An STTN network where nodes represent New York City taxi zones and edges represent individual trips.

### Data

#### Nodes
The nodes GeoPandas dataframe contains the following:
- **index**:
    - `id` (int64): Index, represents taxi zone id.
- **columns**:
    - `borough` (str): Taxi zone borough.
    - `zone` (str): Taxi zone name.
    - `geometry` (shape): Shape object for the zone.

#### Edges
The edges GeoPandas dataframe contains the following columns:
- `origin` (int64): Trip origin taxi zone id.
- `destination` (int64): Trip destination taxi zone id.
- `time` (datetime64[ns]): Trip start time.
- `passenger_count` (int64): Number of passengers.
- `fare_amount` (float64): Trip fare in USD.

# Explaining the *awareness* features
Below, we provide the prompt that was given to the LLM to recognize the awareness features that the code should account for when working with spatio-temporal data.
## Geospatial Awareness
Geospatial awareness requires the code to account for different geospatial features and peculiarities like:

1. **Naming overlap**:
    - Didn't pick the wrong entity (e.g., street or city with a similar/same name) instead of the requested entity (e.g., county or district) with the same name.
2. **Interchangeability and proper unit use**:
    - If our data provider has only official administrative units (e.g., counties) and the query asks for the same interchangeable entity (e.g., borough), the model should pick the right entity that is available in the data provider.
3. **Properties of geographic entities (if requested)**:
    - Each US state has a capital city, cities might have a specified downtown, etc.
4. **An accurate relational understanding of hierarchical and nested geographic entities**:
    - Uses all corresponding and available counties/census tracts/districts/zones to represent a bigger entity like state, city, etc., the relationship between different geographic levels.
5. **Ability to handle various regional features specific to different countries, zones, etc.**:
    - "Okresy" in Czechia, "departments" in France, etc.
6. **Accurate spatial relationships between entities**:
    - Identifying entities within a certain distance, to any cardinal direction, or in between other geographic entities.

## Temporal Awareness
Temporal awareness requires the code to account for different temporal features like:

1. **Temporal features specific to different contexts or regions**:
    - Different public holidays, cultural calendars, etc., in different countries or regions.
2. **Proper filtering and aggregation of temporal features**:
    - Ensuring that all relevant temporal units are considered (e.g., not to miss hours/minutes while filtering by time or summing up daily data to get monthly totals).
3. **Absence of temporal inconsistencies**:
    - Overlapping time periods, mismatched time zones, not accounting for leap years.

# Ensuring Consistency
Below, we provide a description of how to treat certain categories and situations and how we expected the model to treat such tasks to ensure consistency in the data processing.
## Graph Analysis
We have **directed graphs** in both providers. Therefore, any graph-related tasks such as `network_density` and `centrality_degree` must account for this unless stated otherwise.

## Spatial Queries
When handling queries with **cardinal directions** (e.g., "Find commutes from county A to the points north of county A"), use the memorised geographical knowledge or **centroids** of the entire county. For example, *to find counties west of county A*:
1. Find the centroid of the **entire** county.
2. Look for census tracts with centroids where the **x**-coordinate is smaller (to the left) than the **x**-coordinate of the county centroid.
3. **Filter out** the destination census tracts of the origin county’s (county A) census tracts that might be located left of the county’s centroid.

## Community Detection
For `community_detection` tasks, always use the internal **[Combo](https://github.com/Alexander-Belyi/Combo)** algorithm with `random_state=0` (set as the default value).

## Node Filtering
Be cautious when using the `.filter_nodes()` method. For example, in the query *"Find workers commuting from King’s County to other counties,"* using `.filter_nodes("King’s County")` will leave **only** King’s County’s nodes and remove edges leading to other counties, resulting in incorrect data.

## Network Density and `filter_edges` method
When calculating **network density**, ensure to **filter nodes after filtering edges** with `filter_edges`. The `filter_edges` method **does not** delete empty nodes from the graph, which can impact the density formula. For instance, when looking for density in a graph where something originates and ends in the same county/borough/tract, the model might use only 
```python
filter_edges(edges['origin']=='same place' | edges['destination']=='same place')
```
 without `filter_nodes`, resulting in a graph with many irrelevant nodes without edges, thus **affecting the network density calculation**.

## Negative values filtering
If there's a numeric column that should be **positive** or zero (like `fare_price` in *NYC Taxi*), please ensure that the code **filters out negative values**.

## Pandas DateTime Handling
The model might misuse Pandas' datetime string filtering. Example query: *"Get taxi trips from 30 till 31 December 2018 included."* Model's code:
```python
sttn_network.edges[
    (sttn_network.edges['time'] >= '30.12.2018') & 
    (sttn_network.edges['time'] <= '31.12.2018')
]
```
By executing the previous line, 23 hours would be lost  from the 31st of December because Pandas assumes that strings like `'31.12.2018'` are actually `'31.12.2018 00:00:00'`. To avoid this, ensure that the datetime filtering includes the entire day.
