from lib import *
from sqlalchemy.sql import text
import geopandas as gpd
from shapely import wkt

# Prepares and returns a GeoPandas dataframe containing taxi zone geometry
# input: none
# output: GeoPandas dataframe containing taxi zone geometry
def getTaxiGDF():
    gdf = getDF(text('SELECT zone_shape, location_id, location_name, zone FROM taxi_zones'))
    gdf['geometry'] = gdf['zone_shape'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(gdf.dropna(), geometry='geometry')
    gdf['geometry'] = gdf['geometry'].buffer(0)

    return gdf

# Returns trip_count by pu_location_id for a given year and taxi_type
# input: year and taxi_type
# output: dataframe containing trip_count by pu_location_id
def getTripCountByPickup(year, taxi_type):
    cols = ['pu_location_id', 'COUNT(1) as trip_count']
    query = f"""
        {selFrom(cols, year, taxi_type)}
        WHERE (strftime('%Y', pickup_datetime))='{year}'
        GROUP BY pu_location_id
    """

    return getDF(text(query))

# Returns sample data for a given year and taxi_type. 
# input: year, taxi_type, and optionally limit for changing sample size
# output: dataframe containing the sample data
def getSample(year = 2023, taxi_type = YELLOW, limit = 10**3):
    sql = text(f"""
        {selFrom(COMMON_FETCH_COLUMNS, year, taxi_type)}
        where (strftime('%Y', pickup_datetime))='{year}'
        order by random()
        limit {limit}
    """)
    return getDF(sql)

# Returns data for all years for either green or yellow taxi trips
# input: cols to fetch, conditions and taxi_type
# output: dataframe containing specified cols for all years (2020-2023)
def readData(cols = COMMON_FETCH_COLUMNS, conditions = [], taxi_type = YELLOW, grp_ord_by = ''):
    chunks = []
    for year in range(MIN_YEAR, MAX_YEAR + 1):
        all_conditions = commonConditions(year) + conditions
        where_clause = f'''where {' AND '.join(all_conditions)}''' 

        sql = text(f"""
            {selFrom(cols, year, taxi_type)} {where_clause} {grp_ord_by}
        """)

        df = getDF(sql)
        if 'pickup_datetime' in cols:
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

        print(len(df))
        chunks.append(df)

    return pd.concat(chunks, ignore_index=True)

def dataByLocations(pu_location_id, do_location_id, taxi_type = GREEN):
    cols = COMMON_FETCH_COLUMNS + [
        'pu_location',
        'do_location',
        'time_of_day'
    ]
    conditions = [
        f'pu_location_id={pu_location_id}',
        f'do_location_id={do_location_id}'
    ]

    return readData(cols, conditions, taxi_type)

# Prepares and returns data for taxi trips for green or yellow taxi with pickup
#   time between start and end dates.
# input: start_date, end_date, columns to fetch and taxi_type
# output: a dataframe containing taxi trips for specified taxi_type and with 
#   specified columns and pickup time between start and end dates
def getTripsForDateRange(start_date, end_date, columns, taxi_type):
    data_frames = []

    for year in range(start_date.year, end_date.year + 1):
        sd_str = datetime.datetime.strftime(start_date, day_format)
        ed_str = datetime.datetime.strftime(end_date, day_format)

        if year == start_date.year and year == end_date.year:
            # Single-year case
            query = f"""
                {selFrom(columns, year, taxi_type)}
                WHERE pickup_datetime >= '{sd_str}' AND pickup_datetime <= '{ed_str}'
            """
        elif year == start_date.year:
            # First year in range, so filter from startDate to the end of the year
            query = f"""
                {selFrom(columns, year, taxi_type)}
                WHERE pickup_datetime >= '{sd_str}' AND pickup_datetime <= '{year}-12-31 23:59:59'
            """
        elif year == end_date.year:
            # Last year in range, so filter from the beginning of the year to endDate
            query = f"""
                {selFrom(columns, year, taxi_type)}
                WHERE pickup_datetime >= '{year}-01-01' AND pickup_datetime <= '{ed_str}'
            """
        else:
            # Intermediate years (full-year range)
            query = f"""
                {selFrom(columns, year, taxi_type)}
                WHERE pickup_datetime >= '{year}-01-01' AND pickup_datetime <= '{year}-12-31 23:59:59'
            """
        
        # Execute the query and store the result in a DataFrame
        df = getDF(query)
        data_frames.append(df)

    # Concatenate all DataFrames from the list into a single DataFrame
    result_df = pd.concat(data_frames, ignore_index=True)

    return result_df

# Returns Location Name vs Location ID for the dashboard
# input: n/a
# output: a dictionary of location name vs location id
def getLocationOptions():
    df = getDF(text('''
                    SELECT location_id, (location_name || ', ' || zone) as f_location_name FROM taxi_zones
                    ORDER BY f_location_name ASC
                    '''))
    return dict(zip(df["f_location_name"], df["location_id"]))


# # write a function to store result of a sql, a dataframe as a cache with given name
def cacheSql(sql_obj, name, taxi_type):
    # get dataframe from sql
    df = readData(cols=sql_obj['cols'], 
                  conditions = sql_obj['conditions'], 
                  grp_ord_by = sql_obj['grp_ord_by'], 
                  taxi_type=taxi_type)
    
    # store dataframe as cache
    storeObject(df, f'{taxi_type}_{name}', 'cache')

def getCachedSql(name, taxi_type):
    return loadObject(f'{taxi_type}_{name}', 'cache')

# write sql for weekly data
# name: daily_avgs
DAILY_AVGS_CACHE = 'daily_avgs'
DAILY_AVGS = {
    'cols': [
        'DATE(pickup_datetime) AS day',
        f'''
            CASE STRFTIME('%w', pickup_datetime)
                WHEN '0' THEN 'Sunday'
                WHEN '1' THEN 'Monday'
                WHEN '2' THEN 'Tuesday'
                WHEN '3' THEN 'Wednesday'
                WHEN '4' THEN 'Thursday'
                WHEN '5' THEN 'Friday'
                WHEN '6' THEN 'Saturday'
            END AS day_of_week
        ''',
        'AVG(total_amount) AS avg_total_amount',
        'AVG(fare_amount) AS avg_fare_amount',
        'AVG(tip_amount) AS avg_tip_amount',
        'SUM(passenger_count) AS sum_passenger_count',
        'AVG(trip_distance) as avg_trip_distance',
        'AVG(tolls_amount) as avg_tolls_amount',
        f'''
            AVG(unixepoch(dropoff_datetime)-unixepoch(pickup_datetime)) / 60 as avg_trip_duration
        ''',
        'COUNT(1) AS trip_count',
        ],
    'conditions': [],
    'grp_ord_by': f'GROUP BY day ORDER BY day', 
}


# strftime('%H', pickup_datetime) AS hour_of_day

# write sql for hourly data
# name: hourly_avgs
HOURLY_AVGS_CACHE = 'hourly_avgs'
HOURLY_AVGS = {
    'cols': [
        f'''
            strftime('{hour_format}', pickup_datetime) AS hour
        ''',
        f'''
            CASE
                WHEN CAST(strftime('%H', pickup_datetime) AS INTEGER) BETWEEN 0 AND 5 THEN 'Night'
                WHEN CAST(strftime('%H', pickup_datetime) AS INTEGER) BETWEEN 6 AND 11 THEN 'Morning'
                WHEN CAST(strftime('%H', pickup_datetime) AS INTEGER) BETWEEN 12 AND 17 THEN 'Afternoon'
                WHEN CAST(strftime('%H', pickup_datetime) AS INTEGER) BETWEEN 18 AND 23 THEN 'Evening'
            END AS time_of_day
        ''',
        'AVG(total_amount) AS avg_total_amount',
        'AVG(fare_amount) AS avg_fare_amount',
        'AVG(tip_amount) AS avg_tip_amount',
        'SUM(passenger_count) AS sum_passenger_count',
        'AVG(trip_distance) as avg_trip_distance',
        'AVG(tolls_amount) as avg_tolls_amount',
        f'''
            AVG(unixepoch(dropoff_datetime)-unixepoch(pickup_datetime)) / 60 as avg_trip_duration
        ''',
        'COUNT(1) AS trip_count',
    ],
    'conditions': [],
    'grp_ord_by': f'GROUP BY hour ORDER BY hour'
}

# avg fields by pickup location and time of day
PU_TIME_AVGS_CACHE = 'pu_time_avgs'
PU_TIME_AVGS = {
    'cols': [
        'pu_location_id','pu_location',
        f'''
            CASE
                WHEN CAST(strftime('%H', pickup_datetime) AS INTEGER) BETWEEN 0 AND 5 THEN 'Night'
                WHEN CAST(strftime('%H', pickup_datetime) AS INTEGER) BETWEEN 6 AND 11 THEN 'Morning'
                WHEN CAST(strftime('%H', pickup_datetime) AS INTEGER) BETWEEN 12 AND 17 THEN 'Afternoon'
                WHEN CAST(strftime('%H', pickup_datetime) AS INTEGER) BETWEEN 18 AND 23 THEN 'Evening'
            END AS time_of_day
        ''',
        'AVG(total_amount) AS avg_total_amount',
        'AVG(fare_amount) AS avg_fare_amount',
        'AVG(tip_amount) AS avg_tip_amount',
        'SUM(passenger_count) AS sum_passenger_count',
        'AVG(trip_distance) as avg_trip_distance',
        'AVG(tolls_amount) as avg_tolls_amount',
        f'''
            AVG(unixepoch(dropoff_datetime)-unixepoch(pickup_datetime)) / 60 as avg_trip_duration
        ''',
        'COUNT(1) AS trip_count',
    ],
    'conditions': ['total_amount > 0'],
    'grp_ord_by': f'GROUP BY pu_location_id, time_of_day ORDER BY pu_location_id, time_of_day'
}

PU_DO_TIME_AVGS_CACHE = 'pu_do_time_avgs'
PU_DO_TIME_AVGS = {
    'cols': [
        'pu_location_id',
        'do_location_id',
        f'''
            CASE
                WHEN CAST(strftime('%H', pickup_datetime) AS INTEGER) BETWEEN 0 AND 5 THEN 'Night'
                WHEN CAST(strftime('%H', pickup_datetime) AS INTEGER) BETWEEN 6 AND 11 THEN 'Morning'
                WHEN CAST(strftime('%H', pickup_datetime) AS INTEGER) BETWEEN 12 AND 17 THEN 'Afternoon'
                WHEN CAST(strftime('%H', pickup_datetime) AS INTEGER) BETWEEN 18 AND 23 THEN 'Evening'
            END AS time_of_day
        ''',
        'AVG(total_amount) AS avg_total_amount',
        'AVG(fare_amount) AS avg_fare_amount',
        'AVG(tip_amount) AS avg_tip_amount',
        'SUM(passenger_count) AS sum_passenger_count',
        'AVG(trip_distance) as avg_trip_distance',
        'AVG(tolls_amount) as avg_tolls_amount',
        f'''
            AVG(unixepoch(dropoff_datetime)-unixepoch(pickup_datetime)) / 60 as avg_trip_duration
        ''',
        'COUNT(1) AS trip_count',
    ],
    'conditions': ['total_amount > 0'],
    'grp_ord_by': f'GROUP BY pu_location_id, do_location_id, time_of_day ORDER BY pu_location_id, do_location_id, time_of_day'
}


# cacheSql(DAILY_AVGS, DAILY_AVGS_CACHE, GREEN)
# cacheSql(DAILY_AVGS, DAILY_AVGS_CACHE, YELLOW)
# cacheSql(HOURLY_AVGS, HOURLY_AVGS_CACHE, GREEN)
# cacheSql(HOURLY_AVGS, HOURLY_AVGS_CACHE, YELLOW)
# cacheSql(PU_TIME_AVGS, PU_TIME_AVGS_CACHE, GREEN)
# cacheSql(PU_TIME_AVGS, PU_TIME_AVGS_CACHE, YELLOW)
# cacheSql(PU_DO_TIME_AVGS, PU_DO_TIME_AVGS_CACHE, GREEN)
# cacheSql(PU_DO_TIME_AVGS, PU_DO_TIME_AVGS_CACHE, YELLOW)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
from model import *
from data import *

def getSqlForModeling(taxi_type, isTrain):
    trip_count = {
        "green": 3071981,
        "yellow": 120393705
    }
    train_size = {
        "green": math.floor(0.8 * trip_count[GREEN]),
        "yellow": math.floor(0.8 * trip_count[YELLOW])
    }
    test_size = {
        "green": trip_count[GREEN] - train_size[GREEN],
        "yellow": trip_count[YELLOW] - train_size[YELLOW],
    }

    columns = COMMON_FETCH_COLUMNS 
    
    # init with limit clause for test
    limit_clause = f'LIMIT {test_size[taxi_type]} OFFSET {train_size[taxi_type]}'
    
    if isTrain:
        limit_clause = f'LIMIT {train_size[taxi_type]}'
    
    sql = f'''
        {selFrom(columns, 2020, taxi_type)}
        WHERE {' AND '.join(commonConditions(2020))}
        UNION ALL
        {selFrom(columns, 2021, taxi_type)}
        WHERE {' AND '.join(commonConditions(2021))}
        UNION ALL
        {selFrom(columns, 2022, taxi_type)}
        WHERE {' AND '.join(commonConditions(2022))}
        UNION ALL
        {selFrom(columns, 2023, taxi_type)}
        WHERE {' AND '.join(commonConditions(2023))}
        {limit_clause}
    '''

    return text(sql)

# getSqlForModeling(GREEN, isTrain=False)