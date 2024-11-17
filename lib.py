import warnings
import pandas as pd
import datetime
from tabulate import tabulate
from sqlalchemy import MetaData
from sqlalchemy import create_engine
import numpy as np
from sqlalchemy.sql import text
import matplotlib.pyplot as plt
import calendar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import geopandas as gpd
from shapely import wkt

warnings.filterwarnings("ignore")

GREEN = 'green'
YELLOW = 'yellow'
CHUNK_SIZE=10**6
OUT_PATH = 'output/output.txt'
SEPARATOR = '-' * 100
MIN_YEAR = 2020
MAX_YEAR = 2023
# TABLE_FORMAT = 'simple_grid'
TABLE_FORMAT = 'rst'

COMMON_FETCH_COLUMNS = ['pickup_datetime', 'pu_location_id', 'do_location_id', 'passenger_count', 'trip_distance', 'trip_duration', 
                        'total_amount', 'fare_amount', 'tip_amount', 'mta_tax', 'tolls_amount', 'extra', 'improvement_surcharge', 
                        'congestion_surcharge']

# Caches
BOX_PLOT_CACHE = 'box_plot_data'

# Models
RANDOM_FOREST = 'random_forest'
LINEAR_REGRESSION = 'linear_regression'

def getOutputPath(prefix = 'output'):
    today_date = datetime.datetime.today().strftime('%Y-%m-%d')

    return f"output/{prefix}_{today_date}.txt"

class Output:
    def __init__(self, path) -> None:
        self.path = path
    
    def out(self, data, is_df = False, page_size = 20):
        with open(self.path, 'a') as f:
            if is_df:
                l = len(data)
                if l <= page_size:
                    print(tabulate(data, headers='keys', tablefmt=TABLE_FORMAT), file=f)
                else:
                    print(tabulate(data[:page_size], headers='keys', tablefmt=TABLE_FORMAT), file=f)
                print(f'Showing {min(page_size, l)} of {l} rows', file=f)
            else:
                print(data, file=f)
    
    def clear(self):
        open(self.path, 'w').close()

def getSQLiteString():
    return 'sqlite:///db/taxi_db.db'        

class TaxiDBReader:
    def __init__(self):
        self.md = MetaData()
        self.engn = create_engine(getSQLiteString())
        self.md.reflect(self.engn)
        self.year = MAX_YEAR
        self.taxi_type = YELLOW
        print('sql engine ready')

        with self.engn.connect() as conn:
            conn.rollback()

    def setTable(self, year=MAX_YEAR, taxi_type=YELLOW):
        self.year = year
        self.taxi_type = taxi_type

    def getTableName(self):
        return f'{self.taxi_type}_taxi_trips{self.year}'
    
TABLES = [
    (GREEN, 2020), (YELLOW, 2020),
    (GREEN, 2021), (YELLOW, 2021),
    (GREEN, 2022), (YELLOW, 2022),
    (GREEN, 2023), (YELLOW, 2023)
]

TAXI_ZONES='taxi_zones'

# Data Reader can be used by all other modules
DR = TaxiDBReader()
O = Output(getOutputPath())


full_date_format = '%Y-%m-%d %H:%M:%S.%f'
minute_format = '%Y-%m-%d %H:%M'
hour_format = '%Y-%m-%d %H'
day_format = '%Y-%m-%d'

selectMap = {
    'pu_location': f'''(select (zone||', '||location_name) from taxi_zones where location_id=pu_location_id limit 1) as pu_location''',
    'do_location': f'''(select (zone||', '||location_name) from taxi_zones where location_id=do_location_id limit 1) as do_location''',
    'time_of_day': f'''CASE  
            WHEN CAST(strftime('%H', pickup_datetime) as integer) < 6
                THEN 'night' 
            WHEN CAST(strftime('%H', pickup_datetime) as integer) >= 6 AND CAST(strftime('%H', pickup_datetime) as integer) < 12
                THEN 'morning'
            WHEN CAST(strftime('%H', pickup_datetime) as integer) >= 12 AND CAST(strftime('%H', pickup_datetime) as integer) < 18
                THEN 'afternoon'
            ELSE 'evening' 
        END time_of_day''',
    'trip_duration': f'''(unixepoch(dropoff_datetime)-unixepoch(pickup_datetime)) as trip_duration''',
    'year': f'''CAST(strftime('%Y', pickup_datetime) as integer) as year''',
    'f_trip_distance': f'''f_trip_distance as trip_distance''',
    'f_fare_amount': f'''f_fare_amount as fare_amount''',
    'f_mta_tax': f'''f_mta_tax as mta_tax''',
    'f_total_amount': f'''f_total_amount as total_amount''',
    'f_passenger_count': f'''f_passenger_count as passenger_count''',
    'trip_distance': f'''f_trip_distance as trip_distance''',
    'fare_amount': f'''f_fare_amount as fare_amount''',
    'mta_tax': f'''f_mta_tax as mta_tax''',
    'total_amount': f'''f_total_amount as total_amount''',
    'passenger_count': f'''f_passenger_count as passenger_count''',
}

def selFrom(cols, year, taxi_type):
    mappedCols = []
    for c in cols:
        if c in selectMap:
            mappedCols.append(selectMap[c])
        else:
            mappedCols.append(c)
    DR.setTable(year, taxi_type)
    return f'''
    SELECT {', '.join(mappedCols)} FROM {DR.getTableName()}
'''

def getDF(sql):
    with DR.engn.connect() as conn:
        chunks = []
        for chunk in pd.read_sql(sql, conn, chunksize=CHUNK_SIZE):
            chunks.append(chunk)
        
        return pd.concat(chunks, ignore_index=True)

def runSql(sql, commit = False):
    with DR.engn.connect() as conn:
        res = conn.execute(sql)
        if commit:
            conn.commit()
        return res


OBJECT_TYPE_MAP = {
    # object_type : (directory, suffix)
    'cache': ('caches', '_cache'),
    'model': ('models', '_model')
}

def buildObjectPath(name, type):
    dir, suffix = OBJECT_TYPE_MAP[type]
    return f'{dir}/{name}{suffix}.joblib'

def storeModel(model, name):
    storeObject(model, name, 'model')

def loadModel(name):
    return loadObject(name, 'model')

def storeObject(object, name, type):
    joblib.dump(object, buildObjectPath(name, type))

def loadObject(name, type):
    return joblib.load(buildObjectPath(name, type))

def commonConditions(year):
    return [f""" (strftime('%Y', pickup_datetime))='{year}' """, 
                          'passenger_count > 0',
                          'trip_distance > 0',
                          'fare_amount > 0',
                          'trip_duration > 0']

def todFromDate(date):
    hr = date.hour
    if hr < 6: return 'night'
    elif 6 <= hr < 12: return 'morning'
    elif 12 <= hr < 18: return 'afternoon'
    return 'evening'


