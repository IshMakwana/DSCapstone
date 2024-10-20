import warnings
import pandas as pd
from tabulate import tabulate
from sqlalchemy import MetaData
from sqlalchemy import create_engine

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

normalizedColumns = {
    'lpep_pickup_datetime': 'pickup_datetime', 'tpep_pickup_datetime': 'pickup_datetime',
    'lpep_dropoff_datetime': 'dropoff_datetime', 'tpep_dropoff_datetime': 'dropoff_datetime',
    'RatecodeID': 'ratecode_id',
    'PULocationID': 'pu_location_id', 
    'DOLocationID': 'do_location_id',
    'passenger_count': 'passenger_count', 
    'trip_distance': 'trip_distance', 
    'fare_amount': 'fare_amount', 
    'extra': 'extra', 
    'mta_tax': 'mta_tax',
    'tip_amount': 'tip_amount', 
    'tolls_amount': 'tolls_amount', 
    'improvement_surcharge': 'improvement_surcharge',
    'total_amount': 'total_amount', 
    'payment_type': 'payment_type', 
    'congestion_surcharge': 'congestion_surcharge'
}

COLUMNS = [
    'pickup_datetime', 'dropoff_datetime',
    'pu_location_id', 'do_location_id',
    'passenger_count', 'trip_distance', 'total_amount',
    'fare_amount', 'tip_amount', 'mta_tax', 'tolls_amount', 'extra', 'improvement_surcharge', 'congestion_surcharge',
    'payment_type', 'ratecode_id'
]

ALL_COLUMNS = ['pickup_datetime',
 'dropoff_datetime',
 'pu_location_id',
 'do_location_id',
 'passenger_count',
 'trip_distance',
 'total_amount',
 'fare_amount',
 'tip_amount',
 'mta_tax',
 'tolls_amount',
 'extra',
 'improvement_surcharge',
 'congestion_surcharge',
 'payment_type',
 'ratecode_id',
 'f_trip_distance',
 'f_fare_amount',
 'f_mta_tax',
 'f_total_amount',
 'f_passenger_count']

# Functions
def getSQLiteString():
    return 'sqlite:///db/taxi_db.db'

def getDateColumns(taxi_type = YELLOW):
    return ['lpep_pickup_datetime','lpep_dropoff_datetime'] if taxi_type == GREEN else ['tpep_pickup_datetime','tpep_dropoff_datetime']

# def getODBCString():
#     SERVER = 'tcp:nyc-taxi-2024.database.windows.net,1433'
#     DATABASE = 'nyc_taxi_2024'
#     USERNAME = 'ishmakwana'
#     PASSWORD = 'xxx'

#     con_str = f'DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
#     return URL.create("mssql+pyodbc", query={"odbc_connect": con_str})

def getDF(sql):
    with DR.engn.connect() as conn:
        return pd.read_sql(sql, conn)
    
def runSql(sql, commit = False):
    with DR.engn.connect() as conn:
        res = conn.execute(sql)
        if commit:
            conn.commit()
        return res

# Classes
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

    # uniq_ can be added a prefix to refer to distinct tables for the data
    def getTableName(self, prefix = ''):
        return f'{prefix}{self.taxi_type}_taxi_trips{self.year}'
    
TABLES = [
    (GREEN, 2020), (YELLOW, 2020),
    (GREEN, 2021), (YELLOW, 2021),
    (GREEN, 2022), (YELLOW, 2022),
    (GREEN, 2023), (YELLOW, 2023)
]

TAXI_ZONES='taxi_zones'

# Data Reader can be used by all other modules
DR = TaxiDBReader()

# common code templates
# ctables = [
#     (GREEN, 2023), 
#     (YELLOW, 2023)
# ]

# for t in TABLES:
# for t in ctables:
#     taxi_type, year  = t
#     DR.setTable(year, taxi_type)
#     table_name = DR.getTableName()
#     O.out(f'table: {table_name}')
#     uniq_table_name = DR.getTableName('uniq_')

# (strftime('%Y-%m-%d %H', pickup_datetime)) as dt_hr,