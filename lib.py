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

# Caches
BOX_PLOT_CACHE = 'box_plot_data'

# Models
RANDOM_FOREST = 'random_forest'
LINEAR_REGRESSION = 'linear_regression'

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



def getOutputPath(prefix = 'output'):
    today_date = datetime.datetime.today().strftime('%Y-%m-%d')

    return f"output/{prefix}_{today_date}.txt"

def getSQLiteString():
    return 'sqlite:///db/taxi_db.db'

def getDateColumns(taxi_type = YELLOW):
    return ['lpep_pickup_datetime','lpep_dropoff_datetime'] if taxi_type == GREEN else ['tpep_pickup_datetime','tpep_dropoff_datetime']



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


# def getODBCString():
#     SERVER = 'tcp:nyc-taxi-2024.database.windows.net,1433'
#     DATABASE = 'nyc_taxi_2024'
#     USERNAME = 'ishmakwana'
#     PASSWORD = 'xxx'

#     con_str = f'DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
#     return URL.create("mssql+pyodbc", query={"odbc_connect": con_str})

# def getDF(sql):
#     with DR.engn.connect() as conn:
#         return pd.read_sql(sql, conn)
    
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
    
def getManhattanLocIds():
    df = getDF(text(f'select location_id from taxi_zones where location_name=\'Manhattan\''))
    return list(df['location_id'])

manh_locids = getManhattanLocIds()

def getLocationName(location_id):
    # airport location ids = newark=1, JFK Airport,132, LaGuardia Airport,138
    if location_id==1:
        return 'EWR'
    elif location_id==132:
        return 'JFK'
    elif location_id==138:
        return 'LGA'
    elif location_id in manh_locids:
        return 'Manhattan'
    return '-'

def compute_total_amount(row, taxi_type=YELLOW):
    trip_distance = row['trip_distance']
    pickup_datetime = row['pickup_datetime']
    tip_amount = row['tip_amount']
    tolls_amount = row['tolls_amount']
    pickup_location = getLocationName(row['pu_location_id'])
    dropoff_location = getLocationName(row['do_location_id'])
    
    # Initial base fare
    total_amount = 3.00
    
    # Metered fare: 70 cents per 1/5 mile (3.50 per mile)
    total_amount += (trip_distance * 3.50)
    
    # MTA State Surcharge
    total_amount += 0.50
    
    # Improvement Surcharge
    total_amount += 1.00
    
    # Time-based surcharges
    pickup_time = datetime.datetime.strptime(pickup_datetime, '%Y-%m-%d %H:%M:%S.%f')

    # Overnight surcharge: $1.00 (8pm to 6am)
    if pickup_time.hour >= 20 or pickup_time.hour < 6:
        total_amount += 1.00
    
    # Rush hour surcharge: $2.50 (4pm to 8pm on weekdays, excluding holidays)
    if pickup_time.weekday() < 5 and 16 <= pickup_time.hour < 20:
        total_amount += 2.50
    
    # Airport rules
    if pickup_location in ['LGA', 'JFK', 'EWR'] or dropoff_location in ['LGA', 'JFK', 'EWR']:
        
        # Handle LaGuardia (LGA)
        if pickup_location == 'LGA':
            total_amount += 1.75  # Airport Access Fee for pickup at LGA
            total_amount += 5.00  # Additional $5 surcharge for trips from/to LGA
        
        # Handle JFK Airport
        if pickup_location == 'JFK' or dropoff_location == 'JFK':
            if pickup_location == 'Manhattan' or dropoff_location == 'Manhattan':
                # Flat rate for trips between Manhattan and JFK
                total_amount = 70.00  # Flat rate
                total_amount += 0.50  # MTA Surcharge
                total_amount += 1.00  # Improvement Surcharge
                
                # Rush hour surcharge: $5.00 (4pm to 8pm weekdays)
                if pickup_time.weekday() < 5 and 16 <= pickup_time.hour < 20:
                    total_amount += 2.50
                
                # Set on-screen rate to "Rate #2 - JFK Airport"
                # print("Rate #2 - JFK Airport")
        
        # Handle Newark (EWR)
        if pickup_location == 'EWR' or dropoff_location == 'EWR':
            total_amount += 20.00  # Newark Surcharge
            total_amount += tolls_amount  # Include tolls
            
            # Set on-screen rate to "Rate #3 - Newark Airport"
            # print("Rate #3 - Newark Airport")
    
    # Congestion surcharge for trips in Manhattan south of 96th Street
    # if is_shared_ride:
    #     total_amount += 0.75  # Shared ride
    if taxi_type == GREEN:
        total_amount += 2.75  # Green taxi/FHV
    else:
        total_amount += 2.50  # Yellow taxi

    # Add tips and tolls
    total_amount += tip_amount + tolls_amount
    
    return total_amount



import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

LINREG_FEATS = [
    'trip_duration', 'passenger_count', 'trip_distance', 
    'tip_amount', 'tolls_amount', 'extra', 'mta_tax',
    'improvement_surcharge', 'congestion_surcharge',
 ]
LINREG_DEP = 'total_amount'

R_FRST_FTS = ['pu_location_id', 'do_location_id', 'passenger_count', 'trip_distance', 'trip_duration', 'tip_amount', 'mta_tax', 'tolls_amount', 'extra', 'improvement_surcharge', 'congestion_surcharge']
R_FRST_DEP = 'total_amount'

def calculate_vif(X):
    vif = pd.DataFrame()
    vif["Feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

class LinRegForecast:
    def __init__(self, features, dependent_variable, data) -> None:
        self.feats = features
        self.dep = dependent_variable
        self.X = data[self.feats]
        self.y = data[self.dep]

        # Add a constant to the features (for the intercept)
        self.X = sm.add_constant(self.X)

        # Perform stepwise VIF feature selection
        self.stepwise_vif_selection()
        O.out("Selected Features after VIF Selection:")
        O.out(self.X.columns)

        # X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.20)

        # Train the linear regression model
        self.model = LinearRegression()
        self.model.fit(self.X, self.y)

    def getModel(self):
        return self.model

    def stepwise_vif_selection(self, threshold=10):
        while True:
            # Calculate VIF for all features
            vif = calculate_vif(self.X)
            max_vif = vif["VIF"].max()
            if max_vif > threshold:
                # Drop the feature with the highest VIF
                feature_to_remove = vif.loc[vif["VIF"].idxmax(), "Feature"]
                O.out(f"Removing feature: {feature_to_remove} with VIF = {max_vif}")
                self.X = self.X.drop(columns=[feature_to_remove])
            else:
                # If all VIFs are below the threshold, stop the loop
                break

    

import geopandas as gpd
from shapely import wkt

def getTaxiGDF():
    gdf = getDF(text('SELECT zone_shape, location_id, location_name, zone FROM taxi_zones'))
    gdf['geometry'] = gdf['zone_shape'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(gdf.dropna(), geometry='geometry')
    gdf['geometry'] = gdf['geometry'].buffer(0)

    return gdf


def fmtDate(y, date, fmt):
    updated_date = datetime.datetime(y, date.month, date.day, date.hour, date.minute)
    return datetime.datetime.strftime(updated_date, fmt)


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


def getTripCountByPickup(year, taxi_type):
    cols = ['pu_location_id', 'COUNT(1) as trip_count']
    query = f"""
        {selFrom(cols, year, taxi_type)}
        WHERE (strftime('%Y', pickup_datetime))='{year}'
        GROUP BY pu_location_id
    """

    return getDF(text(query))

COMMON_FETCH_COLUMNS = ['pickup_datetime', 'pu_location_id', 'do_location_id', 'f_passenger_count', 'f_trip_distance', 'trip_duration', 
                        'f_total_amount', 'f_fare_amount', 'tip_amount', 'f_mta_tax', 'tolls_amount', 'extra', 'improvement_surcharge', 
                        'congestion_surcharge']

def readData(cols = COMMON_FETCH_COLUMNS, conditions = [], taxi_type = YELLOW):
    chunks = []
    for year in range(MIN_YEAR, MAX_YEAR + 1):
        base_condition = [f""" (strftime('%Y', pickup_datetime))='{year}' """]
        all_conditions = base_condition + conditions
        sql = text(f"""
            {selFrom(cols, year, taxi_type)}
            where {' AND '.join(all_conditions)}
        """)

        df = getDF(sql)
        if 'pickup_datetime' in cols:
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

        print(len(df))

        chunks.append(df)

    return pd.concat(chunks, ignore_index=True)


OBJECT_TYPE_MAP = {
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

def getSample(year = 2023, taxi_type = YELLOW, limit = 10**3):
    sql = text(f"""
        {selFrom(COMMON_FETCH_COLUMNS, year, taxi_type)}
        where (strftime('%Y', pickup_datetime))='{year}'
        order by random()
        limit {limit}
    """)
    return getDF(sql)




def todFromDate(date):
    hr = date.hour
    if hr < 6: return 'night'
    elif 6 <= hr < 12: return 'morning'
    elif 12 <= hr < 18: return 'afternoon'
    return 'evening'

BY_LOC_CATS = ['by_minute', 'by_hour', 'by_time_of_day', 'all']

def getCatsAndCons(year, date, min_fmt=minute_format, hr_fmt=hour_format, day_fmt=day_format):
    return [
        ('by_minute', f'''   strftime('{min_fmt}', pickup_datetime)='{fmtDate(year, date, min_fmt)}'    '''),
        ('by_hour', f'''   strftime('{hr_fmt}', pickup_datetime)='{fmtDate(year, date, hr_fmt)}'    '''),
        (
            'by_time_of_day', 
            f''' 
                time_of_day='{todFromDate(date)}' AND 
                strftime('{day_fmt}', pickup_datetime)='{fmtDate(year, date, day_fmt)}'
            '''
        ),
        ('all', None)
    ]

def displayByCat(data):
    O.out(SEPARATOR)
    for c in BY_LOC_CATS:
        O.out(f'Category: {c}')
        O.out(data[c], True, 10)
        O.out(SEPARATOR)

# we want to do predict total_amount
def get_data_by_loc(pu_location_id, do_location_id, date, taxi_type = GREEN):
    cols = [
        'pu_location_id', 'pu_location',
        'do_location_id', 'do_location',
        'pickup_datetime', 'time_of_day', 
        'trip_duration', 'trip_distance',
        'total_amount', 'fare_amount', 
        'tip_amount', 'tolls_amount', 'passenger_count',
        'extra', 'fare_amount','mta_tax',
        'improvement_surcharge','congestion_surcharge',
    ]

    chunks = {}
    for c in BY_LOC_CATS:
        chunks[c] = []

    for y in range(MIN_YEAR, MAX_YEAR+1):
        DR.setTable(y, taxi_type)
        
        for (cat, con) in getCatsAndCons(y, date):
            conditions = [
                f'''CAST(strftime('%Y', pickup_datetime) as integer)={y}''',
                f'''pu_location_id={pu_location_id}''',
                f'''do_location_id={do_location_id}'''
            ]
            if con is not None:
                conditions.append(con)

            sql = f'''
            {selFrom(cols, y, taxi_type)}
            WHERE {' AND '.join(conditions)}
            '''
            chunks[cat].append(getDF(text(sql)))
    
    for k in BY_LOC_CATS:
        chunks[k] = pd.concat(chunks[k], ignore_index=True)
        for intcol in ['trip_duration', 'passenger_count']:
            chunks[k][intcol] = chunks[k][intcol].astype(float)

    return chunks

def out_sample_tt(train, test, label='', train_page_size=7, test_page_size=3):
    O.out(f'{label} train data: ')
    O.out(train, True, train_page_size)
    O.out(f'{label} test data: ')
    O.out(test, True, test_page_size)

def display_computed_total_amount(data, taxi_type, msg):
    # O.out(f'{taxi_type.capitalize()} Taxi computed_total_amount')
    msg.append(f'{taxi_type.capitalize()} Taxi computed_total_amount')
    data['computed_total_amount'] = data.apply(lambda row: compute_total_amount(row, taxi_type), axis=1)
    train, test = train_test_split(data, test_size=0.1)
    # out_sample_tt(train, test, 'total_amount')
    actual, computed = test['total_amount'].median(), test['computed_total_amount'].median()
    # O.out(f'Actual: {actual}, computed: {computed:.02f}')
    msg.append(f'Actual: {actual}, computed: {computed:.02f}')

def display_lin_reg(data, taxi_type, msg):
    msg.append(f'{taxi_type.capitalize()} Taxi LINEAR REGRESSION Forecast')
    # train vs test
    X = data[LINREG_FEATS]
    y = data[LINREG_DEP]
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.1, random_state=42)
    model = LinearRegression()
    model.fit(X_trn, y_trn)
    pred = model.predict(X_tst)
    actual, computed = np.median(pred), np.median(y_tst)
    msg.append(f'Actual: {actual:.02f}, computed: {computed:.02f}')
    return 0

def display_rand_frst(data, taxi_type, msg):
    msg.append(f'{taxi_type.capitalize()} Taxi RANDOM FOREST Forecast')
    # train vs test
    X = data[R_FRST_FTS]
    y = data[R_FRST_DEP]
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.1, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_trn, y_trn)
    pred = model.predict(X_tst)
    actual, computed = np.median(pred), np.median(y_tst)
    msg.append(f'Actual: {actual:.02f}, computed: {computed:.02f}')
    return 0

def predict_total_amount(green_bc, yellow_bc):
    # make a table of all methods
    # O.out(SEPARATOR)
    SEPARATOR = '-' * 25
    messages = {}
    messages[GREEN] = []
    messages[YELLOW] = []

    for c in BY_LOC_CATS:
        for type in [GREEN, YELLOW]:
            messages[type].append('*' * 10)
            messages[type].append(f'Category: {c}')
            messages[type].append('*' * 10)

        green_df = green_bc[c]
        if len(green_df) > 1:
            display_computed_total_amount(green_df, GREEN, messages[GREEN])
            display_lin_reg(green_df, GREEN, messages[GREEN])
            display_rand_frst(green_df, GREEN, messages[GREEN])
        else:
            messages[GREEN].append(f'Not enough data for {c}')
        
        messages[GREEN].append(SEPARATOR)
            
        yellow_df = yellow_bc[c]
        if len(yellow_df) > 1:
            display_computed_total_amount(yellow_df, YELLOW, messages[YELLOW])
            display_lin_reg(yellow_df, YELLOW, messages[YELLOW])
            display_rand_frst(yellow_df, YELLOW, messages[YELLOW])
        else:
            messages[YELLOW].append(f'Not enough data for {c}')
        
        messages[YELLOW].append(SEPARATOR)

    return messages


def displayPredictionByLocation(pickup, dropoff, pickup_datetime):
    green_bc = get_data_by_loc(pu_location_id=pickup, do_location_id=dropoff, date=pickup_datetime, taxi_type= GREEN)
    yellow_bc = get_data_by_loc(pu_location_id=pickup, do_location_id=dropoff, date=pickup_datetime, taxi_type= YELLOW)

    return predict_total_amount(green_bc, yellow_bc)


def getLocationOptions():
    df = getDF(text('''
                    SELECT location_id, (location_name || ', ' || zone) as f_location_name FROM taxi_zones
                    ORDER BY f_location_name ASC
                    '''))
    return dict(zip(df["f_location_name"], df["location_id"]))

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# common code templates
# ctables = [
#     (GREEN, 2023), 
#     (YELLOW, 2023)
# ]
# ----------------------------------------------------------------------
# for t in TABLES:
# for t in ctables:
#     taxi_type, year  = t
#     DR.setTable(year, taxi_type)
#     table_name = DR.getTableName()
#     O.out(f'table: {table_name}')

# ----------------------------------------------------------------------
# (strftime('%Y-%m-%d %H', pickup_datetime)) as dt_hr,

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# SELECT strftime('%Y-%m-%d', pickup_datetime) as date, 
#     AVG(f_total_amount) as total_amount,
#     AVG(tip_amount) as tip_amount,
#     AVG(tolls_amount) as tolls_amount,
#     AVG(f_trip_distance) as trip_distance,
#     AVG(f_passenger_count) as passenger_count,
#     COUNT(1) as trip_count                    
# FROM {table_name}
# WHERE f_total_amount > 0 AND 
#     f_trip_distance > 0 AND 
#     f_passenger_count > 0
# GROUP BY date
# ORDER BY date

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# from lib import *

# year = 2023
# taxi_type = GREEN

# DR.setTable(year, taxi_type)
# table_name = DR.getTableName()

# sql = f"""
# select *,
#     (unixepoch(dropoff_datetime)-unixepoch(pickup_datetime)) as trip_duration
#  from {u_table_name} 
#  where (strftime('%Y', pickup_datetime))='{year}' and
#     trip_duration > 0 AND trip_duration <= 7200
# """

# df = getDF(text(sql))
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------