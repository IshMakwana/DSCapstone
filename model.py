from lib import *
from data import *
from computed import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Calculates VIF for all features in X
# input: dataframe with all features
# output: dataframe containing VIF scores for each Feature
def calculate_vif(X):
    vif = pd.DataFrame()
    vif["Feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif


# Performs feature selection using stepwise vif selection.
# input: dataframe with all features
# output: dataframe with suggested features
def stepwise_vif_selection(X, threshold=10):
    X = sm.add_constant(X)

    while True:
        # Calculate VIF for all features
        vif = calculate_vif(X)
        max_vif = vif["VIF"].max()
        if max_vif > threshold:
            # Drop the feature with the highest VIF
            feature_to_remove = vif.loc[vif["VIF"].idxmax(), "Feature"]
            O.out(f"Removing feature: {feature_to_remove} with VIF = {max_vif}")
            X = X.drop(columns=[feature_to_remove])
        else:
            # If all VIFs are below the threshold, stop the loop
            break

    O.out("Selected Features after VIF Selection:")
    O.out(X.columns)
    return X

# These constants were made using results from the stepwise selection process above.
FEATURES = ['pu_location_id', 'do_location_id', 'passenger_count', 'trip_distance', 
            'trip_duration', 'tip_amount', 'mta_tax', 'tolls_amount', 'extra', 
            'improvement_surcharge', 'congestion_surcharge']

VARIABLE = 'fare_amount'

def commonConditions(year):
    return [f""" (strftime('%Y', pickup_datetime))='{year}' """, 
                          'passenger_count > 0',
                          'trip_distance > 0',
                          'fare_amount > 0',
                          'trip_duration > 0']

# Builds LinearRegression model for trips from 2020-2023 for either green or yellow taxi
# input: taxi_type (green or hellow), features, variable, columns
# output: a persisted model stored in a file locally - path template: f'{taxi_type}_{variable}_{LINEAR_REGRESSION}'
def buildAndStoreModel_lr(taxi_type, features, variable, columns):
    model = LinearRegression()
    chunks = 0
    
    for year in range(MIN_YEAR, MAX_YEAR + 1):
        sql = text(f"""
            {selFrom(columns, year, taxi_type)}
            where {' AND '.join(commonConditions(year))}
        """)

        with DR.engn.connect() as conn:
            for df in pd.read_sql(sql, conn, chunksize=CHUNK_SIZE):
                df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
                chunks += len(df)
                O.out(f'data processed: {chunks}')

                X = df[features]
                y = df[variable]

                model.fit(X, y)
            O.out(SEPARATOR)

    storeModel(model, f'{taxi_type}_{variable}_{LINEAR_REGRESSION}')
    

# Builds RandomForestRegressor model for trips from 2020-2023 for either green or yellow taxi
# input: taxi_type (green or hellow), features, variable, columns
# output: a persisted model stored in a file locally - path template: f'{taxi_type}_{variable}_{LINEAR_REGRESSION}'
def buildAndStoreModel_rfr(taxi_type, features, variable, columns):
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    chunks = 0
    
    for year in range(MIN_YEAR, MAX_YEAR + 1):
        sql = text(f"""
            {selFrom(columns, year, taxi_type)}
            where {' AND '.join(commonConditions(year))}
        """)

        with DR.engn.connect() as conn:
            for df in pd.read_sql(sql, conn, chunksize=CHUNK_SIZE):
                df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
                chunks += len(df)
                O.out(f'data processed: {chunks}')

                X = df[features]
                y = df[variable]

                model.fit(X, y)
            O.out(SEPARATOR)

    storeModel(model, f'{taxi_type}_{variable}_{RANDOM_FOREST}')

# Returns computed result vs actual with accuracy
# input: data (a dataframe) and taxi_type
# output: a tuple of 4 - method name, actual, computed and accuracy
def findComputed(data, taxi_type):
    data['computed_fare_amount'] = data.apply(lambda row: compute_fare_amount(row, taxi_type), axis=1)
    accuracy = r2_score(data[VARIABLE], data['computed_fare_amount'])
    
    return ('Computed', data[VARIABLE].mean(), data['computed_fare_amount'].mean(), accuracy)

# Returns Linear Regression forecast result vs actual, with accuracy
# input: train and test data, and a flag to use persisted models
# output: a tuple of 4 - method name, actual, computed and accuracy
def findResult_lr(X_train, X_test, y_train, y_test, taxi_type, use_persisted = False):
    if use_persisted:
        model = loadModel(f'{taxi_type}_{VARIABLE}_{LINEAR_REGRESSION}')
        method = f'{LINEAR_REGRESSION}_persisted'
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        method = f'{LINEAR_REGRESSION}_by_location'

    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    return (method, y_test.mean(), y_pred.mean(), accuracy)

# Returns Linear Regression forecast result vs actual, with accuracy
# input: train and test data, and a flag to use persisted models
# output: a tuple of 4 - method name, actual, computed and accuracy
def findResult_rfr(X_train, X_test, y_train, y_test, taxi_type, use_persisted = False):
    if use_persisted:
        model = loadModel(f'{taxi_type}_{VARIABLE}_{RANDOM_FOREST}')
        method = f'{RANDOM_FOREST}_persisted'
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        method = f'{RANDOM_FOREST}_by_location'

    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    return (method, y_test.mean(), y_pred.mean(), accuracy)

# what does this do
# input: ??
# output: ??
def findBestResult(data):
    # result containing all predictions
    result = {}

    for tt in [GREEN, YELLOW]:
        result[tt] = []
        df = data[tt]
        
        if len(df) <= 1:
            continue
        
        X = df[FEATURES]
        y = df[VARIABLE]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        # add computed result with accuracy
        result[tt].append(findComputed(df, tt))
        # add overall linear regression result 
        result[tt].append(findResult_lr(X_train, X_test, y_train, y_test, tt, use_persisted=True))
        # add location based linear regression result 
        result[tt].append(findResult_lr(X_train, X_test, y_train, y_test, tt, use_persisted=False))
        # add overall random forest result 
        result[tt].append(findResult_rfr(X_train, X_test, y_train, y_test, tt, use_persisted=True))
        # add location based random forest result 
        result[tt].append(findResult_rfr(X_train, X_test, y_train, y_test, tt, use_persisted=False))

        # sort result by accuracy in desc order
        result[tt].sort(key=lambda tup: tup[3], reverse=True)
    
    return result

# what does this do
# input: ??
# output: ??
def displayPredictionByLocation(pickup, dropoff):
    data = {}
    for tt in [GREEN, YELLOW]:
        data[tt] = dataByLocations(pickup, dropoff, tt)

    return findBestResult(data)