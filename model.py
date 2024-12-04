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

FEATURES_SET1 = ['trip_distance', 'trip_duration', 
               'tip_amount', 'mta_tax', 'tolls_amount', 'extra', 'improvement_surcharge', 'congestion_surcharge']
SUFFIX_SET1 = '_f_set1'

FEATURES_SET2 = ['trip_distance', 'trip_duration', 'tip_amount', 'tolls_amount']
SUFFIX_SET2 = '_f_set2'

VARIABLE = 'fare_amount'

# Builds LinearRegression model for trips from 2020-2023 for either green or yellow taxi
# input: taxi_type (green or hellow), features, variable, columns
# output: a persisted model stored in a file locally - path template: f'{taxi_type}_{variable}_{LINEAR_REGRESSION}'
def buildAndStoreModel_lr(taxi_type, features, variable, columns, suffix = ''):
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

    storeModel(model, f'{taxi_type}_{variable}_{LINEAR_REGRESSION}{suffix}')
    

# Builds RandomForestRegressor model for trips from 2020-2023 for either green or yellow taxi
# input: taxi_type (green or hellow), features, variable, columns
# output: a persisted model stored in a file locally - path template: f'{taxi_type}_{variable}_{LINEAR_REGRESSION}'
def buildAndStoreModel_rfr(taxi_type, features, variable, columns, suffix = ''):
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

    storeModel(model, f'{taxi_type}_{variable}_{RANDOM_FOREST}{suffix}')

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


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.graph_objects as go

def showPerformance(y_test, y_pred):
    # Calculate Model Accuracy Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Accuracy Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (RMSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (Accuracy): {r2:.2f}")

    # Limit data to 150 samples for visualization
    samples_to_plot = 150
    y_test_limited = y_test[:samples_to_plot].reset_index(drop=True)
    y_pred_limited = y_pred[:samples_to_plot]

    # Create Line Chart with Plotly
    fig = go.Figure()

    # Actual Values
    fig.add_trace(go.Scatter(
        x=y_test_limited.index, 
        y=y_test_limited, 
        mode="lines+markers", 
        name="Actual",
        line=dict(color="blue")
    ))

    # Predicted Values
    fig.add_trace(go.Scatter(
        x=y_test_limited.index, 
        y=y_pred_limited, 
        mode="lines+markers", 
        name="Predicted",
        line=dict(color="orange")
    ))

    # Customize Layout
    fig.update_layout(
        title="Actual vs Predicted Fare Amount (Limited to 150 Samples)",
        xaxis_title="Sample Index",
        yaxis_title="Fare Amount ($)",
        legend=dict(x=0.5, y=1.15, xanchor="center", yanchor="top"),
        template="plotly_white"
    )

    # Show the plot
    fig.show()


import matplotlib.pyplot as plt
import seaborn as sns

def residualAnalysis(X_test, y_test, y_pred):
    residuals = y_test - y_pred

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # 1. Residuals vs Predicted Values
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, color="blue", edgecolor="k")
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Values")
    plt.show()

    # 2. Residual Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=20, color="purple")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Histogram of Residuals")
    plt.show()

    # 3. Q-Q Plot for Residuals
    import scipy.stats as stats
    plt.figure(figsize=(8, 5))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")
    plt.show()

    # 4. Residuals vs Individual Features
    for feature in X_test.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=X_test[feature], y=residuals, color="green", edgecolor="k")
        plt.axhline(y=0, color="red", linestyle="--")
        plt.xlabel(feature)
        plt.ylabel("Residuals")
        plt.title(f"Residuals vs {feature}")
        plt.show()