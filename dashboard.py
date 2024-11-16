# get data
from lib import *

import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import plotly.graph_objs as go

def overallForecastForDashboard(data, taxi_type):
    # Define features and target
    X_LR = data[LINREG_FEATS]
    y_LR = data[LINREG_DEP]

    X_RFR = data[R_FRST_FTS]
    y_RFR = data[R_FRST_DEP]
    
    # load linreg model
    lr_model = loadModel(f'{taxi_type}_{LINEAR_REGRESSION}')
    y_pred_lr = lr_model.predict(X_LR)
    
    # Train Random Forest model
    rf_model = loadModel(f'{taxi_type}_{RANDOM_FOREST}')
    y_pred_rf = rf_model.predict(X_RFR)
    
    # Calculate metrics for Linear Regression
    mae_lr = mean_absolute_error(y_LR, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_LR, y_pred_lr))
    mape_lr = mean_absolute_percentage_error(y_LR, y_pred_lr) * 100
    r_squared_lr = lr_model.score(X_LR, y_LR) * 100
    
    # Calculate metrics for Random Forest
    mae_rf = mean_absolute_error(y_RFR, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_RFR, y_pred_rf))
    mape_rf = mean_absolute_percentage_error(y_RFR, y_pred_rf) * 100
    r_squared_rf = rf_model.score(X_RFR, y_RFR) * 100
    
    # Create line chart for Linear Regression forecast
    fig_lr = go.Figure()
    fig_lr.add_trace(go.Scatter(x=y_LR.index, y=y_LR, mode='lines+markers', name="Actual", line=dict(color='blue')))
    fig_lr.add_trace(go.Scatter(x=y_LR.index, y=y_pred_lr, mode='lines+markers', name="Predicted (LR)", line=dict(color='orange')))
    fig_lr.update_layout(
        title=f"{taxi_type.capitalize()} Taxi - Linear Regression Forecast",
        xaxis_title="Sample Index",
        yaxis_title="Total Amount",
        legend=dict(x=0, y=1.1, orientation="h"),
        margin=dict(l=5, r=5, t=70, b=5)
    )
    
    # Create line chart for Random Forest forecast
    fig_rf = go.Figure()
    fig_rf.add_trace(go.Scatter(x=y_RFR.index, y=y_RFR, mode='lines+markers', name="Actual", line=dict(color='blue')))
    fig_rf.add_trace(go.Scatter(x=y_RFR.index, y=y_pred_rf, mode='lines+markers', name="Predicted (RF)", line=dict(color='green')))
    fig_rf.update_layout(
        title=f"{taxi_type.capitalize()} Taxi - Random Forest Forecast",
        xaxis_title="Sample Index",
        yaxis_title="Total Amount",
        legend=dict(x=0, y=1.1, orientation="h"),
        margin=dict(l=5, r=5, t=70, b=5)
    )
    
    # Create a combined metrics table for both models
    metrics = {
        "Metric": ["R-squared (%)", "MAE", "RMSE", "MAPE (%)"],
        "Linear Regression": [f"{r_squared_lr:.2f}", f"{mae_lr:.2f}", f"{rmse_lr:.2f}", f"{mape_lr:.2f}"],
        "Random Forest": [f"{r_squared_rf:.2f}", f"{mae_rf:.2f}", f"{rmse_rf:.2f}", f"{mape_rf:.2f}"]
    }
    fig_table = go.Figure(data=[go.Table(
        header=dict(values=["Metric", "Linear Regression", "Random Forest"], fill_color='darkgrey', align='center'),
        cells=dict(values=[metrics["Metric"], metrics["Linear Regression"], metrics["Random Forest"]], fill_color='black', align='center')
    )])
    fig_table.update_layout(title=f"{taxi_type.capitalize()} Taxi Model Performance Metrics", margin=dict(l=5, r=5, t=30, b=5))
    
    return fig_lr, fig_rf, fig_table

def createMap(trips_by_location, taxi_type, column):
        # Filter the data for the specified taxi type
        # trips_by_location = data[taxi_type]
        
        # Merge trip counts with polygon GeoDataFrame
        merged_gdf = polygon_gdf.merge(trips_by_location, how='left', left_on='location_id', right_on='pu_location_id')
        merged_gdf['trip_count'] = merged_gdf['trip_count'].fillna(0)  # Fill NaN values with 0 for locations with no trips

        # Convert GeoDataFrame to GeoJSON format for Plotly
        merged_gdf_json = json.loads(merged_gdf.to_json())

        # Create a map visualization with Plotly
        fig = px.choropleth_mapbox(
            merged_gdf,
            geojson=merged_gdf_json,
            locations='location_id',
            color='trip_count',
            featureidkey="properties.location_id",
            hover_name="zone", 
            hover_data={"trip_count": True, "location_id": False},  # Hide location_id in tooltip, show trip_count
            color_continuous_scale="Viridis",
            mapbox_style="carto-positron",
            zoom=10,
            center={"lat": 40.7128, "lon": -74.0060},  # Center on NYC
            opacity=0.6,
            title=f"Trip Counts by Pickup Location ({taxi_type.capitalize()} Taxi)"
        )

        # Display the map in the specified Streamlit column
        column.plotly_chart(fig)

def summaryBoxFig(stats, taxi_type, feature):
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=[stats["min"], stats["q1"], stats["median"], stats["q3"], stats["max"]],
        boxpoints=False,
        name=feature,
        # orientation="h"
    ))
    fig.update_layout(
        title=f"{taxi_type.capitalize()} Taxi Box Plot",
        yaxis_title=feature
    )
    return fig

def getSummary():
    summary = {
        'green': loadObject(f'{GREEN}_{BOX_PLOT_CACHE}', 'cache'),
        'yellow': loadObject(f'{YELLOW}_{BOX_PLOT_CACHE}', 'cache')
    }
    return summary


# ---xx---
st.set_page_config(layout="wide")
available_years = [2020, 2021, 2022, 2023]

st.sidebar.title("NYC Taxi Trips Dashboard")
st.sidebar.markdown("<small>Author: Ishani Makwana</small>", unsafe_allow_html=True)

st.sidebar.markdown("---")

section = st.sidebar.radio("Go to", ["Home", "Viz", "Modeling", "Result"])

if section == "Home":
    # green vs yellow taxi
    summary = getSummary()
    features = summary[GREEN].keys()

    sp_col1, _,_,_ = st.columns(4)
    with sp_col1:
        feature = st.selectbox("Summary for ", options=features, index=0)

    col1, col2 = st.columns(2)
    # show distribution
    #   collect data
    #   make box plot
    with col1:
        g_fig = summaryBoxFig(summary[GREEN][feature], GREEN, feature)
        st.plotly_chart(g_fig, use_container_width=True)
    
    with col2:
        y_fig = summaryBoxFig(summary[YELLOW][feature], YELLOW, feature)
        st.plotly_chart(y_fig, use_container_width=True)

    # show weekly average
    # show by location 
elif section == "Viz":
    # Split the Streamlit layout into two columns

    h_col1, h_col2, h_col3 = st.columns(3)
    year = h_col1.selectbox("Year", options=available_years, index=3)

    col1, col2 = st.columns(2)

    polygon_gdf = getTaxiGDF()

    data = {}
    data[GREEN] = getTripCountByPickup(year, GREEN)
    data[YELLOW] = getTripCountByPickup(year, YELLOW)

    # Function to create a map for each taxi type
    

    # Create the map for green taxis in the first column
    createMap(data[GREEN], GREEN, col1)

    # Create the map for yellow taxis in the second column
    createMap(data[YELLOW], YELLOW, col2)
elif section == "Modeling":
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("## Green Taxi Forecast")
        # get ransom sample of n rows by taxi_type
        df = getSample(2023, GREEN, 150)
        green_fig_lr, green_fig_rf, green_fig_table = overallForecastForDashboard(df, GREEN)
        st.plotly_chart(green_fig_lr, use_container_width=True)
        st.plotly_chart(green_fig_rf, use_container_width=True)
        st.plotly_chart(green_fig_table, use_container_width=True)
        
    with col2:
        st.write("## Yellow Taxi Forecast")
        df = getSample(2023, YELLOW, 150)
        yellow_fig_lr, yellow_fig_rf, yellow_fig_table = overallForecastForDashboard(df, YELLOW)
        st.plotly_chart(yellow_fig_lr, use_container_width=True)
        st.plotly_chart(yellow_fig_rf, use_container_width=True)
        st.plotly_chart(yellow_fig_table, use_container_width=True)
elif section == "Result":
    location_options = getLocationOptions()
    col1, col2, _ = st.columns(3)

    with col1:
        pickup_location_label = st.selectbox("Pickup Location", options=list(location_options.keys()))
    pickup_location_value = location_options[pickup_location_label]

    with col2:
        dropoff_location_label = st.selectbox("Dropoff Location", options=list(location_options.keys()))
    dropoff_location_value = location_options[dropoff_location_label]

    # with col3:
    #     pickup_date = st.date_input("Select Pickup Date", value=datetime.date.today())
    #     pickup_time = st.time_input("Select Pickup Time", value=datetime.time(12, 0))
    # pickup_datetime = datetime.datetime.combine(pickup_date, pickup_time)

    if st.button("Predict Fare"):
        msgs = displayPredictionByLocation(pickup_location_value, dropoff_location_value)
        col1, col2 = st.columns(2)
        with col1:
            st.write(msgs[GREEN])
        with col2:
            st.write(msgs[YELLOW])

else:
    st.header(f'Section: {section}')
