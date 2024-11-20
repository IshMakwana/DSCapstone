
# import geopandas as gpd
# import pandas as pd
# from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from lib import *
from data import *
from model import *

import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import json

GENERAL = 'Home'
MAPS = 'Viz'
COMPARE = 'Modeling'
RESULT = 'Result'

available_years = [2020, 2021, 2022, 2023]
polygon_gdf = getTaxiGDF()

# ------------------------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------------------------
# @st.cache_data
def goLinePlot(title, x_label, y_label, x, y1, y2):
    fig_lr = go.Figure()
    fig_lr.add_trace(go.Scatter(x=x, y=y1, mode='lines+markers', name="Actual", line=dict(color='blue')))
    fig_lr.add_trace(go.Scatter(x=x, y=y2, mode='lines+markers', name="Predicted", line=dict(color='orange')))
    fig_lr.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend=dict(x=0, y=1.1, orientation="h"),
        margin=dict(l=5, r=5, t=70, b=5)
    )
    return fig_lr

# @st.cache_data
def goMapPlot(trips_by_location, taxi_type):
        merged_gdf = polygon_gdf.merge(trips_by_location, how='left', left_on='location_id', right_on='pu_location_id')
        # merged_gdf['trip_count'] = merged_gdf['trip_count'].fillna(0) 
        merged_gdf_json = json.loads(merged_gdf.to_json())

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

        return fig

# @st.cache_data
def goBoxPlot(stats, taxi_type, feature):
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

# @st.cache_data
def buildComparison(data, taxi_type):
    X_LR, y_LR = data[FEATURES], data[VARIABLE] 
    X_RFR, y_RFR = data[FEATURES], data[VARIABLE]
    
    lr_model = loadModel(f'{taxi_type}_{LINEAR_REGRESSION}')
    y_pred_lr = lr_model.predict(X_LR)
    
    rf_model = loadModel(f'{taxi_type}_{RANDOM_FOREST}')
    y_pred_rf = rf_model.predict(X_RFR)
    
    fig_lr = goLinePlot(f"{taxi_type.capitalize()} Taxi - Linear Regression Forecast", 
                        "Sample Index","Total Amount", y_LR.index, y_LR, y_pred_lr)
    
    fig_rf = goLinePlot(f"{taxi_type.capitalize()} Taxi - Random Forest Forecast", 
                        "Sample Index","Total Amount", y_RFR.index, y_RFR, y_pred_rf)
    
    metrics = {
        "Metric": ["R-squared (%)", "MAE", "RMSE", "MAPE (%)"],
        "Linear Regression": errors(y_LR, y_pred_lr),
        "Random Forest": errors(y_RFR, y_pred_rf),
    }
    fig_table = go.Figure(data=[go.Table(
        header=dict(values=["Metric", "Linear Regression", "Random Forest"], fill_color='darkgrey', align='center'),
        cells=dict(values=[metrics["Metric"], metrics["Linear Regression"], metrics["Random Forest"]], fill_color='black', align='center')
    )])
    fig_table.update_layout(title=f"{taxi_type.capitalize()} Taxi Model Performance Metrics", margin=dict(l=5, r=5, t=30, b=5))
    
    return fig_lr, fig_rf, fig_table


# ------------------------------------------------------------------------------------------------
# Cached Data
# ------------------------------------------------------------------------------------------------
@st.cache_data
def getSampleData(taxi_type):
    return getSample(2023, taxi_type, 150)

@st.cache_data
def getTripData(year, taxi_type):
    return getTripCountByPickup(year, taxi_type)

@st.cache_data
def getBoxPlotData(taxi_type):
    return loadObject(f'{taxi_type}_{BOX_PLOT_CACHE}', 'cache')

# ------------------------------------------------------------------------------------------------
# Section Renderers
# ------------------------------------------------------------------------------------------------
# GENERAL
def renderGeneral():
    summary = {
        'green': getBoxPlotData(GREEN),
        'yellow': getBoxPlotData(YELLOW)
    }
    features = summary[GREEN].keys()

    sp_col1, _,_,_ = st.columns(4)
    with sp_col1:
        feature = st.selectbox("Summary for ", options=features, index=0)

    col1, col2 = st.columns(2)
    # show distribution
    #   collect data
    #   make box plot
    with col1:
        g_fig = goBoxPlot(summary[GREEN][feature], GREEN, feature)
        st.plotly_chart(g_fig, use_container_width=True)
    
    with col2:
        y_fig = goBoxPlot(summary[YELLOW][feature], YELLOW, feature)
        st.plotly_chart(y_fig, use_container_width=True)

# MAPS
def renderMaps():
    h_col1, h_col2, h_col3 = st.columns(3)
    year = h_col1.selectbox("Year", options=available_years, index=3)

    col1, col2 = st.columns(2)

    data = {}
    data[GREEN] = getTripData(year, GREEN)
    data[YELLOW] = getTripData(year, YELLOW)
    
    col1.plotly_chart(goMapPlot(data[GREEN], GREEN))
    col2.plotly_chart(goMapPlot(data[YELLOW], YELLOW))

# COMPARE
def renderComparison():
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("## Green Taxi Forecast")
        # get ransom sample of n rows by taxi_type
        df = getSampleData(GREEN)
        green_fig_lr, green_fig_rf, green_fig_table = buildComparison(df, GREEN)
        st.plotly_chart(green_fig_lr, use_container_width=True)
        st.plotly_chart(green_fig_rf, use_container_width=True)
        st.plotly_chart(green_fig_table, use_container_width=True)
        
    with col2:
        st.write("## Yellow Taxi Forecast")
        df = getSampleData(YELLOW)
        yellow_fig_lr, yellow_fig_rf, yellow_fig_table = buildComparison(df, YELLOW)
        st.plotly_chart(yellow_fig_lr, use_container_width=True)
        st.plotly_chart(yellow_fig_rf, use_container_width=True)
        st.plotly_chart(yellow_fig_table, use_container_width=True)

# RESULT
def renderResult():
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


# ------------------------------------------------------------------------------------------------
# Dashboard Layout
# ------------------------------------------------------------------------------------------------
# Set page layout, add sidebar and list menu
st.set_page_config(layout="wide")

st.sidebar.title("NYC Taxi Trips Dashboard")
st.sidebar.markdown("<small>Author: Ishani Makwana</small>", unsafe_allow_html=True)
st.sidebar.markdown("---")

section = st.sidebar.radio("Go to", [GENERAL, MAPS, COMPARE, RESULT])

# Render main section by menu item
if section == GENERAL:
    # green vs yellow taxi
    renderGeneral()

    # show weekly average
    # show by location 
elif section == MAPS:
    renderMaps()

elif section == COMPARE:
    renderComparison()

elif section == RESULT:
    renderResult()
# ------------------------------------------------------------------------------------------------
