# get data
from lib import *

import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import json


st.set_page_config(layout="wide")
available_years = [2020, 2021, 2022, 2023]

# # Streamlit app title
# st.title("NYC Taxi Trips Dashboard")
# st.subheader("By Ishani Makwana")
# Add title and author name in the sidebar
st.sidebar.title("NYC Taxi Trips Dashboard")
st.sidebar.markdown("<small>Author: Ishani Makwana</small>", unsafe_allow_html=True)

# Sidebar separator
st.sidebar.markdown("---")

# Date range selection
# st.sidebar.header("Select Date Range")

section = st.sidebar.radio("Go to", ["Home", "Visualization", "Modeling", "Comparison", "Result"])

if section == 'Visualization':
    # Split the Streamlit layout into two columns

    h_col1, h_col2, h_col3 = st.columns(3)
    year = h_col1.selectbox("Year", options=available_years, index=3)

    col1, col2 = st.columns(2)

    polygon_gdf = getTaxiGDF()

    data = {}
    data[GREEN] = getTripCountByPickup(year, GREEN)
    data[YELLOW] = getTripCountByPickup(year, YELLOW)

    # Function to create a map for each taxi type
    def create_map(taxi_type, column):
        # Filter the data for the specified taxi type
        trips_by_location = data[taxi_type]
        
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

    # Create the map for green taxis in the first column
    create_map(GREEN, col1)

    # Create the map for yellow taxis in the second column
    create_map(YELLOW, col2)
else:
    st.header(f'Section: {section}')
