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
RESIDUALS = 'Residual Analysis'
RESULT = 'Result'

available_years = [2020, 2021, 2022, 2023]
polygon_gdf = getTaxiGDF()

models = [
    ('LinearRegression', 'lr'),
    ('Ridge Regression', 'ridge'),
    ('RandomForestRegressor', 'rfr'),
    ('GradientBoostRegressor', 'gbm'),
    ('LGBMRegressor', 'lgbm'),
    ('XGBRegressor', 'xgbm'),
]

time_category_lists = {
    'time_of_day': ['Night', 'Morning', 'Afternoon', 'Evening'],
    'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
}

field_map_1 = {
    'total_amount': 'avg_total_amount', 
    'fare_amount': 'avg_fare_amount', 
    'tip_amount': 'avg_tip_amount', 
    'passenger_count': 'sum_passenger_count',
    'trip_duration': 'avg_trip_duration',
    'trip_distance': 'avg_trip_distance',
    'tolls_amount': 'avg_tolls_amount'
}
color_list = ["red", "yellow", "green", "blue", "orange", "purple", "pink"]

green_result_df = pd.DataFrame(
    data = [
        ['linear_regression', 0.82, 3.09, 4.95],
        ['random_forest', 0.93, 0.95, 2.97],
        ['gradient_boost', 0.94, 0.96, 2.82],
        ['xgb_regressor', 0.94, 0.86, 2.85],
        ['light_gbm_regressor', 0.94, 0.94, 2.87],
        ['ridge', 0.82, 3.09, 4.95],
        # ['neural_network', 0.89, 1.94, 3.82],
    ],
    columns=['model_name', 'r2', 'mae', 'rmse']
)

yellow_result_df = pd.DataFrame(
    data = [
        ['linear_regression', 0.95, 1.28, 3.14],
        ['random_forest', 0.98, 0.72, 2.25],
        ['gradient_boost', 0.97, 0.81, 2.51],
        ['xgb_regressor', 0.98, 0.61, 2.24],
        ['light_gbm_regressor', 0.98, 0.72, 2.27],
        ['ridge', 0.95, 1.28, 3.14],
        # ['neural_network', 0, 0, 0],
    ],
    columns=['model_name', 'r2', 'mae', 'rmse']
)

# ------------------------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------------------------
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

def plotByTimeCategory(data, category, x_fld, y_fld):
    fig = go.Figure()

    category_list = time_category_lists[category]

    for ind, d in enumerate(category_list):
        d_data = data[data[category] == d]
        # data.reset_index(inplace=True)
        fig.add_trace(go.Scatter(
            x=d_data[x_fld],  # Use the DataFrame index for the x-axis
            y=d_data[y_fld],
            mode="lines+markers",
            name=f"{d} {y_fld}",
            line=dict(color=color_list[ind]),
            marker=dict(size=6)
        ))

    # Update layout
    fig.update_layout(
        title=f"{y_fld} by {category}",
        xaxis_title=x_fld,
        yaxis_title=y_fld,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        template="plotly_white",
        height=600,
        legend=dict(
            x=0.2,  # Horizontal position (0 is left, 1 is right)
            y=1,  # Vertical position (1 is top, >1 is above the plot)
            xanchor="center",  # Anchor the legend horizontally at the center
            yanchor="top",     # Anchor the legend vertically at the top
            bgcolor="rgba(255, 255, 255, 0.5)"  # Semi-transparent white background
        )
    )
    
    return fig

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
            zoom=9,
            center={"lat": 40.7128, "lon": -74.0060},  # Center on NYC
            opacity=0.7,
            height=600,
            title=f"Trip Counts by Pickup Location ({taxi_type.capitalize()} Taxi)"
        )

        return fig

def goMapPlot2(data, field, tod):
    if tod == '':
        data = data.groupby(["pu_location_id"])[field].mean().reset_index()
        title = f"{field} by Pickup Location"
    else:
        data = data.groupby(["pu_location_id", "time_of_day"])[field].mean().reset_index()
        data = data[data['time_of_day'] == tod]
        title = f"{field} by Pickup Location - {tod}"

    merged_gdf = polygon_gdf.merge(data, how='left', left_on='location_id', right_on='pu_location_id')
    # merged_gdf['trip_count'] = merged_gdf['trip_count'].fillna(0) 
    merged_gdf_json = json.loads(merged_gdf.to_json())

    fig = px.choropleth_mapbox(
        merged_gdf,
        geojson=merged_gdf_json,
        locations='location_id',
        color=field,
        featureidkey="properties.location_id",
        hover_name="zone", 
        hover_data={field: True, "location_id": False},  # Hide location_id in tooltip, show trip_count
        color_continuous_scale="Viridis",
        mapbox_style="carto-positron",
        zoom=9,
        center={"lat": 40.7128, "lon": -74.0060},  # Center on NYC
        opacity=0.7,
        height=600,
        title=title
    )

    # fig.show()
    return fig

def goBoxPlot(stats, taxi_type, feature):
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=[stats["min"], stats["q1"], stats["median"], stats["q3"], stats["max"]],
        boxpoints=False,
        name=feature,
        marker=dict(color=taxi_type),  # Box fill color
        line=dict(color=f"{taxi_type}")  # Outline color of the box
        # orientation="h"
    ))
    fig.update_layout(
        title=f"{taxi_type.capitalize()} Taxi Box Plot",
        yaxis_title=feature
    )
    return fig

def plotTable(df):
    # Create Plotly Table
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(df.columns),  
                    fill_color="lightgrey",  
                    align="center",          
                    font=dict(size=15, color="black"),
                    height=30
                ),
                cells=dict(
                    values=[df[col] for col in df.columns], 
                    fill_color="white",                     
                    align="left",                           
                    font=dict(size=14, color="black"),
                    height=28
                )
            )
        ]
    )

    fig.update_layout(title="Model Comparison", #title_x=0.5,
                      margin=dict(l=10, r=10, t=40, b=10),
                      height = 275) 
    return fig

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

@st.cache_data
def getDailyData(taxi_type):
    return getCachedSql(DAILY_AVGS_CACHE, taxi_type)

@st.cache_data
def getHourlyData(taxi_type):
    return getCachedSql(HOURLY_AVGS_CACHE, taxi_type)

@st.cache_data
def getLocationData(taxi_type):
    return getCachedSql(PU_TIME_AVGS_CACHE, taxi_type)

@st.cache_data
def getTripFareEst(pu_location_id, do_location_id, time_of_day):
    data = {}
    for tt in [GREEN, YELLOW]:
        df = getCachedSql(PU_DO_TIME_AVGS_CACHE, tt)
    
        df = df.groupby(["pu_location_id", "do_location_id", "time_of_day"])['avg_fare_amount'].mean().reset_index()
        df = df[(df['pu_location_id'] == pu_location_id )
                            & (df['do_location_id'] == do_location_id )
                            & (df['time_of_day'] == time_of_day )]
        
        data[tt] = df
        print(df)

    if len(data[GREEN]) == 0 and len(data[YELLOW]) == 0:
        return ['''## Neither Green nor Yellow taxis are available for such trips. ''']
    elif len(data[GREEN]) == 0:
        yfa = data[YELLOW]['avg_fare_amount'].iloc[0]
        return [
            f'## Expect the fare amount to be ~{yfa:.2f} with a Yellow Taxi.',
            '### Yellow Taxis will be available for this trip.',
        ]
    
    elif len(data[YELLOW]) == 0:
        gfa = data[GREEN]['avg_fare_amount'].iloc[0]
        return [
            f'## Expect the fare amount to be ~{gfa:.2f} with a Green Taxi.',
            '### Green Taxis will be available for this trip.'
        ]
        
    else:
        gfa = data[GREEN]['avg_fare_amount'].iloc[0]
        yfa = data[YELLOW]['avg_fare_amount'].iloc[0]
        if gfa < yfa:
            return [
                f'## Expect the fare amount to be ~{gfa:.2f} with a Green Taxi.',
                '### Green Taxis should be preferred over Yellow for this trip as they offer lower fares.'
            ]
        else:
            return [
                f'## Expect the fare amount to be ~{yfa:.2f} with a Yellow Taxi.',
                '### Yellow Taxis should be preferred over Green for this trip as they offer lower fares.'
            ]
    

# ------------------------------------------------------------------------------------------------
# Section Renderers
# ------------------------------------------------------------------------------------------------
# GENERAL
def renderGeneral():
    col1, col2 = st.columns(2)
    col1.write("## Green Taxi Data Distribution")
    col2.write("## Yellow Taxi Data Distribution")

    summary = {
        'green': getBoxPlotData(GREEN),
        'yellow': getBoxPlotData(YELLOW)
    }
    daily_data = {
        'green': getDailyData(GREEN),
        'yellow': getDailyData(YELLOW)
    }
    hourly_data = {
        'green': getHourlyData(GREEN),
        'yellow': getHourlyData(YELLOW)
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

        st.write("### Green Taxi By Pickup Time")
        # day of week data plot for selected feature
        fig = plotByTimeCategory(daily_data[GREEN], 'day_of_week', 'day', field_map_1[feature])
        st.plotly_chart(fig, use_container_width=True)
        # hour of day data plot for selected feature
        fig = plotByTimeCategory(hourly_data[GREEN], 'time_of_day', 'hour', field_map_1[feature])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        y_fig = goBoxPlot(summary[YELLOW][feature], YELLOW, feature)
        st.plotly_chart(y_fig, use_container_width=True)

        st.write("### Yellow Taxi By Pickup Time")
        # day of week data plot for selected feature
        fig = plotByTimeCategory(daily_data[YELLOW], 'day_of_week', 'day', field_map_1[feature])
        st.plotly_chart(fig, use_container_width=True)
        # hour of day data plot for selected feature
        fig = plotByTimeCategory(hourly_data[YELLOW], 'time_of_day', 'hour', field_map_1[feature])
        st.plotly_chart(fig, use_container_width=True)

# MAPS
def renderMaps():

    col1, col2 = st.columns(2)
    col1.write("## Green Taxi By Pickup Location")
    col2.write("## Yellow Taxi By Pickup Location")
    
    location_data = {}
    location_data[GREEN] = getLocationData(GREEN)
    location_data[YELLOW] = getLocationData(YELLOW)

    hcol1, hcol2 = st.columns(2)
    feature = hcol1.selectbox("Feature: ", options=field_map_1.keys(), index=0)
    time_of_day = hcol2.selectbox("Time of day: ", options=time_category_lists['time_of_day'], index=0)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(goMapPlot2(location_data[GREEN], field_map_1[feature], time_of_day))
    
    with col2:
        st.plotly_chart(goMapPlot2(location_data[YELLOW], field_map_1[feature], time_of_day))

    h_col1, h_col2, h_col3 = st.columns(3)
    year = h_col1.selectbox("Year", options=available_years, index=3)

    data = {}
    data[GREEN] = getTripData(2023, GREEN)
    data[YELLOW] = getTripData(2023, YELLOW)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(goMapPlot(data[GREEN], GREEN))
    
    with col2:
        st.plotly_chart(goMapPlot(data[YELLOW], YELLOW))

# COMPARE
def renderComparison():
    tab1, tab2 = st.tabs(['Visual Validation', 'Performance'])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.write("## Green Taxi Forecast")

            st.image('img/green_lr.png', caption="LinearRegression Model Forecast", use_column_width=True)
            st.image('img/green_rfr.png', caption="RandomForestRegressor Model Forecast", use_column_width=True)
            st.image('img/green_gbm.png', caption="GradientBoostRegressor Model Forecast", use_column_width=True)
            st.image('img/green_xgbm.png', caption="XGBRegressor Model Forecast", use_column_width=True)
            st.image('img/green_lgbm.png', caption="LGBMRegressor Model Forecast", use_column_width=True)
            st.image('img/green_ridge.png', caption="Ridge Model Forecast", use_column_width=True)

        with col2:
            st.write("## Yellow Taxi Forecast")

            st.image('img/yellow_lr.png', caption="LinearRegression Model Forecast", use_column_width=True)
            st.image('img/yellow_rfr.png', caption="RandomForestRegressor Model Forecast", use_column_width=True)
            st.image('img/yellow_gbm.png', caption="GradientBoostRegressor Model Forecast", use_column_width=True)
            st.image('img/yellow_xgbm.png', caption="XGBRegressor Model Forecast", use_column_width=True)
            st.image('img/yellow_lgbm.png', caption="LGBMRegressor Model Forecast", use_column_width=True)
            st.image('img/yellow_ridge.png', caption="Ridge Model Forecast", use_column_width=True)

            
        # for tt in [GREEN, YELLOW]:
        #     st.write(f'### {tt.capitalize()} Taxi')
        #     cols = st.columns(2)
        #     for i, (name, id) in enumerate(models):
        #         cols[i % 2].image(f'img/{tt}_{id}.png', caption=f"{name} Model Forecast", use_column_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.write("## Green Taxi Forecast")

            # st.image('img/green_lr.png', caption="LinearRegression Model Forecast", use_column_width=True)
            # st.image('img/green_rfr.png', caption="RandomForestRegressor Model Forecast", use_column_width=True)
            # st.image('img/green_gbm.png', caption="GradientBoostRegressor Model Forecast", use_column_width=True)
            # st.image('img/green_xgbm.png', caption="XGBRegressor Model Forecast", use_column_width=True)
            # st.image('img/green_lgbm.png', caption="LGBMRegressor Model Forecast", use_column_width=True)
            # st.image('img/green_ridge.png', caption="Ridge Model Forecast", use_column_width=True)

            fig = plotTable(green_result_df)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.write("## Yellow Taxi Forecast")

            # st.image('img/yellow_lr.png', caption="LinearRegression Model Forecast", use_column_width=True)
            # st.image('img/yellow_rfr.png', caption="RandomForestRegressor Model Forecast", use_column_width=True)
            # st.image('img/yellow_gbm.png', caption="GradientBoostRegressor Model Forecast", use_column_width=True)
            # st.image('img/yellow_xgbm.png', caption="XGBRegressor Model Forecast", use_column_width=True)
            # st.image('img/yellow_lgbm.png', caption="LGBMRegressor Model Forecast", use_column_width=True)
            # st.image('img/yellow_ridge.png', caption="Ridge Model Forecast", use_column_width=True)

            fig = plotTable(yellow_result_df)
            st.plotly_chart(fig, use_container_width=True)

# RESIDUALS
def renderResiduals():
    
    c1,_,_ = st.columns(3)
    tab_options = ['Residual VS Predicted', 'Q-Q Plot']
    tabs = st.tabs(tab_options)
    # plt_type = c1.selectbox("Select Plot", options=['Residual VS Predicted', 'Q-Q Plot'], index=0)
    plots = {
        'Residual VS Predicted': 'rvp',
        'Q-Q Plot': 'qq'
    }

    for tab_idx, tab in enumerate(tabs):
        with tab:
            for tt in [GREEN, YELLOW]:
                st.write(f'### {tt.capitalize()} Taxi')
                cols = st.columns(6)
                for i, (name, id) in enumerate(models):
                    cols[i].image(f'img/{tt}_{id}_{plots[tab_options[tab_idx]]}.png', caption=f"{name}", use_column_width=True)

# RESULT
def renderResult():
    location_options = getLocationOptions()
    col1, col2, col3 = st.columns(3)

    with col1:
        pickup_location_label = st.selectbox("Pickup Location", options=list(location_options.keys()))
    pickup_location_value = location_options[pickup_location_label]

    with col2:
        dropoff_location_label = st.selectbox("Dropoff Location", options=list(location_options.keys()))
    dropoff_location_value = location_options[dropoff_location_label]

    with col3:
        time_of_day = st.selectbox("Time: ", options=time_category_lists['time_of_day'], index=0)

    # if st.button("Find Best Taxi"):
        # st.write(f"## And the winner is: {GREEN}")
    msgs = getTripFareEst(pickup_location_value, dropoff_location_value, time_of_day)
    for msg in msgs:
        st.write(msg)


# ------------------------------------------------------------------------------------------------
# Dashboard Layout
# ------------------------------------------------------------------------------------------------
# Set page layout, add sidebar and list menu
st.set_page_config(layout="wide")

st.sidebar.title("NYC Taxi Trips Dashboard")
st.sidebar.markdown("<small>Author: Ishani Makwana</small>", unsafe_allow_html=True)
st.sidebar.markdown("---")

section = st.sidebar.radio("Go to", [GENERAL, MAPS, RESIDUALS, COMPARE, RESULT])

# Render main section by menu item
if section == GENERAL:
    renderGeneral()

elif section == MAPS:
    renderMaps()

elif section == COMPARE:
    renderComparison()

elif section == RESIDUALS:
    renderResiduals()

elif section == RESULT:
    renderResult()
# ------------------------------------------------------------------------------------------------
