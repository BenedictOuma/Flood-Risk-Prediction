import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import folium
from streamlit_folium import st_folium

#Helper Functions

def simulate_data(n=100):
    """Simulate geospatial village data"""
    np.random.seed(0)
    longitudes = np.random.uniform(36.5, 37.5, n)
    latitudes = np.random.uniform(-1.0, 0.0, n)
    geometries = [Point(xy) for xy in zip(longitudes, latitudes)]
    
    villages = gpd.GeoDataFrame({
        'elevation': np.random.uniform(100.0, 500.0, n),
        'distance_to_river': np.random.uniform(0.0, 10.0, n),
        'rainfall_last_7days': np.random.uniform(0.0, 300.0, n),
        'soil_permeability': np.random.uniform(0.0, 1.0, n),
        'ndvi': np.random.uniform(-0.1, 0.8, n),
        'slope': np.random.uniform(0.0, 30.0, n)
    }, geometry=geometries, crs="EPSG:4326")
    
    return villages


def classify_flood_risk(row, elev_thresh, dist_thresh, rain_thresh, soil_thresh):
    """Binary classification based on thresholds"""
    if (row['elevation'] < elev_thresh and
        row['distance_to_river'] < dist_thresh and
        row['rainfall_last_7days'] > rain_thresh and
        row['soil_permeability'] < soil_thresh):
        return "High"
    else:
        return "Low"


def train_model(villages):
    """Train Random Forest model and return performance report"""
    X = villages[['elevation', 'distance_to_river', 'rainfall_last_7days', 
                  'soil_permeability', 'ndvi', 'slope']]
    y = villages['flood_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42,
                                                        stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    return model, report_dict


def create_map(villages):
    """Create Folium map with predictions"""
    risk_colors = {"High": "red", "Low": "green"}
    m = folium.Map(location=[-0.5, 37.0], zoom_start=8)
    
    for _, row in villages.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color=risk_colors[row['flood_risk']],
            fill=True,
            fill_opacity=0.8
        ).add_to(m)
    return m

#Streamlit UI

st.set_page_config(page_title="Flood Risk Prediction", layout="wide")
st.title("Flood Risk Prediction & Mapping")

# File upload or simulate
uploaded_file = st.file_uploader("Upload CSV with village data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'longitude' in df.columns and 'latitude' in df.columns:
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        villages = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    else:
        st.error("Your CSV must have **'longitude'** and **'latitude'** columns!")
        st.stop()
else:
    st.info("Using **simulated data** (upload your CSV to use real data).")
    villages = simulate_data()

#Setting threshold controls
st.sidebar.header("Adjust the thresholds")

elev_thresh = st.sidebar.number_input(
    "Elevation threshold (m)",
    0.0, 600.0,
    float(villages['elevation'].quantile(0.7)),
    step=10.0
)
dist_thresh = st.sidebar.number_input(
    "Distance to river threshold (km)",
    0.0, 10.0,
    float(villages['distance_to_river'].quantile(0.7)),
    step=0.1
)
rain_thresh = st.sidebar.number_input(
    "Rainfall threshold (mm in last 7 days)",
    0.0, 300.0,
    float(villages['rainfall_last_7days'].quantile(0.4)),
    step=5.0
)
soil_thresh = st.sidebar.number_input(
    "Soil permeability threshold",
    0.0, 1.0,
    float(villages['soil_permeability'].quantile(0.8)),
    step=0.05
)

ndvi_thresh = st.sidebar.number_input(
    "NDVI threshold",
    -0.1, 1.0,
    float(villages['ndvi'].quantile(0.5)),  # default = median
    step=0.05
)

slope_thresh = st.sidebar.number_input(
    "Slope threshold (degrees)",
    0.0, 40.0,
    float(villages['slope'].quantile(0.5)),  # default = median
    step=1.0
)

lon_min, lon_max = st.sidebar.slider(
    "Longitude range",
    float(villages.geometry.x.min()), 
    float(villages.geometry.x.max()),
    (float(villages.geometry.x.min()), float(villages.geometry.x.max())),
    step=0.01
)
lat_min, lat_max = st.sidebar.slider(
    "Latitude range",
    float(villages.geometry.y.min()), 
    float(villages.geometry.y.max()),
    (float(villages.geometry.y.min()), float(villages.geometry.y.max())),
    step=0.01
)

#Applying filters
villages = villages[(villages.geometry.x.between(lon_min, lon_max)) &
                    (villages.geometry.y.between(lat_min, lat_max))]

#Adding a footer in the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align: center; color: grey; font-size: 14px;'>"
    "Built By <b>Benedict Ouma</b> | Your Safety Starts Here"
    "</div>",
    unsafe_allow_html=True
)

#Classifying flood risk
villages['flood_risk'] = villages.apply(
    classify_flood_risk, axis=1,
    elev_thresh=elev_thresh, dist_thresh=dist_thresh,
    rain_thresh=rain_thresh, soil_thresh=soil_thresh
)

#Showing distribution
st.subheader("Flood Risk Distribution")
st.write(villages['flood_risk'].value_counts())

#Training model and showing performance
model, report = train_model(villages)

st.subheader("Model Performance")
st.metric("Overall Accuracy", f"{report['accuracy']:.2%}")

# Display performance as a table
report_df = pd.DataFrame(report).transpose().drop(columns='support', errors='ignore')
st.dataframe(report_df.style.format("{:.2f}"))

# Flood Risk Map
st.subheader("Flood Risk Map")
map_obj = create_map(villages)
st_folium(map_obj, width=800, height=500)

#Saving the model
if st.sidebar.button("Save Model"):
    joblib.dump(model, "flood_risk_model.pkl")
    st.sidebar.success("Model saved as flood_risk_model.pkl")