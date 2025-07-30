# Flood Risk Prediction & Mapping

This project is a **Streamlit-based geospatial machine learning app** that predicts and visualizes **flood risk levels** (High/Low) for villages based on environmental factors such as elevation, distance to rivers, rainfall, soil properties, vegetation index, and slope.

The app allows you to:

* Upload your own CSV data or use simulated data.
* Adjust classification thresholds interactively.
* View model performance metrics.
* Visualize the flood risk distribution on an interactive **Folium map**.
* Save the trained model.

---

## Features

* **Data Upload:** Upload a CSV containing village features & coordinates (longitude, latitude).
* **Simulated Data:** If no file is uploaded, the app auto-generates sample data.
* **Threshold Control:** Adjust thresholds (e.g., elevation, rainfall) via sidebar sliders.
* **Interactive Map:** View risk classification on an interactive map using **Folium**.
* **Model Training:** Train a `RandomForestClassifier` in real-time with updated data.
* **Performance Metrics:** Visualize precision, recall, f1-score, and accuracy of the model.

---

## CSV File Requirements

If uploading a CSV file, it **must contain** the following columns:

* `longitude`
* `latitude`
* `elevation` (m)
* `distance_to_river` (km)
* `rainfall_last_7days` (mm)
* `soil_permeability` (0–1)
* `ndvi` (Normalized Difference Vegetation Index)
* `slope` (degrees)

Example row:

```csv
longitude,latitude,elevation,distance_to_river,rainfall_last_7days,soil_permeability,ndvi,slope
36.82,-1.28,220,1.8,190,0.25,0.45,12
```

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/BenedictOuma/Flood-Risk-Prediction.git
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv flood
   source flood/bin/activate  # Mac/Linux
   flood\Scripts\activate     # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

## Deployment

The app can be deployed easily on:

* **Streamlit Cloud** – [streamlit.io/cloud](https://streamlit.io/cloud)

Ensure that your **requirements.txt** and **app.py** are in the root folder before deploying.

---

## Screenshots

### App Home

![App Home](Pics/Screenshot%202025-07-30%20192552.png)

### Interactive Flood Map

![Flood Map](Pics/Screenshot%202025-07-30%20192612.png)

---

## Author

**Benedict Ouma**
"Your Safety Starts Here"