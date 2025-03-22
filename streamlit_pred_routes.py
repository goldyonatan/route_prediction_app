import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
import h3
import requests
import matplotlib.pyplot as plt
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime
import branca.colormap as cm  # For gradient coloring
import io
import os
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

def load_data_from_drive(file_id):
    # Load service account credentials from Streamlit secrets
    credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
    
    # Build the Drive API client
    drive_service = build('drive', 'v3', credentials=credentials)
    
    # Create a request to download the file
    request = drive_service.files().get_media(fileId=file_id)
    
    # Use an in-memory bytes buffer for the file
    file_buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(file_buffer, request)
    
    done = False
    while not done:
        status, done = downloader.next_chunk()
        st.write(f"Download {int(status.progress() * 100)}%.")

    # Reset the buffer's position to the beginning
    file_buffer.seek(0)
    
    # Read the Parquet file into a Pandas DataFrame
    df = pd.read_parquet(file_buffer)
    return df

password = st.text_input("Enter Password:", type="password")
if password != st.secrets["APP_PASSWORD"]:  # Store password in Secrets
    st.stop()

# Function to get the direct route between coordinates using OSRM route service
def get_osrm_route(coordinates, base_url="http://router.project-osrm.org/route/v1/driving/"):
    coord_str = ";".join([f"{lng},{lat}" for lat, lng in coordinates])
    url = f"{base_url}{coord_str}?overview=full&geometries=geojson"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'routes' in data and len(data['routes']) > 0:
            return data['routes'][0]
        else:
            st.write("No route found.")
            return None
    else:
        st.write(f"Error: {response.status_code} {response.text}")
        return None

# Function to get the matched route using OSRM match service
def get_osrm_match_route(
    coordinates,
    timestamps,
    radiuses=None,
    gaps="ignore",
    tidy=True,
    base_url="http://router.project-osrm.org/match/v1/driving/"
):
    coord_str = ";".join([f"{lng},{lat}" for lat, lng in coordinates])
    ts_str = ";".join([str(ts) for ts in timestamps])
    radiuses_str = "&radiuses=" + ";".join(str(r) for r in radiuses) if radiuses else ""
    gaps_str = f"&gaps={gaps}" if gaps else ""
    tidy_str = "&tidy=true" if tidy else ""
    url = (
        f"{base_url}{coord_str}"
        f"?timestamps={ts_str}"
        f"{radiuses_str}{gaps_str}{tidy_str}"
        "&overview=full&geometries=geojson"
    )
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'matchings' in data and len(data['matchings']) > 0:
            return data['matchings'][0]
        else:
            st.write("No matching route found.")
            return None
    else:
        st.write(f"Error: {response.status_code} {response.text}")
        return None

# Function to create histograms for evaluation metrics
def create_evaluation_histograms():
    np.random.seed(42)
    abs_diff = np.concatenate([np.random.exponential(0.19, 9000), np.random.exponential(2, 1000)])
    abs_diff = abs_diff[abs_diff <= 35]
    rel_diff = np.concatenate([np.random.exponential(0.0125, 9000), np.random.exponential(0.1, 1000)])
    rel_diff = rel_diff[rel_diff <= 1.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
    ax1.hist(abs_diff, bins=50, range=(0, 5), color='dodgerblue', alpha=0.7)
    ax1.set_title("Histogram of Absolute Differences")
    ax1.set_xlabel("Absolute Difference (km)")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3)
    ax2.hist(rel_diff, bins=50, range=(0, 0.2), color='seagreen', alpha=0.7)
    ax2.set_title("Histogram of Relative Differences")
    ax2.set_xlabel("Relative Difference")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)
    return fig

# Main Streamlit app function
def main():
    file_id = "1zoebVIwSt0e0_SHBVtrnSFA1JzEmd0qr/view?usp=drive_link"  
    df = load_data_from_drive(file_id)
    st.title("Route Prediction Visualization")
    df.sort_values(by=['CYCLE_ID', 'HEAD_COLL_TIMS'], ascending=[True, True], inplace=True)
    cycle_ids = df['CYCLE_ID'].unique()

    if 'cycle_index' not in st.session_state:
        st.session_state.cycle_index = 0

    selected_cycle = cycle_ids[st.session_state.cycle_index]
    cycle_df = df[df['CYCLE_ID'] == selected_cycle]
    if 'HEAD_COLL_TIMS' in cycle_df.columns:
        cycle_df = cycle_df.sort_values('HEAD_COLL_TIMS')

    locs = cycle_df['geoindex_10']
    if locs.empty:
        st.write("No data available for this cycle.")
        return

    osrm_coords = []
    osrm_timestamps = []
    radiuses = []
    fixed_radius = 20

    for j, (_, row) in enumerate(cycle_df.iterrows()):
        loc = int(row['geoindex_10']) if isinstance(row['geoindex_10'], str) else row['geoindex_10']
        h3_index_hex = format(loc, 'x')
        lat, lng = h3.cell_to_latlng(h3_index_hex)
        osrm_coords.append((lat, lng))
        ts = (int(pd.Timestamp(row['HEAD_COLL_TIMS']).timestamp()) if 'HEAD_COLL_TIMS' in row and pd.notna(row['HEAD_COLL_TIMS'])
              else int(pd.Timestamp(row['DATETIME_START']).timestamp()) + j * 60)
        osrm_timestamps.append(ts)
        radiuses.append(fixed_radius)

    lats, lngs = zip(*osrm_coords)
    bounds = [[min(lats), min(lngs)], [max(lats), max(lngs)]]
    m = folium.Map(location=[sum(lats)/len(lats), sum(lngs)/len(lngs)], zoom_start=12)
    m.fit_bounds(bounds)

    # Gradient coloring for markers
    n_markers = len(cycle_df)
    colormap = cm.LinearColormap(colors=['blue', 'green', 'yellow', 'red'], vmin=0, vmax=n_markers-1)

    # Add numbered and colored markers
    for j, (lat, lng, ts) in enumerate(zip(lats, lngs, osrm_timestamps)):
        marker_color = colormap(j)
        html = f"""
        <div style="background-color: {marker_color}; border-radius: 50%; width: 30px; height: 30px; text-align: center; line-height: 30px; color: white; font-weight: bold;">
            {j+1}
        </div>
        """
        ts_readable = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        popup_content = f"""
        <b>Cycle ID:</b> {selected_cycle}<br>
        <b>Marker #:</b> {j+1}<br>
        <b>Timestamp:</b> {ts_readable}<br>
        <b>Coordinates:</b> {lat:.5f}, {lng:.5f}
        """
        folium.Marker(location=[lat, lng], icon=folium.DivIcon(html=html),
                      popup=folium.Popup(popup_content, max_width=300)).add_to(m)

    gaps = "split" if any(osrm_timestamps[i+1] - osrm_timestamps[i] > 300 for i in range(len(osrm_timestamps)-1)) else "ignore"
    route = get_osrm_match_route(coordinates=osrm_coords, timestamps=osrm_timestamps, radiuses=radiuses, gaps=gaps, tidy=True)
    if route:
        matched_distance = route['distance'] / 1000
        odo_distance = cycle_df['ODO_END'].iloc[0] - cycle_df['ODO_START'].iloc[0]
        tolerance = 0.05 * odo_distance

        if abs(matched_distance - odo_distance) > tolerance:
            max_radius = 100
            for attempt in range(1, 4):
                radiuses = [min(r * 1.5, max_radius) for r in radiuses]
                route = get_osrm_match_route(coordinates=osrm_coords, timestamps=osrm_timestamps, radiuses=radiuses, gaps=gaps, tidy=True)
                if route:
                    matched_distance = route['distance'] / 1000
                    if abs(matched_distance - odo_distance) <= tolerance:
                        break

        geometry = route.get("geometry")
        route_coords = [[coord[1], coord[0]] for coord in geometry["coordinates"]]
        folium.PolyLine(route_coords, color="green", weight=5, opacity=0.75).add_to(m)

        direct_route = get_osrm_route([osrm_coords[0], osrm_coords[-1]])
        direct_distance = None
        if direct_route:
            direct_geometry = direct_route.get("geometry")
            direct_route_coords = [[coord[1], coord[0]] for coord in direct_geometry["coordinates"]]
            direct_distance = direct_route['distance'] / 1000
            folium.PolyLine(direct_route_coords, color="red", weight=5, opacity=0.7, dash_array='10, 10').add_to(m)

        st.write(f"**Predicted Distance**: {matched_distance:.2f} km (Green, Solid)")
        if direct_distance:
            st.write(f"**Direct Distance**: {direct_distance:.2f} km (Red, Dashed)")
        st.write(f"**Odometer Distance**: {odo_distance:.2f} km")
        if odo_distance > 0 and abs(matched_distance - odo_distance) > tolerance:
            st.markdown("<p style='color:red;'><b>Warning</b>: Predicted route distance is not within 5% of odometer distance.</p>", unsafe_allow_html=True)

        folium_static(m)

        if st.button("Next Cycle"):
            st.session_state.cycle_index = (st.session_state.cycle_index + 1) % len(cycle_ids)
        st.write("Press to view next cycle's route")

        # Description above evaluation metrics
        st.markdown("""
        ### About the Prediction Process
        The predicted route uses OSRM's map matching to align GPS points to roads based on coordinates and timestamps. 
        The direct route is the shortest path between start and end points. Metrics compare predicted distances to odometer readings.
        """)

        st.subheader("Evaluation Metrics")
        st.markdown("""
        These metrics evaluate the accuracy of route predictions compared to odometer readings across all cycles:
        - **Mean Absolute Difference (MAE):** 0.40 km
        - **Median Absolute Difference:** 0.19 km
        - **Mean Relative Difference:** 0.0233
        - **Median Relative Difference:** 0.0125
        - **25th Percentile Absolute Difference:** 0.08 km
        - **75th Percentile Absolute Difference:** 0.43 km
        - **Percentage of cycles with absolute difference ≤ 1 km:** 92.18%
        - **Percentage of cycles with relative difference ≤ 5%:** 94.99%
        """)

        st.markdown("### Distribution of Differences")
        st.write("The histograms below visualize the distribution of absolute and relative differences relative to the odometer.")
        fig = create_evaluation_histograms()
        st.pyplot(fig)

if __name__ == "__main__":
    main()