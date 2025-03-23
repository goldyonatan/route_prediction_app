import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
import h3
import requests
from datetime import datetime
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import matplotlib.pyplot as plt
import numpy as np
import branca.colormap as cm  # For gradient coloring

# Function to load data from Google Drive
def load_data_from_drive(file_id):
    credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
    drive_service = build('drive', 'v3', credentials=credentials)
    request = drive_service.files().get_media(fileId=file_id)
    file_buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(file_buffer, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        st.write(f"Download {int(status.progress() * 100)}%.")
    file_buffer.seek(0)
    return pd.read_parquet(file_buffer)

def load_multi_trip_df_from_drive(file_id):
    credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
    drive_service = build('drive', 'v3', credentials=credentials)
    request = drive_service.files().get_media(fileId=file_id)
    file_buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(file_buffer, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        st.write(f"Download {int(status.progress() * 100)}%.")
    file_buffer.seek(0)
    return pd.read_pickle(file_buffer)

# Function to get the matched route using OSRM match service
def get_osrm_match_route(coordinates, timestamps, radiuses=None, gaps="ignore", tidy=True):
    base_url = "http://router.project-osrm.org/match/v1/driving/"
    coord_str = ";".join([f"{lng},{lat}" for lat, lng in coordinates])
    ts_str = ";".join([str(ts) for ts in timestamps])
    radiuses_str = "&radiuses=" + ";".join(str(r) for r in radiuses) if radiuses else ""
    gaps_str = f"&gaps={gaps}" if gaps else ""
    tidy_str = "&tidy=true" if tidy else ""
    url = f"{base_url}{coord_str}?timestamps={ts_str}{radiuses_str}{gaps_str}{tidy_str}&overview=full&geometries=geojson"
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

# Main Streamlit app
def main():
    st.title("Multi-Trip Sequence Explorer")

    # Load data
    file_id = "1zoebVIwSt0e0_SHBVtrnSFA1JzEmd0qr"
    df = load_data_from_drive(file_id)
    df['DATETIME_START'] = pd.to_datetime(df['DATETIME_START'])
    df['DATETIME_END'] = pd.to_datetime(df['DATETIME_END'])

    # Aggregate df by CYCLE_ID to create trip_df with one row per CYCLE_ID
    trip_df = df.groupby('CYCLE_ID').agg({
        'FAMILY_LABEL': 'first',
        'DATETIME_START': 'first',
        'DATETIME_END': 'first',
        'ODO_START': 'first',
        'ODO_END': 'first',
        'geoindex_10_start': 'first',
        'geoindex_10_end': 'first',
    }).reset_index()

    multi_trip_file_id = "1D351PegNv7WPLnne3mSPuZTYBoYnB5KJ"
    multi_trip_df = load_multi_trip_df_from_drive(multi_trip_file_id)

    # Initialize session state for sequence index
    if 'sequence_index' not in st.session_state:
        st.session_state.sequence_index = 0

    # List of unique sequences
    sequence_ids = multi_trip_df['Sequence ID'].unique()
    total_sequences = len(sequence_ids)

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("Previous Sequence"):
            st.session_state.sequence_index = (st.session_state.sequence_index - 1) % total_sequences
    with col2:
        if st.button("Next Sequence"):
            st.session_state.sequence_index = (st.session_state.sequence_index + 1) % total_sequences
    with col3:
        st.write(f"Sequence {st.session_state.sequence_index + 1} of {total_sequences}")

    # Selected sequence
    selected_sequence_id = sequence_ids[st.session_state.sequence_index]
    selected_sequence = multi_trip_df[multi_trip_df['Sequence ID'] == selected_sequence_id].iloc[0]
    selected_model = selected_sequence['Vehicle model']
    st.subheader(f"Trips for Sequence {selected_sequence_id} (Model: {selected_model})")

    # Get ordered trips for the selected sequence using trip_df
    trip_ids = selected_sequence['Trip ids']
    trip_subset = trip_df[trip_df['CYCLE_ID'].isin(trip_ids)].sort_values('DATETIME_START')

    # Display trip table grouped by CYCLE_ID
    trip_table = trip_subset[[
        'CYCLE_ID', 'FAMILY_LABEL', 'ODO_START', 'ODO_END', 'DATETIME_START'
    ]].copy()
    trip_table['Index'] = range(1, len(trip_table) + 1)
    trip_table = trip_table[[
        'Index', 'CYCLE_ID', 'FAMILY_LABEL', 'ODO_START', 'ODO_END', 'DATETIME_START'
    ]]
    trip_table.columns = ['Index', 'Cycle Code', 'Vehicle Model', 'ODO Start (km)', 'ODO End (km)', 'Datetime']
    trip_table['Datetime'] = trip_table['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    st.table(trip_table)

    # Dropdown for trip selection using grouped CYCLE_IDs
    trip_options = trip_subset['CYCLE_ID'].tolist()
    selected_trip_id = st.selectbox("Select a Trip to View Route", options=trip_options, 
                                    format_func=lambda x: f"Trip {trip_options.index(x) + 1} (ID: {x})")

    # Display route for selected trip using all rows from df
    st.subheader(f"Route for Trip: {selected_trip_id}")
    trip_data = df[df['CYCLE_ID'] == selected_trip_id].sort_values(
        'HEAD_COLL_TIMS' if 'HEAD_COLL_TIMS' in df.columns else 'DATETIME_START'
    )
    
    osrm_coords = []
    osrm_timestamps = []
    radiuses = []
    fixed_radius = 20

    for j, (_, row) in enumerate(trip_data.iterrows()):
        loc = int(row['geoindex_10']) if isinstance(row['geoindex_10'], str) else row['geoindex_10']
        h3_index_hex = format(loc, 'x')
        lat, lng = h3.cell_to_latlng(h3_index_hex)
        osrm_coords.append((lat, lng))
        ts = (int(pd.Timestamp(row['HEAD_COLL_TIMS']).timestamp()) if 'HEAD_COLL_TIMS' in row and pd.notna(row['HEAD_COLL_TIMS'])
              else int(pd.Timestamp(row['DATETIME_START']).timestamp()) + j * 60)
        osrm_timestamps.append(ts)
        radiuses.append(fixed_radius)

    if osrm_coords:
        lats, lngs = zip(*osrm_coords)
        m = folium.Map(location=[sum(lats)/len(lats), sum(lngs)/len(lngs)], zoom_start=12)
        
        # Gradient coloring for markers
        n_markers = len(trip_data)
        colormap = cm.LinearColormap(colors=['blue', 'green', 'yellow', 'red'], vmin=0, vmax=n_markers-1)
        
        for j, (lat, lng, ts) in enumerate(zip(lats, lngs, osrm_timestamps)):
            marker_color = colormap(j)
            html = f"""
            <div style="background-color: {marker_color}; border-radius: 50%; width: 30px; height: 30px; text-align: center; line-height: 30px; color: white; font-weight: bold;">
                {j+1}
            </div>
            """
            ts_readable = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            popup_content = f"""
            <b>Trip ID:</b> {selected_trip_id}<br>
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
            odo_distance = trip_data['ODO_END'].iloc[0] - trip_data['ODO_START'].iloc[0]
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
            
            st.write(f"**Predicted Distance**: {matched_distance:.2f} km")
            st.write(f"**Odometer Distance**: {odo_distance:.2f} km")
            if odo_distance > 0 and abs(matched_distance - odo_distance) > tolerance:
                st.markdown("<p style='color:red;'><b>Warning</b>: Predicted route distance is not within 5% of odometer distance.</p>", unsafe_allow_html=True)
            
            folium_static(m)
        else:
            st.write("Could not find a matching route for this trip.")
    else:
        st.write("No coordinates available for this trip.")

    # Explanation
    st.markdown("""
    ### How This Works
    - **Table**: Displays one row per `CYCLE_ID` for the selected sequence, showing the first recorded values for `Vehicle Model`, `ODO Start`, `ODO End`, and `Datetime`.
    - **Trip Selection**: Allows you to pick a `CYCLE_ID` from the sequence to view its detailed route.
    - **Route Visualization**: Uses all data points from `df` for the selected `CYCLE_ID` to plot the full trip path.
    """)

if __name__ == "__main__":
    main()