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
    df = pd.read_parquet(file_buffer)
    return df

def load_multi_trip_df_from_drive(file_id):
    # Use your Google Drive credentials (stored in Streamlit secrets, for example)
    credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
    drive_service = build('drive', 'v3', credentials=credentials)
    
    # Download the file
    request = drive_service.files().get_media(fileId=file_id)
    file_buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(file_buffer, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        st.write(f"Download {int(status.progress() * 100)}%.")
    file_buffer.seek(0)
    
    # Load the pickle file into a DataFrame
    multi_trip_df = pd.read_pickle(file_buffer)
    return multi_trip_df

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
    st.title("Multi-Trip Car Route Explorer")

    # Load data
    file_id = "1zoebVIwSt0e0_SHBVtrnSFA1JzEmd0qr"
    df = load_data_from_drive(file_id)

    multi_trip_file_id = "1D351PegNv7WPLnne3mSPuZTYBoYnB5KJ"
    multi_trip_df = load_multi_trip_df_from_drive(multi_trip_file_id)

    # Car selection
    car_options = multi_trip_df['Vehicle model'].unique()
    selected_car = st.selectbox("Select a Car", car_options)

    # Filter data for the selected car
    car_data = multi_trip_df[multi_trip_df['Vehicle model'] == selected_car].iloc[0]
    trip_ids = car_data['Trip ids']
    trip_subset = df[df['CYCLE_ID'].isin(trip_ids)].sort_values('DATETIME_START')

    # Display trip sequence in a table
    st.subheader(f"Trip Sequence for {selected_car}")
    trip_table = trip_subset[['CYCLE_ID', 'DATETIME_START', 'DATETIME_END']].copy()
    trip_table['Trip Number'] = range(1, len(trip_table) + 1)
    trip_table = trip_table[['Trip Number', 'CYCLE_ID', 'DATETIME_START', 'DATETIME_END']]
    trip_table['DATETIME_START'] = trip_table['DATETIME_START'].dt.strftime('%Y-%m-%d %H:%M:%S')
    trip_table['DATETIME_END'] = trip_table['DATETIME_END'].dt.strftime('%Y-%m-%d %H:%M:%S')
    st.table(trip_table)

    # Trip selection
    selected_trip_index = st.selectbox("Select a Trip to View Route", options=range(len(trip_ids)), format_func=lambda x: f"Trip {x+1}")
    selected_trip_id = trip_ids[selected_trip_index]

    # Display route for selected trip
    st.subheader(f"Route for Trip {selected_trip_index + 1} (ID: {selected_trip_id})")
    st.write(f"Starting Datetime: {trip_table.iloc[selected_trip_index]['DATETIME_START']}")
    trip_data = df[df['CYCLE_ID'] == selected_trip_id].sort_values('HEAD_COLL_TIMS')
    osrm_coords = []
    osrm_timestamps = []
    radiuses = []
    fixed_radius = 20

    for _, row in trip_data.iterrows():
        loc = int(row['geoindex_10']) if isinstance(row['geoindex_10'], str) else row['geoindex_10']
        h3_index_hex = format(loc, 'x')
        lat, lng = h3.cell_to_latlng(h3_index_hex)
        osrm_coords.append((lat, lng))
        ts = int(pd.Timestamp(row['HEAD_COLL_TIMS']).timestamp()) if 'HEAD_COLL_TIMS' in row and pd.notna(row['HEAD_COLL_TIMS']) else int(pd.Timestamp(row['DATETIME_START']).timestamp())
        osrm_timestamps.append(ts)
        radiuses.append(fixed_radius)

    if osrm_coords:
        lats, lngs = zip(*osrm_coords)
        m = folium.Map(location=[sum(lats)/len(lats), sum(lngs)/len(lngs)], zoom_start=12)
        for j, (lat, lng) in enumerate(osrm_coords):
            folium.Marker(location=[lat, lng], popup=f"Point {j+1}").add_to(m)

        gaps = "split" if any(osrm_timestamps[i+1] - osrm_timestamps[i] > 300 for i in range(len(osrm_timestamps)-1)) else "ignore"
        route = get_osrm_match_route(coordinates=osrm_coords, timestamps=osrm_timestamps, radiuses=radiuses, gaps=gaps, tidy=True)
        if route:
            geometry = route.get("geometry")
            route_coords = [[coord[1], coord[0]] for coord in geometry["coordinates"]]
            folium.PolyLine(route_coords, color="green", weight=5, opacity=0.75).add_to(m)
            folium_static(m)
        else:
            st.write("Could not find a matching route for this trip.")
    else:
        st.write("No coordinates available for this trip.")

if __name__ == "__main__":
    main()