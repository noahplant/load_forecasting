import meteostat
import os
import pandas as pd
import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
import folium
import webbrowser


abs_path = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(abs_path,'data','explore')

STATIONS_FILE = 'stations.csv'

LOAD_FROM_FILE = True

WEATHER_STATIONS = [

    'KLGA', 'KBGM', 'KSYR', 'KGFL', 'KSLK', 'KJFK', 'KART', 'KELZ', 'KFOK', 'KIAG',
    'KMSS', 'KPOU', 

]



def get_meteostat():
    stations = meteostat.Stations()

    stations = stations.region(country='US',state='NY')
    df_stations = stations.fetch()
    return df_stations

def get_ny_map(df):


    # Create a map centered on New York
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10)



    for row in df.itertuples():

        print(row.icao)

        if row.icao in WEATHER_STATIONS:

            # Add a marker
            coord = [row.latitude,row.longitude]
            popup = row.icao
            folium.Marker(coord, popup=popup, tooltip="Click for info").add_to(m)
        else:
            print('else')

    # Bisector Points:

    locations = [
    (40.7128, -74.0060),   # New York City
    (40.73061, -73.935242), # Brooklyn
    (40.785091, -73.968285) # Central Park

    ]

    # Add a PolyLine to connect the points
    folium.PolyLine(locations, color="blue", weight=5, opacity=0.7).add_to(m)

    # Save the map to an HTML file
    map_filename = "map.html"
    m.save(map_filename)

    # Open the file in the default web browser
    webbrowser.open(map_filename)
    


def explore():

    print("Data Exploration Output: ")
    file_dir = os.path.join(DATA_DIR,STATIONS_FILE)

    if LOAD_FROM_FILE:
        df_stations = pd.read_csv(file_dir)
    else:
        df_stations = get_meteostat()

        df_stations.to_csv(file_dir)

    get_ny_map(df_stations)

    
    return df_stations






if __name__ == '__main__':
    explore()
