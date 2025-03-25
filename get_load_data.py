import pandas as pd
import numpy as np
import requests
import time
from zipfile import ZipFile
from io import BytesIO
import datetime as dt
import os

# Get abs path of current file.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get path to data.
DATA_DIR = os.path.join(script_dir, 'data', 'load')

HIST_LOAD_FILE = 'nyiso_hist_load.csv'
CURR_LOAD_FILE = 'nyiso_curr_load.csv'
LOAD_FILE = 'nyiso_load.csv'

# List of load zones to process.
LOAD_ZONES = ['WEST', 'GENESE', 'CENTRL', 'NORTH', 'MHK VL', 'CAPITL', 'HUD VL', 'MILLWD', 'DUNWOD', 'N.Y.C.']

def get_load_data_from_nyiso(start_date, end_date):
    # Data frame for the final raw data
    df_load_raw = pd.DataFrame()

    # Loop through each csv (one per day) from start_date to end_date.
    pointer_date = start_date
    while pointer_date <= end_date:
        print(f'Loading file: {pointer_date}')
        # Format the date to fit the URL to scrape
        date_str = pointer_date.strftime('%Y%m%d')
        url = f'https://mis.nyiso.com/public/csv/palIntegrated/{date_str}palIntegrated_csv.zip'
        
        try:
            response = requests.get(url)
        except Exception as e:
            print("Error in URL request. Try 30 sec later")
            time.sleep(30)
            continue

        if response.status_code == 200:
            print('URL success!')
            # Download and process the zip file
            with ZipFile(BytesIO(response.content)) as z:
                # Iterate through files in the zip (each zip contains multiple CSV files)
                for file_name in z.namelist():
                    if file_name.endswith('.csv'):
                        with z.open(file_name) as f:
                            day_df = pd.read_csv(f)
                            df_load_raw = pd.concat([df_load_raw, day_df], ignore_index=True)
        else:
            print("Failed to download.")
        
        # Go to next month
        next_month = pointer_date.month + 1
        next_year = pointer_date.year + (next_month // 13)
        next_month = next_month % 12 or 12
        pointer_date = dt.datetime(next_year, next_month, 1)
        time.sleep(5)
    
    # Process the raw data:
    df_load_raw['DateTime'] = pd.to_datetime(df_load_raw['Time Stamp'])
    df_load_raw["Hour"] = df_load_raw['DateTime'].dt.floor('h')

    # Build a dictionary of DataFrames for each load zone.
    load_zone_dfs = {}
    for zone in LOAD_ZONES:
        # Use .str.strip() in case there are extra spaces in the Name column
        zone_df = df_load_raw[df_load_raw['Name'].str.strip() == zone].copy()
        zone_df = zone_df[['Hour', 'Time Zone', 'Integrated Load']]
        zone_df.rename(columns={'Hour': 'DateTime', 'Time Zone': 'TZ', 'Integrated Load': 'Load'}, inplace=True)
        load_zone_dfs[zone] = zone_df

    return load_zone_dfs

# Get Historical Data from either a local file or via the URL.
def get_hist_load_data():
    hist_load_file_path = os.path.join(DATA_DIR, HIST_LOAD_FILE)
    if os.path.exists(hist_load_file_path):
        # If the CSV exists, load it and re-create the dictionary structure.
        df_combined = pd.read_csv(hist_load_file_path)
        load_zone_dfs = {}
        for zone in LOAD_ZONES:
            zone_df = df_combined[df_combined['TZ'] == zone].copy()
            load_zone_dfs[zone] = zone_df
    else:
        start_date = dt.datetime(2002, 1, 1)
        end_date = dt.datetime(2025, 1, 1)
        load_zone_dfs = get_load_data_from_nyiso(start_date=start_date, end_date=end_date)
        # Save all zones combined into one CSV, adding a 'Zone' column.
        combined_df = pd.concat([df.assign(Zone=zone) for zone, df in load_zone_dfs.items()])
        combined_df.to_csv(hist_load_file_path, index=False)

    return load_zone_dfs

# Get current data from either a local file or via the URL.
def get_curr_load_data():
    curr_load_file_path = os.path.join(DATA_DIR, CURR_LOAD_FILE)
    start_date = dt.datetime(2025, 2, 1)
    end_date = dt.datetime(dt.datetime.today().year, dt.datetime.today().month, 1)
    load_zone_dfs = get_load_data_from_nyiso(start_date=start_date, end_date=end_date)
    # Save all zones combined into one CSV, adding a 'Zone' column.
    combined_df = pd.concat([df.assign(Zone=zone) for zone, df in load_zone_dfs.items()])
    combined_df.to_csv(curr_load_file_path, index=False)
    return load_zone_dfs

# Get all load data and merge historical and current data for each zone.
def get_load_data():
    curr_dfs = get_curr_load_data()
    hist_dfs = get_hist_load_data()

    # Merge the dictionaries by concatenating DataFrames for each zone.
    load_zone_dfs = {}
    for zone in LOAD_ZONES:
        # Use pd.concat even if one of the DataFrames is empty.
        load_zone_dfs[zone] = pd.concat([hist_dfs.get(zone, pd.DataFrame()), curr_dfs.get(zone, pd.DataFrame())])
    
    # Optionally, save the combined data for all zones into one CSV.
    combined_df = pd.concat([df.assign(Zone=zone) for zone, df in load_zone_dfs.items()])
    combined_path = os.path.join(DATA_DIR, LOAD_FILE)
    combined_df.to_csv(combined_path, index=False)

    return load_zone_dfs

if __name__ == '__main__':
    load_data_by_zone = get_load_data()
    # Now load_data_by_zone is a dictionary where each key is a zone name and
    # its value is the corresponding DataFrame.
    for zone, df in load_data_by_zone.items():
        print(f"Zone: {zone}")
        print(df.head())
