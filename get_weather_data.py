import os
import pandas as pd
import datetime as dt
from meteostat import Stations, Hourly
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, 'data', 'weather')

HIST_FILE = 'nyiso_weather_hist.csv'   # Historical portion
CURR_FILE = 'nyiso_weather_curr.csv'   # Recent portion
MERGED_FILE = 'nyiso_weather.csv'      # Final merged data

WEATHER_STATIONS = [
    'KALB','KART','KBGM','KELM','KELZ',
    'KFOK','KFRG','KGFL','KHPN','KIAG',
    'KISP','KITH','KJFK','KLGA','KMSS',
    'KMSV','KNYC','KPLB','KPOU','KROC',
    'KSLK','KSWF','KSYR','KUCA','KBGM'
]

def get_stations(country='US', state='NY'):
    stations = Stations()
    stations = stations.region(country, state)
    df_stations = stations.fetch()
    return df_stations

def get_nyiso_station_ids():
    df_stations = get_stations(country='US', state='NY')
    # Filter to only the ICAOs in WEATHER_STATIONS
    df_stations = df_stations[df_stations['icao'].isin(WEATHER_STATIONS)]
    df_stations = df_stations[['icao']]
    df_stations.columns = ['station']
    return df_stations

def get_tz_name(dt):
    """Return 'EDT' if dt is in daylight savings time, otherwise 'EST'."""
    if dt.dst() != pd.Timedelta(0):
        return 'EDT'
    else:
        return 'EST'

def get_hourly_data(station, dt_from, dt_to):
    data = Hourly(station, dt_from, dt_to, timezone='America/New_York')
    df = data.fetch()
    return df

def drop_incomplete_stations(df, station_list, missing_threshold=0.9):
    """
    Drop entire stations if:
      1) Any column for that station has more than `missing_threshold` fraction of NaNs
      2) Any column has NaNs in the last 48 hours.
    """
    n_rows = len(df)
    max_missing = int(n_rows * missing_threshold)
    max_datetime = df.index.max()
    time_window_start = max_datetime - pd.Timedelta(hours=48)
    last_48h_index = df.index[df.index >= time_window_start]

    for stn in station_list:
        stn_cols = [f"{stn}_t", f"{stn}_w", f"{stn}_h", f"{stn}_s"]
        stn_cols = [c for c in stn_cols if c in df.columns]
        if not stn_cols:
            continue

        # 1) If any column has more than 'max_missing' NaNs, drop them all
        missing_count = df[stn_cols].isna().sum()
        if (missing_count > max_missing).any():
            df = df.drop(columns=stn_cols)
            continue

        # 2) If there's any NaN in the last 48 hours, drop them
        #if len(last_48h_index) > 0:
        #    if df.loc[last_48h_index, stn_cols].isna().any().any():
        #        df = df.drop(columns=stn_cols)
        #        continue

    return df

def clean_data(df, station_list):
    """
    Remove all-NaN columns, then drop entire stations failing threshold checks,
    then forward-fill missing data.
    """
    df = df.dropna(axis=1, how='all')  # remove columns that are entirely NaN
    df = drop_incomplete_stations(df, station_list, missing_threshold=0.9)
    df = df.ffill()  # forward fill
    return df

def download_weather_data(dt_from, dt_to, csv_filename):
    """
    Download weather data from dt_from to dt_to for all stations
    and save to csv_filename. Returns a DataFrame.
    """
    df_stations = get_nyiso_station_ids()
    station_id_list = df_stations.index.values.tolist()

    df_all = pd.DataFrame()

    for id_ in station_id_list:
        df_temp = get_hourly_data(id_, dt_from, dt_to)
        # Keep only these columns if they exist
        sub_cols = [c for c in ['temp', 'wspd', 'rhum', 'tsun'] if c in df_temp.columns]
        df_sub = df_temp[sub_cols].copy()

        # Rename them: station_t, station_w, station_h, station_s
        station_icao = df_stations.loc[id_].iat[0]
        rename_map = {}
        if 'temp' in df_sub.columns:
            rename_map['temp'] = f'{station_icao}_t'
        if 'wspd' in df_sub.columns:
            rename_map['wspd'] = f'{station_icao}_w'
        if 'rhum' in df_sub.columns:
            rename_map['rhum'] = f'{station_icao}_h'
        if 'tsun' in df_sub.columns:
            rename_map['tsun'] = f'{station_icao}_s'
        df_sub.rename(columns=rename_map, inplace=True)

        if df_all.empty:
            df_all = df_sub
        else:
            df_all = df_all.join(df_sub, how='left')

        time.sleep(2)

    # Add a TZ column
    df_all['TZ'] = df_all.index.map(get_tz_name)
    df_all.index.name = 'DateTime'
    df_all.index = df_all.index.tz_localize(None)

    station_list = df_stations['station'].tolist()
    df_all = clean_data(df_all, station_list)

    # Save to CSV
    df_all.to_csv(os.path.join(DATA_DIR, csv_filename))
    return df_all

def get_nyiso_hourly_weather_data():
    """
    1) If we haven't downloaded historical data, download from 2002-01-01 to 2025-01-01
       and save as nyiso_weather_hist.csv
    2) Download current data from 2025-01-01 to "today" and save as nyiso_weather_curr.csv
    3) Merge both into nyiso_weather.csv
    """
    # 1) Historical portion
    hist_path = os.path.join(DATA_DIR, HIST_FILE)
    if not os.path.exists(hist_path):
        dt_from_hist = dt.datetime(2002, 1, 1)
        dt_to_hist = dt.datetime(2025, 1, 1)
        print("Downloading historical weather data ...")
        df_hist = download_weather_data(dt_from_hist, dt_to_hist, HIST_FILE)
    else:
        df_hist = pd.read_csv(hist_path, parse_dates=['DateTime'], index_col='DateTime')
        print("Loaded historical weather data from file.")

    # 2) Current portion
    curr_path = os.path.join(DATA_DIR, CURR_FILE)
    dt_today = dt.datetime.today()
    dt_from_curr = dt.datetime(2025, 1, 1)
    # Until midnight tomorrow
    dt_to_curr = dt.datetime(dt_today.year, dt_today.month, dt_today.day, 23, 59) + dt.timedelta(days=1)

    print("Downloading current weather data ...")
    df_curr = download_weather_data(dt_from_curr, dt_to_curr, CURR_FILE)

    # 3) Merge historical + current
    # Because both data sets have the same columns, we can concatenate them on the index
    df_merged = pd.concat([df_hist, df_curr])
    df_merged = df_merged.sort_index()
    # Remove duplicates if any overlap
    df_merged = df_merged[~df_merged.index.duplicated(keep='last')]

    # Save final
    merged_path = os.path.join(DATA_DIR, MERGED_FILE)
    df_merged.to_csv(merged_path)
    print(f"Merged file saved to {merged_path}")

    return df_merged

if __name__ == '__main__':
    df = get_nyiso_hourly_weather_data()
    print(df.tail())
