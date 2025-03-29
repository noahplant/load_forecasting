import os
import sys
import datetime as dt
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import itertools

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import streamlit as st

# =============================================================================
# Directory and File Configurations
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_LOAD = os.path.join(script_dir, 'data', 'load')
DATA_DIR_WEATHER = os.path.join(script_dir, 'data', 'weather')

LOAD_FILE = 'nyiso_load.csv'
WEATHER_FILE = 'nyiso_weather.csv'

# List of load zones
LOAD_ZONES = [
    'WEST', 'GENESE', 'CENTRL', 'NORTH', 'MHK VL', 
    'CAPITL', 'HUD VL', 'MILLWD', 'DUNWOD', 'N.Y.C.'
]

best_params = {
    'learning_rate': 0.2,
    'max_depth': 6,
    'n_estimators': 700  # or use the best round from CV if available
}

# Example: updated zone->stations mapping using station codes
zone_weather_map = {
    'WEST':   ['KROC', 'KBGM'],  # Rochester & Binghamton
    'GENESE': ['KROC', 'KUCA'],  # Rochester & Utica
    'CENTRL': ['KSYR', 'KUCA'],  # Syracuse & Utica
    'NORTH':  ['KART', 'KGFL'],  # Watertown & Glens Falls
    'MHK VL': ['KMSS', 'KUCA'],  # Massena & Utica (Mohawk Valley)
    'CAPITL': ['KALB', 'KGFL'],  # Albany & Glens Falls
    'HUD VL': ['KNYC', 'KHPN'],  # NYC (Central Park) & White Plains
    'MILLWD': ['KISP', 'KHPN'],  # Islip & White Plains
    'DUNWOD': ['KBGM', 'KALB'],  # Binghamton & Albany (example)
    'N.Y.C.': ['KJFK', 'KLGA']   # JFK & LaGuardia
}

# =============================================================================
# Data Preparation Functions
# =============================================================================

def load_and_merge_data():
    """
    Load the load data and the weather data, then merge on DateTime and TZ.
    The weather file should have columns like: KLGA_t, KLGA_w, KLGA_h, KLGA_s, etc.
    
    Additional steps:
      - Extract a 24-hour period of weather data (forecast) based on the latest DateTime in the load data.
      - Store that 24-hour weather test data in a global variable (WEATHER_TEST).
      - Remove these 24 hours from df_weather.
      - Merge the remaining weather data with the load data.
    """
    global WEATHER_TEST

    # Load the load and weather CSV files
    df_load = pd.read_csv(os.path.join(DATA_DIR_LOAD, LOAD_FILE), parse_dates=['DateTime'])
    df_weather = pd.read_csv(os.path.join(DATA_DIR_WEATHER, WEATHER_FILE), parse_dates=['DateTime'])
    
    # Determine the forecast start time from the maximum load DateTime
    forecast_end = df_weather['DateTime'].max()
    # Define the 24-hour period starting at forecast_start
    forecast_start = forecast_end - pd.Timedelta(hours=24)
    
    # Extract the 24-hour weather forecast data and store as a global variable
    WEATHER_TEST = df_weather[(df_weather['DateTime'] > forecast_start) &
                              (df_weather['DateTime'] <= forecast_end)].copy()
    
    # Remove the extracted 24-hour forecast from df_weather
    df_weather = df_weather[~((df_weather['DateTime'] > forecast_start) & 
                              (df_weather['DateTime'] <= forecast_end))]
    
    # Merge the remaining weather data with the load data on ['DateTime', 'TZ']
    df = pd.merge(df_weather, df_load, on=['DateTime', 'TZ'], how='left')
    print(df.tail)
    return df

def pivot_load_data(df):
    """
    Pivot the merged dataframe so that each zone's load becomes a separate column.
    Then create a Total_Load column as the sum of all zones.
    We do NOT rename weather columns here, since they are already in station_t/w/h/s format.
    Finally:
      1) Remove any rows where any zone load is negative
      2) Remove any rows where Total_Load is 0
    """
    # Pivot so each zone becomes a column
    df_pivot = df.pivot_table(index=['DateTime', 'TZ'], 
                              columns='Zone', values='Load').reset_index()

    # Ensure every zone is present
    for zone in LOAD_ZONES:
        if zone not in df_pivot.columns:
            df_pivot[zone] = np.nan

    # Remove rows where any zone's load is negative
    df_pivot = df_pivot[(df_pivot[LOAD_ZONES] >= 0).all(axis=1)]

    # Create total load column as the sum of all zones
    df_pivot['Total_Load'] = df_pivot[LOAD_ZONES].sum(axis=1)

    # Remove rows where Total_Load equals 0
    df_pivot = df_pivot[df_pivot['Total_Load'] != 0]

    # Re-merge the station columns (already in station_t/w/h/s format)
    weather_cols = [col for col in df.columns if col not in ['DateTime','TZ','Load','Zone']]
    weather_df = df[['DateTime','TZ'] + weather_cols].drop_duplicates()
    df_pivot = pd.merge(df_pivot, weather_df, on=['DateTime','TZ'], how='left')

    print("PIVOT")
    print(df_pivot)
    
    return df_pivot

def add_time_features(df):
    """Add time-related features to the dataframe."""
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfYear'] = df['DateTime'].dt.dayofyear
    df['DaySin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['DayCos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
    df['Year'] = df['DateTime'].dt.year
    df['IsHoliday'] = (df['DateTime'].dt.weekday >= 5).astype(int)
    return df

def prepare_data():
    """Execute all data preparation steps."""
    df = load_and_merge_data()
    df = pivot_load_data(df)
    df = add_time_features(df)
    # Drop rows missing any load data for any zone or missing total
    df = df.dropna(subset=LOAD_ZONES + ['Total_Load'])
    return df

# =============================================================================
# Hyperparameter Tuning Functions
# =============================================================================

def tune_xgb_cv(X_train, y_train, param_grid, num_boost_round=100, nfold=5, early_stopping_rounds=10):
    """
    For each combination in the param_grid, run cross-validation using xgb.cv
    and return a list of results with the parameter combination, best iteration,
    train RMSE, test RMSE, and the cv results DataFrame.
    """
    results = []
    keys = list(param_grid.keys())
    for values in itertools.product(*param_grid.values()):
        params = dict(zip(keys, values))
        # Set fixed parameters
        params['objective'] = 'reg:squarederror'
        params['seed'] = 42
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            nfold=nfold,
            metrics={'rmse'},
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )
        best_iteration = cv_results.shape[0]
        train_rmse = cv_results['train-rmse-mean'].iloc[-1]
        test_rmse = cv_results['test-rmse-mean'].iloc[-1]
        results.append({
            'params': params,
            'best_iteration': best_iteration,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'cv_results': cv_results
        })
    return results

def plot_cv_results(cv_result, title="CV Learning Curve"):
    """Plot the training and validation RMSE over boosting rounds for one CV result."""
    fig, ax = plt.subplots()
    ax.plot(cv_result.index, cv_result['train-rmse-mean'], label='Train RMSE')
    ax.plot(cv_result.index, cv_result['test-rmse-mean'], label='Validation RMSE')
    ax.set_xlabel("Boosting Rounds")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.legend()
    return fig

# =============================================================================
# ModelResults Class (for forecasting)
# =============================================================================

class ModelResults():
    def __init__(self):
        self.df = None
        self.zone_models = {}
        self.zone_preds = None  # predictions on entire training set
        self.weighted_regressor = None
        self.feature_mapping = {}
        self.average_total_error = None
        self.histories = None
        self.forecast_df = None
        self.forecast_zone_df = None
        self.total_forecast = None

    def train_zone_model(self, df, zone):
        """
        Build the feature list based on the station codes from zone_weather_map,
        add time features, and train on the ENTIRE dataset.
        """
        station_codes = zone_weather_map.get(zone, [])
        station_features = []
        for stn in station_codes:
            for suffix in ['_t', '_w', '_h']:
                col = stn + suffix
                if col in df.columns:
                    station_features.append(col)
        time_features = ['Hour', 'DaySin', 'DayCos', 'Year', 'IsHoliday']
        feature_cols = station_features + time_features
        self.feature_mapping[zone] = feature_cols
        X = df[feature_cols]
        y = df[zone]
        # Train on the entire dataset
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            n_estimators=best_params['n_estimators'],
            verbosity=0
        )
        model.fit(X, y)
        return model

    def train_all_zone_models(self, df):
        """
        Train models for each zone on the entire dataset.
        Save predictions from each zone in self.zone_preds.
        """
        self.zone_models = {}
        zone_preds = pd.DataFrame(index=df.index)
        for zone in LOAD_ZONES:
            st.write(f"Training model for zone: {zone}")
            model = self.train_zone_model(df, zone)
            self.zone_models[zone] = model
            feat_cols = self.feature_mapping[zone]
            X = df[feat_cols]
            preds = pd.Series(model.predict(X), index=df.index)
            zone_preds[zone] = preds
        self.zone_preds = zone_preds

    def train_weighted_sum_regressor(self, df):
        """
        Train the weighted sum regressor on the entire dataset.
        """
        X_train = self.zone_preds[LOAD_ZONES]
        y_train = df['Total_Load']
        lr = LinearRegression(fit_intercept=True)
        lr.fit(X_train, y_train)
        self.weighted_regressor = lr
        st.write("Weighted Sum Regressor Coefficients:")
        for zone, coef in zip(LOAD_ZONES, lr.coef_):
            st.write(f"{zone}: {coef:.4f}")
        y_pred_train = lr.predict(X_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
        st.write(f"Training RMSE: {rmse_train:.2f}, Training MAPE: {mape_train:.2%}")

    def forecast(self):
        """
        Generate a forecast using the global WEATHER_TEST.
        Process WEATHER_TEST to add time features, then for each zone,
        extract the forecast features and predict the load.
        Finally, use the weighted regressor to compute the total forecast load.
        (Note: No true y-value exists in the forecast set.)
        """
        # Ensure WEATHER_TEST has time features
        forecast_df = WEATHER_TEST.copy()
        forecast_df = add_time_features(forecast_df)
        # For each zone, predict using its model
        zone_forecast = {}
        for zone in LOAD_ZONES:
            feat_cols = self.feature_mapping.get(zone, [])
            X_forecast = forecast_df[feat_cols]
            zone_forecast[zone] = self.zone_models[zone].predict(X_forecast)
        forecast_zone_df = pd.DataFrame(zone_forecast, index=forecast_df.index)
        # Use the weighted regressor to get the total forecast
        total_forecast = self.weighted_regressor.predict(forecast_zone_df[LOAD_ZONES])
        return forecast_df, forecast_zone_df, total_forecast

    def run(self):
        st.write("Preparing data...")
        self.df = prepare_data()
        st.write("Data prepared. Number of records:", len(self.df))
        st.write("Training per-zone models on the entire dataset...")
        self.train_all_zone_models(self.df)
        st.write("Training weighted sum regressor...")
        self.train_weighted_sum_regressor(self.df)
        st.write("Generating 24-hour forecast using WEATHER_TEST...")
        forecast_df, forecast_zone_df, total_forecast = self.forecast()
        self.forecast_df = forecast_df
        self.forecast_zone_df = forecast_zone_df
        self.total_forecast = total_forecast
        # Plot the forecasted total load
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(forecast_df['DateTime'], total_forecast, label='Forecast Total Load', linestyle='--')
        ax.set_xlabel('DateTime')
        ax.set_ylabel('Load')
        ax.set_title('24-Hour Forecast')
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Display the forecast in text format below the plot
        # Create a new column for the forecast load
        forecast_df = forecast_df.copy()
        forecast_df['Forecast_Load'] = total_forecast
        st.write("### Forecast Values (Datetime, LoadValue)")

        # Build a single string with all rows
        output_str = "Datetime, LoadValue\n"
        for _, row in forecast_df.iterrows():
            # Format DateTime as YYYY-MM-DD HH:MM and the load value with one decimal
            dt_str = row['DateTime'].strftime("%Y-%m-%d %H:%M")
            load_str = f"{row['Forecast_Load']:.1f}"
            output_str += f"{dt_str}, {load_str}\n"

        st.text(output_str)


    # (Optional: remove evaluate_weighted_regressor and backtest methods if no longer needed)

# =============================================================================
# Visualization Functions
# =============================================================================

def plot_history_streamlit(history):
    if history is None:
        st.write("No training history available.")
        return
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Square Error')
    ax2.plot(hist['epoch'], hist['mse'], label='Train Error')
    ax2.plot(hist['epoch'], hist['val_mse'], label='Val Error')
    ax2.legend()
    st.pyplot(fig2)

@st.cache_resource
def plot_predictions_vs_labels(y_true, y_pred):
    x = np.arange(len(y_true))
    fig, ax = plt.subplots()
    ax.plot(x, y_true, marker='o', linestyle='-', color='blue', label='True Labels')
    ax.plot(x, y_pred, marker='o', linestyle='-', color='red', label='Predictions')
    ax.set_xlabel("Data Point Index")
    ax.set_ylabel("Load")
    ax.set_title("Predictions vs. True Labels")
    ax.legend()
    st.pyplot(fig)

@st.cache_resource
def plot_data(df):
    for col in df.columns:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=col, y='Total_Load', ax=ax)
        ax.set_xlabel(col)
        ax.set_ylabel('Total_Load')
        st.pyplot(fig)

def plot_backtest(train, weighted_regressor, zone_train_preds, labels):
    aggregated_pred = weighted_regressor.predict(zone_train_preds.loc[train.index][LOAD_ZONES])
    df_plot = train.copy()
    df_plot['Aggregated_Prediction'] = aggregated_pred
    df_plot = df_plot.reset_index()
    fig = px.line(df_plot, x='DateTime', y=['Total_Load','Aggregated_Prediction'],
                  title="Aggregated Backtest: Predictions vs. True Total Load Over Time")
    st.plotly_chart(fig)

# =============================================================================
# Hyperparameter Tuning Plotting
# =============================================================================

def tune_and_plot_hyperparameters(X_train, y_train):
    """
    Tune essential hyperparameters (learning_rate and max_depth in this example) 
    using xgb.cv and plot training and validation RMSE for each combination.
    """
    param_grid = {
        'learning_rate': [0.2],
        'max_depth': [9]
    }
    
    tuning_results = []
    for lr, md in itertools.product(param_grid['learning_rate'], param_grid['max_depth']):
        params = {
            'learning_rate': lr,
            'max_depth': md,
            'objective': 'reg:squarederror',
            'seed': 42
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=1000,
            nfold=5,
            metrics={'rmse'},
            early_stopping_rounds=10,
            verbose_eval=False
        )
        best_round = cv_results.shape[0]
        train_rmse = cv_results['train-rmse-mean'].iloc[-1]
        test_rmse = cv_results['test-rmse-mean'].iloc[-1]
        tuning_results.append({
            'params': params,
            'best_round': best_round,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'cv_results': cv_results
        })
    
    # Plot each curve using Plotly for interactivity
    for res in tuning_results:
        title = f"lr={res['params']['learning_rate']}, max_depth={res['params']['max_depth']}"
        fig = plot_cv_results(res['cv_results'], title=title)
        st.plotly_chart(px.line(pd.DataFrame({
            'Round': res['cv_results'].index,
            'Train RMSE': res['cv_results']['train-rmse-mean'],
            'Validation RMSE': res['cv_results']['test-rmse-mean']
        }), x='Round', y=['Train RMSE', 'Validation RMSE'], title=title))
    
    return tuning_results

def plot_cv_results(cv_result, title="CV Learning Curve"):
    fig, ax = plt.subplots()
    ax.plot(cv_result.index, cv_result['train-rmse-mean'], label='Train RMSE')
    ax.plot(cv_result.index, cv_result['test-rmse-mean'], label='Validation RMSE')
    ax.set_xlabel("Boosting Rounds")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.legend()
    return fig

# =============================================================================
# Cache the ModelResults instance
# =============================================================================

@st.cache_resource
def get_model_results():
    results = ModelResults()
    results.run()
    st.write("Finished Training and Forecasting.")
    return results

# =============================================================================
# Streamlit App Layout
# =============================================================================

st.title("Electricity Load Forecasting")

# Option to display hyperparameter tuning results
if st.checkbox("Show Hyperparameter Tuning"):
    st.write("Tuning hyperparameters on the training set of the Total_Load prediction...")
    df = prepare_data()
    # For tuning, use the entire dataset features
    X = df[['Hour', 'DaySin', 'DayCos', 'Year', 'IsHoliday']]
    y = df['Total_Load']
    tuning_results = tune_and_plot_hyperparameters(X, y)
    st.write("Hyperparameter tuning complete.")

# Continue with forecasting
results = get_model_results()

df = results.df

st.write("### Data Overview")
st.write("Full Data:", df)

st.write("### Scatter Plots: Feature vs. Load")
plot_data(df)

st.write("### 24-Hour Forecast (from WEATHER_TEST)")
# The forecast plot is generated in results.run()
