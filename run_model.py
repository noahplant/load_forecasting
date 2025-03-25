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
    """
    df_load = pd.read_csv(os.path.join(DATA_DIR_LOAD, LOAD_FILE), parse_dates=['DateTime'])
    df_weather = pd.read_csv(os.path.join(DATA_DIR_WEATHER, WEATHER_FILE), parse_dates=['DateTime'])
    # Merge on ['DateTime', 'TZ']
    df = pd.merge(df_weather, df_load, on=['DateTime', 'TZ'], how='left')
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
        self.zone_train_preds = None
        self.zone_test_preds = None
        self.weighted_regressor = None
        self.feature_mapping = {}
        self.train_index = None
        self.test_index = None
        self.average_total_error = None
        self.histories = None

    def train_zone_model(self, df, zone):
        """
        Build the feature list based on the station codes from zone_weather_map,
        e.g. for zone 'N.Y.C.' => ['KJFK', 'KLGA'] => columns like 'KJFK_t', etc.
        Then add time features.
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
        # Use time-based split: last 24 hours as test
        cutoff = df['DateTime'].max() - pd.Timedelta(hours=24)
        train_idx = df['DateTime'] < cutoff
        X_train = X[train_idx]
        y_train = y[train_idx]
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            n_estimators=best_params['n_estimators'],
            verbosity=0
        )
        model.fit(X_train, y_train)
        return model

    def train_all_zone_models(self, df):
        self.zone_models = {}
        zone_train_preds = pd.DataFrame(index=df.index)
        zone_test_preds = pd.DataFrame(index=df.index)
        cutoff = df['DateTime'].max() - pd.Timedelta(hours=24)
        self.train_index = df.index[df['DateTime'] < cutoff]
        self.test_index = df.index[df['DateTime'] >= cutoff]
        for zone in LOAD_ZONES:
            st.write(f"Training model for zone: {zone}")
            model = self.train_zone_model(df, zone)
            self.zone_models[zone] = model
            feat_cols = self.feature_mapping[zone]
            X = df[feat_cols]
            train_pred = pd.Series(model.predict(X.loc[self.train_index]), index=self.train_index)
            test_pred = pd.Series(model.predict(X.loc[self.test_index]), index=self.test_index)
            zone_train_preds.loc[self.train_index, zone] = train_pred
            zone_test_preds.loc[self.test_index, zone] = test_pred
        self.zone_train_preds = zone_train_preds
        self.zone_test_preds = zone_test_preds

    def train_weighted_sum_regressor(self, df):
        cutoff = df['DateTime'].max() - pd.Timedelta(hours=24)
        self.train_index = df.index[df['DateTime'] < cutoff]
        X_train = self.zone_train_preds.loc[self.train_index][LOAD_ZONES]
        y_train = df.loc[self.train_index]['Total_Load']
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

    def evaluate_weighted_regressor(self, df):
        cutoff = df['DateTime'].max() - pd.Timedelta(hours=24)
        self.test_index = df.index[df['DateTime'] >= cutoff]
        X_test = self.zone_test_preds.loc[self.test_index][LOAD_ZONES]
        y_test = df.loc[self.test_index]['Total_Load']
        y_pred = self.weighted_regressor.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)
        st.write(f"Test RMSE: {rmse:.2f}, Test MAPE: {mape:.2%}")
        self.average_total_error = np.mean(np.abs(y_pred - y_test))
        return y_pred

    def backtest(self, df):
        backtest_preds = {}
        cutoff = df['DateTime'].max() - pd.Timedelta(hours=24)
        self.train_index = df.index[df['DateTime'] < cutoff]
        for zone in LOAD_ZONES:
            feat_cols = self.feature_mapping[zone]
            X_train = df[feat_cols].loc[self.train_index]
            pred = self.zone_models[zone].predict(X_train)
            backtest_preds[zone] = pred
        return backtest_preds

    def run(self):
        st.write("Preparing data...")
        self.df = prepare_data()
        st.write("Data prepared. Number of records:", len(self.df))
        st.write("Training per-zone models...")
        self.train_all_zone_models(self.df)
        st.write("Training weighted sum regressor...")
        self.train_weighted_sum_regressor(self.df)
        st.write("Evaluating on test set...")
        pred_total = self.evaluate_weighted_regressor(self.df)
        backtest = self.backtest(self.df)
        self.backtest_predictions = backtest
        self.test_predictions = self.zone_test_preds
        cutoff = self.df['DateTime'].max() - pd.Timedelta(hours=24)
        df_test = self.df[self.df['DateTime'] >= cutoff].copy().reset_index(drop=True)
        df_test['Pred_Total_Load'] = pred_total[:len(df_test)]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_test['DateTime'], df_test['Total_Load'], label='Actual Total Load')
        ax.plot(df_test['DateTime'], df_test['Pred_Total_Load'], label='Predicted Total Load', linestyle='--')
        ax.set_xlabel('DateTime')
        ax.set_ylabel('Load')
        ax.set_title('Total Load Forecast on Test Set')
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

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
    ax.set_title("Test Predictions vs. True Labels")
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
    st.write("Finished Training.")
    return results

# =============================================================================
# Streamlit App Layout
# =============================================================================

st.title("Electricity Load Forecasting")

# Option to display hyperparameter tuning results
if st.checkbox("Show Hyperparameter Tuning"):
    st.write("Tuning hyperparameters on the training set of the Total_Load prediction...")
    df = prepare_data()
    cutoff = df['DateTime'].max() - pd.Timedelta(hours=24)
    train_df = df[df['DateTime'] < cutoff]
    # Use time features plus a constant if needed
    X = train_df[['Hour', 'DaySin', 'DayCos', 'Year', 'IsHoliday']]
    y = train_df['Total_Load']
    tuning_results = tune_and_plot_hyperparameters(X, y)
    st.write("Hyperparameter tuning complete.")

# Continue with forecasting
results = get_model_results()

df = results.df
cutoff = df['DateTime'].max() - pd.Timedelta(hours=24)
train = df[df['DateTime'] < cutoff]
test = df[df['DateTime'] >= cutoff]

st.write("### Data Overview")
st.write("Full Data:", df)
st.write("Train Data:", train)
st.write("Test Data:", test)

st.write("### Scatter Plots: Feature vs. Load")
plot_data(df)

st.write("### Test Predictions vs. True Total Load")
X_test = results.zone_test_preds.loc[results.test_index][LOAD_ZONES]
total_load_pred = results.weighted_regressor.predict(X_test)
y_test = df.loc[results.test_index]['Total_Load'].values
plot_predictions_vs_labels(y_true=y_test, y_pred=total_load_pred)
st.write("Average Error in MW: ", results.average_total_error)

st.write("### (Optional) Training History")
selected_hour = st.slider('Select Hour (for training history display)', 0, 23, 0)
if results.histories is not None:
    plot_history_streamlit(history=results.histories[selected_hour])
else:
    st.write("No training history to display.")

st.write("### Backtest Results")
plot_backtest(train=train, 
              weighted_regressor=results.weighted_regressor, 
              zone_train_preds=results.zone_train_preds, 
              labels=train['Total_Load'])
