#SARIMA functions

import os
import sqlite3
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from ipywidgets import interact, Dropdown
from sklearn.metrics import mean_squared_error

############################################# SQLITE database setup #############################################
db_path = 'SKColaco-Model-Data.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
base_path = Path(__file__).resolve().parent 
folder_path = base_path / "../Data"

#cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#print(cursor.fetchall())

queryBOOKING = "SELECT * FROM BOOKING_20250623"
df_BOOKING = pd.read_sql_query(queryBOOKING, conn)

queryBTS = "SELECT * FROM DAILY_CA"
df_BTS = pd.read_sql_query(queryBTS, conn)

queryT200 = """
SELECT * FROM T100_202401_202503
WHERE 'seats' > 50 AND 'passengers' > 10
"""
df_T2 = pd.read_sql_query(queryT200, conn)

queryT100_10 = "SELECT * FROM T100_TOP10_202504"
df = pd.read_sql_query(queryT100_10, conn)

conn.close()
#######################################################################################################################################################

# ------------------------ CONFIG ------------------------
# File paths
base_path = Path(__file__).resolve().parent
figures_path = base_path / "../Figures"
tables_path = base_path / "../Tables"

config_path = base_path / "config_T100.json"
with open(config_path, "r") as f:
    config = json.load(f)

ctry_fm_filter = config["ctry_fm_filter"]
ctry_to_filter = config["ctry_to_filter"]
ctry_label = f"{ctry_fm_filter or 'ALL'}_{ctry_to_filter or 'ALL'}"

#----------------- SARIMA model configurations --------------
#had 2 model configurations but removed 2023
model_configs = [
    {
        'train_to': '2024-01-01',
        'order': (0, 1, 1),
        'seasonal_order': (1, 1, 2, 12),
        'model_label': '0124',
        'forecast_horizon': 18,
        'vline_date': '2025-01-01'
    }
]

#---------------- Full-series forecast config ---------------
full_forecast_config = {
    'order': (2, 1, 0),
    'seasonal_order': (1, 1, 1, 12),
    'steps_ahead': 12,
    'label_suffix': 'full_future'
}

# ------------------------ PREP ------------------------
def filter_df(df, ctry_fm=None, ctry_to=None):
    if ctry_fm:
        df = df[df['ctry_fm'] == ctry_fm]
    if ctry_to:
        df = df[df['ctry_to'] == ctry_to]
    return df.copy()

def prepare_time_series(df, date_col='flt_mth', val_col='pax'):
    df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y%m')
    df = df[df[date_col] >= '2012-01-01']
    ts = df.groupby(date_col)[val_col].sum().sort_index().asfreq('MS')
    return ts


#---------- PLOTS/MODELS--------------------------------

def plot_forecast(full_series, forecast_obj, vline_date, fig_name):
    forecast_mean = forecast_obj.predicted_mean
    forecast_ci = forecast_obj.conf_int()

    plt.figure(figsize=(12, 5))
    plt.plot(full_series, label='Observed')
    plt.plot(forecast_mean, label='Forecast', color='orange')
    plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='orange', alpha=0.3)
    plt.axvline(pd.to_datetime(vline_date), color='red', linestyle='--', label=str(vline_date[:4]))
    plt.xlim(pd.Timestamp('2019-01-01'), pd.Timestamp('2026-01-01'))
    plt.xlabel("Date")
    plt.ylabel("Passenger Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, fig_name), dpi=300)
    plt.close()
    print(f"{fig_name} saved!")
    

def compute_and_plot_residuals(actuals, predictions, fig_name, table_name):
    residuals = actuals - predictions

    plt.figure(figsize=(12, 5))
    plt.plot(residuals.index, residuals.values, marker='o', linestyle='-')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Month')
    plt.ylabel('Residuals')
    plt.axhspan(ymin=plt.ylim()[0], ymax=0, color='red', alpha=0.1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, fig_name), dpi=300)
    plt.close()
    print(f"{fig_name} saved!")

    table_df = pd.DataFrame({'Actual': actuals, 'Forecast': predictions, 'Residual': residuals}).round(1)
    fig, ax = plt.subplots(figsize=(10, min(0.4 * len(table_df), 20)))
    ax.axis('off')

    table_df.index = table_df.index.strftime('%Y-%m')
    table = ax.table(cellText=table_df.values,
                     colLabels=table_df.columns,
                     rowLabels=table_df.index,
                     loc='center',
                     cellLoc='right',
                     rowLoc='center')

    for i, val in enumerate(table_df['Residual'].values):
        cell = table[i + 1, 2]
        if pd.isna(val):
            cell.set_text_props(text='—')
            cell.set_facecolor('lightgrey')
        elif val > 0:
            cell.set_facecolor('#c6f7c3')
        elif val < 0:
            cell.set_facecolor('#f7c6c6')
        else:
            cell.set_facecolor('white')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(tables_path, table_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{table_name} saved!")

    return residuals

#-----one total function that runs the sarima, plots the residuals, and makes residual table----------------
def run_sarima_model(series, train_to, order, seasonal_order,
                     model_label, forecast_horizon, vline_date):
    train_series = series[:train_to]
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit()


    forecast = results.get_forecast(steps=forecast_horizon)
    forecast_mean = forecast.predicted_mean

    plot_forecast(
        full_series=series,
        forecast_obj=forecast,
        vline_date=vline_date,
        fig_name=f"sarima_forecast_{model_label}_{ctry_label}.png"
    )

    actuals = series[forecast_mean.index.intersection(series.index)]
    predictions = forecast_mean[actuals.index]
    compute_and_plot_residuals(
        actuals, predictions,
        fig_name=f"sarima_residuals_{model_label}_{ctry_label}_fig.png",
        table_name=f"sarima_residuals_{model_label}_{ctry_label}_table.png"
    )



def forecast_full_series(series, order, seasonal_order, steps_ahead, label_suffix):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit()
    forecast = results.get_forecast(steps=steps_ahead)
    plot_forecast(
        full_series=series,
        forecast_obj=forecast,
        vline_date=str(series.index[-1].date()),
        fig_name=f"sarima_forecast_{label_suffix}_{ctry_label}.png"
    )


# ------------------------ EXECUTION ---------------------------------------------------
df_filtered = filter_df(df, ctry_fm_filter, ctry_to_filter)
ts = prepare_time_series(df_filtered)

model_metrics_list = []
for config in model_configs:
    metrics = run_sarima_model(
        series=ts,
        train_to=config['train_to'],
        order=config['order'],
        seasonal_order=config['seasonal_order'],
        model_label=config['model_label'],
        forecast_horizon=config['forecast_horizon'],
        vline_date=config['vline_date']
    )
    model_metrics_list.append(metrics)


forecast_full_series(
    series=ts,
    order=full_forecast_config['order'],
    seasonal_order=full_forecast_config['seasonal_order'],
    steps_ahead=full_forecast_config['steps_ahead'],
    label_suffix=full_forecast_config['label_suffix']
)