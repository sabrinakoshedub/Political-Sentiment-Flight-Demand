#T100-Functions
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

##########################################################################################################################

# ------------------------ CONFIG ------------------------

base_path = Path(__file__).resolve().parent
figures_path = base_path / "../Figures"
tables_path = base_path / "../Tables"

config_path = base_path / "config_T100.json"
with open(config_path, "r") as f:
    config = json.load(f)

ctry_fm_filter = config["ctry_fm_filter"]
ctry_to_filter = config["ctry_to_filter"]
ctry_label = f"{ctry_fm_filter or 'ALL'}_{ctry_to_filter or 'ALL'}"

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

# ------------------------ PLOTS ------------------------------------------------------------------------
def plot_raw_series(series, fname):
    plt.figure(figsize=(10, 4))
    plt.plot(series)
    plt.ylabel("Passenger Count")
    plt.xlabel("Date")
    plt.axvline(x=pd.Timestamp('2025-01-01'), color='red', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, fname), dpi=300)
    plt.close()
    print(f"{fname} saved!")


def plot_acf_pacf(series, acf_fname, pacf_fname, lags=40):
    # ACF plot
    plot_acf(series.dropna(), lags=lags, title=None)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, acf_fname), dpi=300)
    plt.close()
    print(f"{acf_fname} saved!")

    # PACF plot
    plot_pacf(series.dropna(), lags=lags, title=None)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, pacf_fname), dpi=300)
    plt.close()
    print(f"{pacf_fname} saved!")


def plot_avg_monthly_seasonality(series, fname):
    df = series.to_frame(name='pax')
    df['month'] = df.index.month
    avg_seasonality = df.groupby('month')['pax'].mean()
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    avg_seasonality.index = month_labels

    plt.figure(figsize=(10, 4))
    avg_seasonality.plot(marker='o', linestyle='-', color='cornflowerblue')
    plt.ylabel("Average Passenger Count")
    plt.xlabel("Month")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, fname), dpi=300)
    plt.close()
    print(f"{fname} saved!")


def plot_yoy_growth(series, title, fname, months=range(1,13)):
    df = series[series.index.month.isin(months)].to_frame(name='pax')
    df['month'] = df.index.month_name()
    df['year'] = df.index.year

    pivot = df.pivot(index='year', columns='month', values='pax')
    pivot = pivot[[m for m in [
        'January','February','March','April','May','June',
        'July','August','September','October','November','December'
    ] if m in pivot.columns]]

    growth = pivot.pct_change(fill_method=None) * 100
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    table = ax.table(
        cellText=np.round(growth.values, 1),
        rowLabels=growth.index.tolist(),
        colLabels=growth.columns.tolist(),
        loc='center',
        cellLoc='center'
    )

    for (i, j), val in np.ndenumerate(growth.values):
        cell = table[i+1, j]
        if np.isnan(val):
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
    table.scale(1.2, 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(tables_path, fname), dpi=300)
    plt.close()
    print(f"{fname} saved!")



#-----------------------------------------------------------------------------------------------------------------------------------
def plot_yoy_totals(series, fname, months=range(1,13)):
    df = series[series.index.month.isin(months)].to_frame(name='pax')
    df['month'] = df.index.month_name()
    df['year'] = df.index.year

    pivot = df.pivot_table(index='month', columns='year', values='pax', aggfunc='sum')

    # Limit to January–April in correct order
    month_order = ['January', 'February', 'March', 'April']
    pivot = pivot.loc[[m for m in month_order if m in pivot.index]]

    # Compute difference and % change from 2024 to 2025
    if 2024 in pivot.columns and 2025 in pivot.columns:
        pivot['Diff'] = pivot[2025] - pivot[2024]
        pivot['% Change'] = (pivot['Diff'] / pivot[2024]) * 100
    else:
        raise ValueError("Both 2024 and 2025 must be in the data to compute totals")

    display_cols = [2024, 2025, 'Diff', '% Change']
    table_data = pivot[display_cols].copy()
    table_data['% Change'] = table_data['% Change'].round(1)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    cell_values = table_data.values.astype(object)

    for i in range(cell_values.shape[0]):
        for j in range(cell_values.shape[1]):
            if pd.isna(cell_values[i, j]):
                cell_values[i, j] = '—'

    table = ax.table(
        cellText=cell_values,
        rowLabels=table_data.index.tolist(),
        colLabels=[str(c) for c in display_cols],
        loc='center',
        cellLoc='center',
        colLoc='center'
    )

    for (i, j), val in np.ndenumerate(table_data.values):
        cell = table[i + 1, j]
        if j == 3:  # % Change column
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
    table.scale(1.2, 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(tables_path, fname), dpi=300)
    plt.close()
    print(f"{fname} saved!")
#-----------------------------------------------------------------------------------------------------------------------------------




def plot_intl_passenger_share_pie(df, figures_path, fname='intl_passenger_share_pie.png'):

    df_filtered = df.query("ctry_fm != 'US' and ctry_to == 'US'")

    grouped = (
        df_filtered.groupby('ctry_fm', as_index=False)['pax']
        .sum()
    )
    grouped['proportion'] = grouped['pax'] / grouped['pax'].sum() * 100

    top10 = grouped.nlargest(10, 'proportion').copy()
    other_prop = 100 - top10['proportion'].sum()
    top10_plus_other = pd.concat([
        top10,
        pd.DataFrame([{'ctry_fm': 'Other', 'pax': None, 'proportion': other_prop}])
    ], ignore_index=True)
    top10_plus_other['proportion'] = top10_plus_other['proportion'].round(2)

    colors = []
    for country in top10_plus_other['ctry_fm']:
        if country == ctry_fm_filter:
            colors.append('#fbf239')  
        else:
            i = top10_plus_other[top10_plus_other['ctry_fm'] == country].index[0]
            colors.append(mcolors.to_hex(plt.cm.Blues(0.1 + 0.4 * (1 - (i / 10)))))

    # Plot pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(
        top10_plus_other['proportion'],
        labels=top10_plus_other['ctry_fm'],
        colors=colors,
        autopct='%1.1f%%',
        startangle=140,
        wedgeprops={'edgecolor': 'white'}
    )

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, fname), dpi=300)
    plt.close()
    print(f"{fname} saved!")





# ------------------------ EXECUTION ------------------------
df_filtered = filter_df(df, ctry_fm_filter, ctry_to_filter)
pax_series = prepare_time_series(df_filtered)

# Raw Series 
plot_raw_series(
    pax_series,
    fname=f"Original_TS_T100_{ctry_label}.png"
)


# ACF/PACF
plot_acf_pacf(
    pax_series,
    acf_fname=f"T100_ACF_{ctry_label}.png",
    pacf_fname=f"T100_PACF_{ctry_label}.png"
)

# Seasonality
plot_avg_monthly_seasonality(
    pax_series,
    fname=f"Avg_Monthly_Seasonality_Line_{ctry_label}.png"
)

# Year-over-Year Growth (Jan–Mar)
plot_yoy_growth(
    pax_series,
    title=f"Year-over-Year % Growth (Jan–Mar) ({ctry_label})",
    fname=f"yoy_growth_q1_table_{ctry_label}.png",
    months=[1, 2, 3, 4]
)


#Top 10 international passengers
plot_intl_passenger_share_pie(df_T2, figures_path)


plot_yoy_totals(
    series=pax_series, 
    fname=f"totals_table_2024_2025_{ctry_label}.png"
)

