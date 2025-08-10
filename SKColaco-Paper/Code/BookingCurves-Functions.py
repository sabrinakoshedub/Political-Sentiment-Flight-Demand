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

# Load CSVs into SQLite
for file_name in os.listdir(folder_path):
    if file_name.endswith('.CSV'):
        file_path = os.path.join(folder_path, file_name)
        table_name = os.path.splitext(file_name)[0]
        df = pd.read_csv(file_path)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Loaded {file_name} into table {table_name}")


cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

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

##############Booking Curves##################################################################################################

# ------------------------ CONFIG ------------------------
base_path = Path(__file__).resolve().parent
figures_path = base_path / "../Figures"
tables_path = base_path / "../Tables"

#show available country pairs
country_pairs = df_BOOKING['ndir_ond_ctry'].dropna().unique().tolist()
print("Available country pairs:", country_pairs)

config_path = base_path / "config_booking.json"
with open(config_path, "r") as f:
    config = json.load(f)

selected_country_pair = config["selected_country_pair"]
target_months = config["target_months"]

#complete booking curve function to make curves AND table
def generate_booking_curve(df, selected_country_pair, target_month):
    # Filter data
    df_filtered = df[
        (df['ndir_ond_ctry'] == selected_country_pair) &
        (df['flt_mth'].astype(str).str.endswith(target_month))
    ]

    if df_filtered.empty:
        print(f"No data for {selected_country_pair} in month {target_month}.")
        return

    booking_curves = df_filtered.groupby(['fltyr', 'mths_out'])['bkd_pax'].sum().reset_index()
    pivot_df = booking_curves.pivot(index='mths_out', columns='fltyr', values='bkd_pax').sort_index()

    if pivot_df.empty:
        print(f"No data to plot for {selected_country_pair}, month {target_month}")
        return

    plt.figure(figsize=(10, 6))
    pivot_df.plot(marker='o', ax=plt.gca())
    plt.gca().invert_xaxis()
    plt.xlabel("Months Out from Flight")
    plt.ylabel("Booked Passengers")
    plt.grid(True)
    plt.legend(title="Flight Year")
    plt.tight_layout()
    fig_name = f'booking_curve_{selected_country_pair}_{target_month}.png'
    plt.savefig(os.path.join(figures_path, fig_name), dpi=300)
    plt.close()
    print(f"{fig_name} saved!")

    for yr in [2023, 2024, 2025]:
        if yr not in pivot_df.columns:
            pivot_df[yr] = 0

    pivot_df['% Change (2023-2024)'] = (
        (pivot_df[2024] - pivot_df[2023]) / pivot_df[2023].replace(0, np.nan) * 100
    ).round(1)

    pivot_df['% Change (2024-2025)'] = (
        (pivot_df[2025] - pivot_df[2024]) / pivot_df[2024].replace(0, np.nan) * 100
    ).round(1)

    pivot_df.reset_index(inplace=True)
    pivot_df.rename(columns={'mths_out': 'Months Before Departure'}, inplace=True)
    pivot_df = pivot_df.sort_values(by='Months Before Departure', ascending=False).fillna("N/A")


    def booking_curve_table(df):
        
        fig, ax = plt.subplots(figsize=(14, 0.45 * len(df)))
        ax.axis('off')

        data = df.values
        row_labels = df['Months Before Departure'].tolist()
        col_labels = df.columns.tolist()

        table = ax.table(
            cellText=data,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
            colLoc='center'
        )


        percent_cols = ['% Change (2023-2024)', '% Change (2024-2025)']
        col_indices = [col_labels.index(col) for col in percent_cols]

        for (i, j), val in np.ndenumerate(data):
            cell = table[i + 1, j]
            try:
                val = float(val)
                if j in col_indices:
                    if val > 0:
                        cell.set_facecolor('#c6f7c3')
                    elif val < 0:
                        cell.set_facecolor('#f7c6c6')
                    else:
                        cell.set_facecolor('white')
            except:
                if val in ["N/A", "nan"]:
                    cell.set_text_props(text='—')
                    cell.set_facecolor('lightgrey')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        plt.tight_layout()

        table_name = f'booking_curve_table_{selected_country_pair}_{target_month}.png'
        plt.savefig(os.path.join(tables_path, table_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{table_name} saved!")

    booking_curve_table(pivot_df)

for month in target_months:
    generate_booking_curve(df_BOOKING, selected_country_pair=selected_country_pair, target_month=month)



