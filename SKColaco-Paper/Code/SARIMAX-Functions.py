#SARIMAX Functions

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

#----------------------------------------------------------------------------------------------------------------
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
exog_name = config["exog_name"]
ramp_start_date = config["ramp_start_date"]
ramp_start_date_nc = config["ramp_start_date_nc"]

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

def create_ramp(series, start_date, name='Ramp'):
    ramp = pd.Series(0, index=series.index)
    ramp[ramp.index >= start_date] = range(1, len(ramp[ramp.index >= start_date]) + 1)
    ramp.name = name
    return ramp



def rename_param(name):
    return (
        name.replace('ar.L1', r'$\phi_1$')
            .replace('ar.L2', r'$\phi_2$')
            .replace('ma.L1', r'$\theta_1$')
            .replace('ma.L2', r'$\theta_2$')
            .replace('ar.S.L12', r'$\Phi_1$')
            .replace('ma.S.L12', r'$\Theta_1$')
            .replace('sigma2', r'$\sigma^2$')
    )

def smart_format(val):
    try:
        val = float(val)
        if val == 0:
            return "0.000 $\\times$ 10$^{0}$"
        exponent = int(np.floor(np.log10(abs(val))))
        base = val / (10 ** exponent)
        return f"{base:.3f} $\\times$ 10$^{{{exponent}}}$"
    except:
        return str(val)

def format_row(row):
    name = rename_param(row[0])
    try:
        coef = smart_format(row[1])
        stderr = smart_format(row[2])
        ci_low = smart_format(row[5])
        ci_high = smart_format(row[6])
        ci = f"[{ci_low}, {ci_high}]"
    except:
        coef, stderr, ci = "NA", "NA", "[NA, NA]"
    return [name, coef, stderr, ci]


def run_sarimax(series, exog, order, seasonal_order, label, fname_suffix):
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        exog=exog,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    aic = results.aic
    rmse = np.sqrt(np.mean(results.resid ** 2))

    coeff_table = results.summary().tables[1].data
    rows = coeff_table[1:] 
    formatted_rows = [format_row(row) for row in rows]

    #latex table
    tex_lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{label}}}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Variable & Coef & Std Err & 95\\% CI \\\\",
        "\\midrule"
    ]

    for row in formatted_rows:
        tex_lines.append(" & ".join(row) + " \\\\")

    tex_lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\label{{tab:sarimax-{fname_suffix}}}",
        "\\vspace{0.3em}",
        "\\begin{flushleft}\\footnotesize",
        f"AIC = {smart_format(aic)}, \\quad RMSE = {smart_format(rmse)}",
        "\\end{flushleft}",
        "\\end{table}"
    ]

    os.makedirs(tables_path, exist_ok=True)
    tex_path = os.path.join(tables_path, f'sarimax_summary_{fname_suffix}_{ctry_label}.tex')
    with open(tex_path, 'w') as f:
        f.write("\n".join(tex_lines))

    print("LaTeX table saved!")




def plot_stl_decomposition(series, seasonal, fname):
    result = STL(series, seasonal=seasonal).fit()
    result.plot()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, fname), dpi=300)
    plt.close()
    print(f"{fname} saved!")



# ------------------------ EXECUTION ------------------------

df_filtered = filter_df(df, ctry_fm_filter, ctry_to_filter)
ts = prepare_time_series(df_filtered)

# No-COVID series
pre_covid = ts[ts.index < '2020-01-01']
post_covid = ts[ts.index >= '2023-02-01']
no_covid_series = pd.concat([pre_covid, post_covid]).asfreq('MS').interpolate(method='linear')

# Ramp function NC
ramp_full = create_ramp(ts, ramp_start_date, exog_name)
ramp_nc = create_ramp(no_covid_series, ramp_start_date_nc, exog_name)

#Full SARIMAX 
run_sarimax(
    series=ts,
    exog=ramp_full,
    order=(1, 1, 0),
    seasonal_order=(1, 1, 0, 12),
    label="FULL T100 TIME SERIES SARIMAX",
    fname_suffix="FULL_T100"
)

#Full Trend SARIMAX
stl_full = STL(ts, seasonal=13).fit()
run_sarimax(
    series=stl_full.trend.dropna(),
    exog=ramp_full,
    order=(1, 1, 0),
    seasonal_order=(1, 1, 0, 12),
    label="FULL T100 TREND COMPONENT SARIMAX",
    fname_suffix="TRENDFULL_T100"
)

#No-COVID SARIMAX
run_sarimax(
    series=no_covid_series,
    exog=ramp_nc,
    order=(1, 1, 0),
    seasonal_order=(0, 1, 0, 12),
    label="No-COVID T100 TIME SERIES SARIMAX",
    fname_suffix="NC_T100"
)

#No-COVID Trend SARIMAX
stl_nc = STL(no_covid_series, seasonal=13).fit()
run_sarimax(
    series=stl_nc.trend.dropna(),
    exog=ramp_nc,
    order=(1, 1, 0),
    seasonal_order=(0, 1, 0, 12),
    label="No-COVID TREND T100 TIME SERIES SARIMAX",
    fname_suffix="NC_TREND_T100"
)


#---------------STL Decomposition Plots----------------------

# Full STL
plot_stl_decomposition(
    series=ts,
    seasonal=13,
    fname=f"stl_decomp_FULL_T100_{ctry_label}.png"
)

# No-COVID STL
plot_stl_decomposition(
    series=no_covid_series,
    seasonal=13,
    fname=f"stl_decomp_NC_T100_{ctry_label}.png"
)