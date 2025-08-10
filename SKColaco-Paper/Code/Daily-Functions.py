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
base_path = Path(__file__).resolve().parent
figures_path = base_path / "../Figures"
tables_path = base_path / "../Tables"

segment_colors = {
    'CA-US': '#4490dc',
    'Domestic CA': '#ffa727',
    'Oth Intl': '#9adf42'}


# Ensure correct types
df_BTS['flt_mth'] = df_BTS['flt_mth'].astype(str)
df_BTS['snapshot'] = pd.to_datetime(df_BTS['snapshot'], format='%Y%m%d')
df_BTS.columns = df_BTS.columns.str.strip()

segment_data = {}
segments = df_BTS['itin_type'].unique()

for segment in segments:
    df_segment = df_BTS[df_BTS['itin_type'] == segment].copy()

    df_segment['snapshot_month'] = df_segment['snapshot'].dt.strftime('%Y%m')

    # Label current vs future bookings
    df_segment['curr_or_future'] = df_segment.apply(
        lambda row: 'current' if row['flt_mth'] <= row['snapshot_month'] else 'future',
        axis=1
    )

    #aggregate bookings
    grouped = (
        df_segment.groupby(['snapshot', 'curr_or_future'])['bkgs']
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    grouped['total'] = grouped['current'] + grouped['future']
    grouped = grouped.sort_values('snapshot')

    #accumulate based on future/current
    net_new = []
    prev_total = None
    prev_future = None
    prev_month = None

    for _, row in grouped.iterrows():
        curr_total = row['total']
        curr_future = row['future']
        curr_month = row['snapshot'].strftime('%Y%m')

        if prev_total is None:
            net_new.append(0)
        elif curr_month == prev_month:
            net_new.append(curr_total - prev_total)
        else:
            net_new.append(curr_total - prev_future)

        prev_total = curr_total
        prev_future = curr_future
        prev_month = curr_month

    grouped['net_new'] = net_new
    grouped.loc[(grouped['net_new'] < -100000) | (grouped['net_new'] > 100000), 'net_new'] = np.nan

    segment_data[segment] = grouped[['snapshot', 'net_new']]


#-----------------Total Net New by Segment------------------------------
plt.figure(figsize=(14, 5))
for segment in segments:
    df_seg = segment_data[segment]
    df_filtered = df_seg[df_seg['snapshot'] >= pd.Timestamp('2024-07-01')]
    plt.plot(df_filtered['snapshot'], df_filtered['net_new'], label=segment)

plt.xlabel('Snapshot Date')
plt.ylabel('Net New Bookings')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'net_new_by_seg'), dpi=300)
plt.close()
print(f"Net New bookings by segemnt saved!")




#weekly trend
#filter for April 6–14
april_week = grouped[
    (grouped['snapshot'] >= pd.Timestamp('2024-04-06')) &
    (grouped['snapshot'] <= pd.Timestamp('2024-04-21'))
]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(april_week['snapshot'], april_week['net_new'], marker='o', color='#4490dc')
plt.xlabel('Snapshot Date')
plt.ylabel('Net New Bookings')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'weekly'), dpi=300)
plt.close()
print(f"weekly trend saved!")



#7 day moving average plots
plt.figure(figsize=(14, 6))

for segment in segments:
    df_seg = segment_data[segment].copy()
    df_seg['ma7'] = df_seg['net_new'].rolling(window=7, min_periods=1).mean()

    color = segment_colors.get(segment, 'black')  # fallback color if not found
    plt.plot(df_seg['snapshot'], df_seg['ma7'], label=f'{segment}', color=color, linewidth=2)

plt.xlabel('Snapshot Date')
plt.ylabel('Net New Bookings')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, '7ma'), dpi=300)
plt.close()
print(f"7 day MA saved!")


#ca-us proportion time series
#all segemnts divide, ensure clean, percentage share, make TS
combined_df = None

for seg_name, df_seg in segment_data.items():
    df = df_seg[['snapshot', 'net_new']].copy()
    df = df.rename(columns={'net_new': f'net_new_{seg_name.replace(" ", "_")}'})
    
    if combined_df is None:
        combined_df = df
    else:
        combined_df = combined_df.merge(df, on='snapshot', how='outer')

combined_df = combined_df.fillna(0)

segment_cols = [col for col in combined_df.columns if col.startswith('net_new_')]
combined_df['net_new_total'] = combined_df[segment_cols].sum(axis=1)

combined_df['pct_CA_US'] = (
    combined_df['net_new_CA-US'] / combined_df['net_new_total']
) * 100

share_series = combined_df.set_index('snapshot')['pct_CA_US'].asfreq('D')

#dealing with some outliers

q1 = share_series.quantile(0.25)
q3 = share_series.quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

cleaned_share = share_series.copy()
cleaned_share[(cleaned_share < lower_bound) | (cleaned_share > upper_bound)] = np.nan



#percent share with trend lines ca-us
cutoff_date = pd.to_datetime('2024-12-31')

before = cleaned_share[cleaned_share.index <= cutoff_date].dropna()
after = cleaned_share[cleaned_share.index > cutoff_date].dropna()

x_before_fit = before.index.map(pd.Timestamp.toordinal)
x_after_fit = after.index.map(pd.Timestamp.toordinal)
y_before = before.values
y_after = after.values

slope_before, intercept_before = np.polyfit(x_before_fit, y_before, 1)
slope_after, intercept_after = np.polyfit(x_after_fit, y_after, 1)

trend_before = slope_before * x_before_fit + intercept_before
trend_after = slope_after * x_after_fit + intercept_after

plt.figure(figsize=(12, 5))
plt.plot(cleaned_share, label='CA-US Market Share', color='#4490dc')
plt.plot(before.index, trend_before, color='green', linestyle='--', label='Trend Before 2025')
plt.plot(after.index, trend_after, color='red', linestyle='--', label='Trend After 2025')
plt.axvline(cutoff_date + pd.Timedelta(days=1), color='gray', linestyle=':', label='Jan 1, 2025')
plt.xlabel('Date')
plt.ylabel('Share of Bookings (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'share_with_trends'), dpi=300)
plt.close()
print(f"CA-US share with trends saved!")


#STL then sarimax on trend
ts = cleaned_share.dropna()  # critical step
stl = STL(ts, period=7, robust=True)
stl_result = stl.fit()


stl_result.plot()
plt.tight_layout()
plt.title("")
plt.savefig(os.path.join(figures_path, 'share_stl'), dpi=300)
plt.close()
print(f"CA-US share stl saved!")


trend = stl_result.trend.dropna()

#ramp
trump_effect = pd.Series(0, index=trend.index)
ramp_start = pd.to_datetime("2025-01-01")
trump_effect[trump_effect.index >= ramp_start] = np.arange(1, (trend.index >= ramp_start).sum() + 1)
trump_effect.name = "TrumpEffect"


combined = pd.concat([trend, trump_effect], axis=1).dropna()
y = combined.iloc[:, 0]
exog = combined[['TrumpEffect']]

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




#------------------------------stl trend pct share sarimax-----------------------------------------
def run_sarimax(series, exog, order, seasonal_order, label, fname_suffix, tables_path):

    # Fit SARIMAX model
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


    # Extract coefficients and format
    coeff_table = results.summary().tables[1].data
    rows = coeff_table[1:] 
    formatted_rows = [format_row(row) for row in rows]

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
    tex_path = os.path.join(tables_path, f'sarimax_summary_{fname_suffix}.tex')
    with open(tex_path, 'w') as f:
        f.write("\n".join(tex_lines))

    print(f"LaTeX table saved to: {tex_path}")


#----------------------------run it------------------------
run_sarimax(
    series=trend,
    exog=combined[['TrumpEffect']],
    order=(1, 1, 0),
    seasonal_order=(0, 1, 0, 7),
    label="STL-Trend SARIMAX on CA-US Share",
    fname_suffix="ca_us_share",
    tables_path=tables_path
)



#------------------------pies----------------------
def save_dual_pie(pre, post, labels, colors, fname):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].pie(pre, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[0].axis('equal')

    axes[1].pie(post, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[1].axis('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, fname), dpi=300)
    plt.close()
    print(f"{fname} saved!")


#pie
cutoff_date = pd.to_datetime('2025-01-01')
pre_totals = {}
post_totals = {}

for segment in segments:
    df = segment_data[segment]
    pre = df[df['snapshot'] < cutoff_date]['net_new'].mean()
    post = df[df['snapshot'] >= cutoff_date]['net_new'].mean()
    pre_totals[segment] = pre
    post_totals[segment] = post

save_dual_pie(
    pre=[pre_totals[s] for s in segments],
    post=[post_totals[s] for s in segments],
    labels=segments,
    colors=[segment_colors[s] for s in segments],
    fname='net_new_booking_share_pre_post.png'
)

