import numpy as np
import pandas as pd

from data_plotter import create_well_plot, \
    calc_relative_error, \
    draw_statistics, \
    draw_performance, \
    draw_wells_model, \
    draw_histogram_model
from config import oilfield, bin_size, ignore_wells, read_columns, skiprows, rhoo, path_read, dates


# %% Defining methods

def read_RGD(cols, skip):
    prod_rgd = np.array([])
    for sheet in [0, 1, 2]:
        s_sheet = pd.read_excel(
            io=path_read / 'ргд.xlsx',
            sheet_name=sheet,
            usecols=[cols[sheet]],
            squeeze=True,
            skiprows=skip,
            engine='openpyxl',
        )
        idx = s_sheet.loc[s_sheet == 'Итого:'].index[0] - 1
        s_sheet = s_sheet.loc[:idx]
        s_sheet.dropna(inplace=True)
        prod = s_sheet.to_list()
        prod_rgd = np.append(prod_rgd, prod)
    # TODO: now returns [m3] with variable rhoo
    return prod_rgd / rhoo


# %% Read data

# Store dataframes of each model
xlsx_files = path_read.glob('*.xlsx')
dfs = dict()
rgd_exists = False
for filepath in xlsx_files:
    if filepath.stem == 'ргд':
        rgd_exists = True
        continue
    df = pd.read_excel(filepath, engine='openpyxl')
    df.name = filepath.stem
    dfs[filepath.stem] = df

# Obtain all well names
columns = []
for df in dfs.values():
    df.rename(columns={f'{df.columns[0]}': 'date'}, inplace=True)
    df.date = df.date.apply(lambda x: x.date())
    df.set_index('date', inplace=True)
    df[df <= 0] = np.nan
    df = df.reindex(dates)
    columns.extend(df.columns[::4])
well_names = [col.split('_')[0] for col in columns]
well_names = list(dict.fromkeys(well_names))  # Remove duplicates
well_names = [name for name in well_names if name not in ignore_wells]

# %% Initialize data
models = list(dfs.keys())
df_perf = {key: pd.DataFrame(data=0, index=dates, columns=['факт', 'модель']) for key in models}

# Read 'РГД' if exists
if rgd_exists:
    df_perf['ргд'] = read_RGD(read_columns, skiprows)

df_perf_liq = {key: pd.DataFrame(data=0, index=dates, columns=['факт', 'модель']) for key in models}
df_err_liq = {key: pd.DataFrame(data=0, index=dates, columns=['модель']) for key in models}

df_err = {key: pd.DataFrame(data=0, index=dates, columns=['модель']) for key in models}

# Daily model error
df_err_model = {key: pd.DataFrame(index=dates) for key in models}
df_err_model_liq = {key: pd.DataFrame(index=dates) for key in models}
# Cumulative model error
df_cumerr_model = {key: pd.DataFrame(index=dates) for key in models}
df_cumerr_model_liq = {key: pd.DataFrame(index=dates) for key in models}

model_mean = dict.fromkeys(models)
model_std = dict.fromkeys(models)
model_mean_daily = dict.fromkeys(models)
model_std_daily = dict.fromkeys(models)

# %% Calculations
print(f'Oilfield: {oilfield}')
print(f'Total number of different wells: {len(well_names)}')

for model, df in dfs.items():
    print(f'{model} number of wells: {df.shape[1] // 4}')

for name in well_names:
    create_well_plot(name, dfs, mode='liq')
    create_well_plot(name, dfs, mode='oil')

for model in models:
    for name in well_names:
        # Check if current model has this well
        if f'{name}_oil_true' not in dfs[model].columns:
            continue

        q_fact = dfs[model][f'{name}_oil_true']
        q_model = dfs[model][f'{name}_oil_pred']
        q_fact_liq = dfs[model][f'{name}_liq_true']
        q_model_liq = dfs[model][f'{name}_liq_pred']
        df_err_model[model][f'{name}'] = np.abs(q_model - q_fact) / np.maximum(q_model, q_fact) * 100
        df_err_model_liq[model][f'{name}'] = np.abs(q_model_liq - q_fact_liq) / np.maximum(q_model_liq, q_fact_liq)*100

        # Cumulative q
        Q_model = q_model.cumsum()
        Q_fact = q_fact.cumsum()
        df_cumerr_model[model][f'{name}'] = (Q_model - Q_fact) / np.maximum(Q_model, Q_fact) * 100

        Q_model_liq = q_model_liq.cumsum()
        Q_fact_liq = q_fact_liq.cumsum()
        df_cumerr_model_liq[model][f'{name}'] = (Q_model_liq - Q_fact_liq) / np.maximum(Q_model_liq, Q_fact_liq) * 100

        df_perf[model]['факт'] += q_fact.fillna(0)
        df_perf[model]['модель'] += q_model.fillna(0)
        df_perf_liq[model]['факт'] += q_fact_liq.fillna(0)
        df_perf_liq[model]['модель'] += q_model_liq.fillna(0)

for model in models:
    df_err[model]['модель'] = calc_relative_error(df_perf[model]['факт'], df_perf[model]['модель'])
    df_err_liq[model]['модель'] = calc_relative_error(df_perf_liq[model]['факт'], df_perf_liq[model]['модель'])

    if rgd_exists:
        df_err['ргд'] = calc_relative_error(df_perf[model]['факт'], df_perf['ргд'])

    model_mean[model] = df_cumerr_model[model].mean(axis=1)
    model_std[model] = df_cumerr_model[model].std(axis=1)

    model_mean_liq = df_cumerr_model_liq[model].mean(axis=1)
    model_std_liq = df_cumerr_model_liq[model].std(axis=1)

    model_mean_daily[model] = df_err_model[model].mean(axis=1)
    model_std_daily[model] = df_err_model[model].std(axis=1)

    # TODO: сейчас строится для жидкости. Если надо для нефти, то убрать "_liq"
    draw_histogram_model(df_err_model_liq[model], model, bin_size)
    draw_wells_model(df_err_model[model], model)

# %% Draw common statistics

draw_performance(dfs, df_perf, df_err, mode='oil', rgd_exists=rgd_exists)
draw_performance(dfs, df_perf_liq, df_err_liq, mode='liq')

draw_statistics(
    models,
    model_mean,
    model_std,
    model_mean_daily,
    model_std_daily,
)
