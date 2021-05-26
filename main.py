import datetime
import pathlib
import numpy as np
import pandas as pd
import plotly as pl
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from plotly.subplots import make_subplots


# %% Config

oilfield = 'kraynee'
date_test = datetime.date(2019, 2, 1)
date_end = datetime.date(2019, 4, 30)
# Settings for РГД
read_columns = {
    0: 27,
    1: 32,
    2: 29,
}
skiprows = 11


# oilfield = 'valyntoyskoe'
# date_test = datetime.date(2019, 3, 1)
# date_end = datetime.date(2019, 5, 31)
# # Settings for РГД
# read_columns = {
#     0: 12,
#     1: 16,
#     2: 14,
# }
# skiprows = 10


# %% Defining methods

def read_RGD(cols, skiprows):
    prod_rgd = list()
    for sheet in [0, 1, 2]:
        s_sheet = pd.read_excel(
            io=path_read / 'ргд.xlsx',
            sheet_name=sheet,
            usecols=[cols[sheet]],
            squeeze=True,
            skiprows=skiprows,
            engine='openpyxl',
        )
        idx = s_sheet.loc[s_sheet == 'Итого:'].index[0] - 1
        s_sheet = s_sheet.loc[:idx]
        s_sheet.dropna(inplace=True)
        prod = s_sheet.to_list()
        prod_rgd.extend(prod)
    return prod_rgd


def convert_day_date(x: str) -> datetime.date:
    return datetime.datetime.strptime(x, '%Y-%m-%d').date()


def calc_relative_error(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    diff = np.abs(y_pred - y_true)
    err = diff.div(y_true)
    err.replace(np.inf, np.nan)
    return err * 100


def create_well_plot(name: str, dfs: dict, liq=False) -> None:
    figure = go.Figure(layout=go.Layout(
        font=dict(size=10),
        hovermode='x',
        template='seaborn',
        title=dict(text=f'Скважина {name}', x=0.05, xanchor='left'),
    ))
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.6, 0.4],
        figure=figure,
    )

    mark = dict(size=4)
    m = 'markers'
    ml = 'markers+lines'
    colors = px.colors.qualitative.Safe

    # TODO: сейчас факт берется по одной из моделей произвольно
    mode = 'liq' if liq else 'oil'
    for df in dfs.values():
        if f'{name}_fact_{mode}' in df.columns:
            trace = go.Scatter(name='факт', x=df.index, y=df[f'{name}_fact_{mode}'], mode=m, marker=mark)
            fig.add_trace(trace, row=1, col=1)
            break

    for ind, (model, df) in enumerate(dfs.items()):
        if f'{name}_oil' in df.columns:
            clr = colors[ind]
            relative_error = calc_relative_error(df[f'{name}_fact_{mode}'], df[f'{name}_{mode}'])
            trace = go.Scatter(name=model, x=df.index, y=df[f'{name}_{mode}'],
                               mode=ml, marker=mark, line=dict(width=1, color=clr))
            fig.add_trace(trace, row=1, col=1)

            trace = go.Scatter(name=f're_{model}', x=df.index, y=relative_error,
                               mode=ml, marker=mark, line=dict(width=1, color=clr))
            fig.add_trace(trace, row=2, col=1)

    if not Path(f'{path_save}/well plots/{mode}').exists():
        Path(f'{path_save}/well plots/{mode}').mkdir(parents=True, exist_ok=True)

    pl.io.write_image(fig, file=f'{path_save}/well plots/{mode}/{name}.png',
                      width=1450, height=700, scale=2, engine='kaleido')


def draw_histogram_model(df_cumerr_model, model):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=[
            '30-е сутки',
            '60-е сутки',
            'Последние сутки',
        ],
    )
    fig.layout.template = 'seaborn'

    text = f'Месторождение {oilfield}; Скважин: {len(well_names)} ; Распределение ошибки'
    fig.update_layout(title=dict(text=text, x=0.05, xanchor='left'), font=dict(size=10))

    fig.add_trace(
        go.Histogram(
            x=df_cumerr_model.iloc[29],
            name='30-е сутки',
            opacity=0.9,
            xbins=dict(
                size=5
            ),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=df_cumerr_model.iloc[59],
            name='60-е сутки',
            opacity=0.9,
            xbins=dict(
                size=5
            ),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=df_cumerr_model.iloc[-1],
            name='Последние сутки',
            opacity=0.9,
            xbins=dict(
                size=5
            ),
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        title_text=f'Месторождение {oilfield} ({len(well_names)} скважин);'
                   f' Распределение ошибки по накопленной добыче нефти',
        bargap=0.005,
    )

    fig.update_xaxes(dtick=5, row=1, col=1)
    fig.update_xaxes(dtick=5, row=2, col=1)
    fig.update_xaxes(dtick=5, row=3, col=1)
    fig.update_xaxes(title_text="Относительная ошибка по накопленной добыче нефти, %", row=3, col=1)
    fig.update_yaxes(title_text="Число скважин", row=1, col=1)
    fig.update_yaxes(title_text="Число скважин", row=2, col=1)
    fig.update_yaxes(title_text="Число скважин", row=3, col=1)

    if not Path(f'{path_save}/{model}').exists():
        Path(f'{path_save}/{model}').mkdir(parents=True, exist_ok=True)

    pl.io.write_image(fig, file=f'{path_save}/{model}/histogram_model.png',
                      width=1450, height=700, scale=2, engine='kaleido')


def draw_wells_model(df_cumerr_model, model):
    fig = make_subplots(
        rows=1,
        cols=1,
        vertical_spacing=0.05,
        subplot_titles=['Относит. ошибка по накопленной добыче нефти на последние сутки, %']
    )
    fig.layout.template = 'seaborn'

    df_cumerr_model.sort_values(by=df_cumerr_model.index[-1], axis=1, inplace=True)
    trace = go.Bar(x=df_cumerr_model.columns, y=df_cumerr_model.iloc[-1])
    fig.add_trace(trace, row=1, col=1)

    fig.update_xaxes(title_text="Номер скважины", row=1, col=1)
    fig.update_yaxes(title_text="Относит. ошибка, %", row=1, col=1)

    if not Path(f'{path_save}/{model}').exists():
        Path(f'{path_save}/{model}').mkdir(parents=True, exist_ok=True)

    pl.io.write_image(fig, file=f'{path_save}/{model}/wells_model.png',
                      width=2050, height=700, scale=2, engine='kaleido')
    fig.write_html(f'{path_save}/{model}/wells_model.html')


def draw_performance(dfs, df_perf, df_err, liq=False, rgd_exists=False):
    mode = 'liq' if liq else 'oil'

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f'Суммарная суточная добыча {mode}, т', 'Относительное отклонение от факта, д. ед.'],
    )
    fig.layout.template = 'seaborn'

    text = f'Месторождение {oilfield} ({len(well_names)} скважин)'
    fig.update_layout(title=dict(text=text, x=0.05, xanchor='left'), font=dict(size=10))

    mark = dict(size=4)
    m = 'markers'
    ml = 'markers+lines'
    colors = px.colors.qualitative.Safe

    # TODO: сейчас факт берется по одной из моделей произвольно
    model = list(dfs.keys())[0]
    x = df_perf[model].index
    trace = go.Scatter(name='факт', x=x, y=df_perf[model]['факт'], mode=m, marker=mark)
    fig.add_trace(trace, row=1, col=1)

    # Model errors
    for ind, model in enumerate(dfs.keys()):
        clr = colors[ind]

        trace1 = go.Scatter(name=model, x=x, y=df_perf[model]['модель'],
                            mode=ml, marker=mark, line=dict(width=1, color=clr))
        trace2 = go.Scatter(name=f're_{model}', x=x, y=df_err[model]['модель'],
                            mode=ml, marker=mark, line=dict(width=1, color=clr))

        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=2, col=1)

    if rgd_exists and not liq:
        clr = colors[-1]
        trace1 = go.Scatter(name='РГД', x=x, y=df_perf['ргд'],
                            mode=ml, marker=mark, line=dict(width=1, color=clr))
        trace2 = go.Scatter(name=f're_РГД', x=x, y=df_err['ргд'],
                            mode=ml, marker=mark, line=dict(width=1, color=clr))

        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=2, col=1)

    pl.io.write_image(fig, file=f'{path_save}/performance_{mode}.png',
                      width=1450, height=700, scale=2, engine='kaleido')


def draw_statistics(
        model_mean,
        model_std,
        model_mean_daily,
        model_std_daily,
):
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            'Относит. ошибка по накопленной добыче, %',
            'Стандартное отклонение по накопленной добыче, %',
            'Относит. ошибка суточной добычи, %',
            'Стандартное отклонение по суточной добыче, %',
        ],
    )
    fig.layout.template = 'seaborn'

    text = f'Месторождение {oilfield}; Скважин: {len(well_names)} ; Добыча нефти, т'
    fig.update_layout(title=dict(text=text, x=0.05, xanchor='left'), font=dict(size=10))

    mark = dict(size=4)
    ml = 'markers+lines'
    colors = px.colors.qualitative.Safe

    # Model errors
    for ind, model in enumerate(dfs.keys()):
        clr = colors[ind]
        trace1 = go.Scatter(name=model, x=dates, y=model_mean[model], mode=ml,
                            marker=mark, line=dict(width=1, color=clr))
        trace2 = go.Scatter(name='', x=dates, y=model_std[model], mode=ml,
                            marker=mark, line=dict(width=1, color=clr))
        trace3 = go.Scatter(name=f'', x=dates, y=model_mean_daily[model],
                            mode=ml, marker=mark, line=dict(width=1, color=clr))
        trace4 = go.Scatter(name=f'', x=dates, y=model_std_daily[model],
                            mode=ml, marker=mark, line=dict(width=1, color=clr))

        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=2, col=1)
        fig.add_trace(trace3, row=3, col=1)
        fig.add_trace(trace4, row=4, col=1)

    pl.io.write_image(fig, file=f'{path_save}/statistics_oil.png', width=1450, height=700, scale=2, engine='kaleido')
    fig.write_html(f'{path_save}/statistics_oil.html')


# %% Read data
dates = pd.date_range(date_test, date_end, freq='D').date

path_read = pathlib.Path.cwd() / 'input_data' / oilfield
path_save = pathlib.Path.cwd() / 'output' / oilfield
if not Path(path_save).exists():
    Path(path_save).mkdir(parents=True, exist_ok=True)

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
    df = df.reindex(dates)
    columns.extend(df.columns[::4])
columns = list(dict.fromkeys(columns))  # Remove duplicates
well_names = [col.split('_')[0] for col in columns]

# %% Initialize data
df_perf = {key: pd.DataFrame(data=0, index=dates, columns=['факт', 'модель']) for key in dfs.keys()}

# Read 'РГД' if exists
if rgd_exists:
    df_perf['ргд'] = read_RGD(read_columns, skiprows)

df_perf_liq = {key: pd.DataFrame(data=0, index=dates, columns=['факт', 'модель']) for key in dfs.keys()}
df_err_liq = {key: pd.DataFrame(data=0, index=dates, columns=['модель']) for key in dfs.keys()}

df_err = {key: pd.DataFrame(data=0, index=dates, columns=['модель']) for key in dfs.keys()}

# Daily model error
df_err_model = {key: pd.DataFrame(index=dates) for key in dfs.keys()}
# Cumulative model error
df_cumerr_model = {key: pd.DataFrame(index=dates) for key in dfs.keys()}

model_mean = dict.fromkeys(dfs.keys())
model_std = dict.fromkeys(dfs.keys())
model_mean_daily = dict.fromkeys(dfs.keys())
model_std_daily = dict.fromkeys(dfs.keys())

# %% Calculations
for name in well_names:
    create_well_plot(name, dfs, liq=True)
    create_well_plot(name, dfs, liq=False)


for model in dfs.keys():
    for name in well_names:
        # Check if current model has this well
        if f'{name}_oil' not in dfs[model].columns:
            continue

        q_fact = dfs[model][f'{name}_fact_oil']
        q_model = dfs[model][f'{name}_oil']
        q_fact_liq = dfs[model][f'{name}_fact_liq']
        q_model_liq = dfs[model][f'{name}_liq']
        # q_rgd = df_perf['ргд'] * ratios[name]

        df_err_model[model][f'{name}'] = np.abs(q_model - q_fact) / q_fact * 100
        # df_err_rgd[f'{name}'] = np.abs(q_rgd - q_fact) / q_fact * 100

        # Cumulative q
        Q_model = q_model.cumsum()
        Q_fact = q_fact.cumsum()
        # Q_rgd = q_rgd.cumsum()

        df_cumerr_model[model][f'{name}'] = (Q_model - Q_fact) / Q_fact * 100
        # df_cumerr_rgd[f'{name}'] = (Q_rgd - Q_fact) / Q_fact * 100

        df_perf[model]['факт'] += q_fact
        df_perf[model]['модель'] += q_model
        df_perf_liq[model]['факт'] += q_fact_liq
        df_perf_liq[model]['модель'] += q_model_liq


for model in dfs.keys():
    df_err[model]['модель'] = calc_relative_error(df_perf[model]['факт'], df_perf[model]['модель'])
    df_err_liq[model]['модель'] = calc_relative_error(df_perf_liq[model]['факт'], df_perf_liq[model]['модель'])

    if rgd_exists:
        df_err['ргд'] = calc_relative_error(df_perf[model]['факт'], df_perf['ргд'])

    model_mean[model] = df_cumerr_model[model].mean(axis=1)
    model_std[model] = df_cumerr_model[model].std(axis=1)

    # rgd_mean = df_cumerr_rgd.mean(axis=1)
    # rgd_std = df_cumerr_rgd.std(axis=1)

    model_mean_daily[model] = df_err_model[model].mean(axis=1)
    model_std_daily[model] = df_err_model[model].std(axis=1)

    # rgd_mean_daily = df_err_rgd.mean(axis=1)
    # rgd_std_daily = df_err_rgd.std(axis=1)

    draw_histogram_model(df_cumerr_model[model], model)
    draw_wells_model(df_cumerr_model[model], model)

# %% Draw common statistics

draw_performance(dfs, df_perf, df_err, liq=False, rgd_exists=rgd_exists)
draw_performance(dfs, df_perf_liq, df_err_liq, liq=True)

draw_statistics(
    model_mean,
    model_std,
    model_mean_daily,
    model_std_daily,
)
