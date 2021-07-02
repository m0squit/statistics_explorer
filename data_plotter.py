import pandas as pd
import numpy as np
import plotly as pl
import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots
from pathlib import Path

from config import oilfield, path_save, dates


def calc_relative_error(y_true: pd.Series,
                        y_pred: pd.Series) -> pd.Series:
    err = np.abs(y_pred - y_true) / np.maximum(y_pred, y_true)
    return err * 100


def create_well_plot(name: str,
                     dfs: dict,
                     mode='oil') -> None:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=[
            f'Дебит: {mode}, м3',
            'Относительная ошибка, %',
        ]
    )
    fig.layout.template = 'seaborn'
    fig.update_layout(
        font=dict(size=15),
        title_text=f'Скважина "{name}"; {oilfield};',
        legend=dict(orientation="h",
                    font=dict(size=15)
                    ),
    )

    mark = dict(size=4)
    m = 'markers'
    ml = 'markers+lines'
    colors = px.colors.qualitative.Safe

    # TODO: сейчас факт строится по всем моделям
    for ind, (model, df) in enumerate(dfs.items()):
        if f'{name}_{mode}_pred' in df.columns:
            clr = colors[ind]

            trace = go.Scatter(name=f'факт_{model}', x=df.index, y=df[f'{name}_{mode}_true'],
                               mode=m, marker=mark, marker_color=clr)
            fig.add_trace(trace, row=1, col=1)

            trace = go.Scatter(name=model, x=df.index, y=df[f'{name}_{mode}_pred'],
                               mode=ml, marker=mark, line=dict(width=1, color=clr))
            fig.add_trace(trace, row=1, col=1)

            relative_error = calc_relative_error(df[f'{name}_{mode}_true'], df[f'{name}_{mode}_pred'])
            trace = go.Scatter(name=f're_{model}', x=df.index, y=relative_error,
                               mode=ml, marker=mark, line=dict(width=1, color=clr),
                               showlegend=False)
            fig.add_trace(trace, row=2, col=1)

    if not Path(f'{path_save}/__well plots__/{mode}').exists():
        Path(f'{path_save}/__well plots__/{mode}').mkdir(parents=True, exist_ok=True)

    print(f'Saving {mode} plot for well {name}...')
    pl.io.write_image(fig, file=f'{path_save}/__well plots__/{mode}/{name}.png',
                      width=1450, height=700, scale=2, engine='kaleido')


def draw_histogram_model(df_err: pd.DataFrame,
                         model: str,
                         bin_size: int):
    length = len(df_err)
    days = [length // 3, 2 * length // 3, -1]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=[
            f'За {days[0]} суток',
            f'За {days[1]} суток',
            f'За весь период прогноза',
        ],
    )
    fig.layout.template = 'seaborn'
    # TODO: по добыче {чего}
    fig.update_layout(
        title_text=f'Распределение средней ошибки за n дней по добыче жидкости'
                   f'; {oilfield}; Скважин: {len(df_err)}',
        bargap=0.005,
        font=dict(size=15),
        showlegend=False,
    )

    for ind, day in enumerate(days):
        x = df_err.iloc[:day].mean()
        fig.add_trace(
            go.Histogram(
                x=x,
                opacity=0.9,
                histnorm='percent',
                xbins=dict(
                    size=bin_size,
                ),
            ),
            row=ind + 1,
            col=1,
        )

        fig.update_xaxes(dtick=bin_size, row=ind + 1, col=1)
        fig.update_yaxes(title_text="Процент скважин", title_font_size=15, row=ind + 1, col=1)

    fig.update_xaxes(title_text="Усредненная относительная ошибка по добыче нефти, %",
                     title_font_size=16,
                     dtick=bin_size,
                     row=3, col=1)

    if not Path(f'{path_save}/{model}').exists():
        Path(f'{path_save}/{model}').mkdir(parents=True, exist_ok=True)

    pl.io.write_image(fig, file=f'{path_save}/{model}/histogram_model.png',
                      width=1450, height=700, scale=2, engine='kaleido')


def draw_wells_model(df_err_model: pd.DataFrame,
                     model: str):
    fig = make_subplots(
        rows=1,
        cols=1,
        vertical_spacing=0.05,
        subplot_titles=['Средняя относит. ошибка по добыче нефти на периоде прогноза, %']
    )
    fig.layout.template = 'seaborn'

    mean_err = df_err_model.mean(axis=0)
    mean_err = mean_err.sort_values()
    trace = go.Bar(x=mean_err.index, y=mean_err)
    fig.add_trace(trace, row=1, col=1)

    fig.update_xaxes(title_text="Номер скважины", row=1, col=1)
    fig.update_yaxes(title_text="Относит. ошибка, %", row=1, col=1)

    if not Path(f'{path_save}/{model}').exists():
        Path(f'{path_save}/{model}').mkdir(parents=True, exist_ok=True)

    pl.io.write_image(fig, file=f'{path_save}/{model}/wells_model.png',
                      width=2050, height=700, scale=2, engine='kaleido')
    fig.write_html(f'{path_save}/{model}/wells_model.html')


def draw_performance(dfs: dict,
                     df_perf: dict,
                     df_err: dict,
                     mode='oil',
                     rgd_exists=False):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f'Суммарная суточная добыча {mode}, м3',
                        'Относительное отклонение от факта, %'],
    )
    fig.layout.template = 'seaborn'

    text = f'Месторождение {oilfield}'
    fig.update_layout(
        title=dict(text=text, x=0.05, xanchor='left'),
        font=dict(size=10),
        legend=dict(
            orientation="h",
            font=dict(size=15)
        )
    )

    mark = dict(size=4)
    m = 'markers'
    ml = 'markers+lines'
    colors = px.colors.qualitative.Safe

    # TODO: сейчас факт строится по всем моделям
    for ind, model in enumerate(dfs.keys()):
        clr = colors[ind]
        x = df_perf[model].index
        trace = go.Scatter(name=f'факт_{model}', x=x, y=df_perf[model]['факт'],
                           mode=m, marker=mark, marker_color=clr)
        fig.add_trace(trace, row=1, col=1)

    # Model errors
    x = None
    for ind, model in enumerate(dfs.keys()):
        clr = colors[ind]
        x = df_perf[model].index
        trace1 = go.Scatter(name=model, x=x, y=df_perf[model]['модель'],
                            mode=ml, marker=mark, line=dict(width=1, color=clr))
        trace2 = go.Scatter(name=f'', x=x, y=df_err[model]['модель'],
                            mode=ml, marker=mark, line=dict(width=1, color=clr),
                            showlegend=False)

        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=2, col=1)

    if rgd_exists and mode == 'oil':
        clr = colors[-2]
        trace1 = go.Scatter(name='РГД', x=x, y=df_perf['ргд'],
                            mode=ml, marker=mark, line=dict(width=1, color=clr))
        trace2 = go.Scatter(name=f'', x=x, y=df_err['ргд'],
                            mode=ml, marker=mark, line=dict(width=1, color=clr),
                            showlegend=False)

        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=2, col=1)

    pl.io.write_image(fig, file=f'{path_save}/performance_{mode}.png',
                      width=1450, height=700, scale=2, engine='kaleido')
    fig.write_html(f'{path_save}/performance_{mode}.html')


def draw_statistics(
        models: list,
        model_mean: dict,
        model_std: dict,
        model_mean_daily: dict,
        model_std_daily: dict,
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

    text = f'Месторождение {oilfield} ; Добыча нефти, т'
    fig.update_layout(title=dict(text=text, x=0.05, xanchor='left'), font=dict(size=10))

    mark = dict(size=4)
    ml = 'markers+lines'
    colors = px.colors.qualitative.Safe

    # Model errors
    for ind, model in enumerate(models):
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
