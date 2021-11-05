import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def calc_relative_error(y_true: pd.Series,
                        y_pred: pd.Series,
                        use_abs: bool = True) -> pd.Series:
    if use_abs:
        err = np.abs(y_pred - y_true) / np.maximum(y_pred, y_true)
    else:
        err = (y_pred - y_true) / np.maximum(y_pred, y_true)
    # Ошибка может быть больше 100%, если одно из значений отрицательное. Исключаем такие случаи.
    err[err > 1] = 1
    err[err < -1] = -1
    return err * 100


def create_well_plot(name: str,
                     dfs: dict,
                     oilfield: str,
                     mode: str = 'oil'):
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
        legend=dict(
            # orientation="h",
            font=dict(size=15)
        ),
    )

    mark = dict(size=4)
    m = 'markers'
    ml = 'markers+lines'
    colors = px.colors.qualitative.Safe

    # сейчас факт строится по всем моделям
    for ind, (model, df) in enumerate(dfs.items()):
        if f'{name}_{mode}_pred' in df.columns:
            clr = colors[ind]

            trace = go.Scatter(name=f'факт_{model}', x=df.index, y=df[f'{name}_{mode}_true'],
                               mode=m, marker=mark, marker_color=clr)
            fig.add_trace(trace, row=1, col=1)

            trace = go.Scatter(name=model, x=df.index, y=df[f'{name}_{mode}_pred'],
                               mode=ml, marker=mark, line=dict(width=1, color=clr))
            fig.add_trace(trace, row=1, col=1)

            relative_error = calc_relative_error(df[f'{name}_{mode}_true'], df[f'{name}_{mode}_pred'], use_abs=False)
            trace = go.Scatter(name=f're_{model}', x=df.index, y=relative_error,
                               mode=ml, marker=mark, line=dict(width=1, color=clr),
                               showlegend=False)
            fig.add_trace(trace, row=2, col=1)
    return fig


def create_well_plot_UI(statistics: dict,
                        df_chess: pd.DataFrame,
                        dates: pd.date_range,
                        date_test: datetime.date,
                        date_test_period: datetime.date,
                        wellname: str,
                        MODEL_NAMES: dict,
                        ensemble_interval: pd.DataFrame = pd.DataFrame()):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=['Дебит жидкости, м3',
                                        'Дебит нефти, м3',
                                        'Относительная ошибка по нефти, %',
                                        'Забойное давление, атм'])
    fig.update_layout(font=dict(size=15),
                      template='seaborn',
                      title_text=f'Скважина {wellname}',
                      legend=dict(orientation="v",
                                  font=dict(size=15),
                                  traceorder='normal'),
                      height=760,
                      width=1300)
    mark, m, ml = dict(size=4), 'markers', 'markers+lines'
    colors = {'ftor': px.colors.qualitative.Pastel[1],
              'wolfram': 'rgba(248, 156, 116, 0.6)',
              'CRM': px.colors.qualitative.Pastel[6],
              'ensemble': 'rgba(115, 175, 72, 0.7)',
              'ensemble_interval': 'rgba(184, 247, 212, 0.7)',
              'true': 'rgba(99, 110, 250, 0.7)',
              'pressure': '#C075A6'}
    if not ensemble_interval.empty:
        trace = go.Scatter(name=f'OIL: Доверит. интервал', x=ensemble_interval.index, y=ensemble_interval[f'{wellname}_lower'],
                           mode='lines', line=dict(width=1, color=colors['ensemble_interval']))
        fig.add_trace(trace, row=2, col=1)
        trace = go.Scatter(name=f'OIL: Доверит. интервал', x=ensemble_interval.index, y=ensemble_interval[f'{wellname}_upper'],
                           fill='tonexty', mode='lines', line=dict(width=1, color=colors['ensemble_interval']))
        fig.add_trace(trace, row=2, col=1)
        fig.add_vline(x=date_test_period, line_width=1, line_dash='dash', exclude_empty_subplots=False)
    x = dates
    y_liq_true = df_chess['Дебит жидкости']
    y_oil_true = df_chess['Дебит нефти']
    # Факт
    trace = go.Scatter(name=f'LIQ: {MODEL_NAMES["true"]}', x=x, y=y_liq_true,
                       mode=m, marker=dict(size=5, color=colors['true']))
    fig.add_trace(trace, row=1, col=1)
    trace = go.Scatter(name=f'OIL: {MODEL_NAMES["true"]}', x=x, y=y_oil_true,
                       mode=m, marker=dict(size=5, color=colors['true']))
    fig.add_trace(trace, row=2, col=1)
    for model in statistics:
        if f'{wellname}_oil_pred' in statistics[model]:
            clr = colors[model]
            y_liq = statistics[model][f'{wellname}_liq_pred']
            y_oil = statistics[model][f'{wellname}_oil_pred']
            trace_liq = go.Scatter(name=f'LIQ: {MODEL_NAMES[model]}', x=x, y=y_liq,
                                   mode=ml, marker=mark, line=dict(width=1, color=clr))
            fig.add_trace(trace_liq, row=1, col=1)  # Дебит жидкости
            trace_oil = go.Scatter(name=f'OIL: {MODEL_NAMES[model]}', x=x, y=y_oil,
                                   mode=ml, marker=mark, line=dict(width=1, color=clr))
            fig.add_trace(trace_oil, row=2, col=1)  # Дебит нефти
            deviation = calc_relative_error(y_oil_true, y_oil, use_abs=False)
            trace_err = go.Scatter(name=f'OIL ERR: {MODEL_NAMES[model]}', x=deviation.index, y=deviation,
                                   mode=ml, marker=mark, line=dict(width=1, color=clr))
            fig.add_trace(trace_err, row=3, col=1)  # Ошибка по нефти
    # Забойное давление
    pressure = df_chess['Давление забойное']
    trace_pressure = go.Scatter(name=f'Заб. давление', x=pressure.index, y=pressure,
                                mode=m, marker=dict(size=4, color=colors['pressure']))
    fig.add_trace(trace_pressure, row=4, col=1)
    # Мероприятия
    events = df_chess['Мероприятие']
    _events = events.dropna()
    trace_events = go.Scatter(name='Мероприятие', x=_events.index, y=[0.2] * len(_events),
                              mode='markers+text', marker=dict(size=8), text=_events.array,
                              textposition='top center', textfont=dict(size=12))
    fig.add_trace(trace_events, row=4, col=1)
    fig.add_vline(x=date_test, line_width=1, line_dash='dash')
    return fig


def draw_histogram_model(df_err: pd.DataFrame,
                         bin_size: int,
                         oilfield: str,
                         ):
    length = len(df_err)
    days = [length // 3, 2 * length // 3, -1]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            f'За {days[0]} суток',
            f'За {days[1]} суток',
            f'За весь период прогноза',
        ],
    )
    fig.layout.template = 'seaborn'
    fig.update_layout(
        title_text=f'Распределение средней ошибки за n дней',
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
                # histnorm='percent',
                xbins=dict(size=bin_size),
                # nbinsx=8,
            ),
            row=ind + 1,
            col=1,
        )

        fig.update_xaxes(dtick=bin_size, row=ind + 1, col=1)
        fig.update_yaxes(title_text="Скважин", title_font_size=15, row=ind + 1, col=1)

    err_all = df_err.mean()
    fig.update_xaxes(
        title_text=f"Усредненная относительная ошибка, %<br><br>"
                   f"<i>Среднее значение ошибки за весь период: <em>{err_all.mean():.2f}</i></em><br>"
                   f"<i>Стандартное отклонение ошибки за весь период: <em>{err_all.std():.2f}</i></em><br>"
                   f"Месторождение: <em>{oilfield}</em>. Количество скважин: <em>{df_err.shape[1]}</em>",
        title_font_size=16,
        row=3,
        col=1
    )
    return fig


def draw_wells_model(df_err_model: pd.DataFrame):
    fig = make_subplots(
        rows=1,
        cols=1,
    )
    fig.layout.template = 'seaborn'
    fig.update_layout(
        title_text=f'Средняя относит. ошибка на периоде прогноза, %',
        # bargap=0.005,
        font=dict(size=15),
    )

    mean_err = df_err_model.mean(axis=0)
    mean_err = mean_err.sort_values()
    trace = go.Bar(x=mean_err.index, y=mean_err)
    fig.add_trace(trace, row=1, col=1)

    fig.update_xaxes(title_text=f"Номер скважины<br><br>"
                                f"<i>Среднее значение ошибки: <em>{mean_err.mean():.2f}</em></i>", row=1, col=1)
    fig.update_yaxes(title_text="Относит. ошибка, %", row=1, col=1)
    return fig


def draw_performance(dfs: dict,
                     df_perf: dict,
                     df_err: dict,
                     oilfield: str,
                     MODEL_NAMES: dict,
                     mode='oil'):
    modes_decode = {
        'oil': 'нефти',
        'liq': 'жидкости',
    }
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f'Суммарная суточная добыча {modes_decode[mode]}, м3',
                        'Относительное отклонение от факта, %'],
    )
    fig.layout.template = 'seaborn'

    text = f'Месторождение {oilfield}'
    fig.update_layout(
        title=dict(text=text, x=0.05, xanchor='left'),
        font=dict(size=10),
        legend=dict(
            # orientation="h",
            font=dict(size=15)
        )
    )

    mark = dict(size=4)
    m = 'markers'
    ml = 'markers+lines'
    colors = px.colors.qualitative.Safe

    # сейчас факт строится по всем моделям
    for ind, model in enumerate(dfs.keys()):
        clr = colors[ind]
        x = df_perf[model].index
        trace = go.Scatter(name=f'факт {MODEL_NAMES[model]}', x=x, y=df_perf[model]['факт'],
                           mode=m, marker=mark, marker_color=clr)
        fig.add_trace(trace, row=1, col=1)

    annotation_text = ''
    # Model errors
    for ind, model in enumerate(dfs.keys()):
        clr = colors[ind]
        x = df_perf[model].index
        trace1 = go.Scatter(name=f'{MODEL_NAMES[model]}', x=x, y=df_perf[model]['модель'],
                            mode=ml, marker=mark, line=dict(width=1, color=clr))
        trace2 = go.Scatter(name=f'ERR: {MODEL_NAMES[model]}', x=x, y=df_err[model]['модель'],
                            mode=ml, marker=mark, line=dict(width=1, color=clr))
        annotation_text += f'<i>Среднее значение ошибки <em>{MODEL_NAMES[model]}</em>: ' \
                           f'{df_err[model]["модель"].mean():.2f}</i><br>'
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=2, col=1)

    fig.update_xaxes(
        title_text=annotation_text,
        title_font_size=16,
        row=2,
        col=1
    )
    return fig


def draw_statistics(
        models: list,
        model_mean: dict,
        model_std: dict,
        model_mean_daily: dict,
        model_std_daily: dict,
        oilfield: str,
        dates: pd.date_range,
        MODEL_NAMES: dict,
):
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            'Средняя относит. ошибка по накопленной добыче, %',
            'Стандартное отклонение по накопленной добыче, %',
            'Средняя относит. ошибка суточной добычи, %',
            'Стандартное отклонение по суточной добыче, %',
        ],
    )
    fig.layout.template = 'seaborn'

    text = f'Месторождение <em>{oilfield}</em>'
    fig.update_layout(title=dict(text=text, x=0.05, xanchor='left'), font=dict(size=10))

    mark = dict(size=4)
    ml = 'markers+lines'
    colors = px.colors.qualitative.Safe

    # Model errors
    for ind, model in enumerate(models):
        clr = colors[ind]
        trace1 = go.Scatter(name=f'{MODEL_NAMES[model]}', x=dates, y=model_mean[model], mode=ml,
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

    return fig
