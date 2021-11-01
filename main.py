import datetime
import numpy as np
import pandas as pd
import plotly as pl
from pathlib import Path

from statistics_explorer.config import ConfigStatistics
from statistics_explorer.plots import create_well_plot, \
    draw_statistics, \
    draw_performance, \
    draw_wells_model, \
    draw_histogram_model


# %% Defining methods
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


def calculate_statistics(dfs: dict, config: ConfigStatistics):
    # Initialize data
    analytics_plots = {}
    models = list(dfs.keys())
    df_perf = {key: pd.DataFrame(data=0, index=config.dates, columns=['факт', 'модель']) for key in models}
    df_perf_liq = {key: pd.DataFrame(data=0, index=config.dates, columns=['факт', 'модель']) for key in models}
    df_err_liq = {key: pd.DataFrame(data=0, index=config.dates, columns=['модель']) for key in models}
    df_err = {key: pd.DataFrame(data=0, index=config.dates, columns=['модель']) for key in models}
    # Daily model error
    df_err_model = {key: pd.DataFrame(index=config.dates) for key in models}
    df_err_model_liq = {key: pd.DataFrame(index=config.dates) for key in models}
    # Cumulative model error
    df_cumerr_model = {key: pd.DataFrame(index=config.dates) for key in models}
    df_cumerr_model_liq = {key: pd.DataFrame(index=config.dates) for key in models}

    model_mean = dict.fromkeys(models)
    model_std = dict.fromkeys(models)
    model_mean_daily = dict.fromkeys(models)
    model_std_daily = dict.fromkeys(models)

    # Calculations
    print(f'Месторождение: {config.oilfield}')
    print(f'Количество различных скважин: {len(config.well_names)}')
    for model, _df in dfs.items():
        print(f'{model} число скважин: {_df.shape[1] // 4}')

    for model in models:
        for _well_name in config.well_names:
            # Check if current model has this well
            if f'{_well_name}_oil_true' not in dfs[model].columns:
                continue

            q_fact = dfs[model][f'{_well_name}_oil_true']
            q_model = dfs[model][f'{_well_name}_oil_pred']
            q_fact_liq = dfs[model][f'{_well_name}_liq_true']
            q_model_liq = dfs[model][f'{_well_name}_liq_pred']
            df_err_model[model][f'{_well_name}'] = calc_relative_error(q_fact, q_model)
            df_err_model_liq[model][f'{_well_name}'] = calc_relative_error(q_fact_liq, q_model_liq)

            Q_model = q_model.cumsum()
            Q_fact = q_fact.cumsum()
            df_cumerr_model[model][f'{_well_name}'] = calc_relative_error(Q_fact, Q_model, use_abs=False)
            Q_model_liq = q_model_liq.cumsum()
            Q_fact_liq = q_fact_liq.cumsum()
            df_cumerr_model_liq[model][f'{_well_name}'] = calc_relative_error(Q_fact_liq, Q_model_liq, use_abs=False)

            df_perf[model]['факт'] += q_fact.fillna(0)
            df_perf[model]['модель'] += q_model.fillna(0)
            df_perf_liq[model]['факт'] += q_fact_liq.fillna(0)
            df_perf_liq[model]['модель'] += q_model_liq.fillna(0)

    for model in models:
        df_err[model]['модель'] = calc_relative_error(df_perf[model]['факт'], df_perf[model]['модель'])
        df_err_liq[model]['модель'] = calc_relative_error(df_perf_liq[model]['факт'], df_perf_liq[model]['модель'])

        model_mean[model] = df_cumerr_model[model].mean(axis=1)
        model_std[model] = df_cumerr_model[model].std(axis=1)

        # model_mean_liq = df_cumerr_model_liq[model].mean(axis=1)
        # model_std_liq = df_cumerr_model_liq[model].std(axis=1)

        model_mean_daily[model] = df_err_model[model].mean(axis=1)
        model_std_daily[model] = df_err_model[model].std(axis=1)

        # TODO: строится для жидкости/нефти. Если надо для жидкости, то подать "df_err_model_liq"
        temp_name = f'Распределение ошибки "{config.MODEL_NAMES[model]}"'
        analytics_plots[temp_name] = draw_histogram_model(df_err_model[model],
                                                          config.bin_size,
                                                          config.oilfield
                                                          )
        temp_name = f'Ошибка прогноза "{config.MODEL_NAMES[model]}"'
        analytics_plots[temp_name] = draw_wells_model(df_err_model[model])

    # Draw common statistics
    analytics_plots['Суммарная добыча нефти'] = draw_performance(dfs,
                                                                 df_perf,
                                                                 df_err,
                                                                 config.oilfield,
                                                                 config.MODEL_NAMES,
                                                                 mode='oil')
    analytics_plots['Суммарная добыча жидкости'] = draw_performance(dfs,
                                                                    df_perf_liq,
                                                                    df_err_liq,
                                                                    config.oilfield,
                                                                    config.MODEL_NAMES,
                                                                    mode='liq')

    analytics_plots['Статистика'] = draw_statistics(models,
                                                    model_mean,
                                                    model_std,
                                                    model_mean_daily,
                                                    model_std_daily,
                                                    config.oilfield,
                                                    config.dates,
                                                    config.MODEL_NAMES)

    return analytics_plots


if __name__ == '__main__':
    # Конфиг
    config_stats = ConfigStatistics(
        oilfield='Крайнее',
        dates=pd.date_range(datetime.date(2019, 2, 1), datetime.date(2019, 4, 30), freq='D').date,
        ignore_wells=(),
        bin_size=10,
    )
    path_read = Path.cwd() / 'input_data' / config_stats.oilfield
    path_save = Path.cwd() / 'output' / config_stats.oilfield

    # %% Read data
    # Store dataframes of each model
    xlsx_files = path_read.glob('*.xlsx')
    dfs = dict()
    for filepath in xlsx_files:
        df = pd.read_excel(filepath, engine='openpyxl')
        df.name = filepath.stem
        dfs[filepath.stem] = df

    # Obtain all well names
    columns = []
    for df in dfs.values():
        df.rename(columns={f'{df.columns[0]}': 'date'}, inplace=True)
        df.date = df.date.apply(lambda x: x.date())
        df.set_index('date', inplace=True)
        df[df < 0] = 0
        df = df.reindex(config_stats.dates)
        columns.extend(df.columns[::])
    well_names = [col.split('_')[0] for col in columns]
    well_names = list(dict.fromkeys(well_names))  # Remove duplicates
    well_names = [name for name in well_names if name not in config_stats.ignore_wells]
    config_stats.well_names = well_names

    analytics_plots = calculate_statistics(dfs, config_stats)

    # Сохранение поскважинных графиков
    for mode in ['liq', 'oil']:
        if not Path(f'{path_save}/well plots/{mode}').exists():
            Path(f'{path_save}/well plots/{mode}').mkdir(parents=True, exist_ok=True)
        for name in well_names:
            fig = create_well_plot(name, dfs, oilfield=config_stats.oilfield, mode=mode)
            pl.io.write_image(fig, file=f'{path_save}/well plots/{mode}/{name}.png',
                              width=1450, height=700, scale=2, engine='kaleido')
            # fig.write_html(f'{path_save}/well plots/{mode}/{name}.html')

    # Сохранение графиков статистики
    for plot_name in analytics_plots:
        savename = plot_name.replace('"', '')
        pl.io.write_image(analytics_plots[plot_name], file=f'{path_save}/{savename}.png',
                          width=1450, height=700, scale=2, engine='kaleido')
        analytics_plots[plot_name].write_html(f'{path_save}/{savename}.html')
