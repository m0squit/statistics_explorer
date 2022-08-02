import datetime
from pathlib import Path

import pandas as pd
import plotly as pl

from statistics_explorer.config import ConfigStatistics
from statistics_explorer.plots import (
    calc_relative_error,
    create_well_plot,
    draw_histogram_model,
    draw_histogram_model_multi,
    draw_performance,
    draw_statistics,
    draw_wells_model,
    draw_wells_model_multi,
)


def calculate_statistics(dfs: dict, config: ConfigStatistics):
    # Initialize data
    analytics_plots = {}
    models = list(dfs.keys())
    df_perf = {
        key: pd.DataFrame(data=0, index=config.dates, columns=["факт", "модель"])
        for key in models
    }
    df_perf_liq = {
        key: pd.DataFrame(data=0, index=config.dates, columns=["факт", "модель"])
        for key in models
    }
    df_err = {
        key: pd.DataFrame(data=0, index=config.dates, columns=["модель"])
        for key in models
    }
    df_err_liq = {
        key: pd.DataFrame(data=0, index=config.dates, columns=["модель"])
        for key in models
    }
    # Daily model error
    df_err_model = {key: pd.DataFrame(index=config.dates) for key in models}
    df_err_model_liq = {key: pd.DataFrame(index=config.dates) for key in models}
    # Cumulative model error
    df_cumerr_model = {key: pd.DataFrame(index=config.dates) for key in models}
    df_cumerr_model_liq = {key: pd.DataFrame(index=config.dates) for key in models}

    model_mean = dict.fromkeys(models)
    model_std = dict.fromkeys(models)
    model_mean_liq = dict.fromkeys(models)
    model_std_liq = dict.fromkeys(models)
    model_mean_daily = dict.fromkeys(models)
    model_std_daily = dict.fromkeys(models)
    model_mean_daily_liq = dict.fromkeys(models)
    model_std_daily_liq = dict.fromkeys(models)

    # Calculations
    for model in models:
        for _well_name in config.well_names:
            # Check if current model has this well
            if f"{_well_name}_oil_true" not in dfs[model].columns:
                continue
            q_fact = dfs[model][f"{_well_name}_oil_true"]
            q_model = dfs[model][f"{_well_name}_oil_pred"]
            q_fact_liq = dfs[model][f"{_well_name}_liq_true"]
            q_model_liq = dfs[model][f"{_well_name}_liq_pred"]
            # Ошибка по суточной добыче
            df_err_model[model][f"{_well_name}"] = calc_relative_error(
                q_fact, q_model, use_abs=config.use_abs
            )
            df_err_model_liq[model][f"{_well_name}"] = calc_relative_error(
                q_fact_liq, q_model_liq, use_abs=config.use_abs
            )
            # Ошибка по накопленной добыче
            Q_model = q_model.cumsum()
            Q_fact = q_fact.cumsum()
            df_cumerr_model[model][f"{_well_name}"] = calc_relative_error(
                Q_fact, Q_model, use_abs=config.use_abs
            )
            Q_model_liq = q_model_liq.cumsum()
            Q_fact_liq = q_fact_liq.cumsum()
            df_cumerr_model_liq[model][f"{_well_name}"] = calc_relative_error(
                Q_fact_liq, Q_model_liq, use_abs=config.use_abs
            )
            df_perf[model]["факт"] += q_fact.fillna(0)
            df_perf[model]["модель"] += q_model.fillna(0)
            df_perf_liq[model]["факт"] += q_fact_liq.fillna(0)
            df_perf_liq[model]["модель"] += q_model_liq.fillna(0)

        # for model in models:
        # Ошибка по суммарной добыче на каждые сутки
        df_err[model]["модель"] = calc_relative_error(
            df_perf[model]["факт"], df_perf[model]["модель"], use_abs=config.use_abs
        )
        df_err_liq[model]["модель"] = calc_relative_error(
            df_perf_liq[model]["факт"],
            df_perf_liq[model]["модель"],
            use_abs=config.use_abs,
        )

        model_mean[model] = df_cumerr_model[model].mean(axis=1)
        model_std[model] = df_cumerr_model[model].std(axis=1)

        model_mean_liq[model] = df_cumerr_model_liq[model].mean(axis=1)
        model_std_liq[model] = df_cumerr_model_liq[model].std(axis=1)

        model_mean_daily[model] = df_err_model[model].mean(axis=1)
        model_std_daily[model] = df_err_model[model].std(axis=1)

        model_mean_daily_liq[model] = df_err_model_liq[model].mean(axis=1)
        model_std_daily_liq[model] = df_err_model_liq[model].std(axis=1)

        temp_name = f'Распределение ошибки (нефть) "{config.MODEL_NAMES[model]}"'
        analytics_plots[temp_name] = draw_histogram_model(
            df_err_model[model], config.bin_size, config.oilfield, "Дебит нефти"
        )
        temp_name = f'Ошибка прогноза (нефть) "{config.MODEL_NAMES[model]}"'
        analytics_plots[temp_name] = draw_wells_model(
            df_err_model[model], "Дебит нефти"
        )

        temp_name = f'Распределение ошибки (жидкость) "{config.MODEL_NAMES[model]}"'
        analytics_plots[temp_name] = draw_histogram_model(
            df_err_model_liq[model], config.bin_size, config.oilfield, "Дебит жидкости"
        )
        temp_name = f'Ошибка прогноза (жидкость) "{config.MODEL_NAMES[model]}"'
        analytics_plots[temp_name] = draw_wells_model(
            df_err_model_liq[model], "Дебит жидкости"
        )

    temp_name = f'Ошибка прогноза (нефть) "{models}"'
    analytics_plots[temp_name] = draw_wells_model_multi(
        df_err_model, models, config.MODEL_NAMES, "Дебит нефти"
    )
    temp_name_multi_oil = (
        f"Распределение ошибки (нефть) {list(config.MODEL_NAMES.values())}"
    )
    analytics_plots[temp_name_multi_oil] = draw_histogram_model_multi(
        df_err_model,
        config.bin_size,
        config.oilfield,
        models,
        config.MODEL_NAMES,
        "Дебит нефти",
    )
    temp_name = f'Ошибка прогноза (жидкость) "{list(config.MODEL_NAMES.values())}"'
    analytics_plots[temp_name] = draw_wells_model_multi(
        df_err_model_liq, models, config.MODEL_NAMES, "Дебит жидкости"
    )
    temp_name_multi_liq = (
        f"Распределение ошибки (жидкость) {list(config.MODEL_NAMES.values())}"
    )
    analytics_plots[temp_name_multi_liq] = draw_histogram_model_multi(
        df_err_model_liq,
        config.bin_size,
        config.oilfield,
        models,
        config.MODEL_NAMES,
        "Дебит жидкости",
    )
    # Draw common statistics
    analytics_plots["Суммарная добыча нефти"] = draw_performance(
        dfs, df_perf, df_err, config.oilfield, config.MODEL_NAMES, mode="oil"
    )
    analytics_plots["Суммарная добыча жидкости"] = draw_performance(
        dfs, df_perf_liq, df_err_liq, config.oilfield, config.MODEL_NAMES, mode="liq"
    )

    analytics_plots["Статистика по нефти"] = draw_statistics(
        models,
        model_mean,
        model_std,
        model_mean_daily,
        model_std_daily,
        config.oilfield,
        config.dates,
        config.MODEL_NAMES,
        "Дебит нефти",
    )
    analytics_plots["Статистика по жидкости"] = draw_statistics(
        models,
        model_mean_liq,
        model_std_liq,
        model_mean_daily_liq,
        model_std_daily_liq,
        config.oilfield,
        config.dates,
        config.MODEL_NAMES,
        "Дебит жидкости",
    )
    return analytics_plots


if __name__ == "__main__":
    # Конфиг
    config_stats = ConfigStatistics(
        oilfield="Отдельное",
        dates=pd.date_range(
            datetime.date(2022, 2, 1), datetime.date(2022, 4, 30), freq="D"
        ).date,
        use_abs=True,
        bin_size=10,
    )
    # Задание имен моделей на графиках. Ключ - название .xlsx файла, значение - название на графике.
    config_stats.MODEL_NAMES = {
        "ftor": "Пьезо",
        "wolfram": "ML",
        "ensemble": "Ансамбль",
        "CRM": "CRM",
        "fedot": "CRM+ML",
        "shelf": "ППТП",
    }
    path_read = Path.cwd() / config_stats.oilfield / "input_data"
    path_save = Path.cwd() / config_stats.oilfield / "output_data"

    # %% Read data
    # Store dataframes of each model
    xlsx_files = path_read.glob("*.xlsx")
    dfs = dict()
    for filepath in xlsx_files:
        df = pd.read_excel(filepath, engine="openpyxl")
        df.name = filepath.stem
        dfs[filepath.stem] = df

    # Obtain all well names
    wells_in_model = []
    for df in dfs.values():
        df.rename(columns={f"{df.columns[0]}": "date"}, inplace=True)
        df.date = df.date.apply(lambda x: x.date())
        df.set_index("date", inplace=True)
        df[df < 0] = 0
        df = df.reindex(config_stats.dates)
        wells_in_model.append(set([col.split("_")[0] for col in df.columns]))
    well_names_common = list(set.intersection(*wells_in_model))
    well_names_all = list(set.union(*wells_in_model))
    # Можно строить статистику только для общего набора скважин (скважина рассчитана всеми моделями),
    # либо для всех скважин (скважина рассчитана хотя бы одной моделью).
    # Выберите, что подать в конфиг ниже: well_names_common или well_names_all.
    config_stats.well_names = well_names_common
    # config_stats.exclude_wells(['2860242300'])  # Список названий скважин, которые нужно исключить из статистики

    analytics_plots = calculate_statistics(dfs, config_stats)
    available_plots = [*analytics_plots]
    plots_to_save = [
        plot_name
        for plot_name in available_plots
        if plot_name not in config_stats.ignore_plots
    ]

    # Сохранение поскважинных графиков
    # for mode in ["liq", "oil"]:
    #     if not Path(f"{path_save}/well plots/{mode}").exists():
    #         Path(f"{path_save}/well plots/{mode}").mkdir(parents=True, exist_ok=True)
    #     for name in config_stats.well_names:
    #         fig = create_well_plot(name, dfs, oilfield=config_stats.oilfield, mode=mode)
    #         pl.io.write_image(
    #             fig,
    #             file=f"{path_save}/well plots/{mode}/{name}.png",
    #             width=1450,
    #             height=700,
    #             scale=2,
    #             engine="kaleido",
    #         )
    # fig.write_html(f"{path_save}/well plots/{mode}/{name}.html")

    # Сохранение графиков статистики
    for plot_name in plots_to_save:
        savename = plot_name.replace('"', "")
        if not Path(f"{path_save}").exists():
            Path(f"{path_save}").mkdir(parents=True, exist_ok=True)
        pl.io.write_image(
            analytics_plots[plot_name],
            file=f"{path_save}/{savename}.png",
            width=1450,
            height=700,
            scale=2,
            engine="kaleido",
        )
        # analytics_plots[plot_name].write_html(f'{path_save}/{savename}.html')
