from pandas import date_range


class ConfigStatistics:
    MODEL_NAMES = {
        'ftor': 'Пьезо',
        'wolfram': 'ML',
        'ensemble': 'Ансамбль',
        'CRM': 'CRM',
        'true': 'Факт',
    }

    ignore_plots = [
        'Распределение ошибки (жидкость) "CRM"',
        'Ошибка прогноза (жидкость) "CRM"',
        'Распределение ошибки (жидкость) "Ансамбль"',
        'Ошибка прогноза (жидкость) "Ансамбль"',
    ]

    def __init__(
            self,
            oilfield: str,
            dates: date_range,
            use_abs: bool,
            well_names: tuple = (),
            bin_size: int = 10,
    ):
        self.oilfield = oilfield
        self.dates = dates
        self.use_abs = use_abs
        self.bin_size = bin_size
        self.well_names = well_names

    def exclude_wells(self, exclude_wells: list):
        self.well_names = [elem for elem in self.well_names if elem not in exclude_wells]
