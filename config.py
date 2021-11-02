from pandas import date_range


class ConfigStatistics:
    MODEL_NAMES = {
        'ftor': 'Пьезо',
        'Piezo': 'Пьезо',
        'Модель пьезопроводности': 'Пьезо',
        'wolfram': 'ML',
        'Модель ML': 'ML',
        'ensemble': 'Ансамбль',
        'Ансамбль': 'Ансамбль',
        'CRM': 'CRM',
        'CRM_ML': 'CRM',
        'true': 'Факт',
        'Ela_Xgb': 'Ela_Xgb',
        'Xgb_SVR': 'Xgb_SVR',
        'ргд': 'ргд',
    }

    def __init__(
            self,
            oilfield: str,
            dates: date_range,
            use_abs: bool,
            well_names: tuple = (),
            ignore_wells: tuple = (),
            bin_size: int = 10,
    ):
        self.oilfield = oilfield
        self.dates = dates
        self.use_abs = use_abs
        self.bin_size = bin_size
        self.well_names = well_names
        self.ignore_wells = ignore_wells
