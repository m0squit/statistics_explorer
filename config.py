import datetime
import pandas as pd
from pathlib import Path


# oilfield = 'kraynee'
# date_test = datetime.date(2019, 2, 1)
# date_end = datetime.date(2019, 4, 30)
# ignore_wells = []
# bin_size = 10
# # Settings for РГД
# read_columns = {
#     0: 27,
#     1: 32,
#     2: 29,
# }
# skiprows = 11
# rhoo = 0.847


# oilfield = 'valyntoyskoe'
# date_test = datetime.date(2019, 3, 1)
# date_end = datetime.date(2019, 5, 31)
# bin_size = 20
# ignore_wells = []
# # Settings for РГД
# read_columns = {
#     0: 12,
#     1: 16,
#     2: 14,
# }
# skiprows = 10
# rhoo = 0.794


oilfield = 'vyngayakhinskoe'
date_test = datetime.date(2019, 4, 1)
date_end = datetime.date(2019, 6, 30)
bin_size = 10
ignore_wells = ['2860424700']
# Settings for РГД
read_columns = {
    0: 32,
    1: 34,
    2: 39,
}
skiprows = 10
rhoo = 0.832


# %%
path_read = Path.cwd() / 'input_data' / oilfield
path_save = Path.cwd() / 'output' / oilfield
if not Path(path_save).exists():
    Path(path_save).mkdir(parents=True, exist_ok=True)

dates = pd.date_range(date_test, date_end, freq='D').date
