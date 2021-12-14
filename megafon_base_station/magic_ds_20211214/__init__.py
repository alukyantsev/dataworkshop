import pandas as pd
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 110)
pd.set_option('display.precision', 3)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import numpy as np
import pickle

import math
import time
from tqdm import tqdm

# test = reload(sys.modules['magic_ds.ml.common']).test
from importlib import reload

# warnings.filterwarnings("ignore")
import warnings
