import pandas as pd
from .utils import *

def count_nas(df):

  tot = len(df)
  cols = list(df.columns)
  nas = [df[col].isna().sum() for col in cols]
  percent_nas = [na/tot * 100 for na in nas]

  na_df = pd.DataFrame.from_dict({"col": cols, "nas": nas, "percent_nas": percent_nas})
  na_df.sort_values(by = ["percent_nas"], ascending = False, inplace = True)
  na_df = na_df.reset_index(drop = True)

  return na_df