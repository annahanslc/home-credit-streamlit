import pandas as pd
import numpy as np
import joblib

from sklearn import set_config
set_config(transform_output='pandas')

def get_data():
  df = pd.read_csv('cleaned_sample.csv')
  return df
