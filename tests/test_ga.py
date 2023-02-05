from absplit import ABSplit, Match
import pandas as pd
import datetime as dt
import numpy as np

# Build test data
data_dct = {
    'date': [dt.date(2030,4,1) + dt.timedelta(days=x) for x in range(3)]*5,
    'country': ['UK'] * 15,
    'region': [item for sublist in [[x]*6 for x in ['z', 'y']] for item in sublist] + ['x']*3,
    'city': [item for sublist in [[x]*3 for x in ['a', 'b', 'c', 'd', 'e']] for item in sublist],
    'metric1': np.arange(0, 15, 1),
    'metric2': np.arange(0, 150, 10)
}
df = pd.DataFrame(data_dct)

# Column kwargs
kwargs = {
    'metrics': ['metric1', 'metric2'],
    'date_col': 'date',
    'split_by': 'region',
    'splitting': 'city'
}

# Initialise and run
ab = ABSplit(
    df=df,
    **kwargs,
)
ab.run()

# Results
data_dct1 = {
    'country': {0: 'UK', 1: 'UK', 2: 'UK', 3: 'UK', 4: 'UK'},
    'region': {0: 'x', 1: 'y', 2: 'y', 3: 'z', 4: 'z'},
    'city': {0: 'e', 1: 'c', 2: 'd', 3: 'a', 4: 'b'},
    'bin': {0: 1, 1: 0, 2: 0, 3: 1, 4: 1}
}
data_dct2 = {
    'country': {0: 'UK', 1: 'UK', 2: 'UK', 3: 'UK', 4: 'UK'},
    'region': {0: 'x', 1: 'y', 2: 'y', 3: 'z', 4: 'z'},
    'city': {0: 'e', 1: 'c', 2: 'd', 3: 'a', 4: 'b'},
    'bin': {0: 0, 1: 1, 2: 1, 3: 0, 4: 0}
}
df_data1 = pd.DataFrame(data_dct1)
df_data2 = pd.DataFrame(data_dct2)


def test_ab_results():
    """GA output ab.results"""
    r1 = len(ab.results.reset_index().compare(df_data1)) == 0
    r2 = len(ab.results.reset_index().compare(df_data2)) == 0
    assert True in (r1, r2)


# Generate sample dataframe
data_dct = {
    'date': [dt.date(2030,4,1) + dt.timedelta(days=x) for x in range(3)],
    'country': ['UK'] * 3,
    'region': ['w'] * 3,
    'city': ['f'] * 3,
    'metric1': [3, 6, 7],
    'metric2': [30, 45, 70]
}
df_sample = pd.DataFrame(data_dct)

# Initialise match and run
m = Match(
    population=df,
    sample=df_sample,
    **kwargs
)
m.run()


def test_match_results():
    dct = {'bin': {('UK', 'w', 'f'): 0, ('UK', 'z', 'a'): 1, ('UK', 'z', 'b'): 1}}
    assert m.results.to_dict() == dct
