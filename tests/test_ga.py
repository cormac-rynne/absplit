import pytest

from absplit import ABSplit, Match, tutorials
import pandas as pd
import datetime as dt
import numpy as np

@pytest.fixture
def df():
    # Build test data
    data_dct = {
        'date': [dt.date(2030, 4, 1) + dt.timedelta(days=x) for x in range(3)] * 5,
        'country': ['UK'] * 15,
        'region': [item for sublist in [[x]*6 for x in ['z', 'y']] for item in sublist] + ['x']*3,
        'city': [item for sublist in [[x]*3 for x in ['a', 'b', 'c', 'd', 'e']] for item in sublist],
        'metric1': np.arange(0, 15, 1),
        'metric2': np.arange(0, 150, 10)
    }
    return pd.DataFrame(data_dct)

@pytest.fixture
def kwargs():
    return  {
        'metrics': ['metric1', 'metric2'],
        'date_col': 'date',
        'splitting': 'city'
    }

@pytest.fixture
def ab(df, kwargs):
    # Initialise and run
    ab = ABSplit(
        df=df,
        size_penalty=0,
        **kwargs,
    )
    ab.run()
    return ab

@pytest.fixture
def results():
    data_dct1 = {
        'country': {0: 'UK', 1: 'UK', 2: 'UK', 3: 'UK', 4: 'UK'},
        'region': {0: 'x', 1: 'y', 2: 'y', 3: 'z', 4: 'z'},
        'city': {0: 'e', 1: 'c', 2: 'd', 3: 'a', 4: 'b'},
        'bin': {0: '1', 1: '0', 2: '0', 3: '1', 4: '1'}
    }
    data_dct2 = {
        'country': {0: 'UK', 1: 'UK', 2: 'UK', 3: 'UK', 4: 'UK'},
        'region': {0: 'x', 1: 'y', 2: 'y', 3: 'z', 4: 'z'},
        'city': {0: 'e', 1: 'c', 2: 'd', 3: 'a', 4: 'b'},
        'bin': {0: '0', 1: '1', 2: '1', 3: '0', 4: '0'}
    }
    df_data1 = pd.DataFrame(data_dct1)
    df_data2 = pd.DataFrame(data_dct2)
    return df_data1, df_data2

def test_ab_results(ab, results):
    """GA output ab.results"""
    df_data1, df_data2 = results
    r1 = len(ab.results.reset_index().compare(df_data1)) == 0
    r2 = len(ab.results.reset_index().compare(df_data2)) == 0
    assert True in (r1, r2)


@pytest.fixture
def sample():
    # Generate sample dataframe
    data_dct = {
        'date': [dt.date(2030,4,1) + dt.timedelta(days=x) for x in range(3)],
        'country': ['UK'] * 3,
        'region': ['w'] * 3,
        'city': ['f'] * 3,
        'metric1': [3, 6, 7],
        'metric2': [30, 45, 70]
    }
    return pd.DataFrame(data_dct)

@pytest.fixture
def m(df, sample, kwargs):
    # Initialise match and run
    m = Match(
        population=df,
        sample=sample,
        **kwargs
    )
    m.run()
    return m


def test_match_results(m):
    dct = {'bin': {('UK', 'w', 'f'): '0', ('UK', 'z', 'a'): '1', ('UK', 'z', 'b'): '1'}}
    assert m.results.to_dict() == dct

@pytest.fixture
def df2():
    df = tutorials.covid(test=True)
    return df[df['state'] == 'Alabama']

def test_absplit_static(df2):
    df2 = df2[df2['date'] == '2022-04-17'].reset_index(drop=True).drop('date', axis=1)
    kwargs = {
        'metrics': ['cases', 'deaths'],
        'splitting': 'county',
    }
    ab = ABSplit(
        df=df2,
        ga_params={'num_generations': 10, 'sol_per_pop': 10},
        **kwargs
    )
    ab.run()


def test_absplit_static_onemetric(df2):
    df2 = df2[df2['date'] == '2022-04-17'].reset_index(drop=True).drop(['deaths', 'date'], axis=1)
    kwargs = {
        'metrics': ['cases'],
        'splitting': 'county',
    }
    ab = ABSplit(
        df=df2,
        ga_params={'num_generations': 10, 'sol_per_pop': 10},
        **kwargs
    )
    ab.run()


def test_absplit_onemetric(df2):
    df2 = df2.drop('deaths', axis=1)
    kwargs = {
        'metrics': ['cases'],
        'splitting': 'county',
        'date_col': 'date'
    }
    ab = ABSplit(
        df=df2,
        ga_params={'num_generations': 10, 'sol_per_pop': 10},
        **kwargs
    )
    ab.run()

def test_absplit_splits(df2):
    kwargs = {
        'metrics': ['cases', 'deaths'],
        'splitting': 'county',
        'date_col': 'date'
    }
    ab = ABSplit(
        df=df2,
        ga_params={'num_generations': 10, 'sol_per_pop': 10},
        splits=[.25, .25, .5],
        **kwargs
    )
    ab.run()


def test_absplit_cutoff(df2):
    kwargs = {
        'metrics': ['cases', 'deaths'],
        'splitting': 'county',
        'date_col': 'date'
    }
    ab = ABSplit(
        df=df2,
        ga_params={'num_generations': 10, 'sol_per_pop': 10},
        cutoff_date='2021-10-01',
        splits=[.25, .25, .5],
        **kwargs
    )
    ab.run()

def test_absplit_cutoff_fail(df2):
    kwargs = {
        'metrics': ['cases', 'deaths'],
        'splitting': 'county',
        'date_col': 'date'
    }
    with pytest.raises(ValueError):
        ab = ABSplit(
            df=df2,
            ga_params={'num_generations': 10, 'sol_per_pop': 10},
            cutoff_date='2030-10-01',
            splits=[.25, .25, .5],
            **kwargs
        )

def test_absplit_penalties(df2):
    kwargs = {
        'metrics': ['cases', 'deaths'],
        'splitting': 'county',
        'date_col': 'date'
    }
    ab = ABSplit(
        df=df2,
        ga_params={'num_generations': 10, 'sol_per_pop': 10},
        metric_weights={'cases': 20},
        sum_penalty=1,
        size_penalty=1,
        splits=[.25, .25, .5],
        **kwargs
    )
    ab.run()