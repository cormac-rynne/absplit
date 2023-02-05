import datetime
import numpy as np
import pandas as pd
from absplit.data import Data

# Build data
data_dct = {
    'date': [datetime.date(2030,4,1) + datetime.timedelta(days=x) for x in range(3)]*5,
    'country': ['UK'] * 15,
    'region': [item for sublist in [[x]*6 for x in ['z', 'y']] for item in sublist] + ['x']*3,
    'city': [item for sublist in [[x]*3 for x in ['a', 'b', 'c', 'd', 'e']] for item in sublist],
    'metric1': np.arange(0, 15, 1),
    'metric2': np.arange(0, 150, 10)
}
df = pd.DataFrame(data_dct)

# Create data object
kwargs = {
    'metrics': ['metric1', 'metric2'],
    'date_col': 'date',
    'splitting': 'city'
}
data1 = Data(df=df, **kwargs)


def test_remainder_cols():
    assert data1.remainder_cols == ['country', 'region']


def test_stacked():
    dct = {
        'metric1': {('UK', 'x', 'e', datetime.date(2030, 4, 1)): 12,
        ('UK', 'x', 'e', datetime.date(2030, 4, 2)): 13,
        ('UK', 'x', 'e', datetime.date(2030, 4, 3)): 14,
        ('UK', 'y', 'c', datetime.date(2030, 4, 1)): 6,
        ('UK', 'y', 'c', datetime.date(2030, 4, 2)): 7,
        ('UK', 'y', 'c', datetime.date(2030, 4, 3)): 8,
        ('UK', 'y', 'd', datetime.date(2030, 4, 1)): 9,
        ('UK', 'y', 'd', datetime.date(2030, 4, 2)): 10,
        ('UK', 'y', 'd', datetime.date(2030, 4, 3)): 11,
        ('UK', 'z', 'a', datetime.date(2030, 4, 1)): 0,
        ('UK', 'z', 'a', datetime.date(2030, 4, 2)): 1,
        ('UK', 'z', 'a', datetime.date(2030, 4, 3)): 2,
        ('UK', 'z', 'b', datetime.date(2030, 4, 1)): 3,
        ('UK', 'z', 'b', datetime.date(2030, 4, 2)): 4,
        ('UK', 'z', 'b', datetime.date(2030, 4, 3)): 5},
        'metric2': {('UK', 'x', 'e', datetime.date(2030, 4, 1)): 120,
        ('UK', 'x', 'e', datetime.date(2030, 4, 2)): 130,
        ('UK', 'x', 'e', datetime.date(2030, 4, 3)): 140,
        ('UK', 'y', 'c', datetime.date(2030, 4, 1)): 60,
        ('UK', 'y', 'c', datetime.date(2030, 4, 2)): 70,
        ('UK', 'y', 'c', datetime.date(2030, 4, 3)): 80,
        ('UK', 'y', 'd', datetime.date(2030, 4, 1)): 90,
        ('UK', 'y', 'd', datetime.date(2030, 4, 2)): 100,
        ('UK', 'y', 'd', datetime.date(2030, 4, 3)): 110,
        ('UK', 'z', 'a', datetime.date(2030, 4, 1)): 0,
        ('UK', 'z', 'a', datetime.date(2030, 4, 2)): 10,
        ('UK', 'z', 'a', datetime.date(2030, 4, 3)): 20,
        ('UK', 'z', 'b', datetime.date(2030, 4, 1)): 30,
        ('UK', 'z', 'b', datetime.date(2030, 4, 2)): 40,
        ('UK', 'z', 'b', datetime.date(2030, 4, 3)): 50}}
    assert data1.stacked.to_dict() == dct


def test_unstacked():
    dct = {('metric1', datetime.date(2030, 4, 1)): {('UK', 'x', 'e'): 0.8571428571428571,
        ('UK', 'y', 'c'): 0.42857142857142855,
        ('UK', 'y', 'd'): 0.6428571428571428,
        ('UK', 'z', 'a'): 0.0,
        ('UK', 'z', 'b'): 0.21428571428571427},
        ('metric1', datetime.date(2030, 4, 2)): {('UK', 'x', 'e'): 0.9285714285714285,
        ('UK', 'y', 'c'): 0.5,
        ('UK', 'y', 'd'): 0.7142857142857142,
        ('UK', 'z', 'a'): 0.07142857142857142,
        ('UK', 'z', 'b'): 0.2857142857142857},
        ('metric1', datetime.date(2030, 4, 3)): {('UK', 'x', 'e'): 1.0,
        ('UK', 'y', 'c'): 0.5714285714285714,
        ('UK', 'y', 'd'): 0.7857142857142857,
        ('UK', 'z', 'a'): 0.14285714285714285,
        ('UK', 'z', 'b'): 0.3571428571428571},
        ('metric2', datetime.date(2030, 4, 1)): {('UK', 'x', 'e'): 0.8571428571428571,
        ('UK', 'y', 'c'): 0.42857142857142855,
        ('UK', 'y', 'd'): 0.6428571428571428,
        ('UK', 'z', 'a'): 0.0,
        ('UK', 'z', 'b'): 0.21428571428571427},
        ('metric2', datetime.date(2030, 4, 2)): {('UK', 'x', 'e'): 0.9285714285714285,
        ('UK', 'y', 'c'): 0.5,
        ('UK', 'y', 'd'): 0.7142857142857143,
        ('UK', 'z', 'a'): 0.07142857142857142,
        ('UK', 'z', 'b'): 0.2857142857142857},
        ('metric2', datetime.date(2030, 4, 3)): {('UK', 'x', 'e'): 1.0,
        ('UK', 'y', 'c'): 0.5714285714285714,
        ('UK', 'y', 'd'): 0.7857142857142857,
        ('UK', 'z', 'a'): 0.14285714285714285,
        ('UK', 'z', 'b'): 0.35714285714285715}}
    assert data1.unstacked.to_dict() == dct


def test_filter():
    df_index = pd.DataFrame({'country': ['UK'], 'region': ['x'], 'city': ['e']}).set_index(['country', 'region', 'city'])
    dct = {'metric1': {('UK', 'x', 'e', datetime.date(2030, 4, 1)): 0.8571428571428571,
        ('UK', 'x', 'e', datetime.date(2030, 4, 2)): 0.9285714285714285,
        ('UK', 'x', 'e', datetime.date(2030, 4, 3)): 1.0},
        'metric2': {('UK', 'x', 'e', datetime.date(2030, 4, 1)): 0.8571428571428571,
        ('UK', 'x', 'e', datetime.date(2030, 4, 2)): 0.9285714285714285,
        ('UK', 'x', 'e', datetime.date(2030, 4, 3)): 1.0}}
    assert data1.filter(df_index).to_dict() == dct


def test_matrix():
    matrix = np.array([[[0.85714286, 0.92857143, 1.        ],
        [0.42857143, 0.5       , 0.57142857],
        [0.64285714, 0.71428571, 0.78571429],
        [0.        , 0.07142857, 0.14285714],
        [0.21428571, 0.28571429, 0.35714286]],

       [[0.85714286, 0.92857143, 1.        ],
        [0.42857143, 0.5       , 0.57142857],
        [0.64285714, 0.71428571, 0.78571429],
        [0.        , 0.07142857, 0.14285714],
        [0.21428571, 0.28571429, 0.35714286]]])
    assert (matrix == np.round(data1.matrix, 8)).all()


def test_assign():
    dct = {'bin': {('UK', 'x', 'e'): 0,
        ('UK', 'y', 'c'): 1,
        ('UK', 'y', 'd'): 0,
        ('UK', 'z', 'a'): 1,
        ('UK', 'z', 'b'): 0}}
    assert data1.assign(np.array([0,1,0,1,0])).to_dict()


def test_dim_spec():
    assert data1.dim_spec_columns == ['city', 'date']


def test_all_spec():
    assert data1.all_spec_columns == ['city', 'date', 'metric1', 'metric2']