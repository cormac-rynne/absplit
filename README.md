<img src="https://raw.githubusercontent.com/cormac-rynne/absplit/main/images/logo.jpeg" width="460" height="140">

![license](https://img.shields.io/badge/License-MIT-blue.svg)
![version](https://img.shields.io/badge/version-0.1.2-blue.svg)
![version](https://img.shields.io/badge/python-<=3.9.13-orange.svg)



ABSplit is a python package that uses a genetic algorithm to generate equal A/B test splits across time and metrics.

This covers the following use cases:
1. Splitting an entire population into 2 groups
2. Finding a matching set in the population for a given sample

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install ABSplit.

```bash
pip install absplit
```

## Tutorial
Please see [this colab](https://colab.research.google.com/drive/1gL7dxDJrtVoO5m1mSUWutdr7yas7sZwI?usp=sharing) for 
a range of examples on how to use ABSplit

## Usage example

```python
from absplit import ABSplit
import pandas as pd
import datetime
import numpy as np

# Synthetic data
data_dct = {
    'date': [datetime.date(2030,4,1) + datetime.timedelta(days=x) for x in range(3)]*5,
    'country': ['UK'] * 15,
    'region': [item for sublist in [[x]*6 for x in ['z', 'y']] for item in sublist] + ['x']*3,
    'city': [item for sublist in [[x]*3 for x in ['a', 'b', 'c', 'd', 'e']] for item in sublist],
    'metric1': np.arange(0, 15, 1),
    'metric2': np.arange(0, 150, 10)
}
df = pd.DataFrame(data_dct)

# Identify which columns are metrics, which is the the time period, and what to split on
kwargs = {
    'metrics': ['metric1', 'metric2'],
    'date_col': 'date',
    'splitting': 'city'
}

# Initialise
ab = ABSplit(
    df=df,
    **kwargs,
)

# Generate split
ab.run()

# Visualise generation fitness
ab.fitness()

# Visualise data
ab.visualise()

# Extract results
ab.results
```

## API Reference
### Absplit 
`ABSplit(df, ga_params={}, metric_weights={}, **kwargs)`

Splits population into 2 groups. Mutually exclusive, completely exhaustive

Arguments:
* `df` (pd.DataFrame): Dataframe to be split
* `metrics` (str, list): Name of, or list of names of, metric columns in DataFrame
* `splitting` (str): Name of column that represents individual in the population that is getting split
* `date_col` (str, optional): Name of column that represents time periods, if applicable.
* `ga_params` (dict, optional): Parameters for the genetic algorithm `pygad.GA` module parameters (default: {})
* `metric_weight` (dict, optional): Weights for each metric in the data. If you want the splitting to focus on one metrics more than the other, you can prioritise this here (default: {})


### Match 
`Match(population, sample, ga_params={}, metric_weights={}, **kwargs)`

Takes DataFrame `sample` and finds a comparable group in `population`.

Arguments:
* `population` (pd.DataFrame): Population to search  for comparable group. Must exclude sample data.
* `sample` (pd.DataFrame): Sample we are looking to find a match for.
* `metrics` (str, list): Name of, or list of names of, metric columns in DataFrame
* `splitting` (str): Name of column that represents individual in the population that is getting split
* `date_col` (str, optional): Name of column that represents time periods, if applicable.
* `ga_params` (dict, optional): Parameters for the genetic algorithm `pygad.GA` module parameters (default: {})
* `metric_weight` (dict, optional): Weights for each metric in the data. If you want the splitting to focus on one metrics more than the other, you can prioritise this here (default: {})

## Contributing

I welcome contributions to absplit! For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)