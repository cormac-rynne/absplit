<a name="readme-top"></a>

<div align="center">
<img src="https://raw.githubusercontent.com/cormac-rynne/absplit/main/images/logo.jpeg" width="460" height="140">
<h3><strong>ABSplit</strong></h3>
Split your data into matching A/B/n groups

![license](https://img.shields.io/badge/License-MIT-blue.svg)
![version](https://img.shields.io/badge/version-1.4.4-blue.svg)
![version](https://img.shields.io/badge/python-3-orange.svg)
[![PyPI Downloads](https://img.shields.io/pypi/dm/absplit.svg)](https://pypi.org/project/absplit/)

</div>

<details open>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#calculation">Calculation</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#tutorial">Tutorials</a></li>
    <ul>
        <li><a href="#do-it-yourself">Do it yourself</a></li>
    </ul>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#api-reference">API Reference</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About the project
ABSplit is a python package that uses a genetic algorithm to generate as equal as possible A/B, A/B/C, or A/B/n test splits.

The project aims to provide a convenient and efficient way for splitting population data into distinct 
groups (ABSplit), as well as and finding matching samples that closely resemble a given original sample (Match).


Whether you have static population data or time series data, this Python package simplifies the process and allows you to 
analyze and manipulate your population data.

This covers the following use cases:
1. **ABSplit class**: Splitting an entire population into n groups by given proportions
2. **Match class**: Finding a matching group in a population for a given sample

### Calculation

ABSplit standardises the population data (so each metric is weighted as equally as possible), then pivots it into a 
three-dimensional array, by metrics, individuals, and dates. 

The selection from the genetic algorithm, along with its inverse, is applied across this array with broadcasting to 
compute the dot products between the selection and the population data.

As a result, aggregated metrics for each group are calculated. The Mean Squared Error is calculated 
for each metric within the groups and then summed for each metric. The objective of the cost function is to minimize the 
overall MSE between these two groups, ensuring the metrics of both groups track each other as similarly across time
as possible.

<div align="center">
  <img src="https://raw.githubusercontent.com/cormac-rynne/absplit/main/images/calculation_diagram.png" width="80%">
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install ABSplit and it's prerequisites.

ABSplit requires `pygad==3.0.1`

### Installation

```bash
pip install absplit
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Tutorials
Please see [this colab](https://colab.research.google.com/drive/1gL7dxDJrtVoO5m1mSUWutdr7yas7sZwI?usp=sharing) for 
a range of examples on how to use ABSplit and Match

### Do it yourself
See [this colab](https://colab.research.google.com/drive/1SlCNnOtN4WCDTSJHsFrZtI7gKcXEl8-C?usp=sharing) to learn how 
ABSplit works under the hood, and how to build your own group splitting tool using 
[PyGAD](https://pypi.org/project/pygad/),


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

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

# Identify which columns are metrics, which is the time period, and what to split on
kwargs = {
    'metrics': ['metric1', 'metric2'],
    'date_col': 'date',
    'splitting': 'city'
}

# Initialise
ab = ABSplit(
    df=df,
    split=[.5, .5],  # Split into 2 groups of equal size
    **kwargs,
)

# Generate split
ab.run()

# Visualise generation fitness
ab.fitness()

# Visualise data
ab.visualise()

# Extract bin splits
df = ab.results

# Extract data aggregated by bins
df_agg = ab.aggregations

# Extract summary statistics
df_dist = ab.distributions    # Population counts between groups
df_rmse = ab.rmse             # RMSE between groups for each metric
df_mape = ab.mape             # MAPE between groups for each metric
df_totals = ab.totals         # Total sum of each metric for each group

```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## API Reference
### Absplit 
`ABSplit(df, metrics, splitting, date_col=None, ga_params={}, metric_weights={}, splits=[0.5, 0.5], size_penalty=0)`

Splits population into n groups. Mutually exclusive, completely exhaustive

Arguments:
* `df` (pd.DataFrame): Dataframe of population to be split
* `metrics` (str, list): Name of, or list of names of, metric columns in DataFrame to be considered in split
* `splitting` (str): Name of column that represents individuals in the population that is getting split. For example, if 
you wanted to split a dataframe of US counties, this would be the county name column
* `date_col` (str, optional): Name of column that represents time periods, if applicable. If left empty, it will
perform a static split, i.e. not across timeseries, (default `None`)
* `ga_params` (dict, optional): Parameters for the genetic algorithm `pygad.GA` module parameters, see 
[here](https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class) for arguments you can pass
(default: `{}`)
* `splits` (list, optional): How many groups to split into, and relative size of the groups (default: `[0.5, 0.5]`,
2 groups of equal size)
* `size_penalty` (float, optional): Penalty weighting for differences in the population count between groups 
(default: `0`)
* `sum_penalty` (float, optional): Penalty weighting for the sum of metrics over time. If this is greater than zero,
it will add a penalty to the cost function that will try and make the sum of each metric the same for each group 
(default: `0`)
* `cutoff_date` (str, optional): Cutoff date between fitting and validation data. For example, if you have data between 
2023-01-01 and 2023-03-01, and the cutoff date is 2023-02-01, the algorithm will only perform the fit on data between 
2023-01-01 and 2023-02-01. If `None`, it will fit on all available data. If cutoff date is provided, RMSE scores
  (gotten by using the `ab.rmse` attribute) will only be for validation period (i.e., from 2023-02-01 to end of 
timeseries)
* `missing_dates` (str, optional): How to deal with missing dates in time series data, options: `['drop_dates',
'drop_population', '0', 'median']` (default: `median`)
* `metric_weights` (dict, optional): Weights for each metric in the data. If you want the splitting to focus on 
one metrics more than the other, you can prioritise this here (default: `{}`)


### Match 
`Match(population, sample, metrics, splitting, date_col=None, ga_params={}, metric_weights={})`

Takes DataFrame `sample` and finds a comparable group in `population`.

Arguments:
* `population` (pd.DataFrame): Population to search  for comparable group (**Must exclude sample data**)
* `sample` (pd.DataFrame): Sample we are looking to find a match for.
* `metrics` (str, list): Name of, or list of names of, metric columns in DataFrame
* `splitting` (str): Name of column that represents individuals in the population that is getting split
* `date_col` (str, optional): Name of column that represents time periods, if applicable. If left empty, it will
perform a static split, i.e. not across timeseries, (default `None`)
* `ga_params` (dict, optional): Parameters for the genetic algorithm `pygad.GA` module parameters, see 
[here](https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class) for arguments you can pass
(default: `{}`)
* `splits` (list, optional): How many groups to split into, and relative size of the groups (default: `[0.5, 0.5]`,
2 groups of equal size)
* `metric_weights` (dict, optional): Weights for each metric in the data. If you want the splitting to focus on 
one metrics more than the other, you can prioritise this here (default: `{}`)
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

I welcome contributions to ABSplit! For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

[MIT](https://choosealicense.com/licenses/mit/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
