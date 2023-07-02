import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

def generate_time_series(days=60, pop=400, metrics=2, drift=True):
    time_periods = np.arange(0, days, 1)

    # Generate random numbers
    trend = np.random.uniform(-0.3, 0.3, (pop, metrics))  # For each pop-metric
    const = np.ones((pop, metrics))
    const[np.where(trend < 0)] = (-trend * days * np.random.uniform(1, 1.5))[np.where(trend < 0)]
    const[np.where(trend > 0)] = (trend * days * np.random.uniform(0.2, 2))[np.where(trend > 0)]
    # Scale, for each metric
    scale = np.random.uniform(1, 5, metrics).reshape(1, 1, -1)
    # Vertical distance applied, pop-metric
    mean = np.random.uniform(0, 5, (pop, 1, metrics))
    # Start of sine wave period in sin wave, pop-metric
    start = np.random.uniform(0, days, (pop, 1, metrics))
    # How many oscilations during total time period, pop-metric
    freq_rand = np.random.uniform(0.01, 0.3, (pop, 1, metrics))

    # Base trend
    base = ((time_periods.reshape(-1, 1) @ trend[:, np.newaxis]) + const[:, np.newaxis, :]) * scale

    # Apply different starting sin wave periods to time periods
    numerator = (np.tile(time_periods.reshape(1, days, 1), (pop, 1, metrics)) + start)

    # Apply frequency modifiers
    z1 = numerator / (days * freq_rand)
    z2 = numerator / (days * freq_rand / 3)

    # Generate oscilations
    oscil = mean + scale * np.sin(z1) * 3
    oscil2 = mean + (scale * np.sin(z2))

    # Add noise
    volatility = np.random.uniform(-0.8, 0.8, (pop, days, metrics))
    noise = np.random.uniform(-scale * volatility, scale * volatility)
    random_walk = np.random.uniform(-1, 1, (pop, days, metrics)).cumsum(axis=1) * const[:, np.newaxis, :] \
                  * 0.1 * drift

    # Create dataframe
    data = (base + oscil + oscil2 + noise + random_walk).reshape(-1, metrics)
    columns = [f'metric{i + 1}' for i in range(metrics)]
    df = pd.DataFrame(data, columns=columns)
    df['date'] = np.tile(pd.date_range(start='2023-01-01', periods=days), pop).reshape(-1, 1)
    df['id'] = np.repeat(np.arange(0, pop), days)
    df = df[['date', 'id'] + columns]

    return df


def covid(test=False, drop_fips=True):
    url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
    if test:
        url = '../data/us-counties.csv'
        print(f'TEST mode, pulling file')
    else:
        print(f'Pulling data from {url}')
    df = pd.read_csv(url)
    print(f'Cleaning and resampling data..')

    # Remove Puerto Rico because of NaNs
    df = df.query("state != 'Puerto Rico'")

    # Clean
    df.loc[:, 'date'] = pd.to_datetime(df['date'])
    df.loc[:, 'deaths'] = df['deaths'].astype(int)
    if drop_fips:
        df = df.drop('fips', axis=1)
    else:
        df = df[~df['fips'].isna()].reset_index(drop=True)
        df['fips'] = df['fips'].astype('int64').astype(str)
    cols = ['state', 'county', 'date']
    cols = cols + ['fips'] if not drop_fips else cols
    df = df.set_index(cols).sort_index()
    df = df.groupby(level=[0, 1]).diff().fillna(0).clip(lower=0)
    resampler = lambda x: x.set_index('date').resample('w').sum()
    df = df.reset_index(level=2).groupby([x for x in cols if x != 'date']).apply(resampler).reset_index()
    return df

def retail(test=False):
    url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
    if test:
        url = '../data/online-retail.csv'
        print(f'TEST mode, pulling file')
    else:
        print(f'Pulling data from {url}')
    df = pd.read_csv(url)
    print(f'Cleaning and resampling data..')

    df.columns = [x.lower().replace(' ', '_') for x in df.columns]
    df.loc[:, 'invoicedate'] = pd.to_datetime(df['invoicedate'])
    df['date'] = df['invoicedate'].dt.date
    df['week'] = df['invoicedate'].dt.to_period('W').dt.start_time
    df['cost'] = df['quantity'] * df['price']
    df = df[~df['customer_id'].isna()].reset_index(drop=True)
    df['customer_id'] = df['customer_id'].astype('int64').astype(str)

    agg_dct = {
        'cost': 'sum',
        'quantity': 'sum',
        'stockcode': pd.Series.nunique
    }
    df2 = df.groupby(['week', 'customer_id']).agg(agg_dct).reset_index().drop('stockcode', axis=1)
    return df2

def display(df):
    for i in range(5):
        df_ = df[df['id'] == i]
        plt.plot(df_['date'].values, df_['metric1'].values, label=f'id: {i}')

    plt.xticks(rotation=45)
    plt.title('metric1')
    plt.legend()
    plt.show()

    metrics = len([x for x in df.columns if x not in ['date', 'id']])

    for i in range(metrics):
        df_ = df[df['id'] == 0]
        col = f'metric{i + 1}'
        plt.plot(df_['date'].values, df_[col], label=col)

    plt.xticks(rotation=45)
    plt.title(f'ID: 0')
    plt.legend()
    plt.show()
