from absplit.param import ParamMixin
import pandas as pd
import numpy as np
from sklearn import preprocessing
from absplit import log

logger = log.get_logger(__name__)


class Data(ParamMixin):
    """Takes population data and manages all the transformations necessary to use it in the genetic algorithm.

    Processes:
        1) Run checks - Date checks and datatype checks
        2) Set indexes as all non-metric columns
        3) Scale data (if required)
            So that metrics are weighted more fairly against each other if they're of different scales
        4) Apply cutoff filter (if requireid)
        5) Unstack data by time periods (if required)
        6) Extract metadata needed for 3D transformation
            Final output shape needed of dimensions (number of metrics, size of population, number of time periods)

    Attributes:
        _df_stacked (pd.DataFrame): Data input from user, indexed on non-metric columns
        _pre_fit_scaler (object): Scaling object
        remainder_cols (list): Identifies any columns not specified by the user
        _df_unstacked (pd.DataFrame): Data input from user, unstacked by time periods
        _df_index (pd.DataFrame): Empty index from _df_unstacked
        num_metrics (int): Number of metrics specified
        time_periods (int): Number of time periods in data
        pop_size (int): Size of population that is getting split/matched
        splitting_values (np.array): IDs of each member of the population being split
        reshape_tup (tuple): Tuple of ints, representing 3D array dimensions

    Properties:
        stacked (pd.DataFrame): Data as entered, but indexed
        unstacked (pd.DataFrame): Data scaled (if required) and unstacked by time period (if required)
        index (pd.DataFrame): Empty index from unstacked dataframe
        matrix (np.array): 3D matrix of transformed data

    Methods:
        _check_columns_object(): Prints warning if any remainder columns are of type object
        _extract_metadata(): Extracts metadata of dataframe dimensions
        _set_indexes(): Sets indexes for data passed
        _unstack(): Unstacks data by time periods (if applicable)
        _scale(): Scales the metrics
        filter(index): Uses index to filter dataframe
        assign(solution): Maps solution (binary numpy array) to unstacked dataframe index
    """

    def __init__(self, df, scaler=None, scale=True, cutoff_date=None, missing_dates='median', **kwargs):
        """Initializes the class and sets the attributes

        Args:
            df (pd.DataFrame): Dataframe to process
            scaler (object, optional): Scaling function object. Default None
            scale (bool, optional): To apply scaling to data or not. Default True
            cutoff_date (str, optional): Max date to apply fitting to.
            missing_dates (str, optional): Config for what to do with missing dates
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self._df_stacked = df
        self._pre_fit_scaler = scaler
        self._scaler_flag = scale
        self._cutoff_date = pd.to_datetime(cutoff_date)
        self._missing_dates = missing_dates
        self.remainder_cols = [x for x in self._df_stacked.columns if x not in self.all_spec_columns]
        self._df_unstacked = None
        self._df_index = None
        self.num_metrics = None
        self.time_periods = None
        self.pop_size = None
        self.splitting_values = None
        self.reshape_tup = None

        logger.debug(f'stacked shape: {self._df_stacked.shape}, scale: {self._scaler_flag}, '
                     f'cutoff_date: {self._cutoff_date}')
        logger.debug(f'Remainder: {self.remainder_cols}, dim_spec: {self.dim_spec_columns}, metric:{self.metrics}')

        self._check_columns_object()
        self._check_dates()
        self._set_indexes()
        self._df_unstacked = self._df_stacked.copy()
        self._scale()
        self._cutoff_filter()
        self._unstack()

        # Extract unstacked index
        self._df_index = pd.DataFrame(index=self._df_unstacked.index)
        self._extract_metadata()

        logger.debug(f'unstacked shape: {self._df_unstacked.shape}, reshape dims: {self.reshape_tup}, '
                     f'population count: {self.pop_size}')

    def _check_dates(self):
        """Checks the cutoff date against the maximum and minimum date values of a DataFrame.

        This method verifies that the cutoff date provided is within the date range of the DataFrame,
        i.e., it is not before the earliest date or after the latest date in the data.

        Raises:
            ValueError: If the `cutoff_date` is not within the range of dates in the DataFrame.
        """
        # Transformations
        if self.date_col and self._cutoff_date:
            self._df_stacked[self.date_col] = pd.to_datetime(self._df_stacked[self.date_col])
            max_date = self._df_stacked[self.date_col].max()
            min_date = self._df_stacked[self.date_col].min()
            logger.debug(f'Max date: {max_date}')
            if (self._cutoff_date > max_date) | (self._cutoff_date < min_date):
                raise ValueError(f'Arg `cutoff_date` ({self._cutoff_date}) is outside of maximum ({max_date}) and '
                                 f'miniumum ({min_date}) values')

    def _check_columns_object(self):
        """Prints warning if any remainder columns (i.e. any columns that are in dataframe which are not passed
         in the kwargs, are not of type object

        Returns:
            None
        """
        for col in self.remainder_cols:
            dtype = self._df_stacked[col].dtype
            if not dtype == 'object':
                print(f'Column \'{col}\' will be treated as a variable to split on. Normally additional splitting'
                      f'columns are of type object, this is type {dtype}, this could cause problems in splitting. '
                      'Please check your kwargs are correct and remove any metrics you dont intend on splitting on.')

    def _set_indexes(self):
        """Sets indexes for the dataframe passed.

        If user passes dataframe with a column which is not specified by one of the input
        args (date_col, split_by etc) then it is considered a remainder col.

        This is indexed because it is assumed this is important for unique identification.
        For example if users wants to split all counties in the US, they will also need to pass
        the state name as some county names will be duplicates. Even though this column isn't
        specified as an argument, it is still relevant.
        """

        columns = self.remainder_cols + self.dim_spec_columns  # Specifies order of columns
        logger.debug(f'Indexes set: {columns}')
        self._df_stacked = self._df_stacked.set_index(columns).sort_index()

    def _scale(self):
        """Standardise all metrics so all metrics weighted as equally as possible

        Returns:
            pd.DataFrame
        """

        if self._pre_fit_scaler:
            scaler = self._pre_fit_scaler
            logger.debug(f'Prefit scaler')
        else:
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(self._df_stacked[self.metrics])

        self._df_unstacked = self._df_stacked.copy()
        if self._scaler_flag:
            self._df_unstacked[self.metrics] = scaler.transform(self._df_stacked[self.metrics])
            logger.debug(f'Scaled')
        else:
            logger.debug(f'Not scaled')

    def _cutoff_filter(self):
        """Applies a cutoff filter to the DataFrame based on a specified cutoff date.

        This method filters out the rows from the DataFrame (`_df_unstacked`) whose date is greater than
        the specified cutoff date (`_cutoff_date`). The filtering is done only if both `date_col` attribute
        and `_cutoff_date` are set. It assumes that the DataFrame's index includes a level with dates.
        """
        if self.date_col and self._cutoff_date:
            logger.debug(f'Unstack cutoff: {self._cutoff_date}')
            self._df_unstacked = self._df_unstacked[
                self._df_unstacked.index.get_level_values(self.date_col) <= pd.to_datetime(self._cutoff_date)
            ]

    def _unstack(self):
        """Unstack date column if date columns is passed, filters on any cutoff dates, runs checks

        Returns:
            pd.DataFrame
        """
        df_unstacked = self._df_unstacked
        logger.debug(f'Stacked shape: {df_unstacked.shape}')

        if self.date_col is None:
            self._df_unstacked = df_unstacked
            if self._df_unstacked.isna().sum().sum() > 0:
                raise ValueError('Dataframe contains NaNs, please remove or replace and run again')
        else:
            self._df_unstacked = df_unstacked.unstack(self.date_col)
            self._run_timeseries_checks()

        logger.debug(f'Unstacked shape: {self._df_unstacked.shape}')

    def _run_timeseries_checks(self):
        """Performs time series checks on the DataFrame (`_df_unstacked`), applying specific strategies to handle
        missing data.

        Depending on the method specified in `_missing_dates`, this method either drops dates or population with missing
        data,  fills missing values with zero, or fills missing values with the median of the DataFrame.

        If all dates or population are dropped due to missing data, the function raises an error, suggesting alternative
        strategies for handling missing data.

        Raises:
            ValueError: If the `_missing_dates` attribute is not set to 'drop_dates', 'drop_population', '0', or 'median'.
            ValueError: If all population or dates have been dropped due to missing data.
        """

        missing_values = self._df_unstacked.isna().sum().sum()
        if missing_values:
            print(f'{missing_values} NaN values from incomplete dates for population')
            if self._missing_dates == 'drop_dates':
                dates = list(str(x) for x in self._df_unstacked.isna().sum(axis=0).index.get_level_values(self.date_col).unique())
                print(f'Dropping {len(dates)} dates')
                self._df_unstacked = self._df_unstacked.dropna(axis=1)
            elif self._missing_dates == 'drop_population':
                population = list(str(x) for x in self._df_unstacked.isna().sum(axis=1).index.get_level_values(self.splitting).unique())
                print(f'Dropping {len(population)} dates')
                self._df_unstacked = self._df_unstacked.dropna(axis=0)
            elif self._missing_dates == '0' or self._missing_dates == 0:
                print(f'Filling NaNs 0')
                self._df_unstacked = self._df_unstacked.fillna(0)
            elif self._missing_dates == 'median':
                print(f'Filling NaNs with column median values')
                self._df_unstacked = self._df_unstacked.fillna(self._df_unstacked.median(numeric_only=True))
            else:
                raise ValueError('missing_dates arg must be \'drop_dates\', \'drop_population\', \'median\' or \'0\'')

            if self._df_unstacked.shape[0] == 0:
                raise ValueError('All population dropped because of missing data. Try using missing_dates=\'drop_date\', '
                                 'missing_dates=\'0\', missing_dates=\'median\'')
            if self._df_unstacked.shape[1] == 0:
                raise ValueError('All dates dropped because of missing data. Try using missing_dates=\'drop_population\', '
                                 'missing_dates=\'0\', missing_dates=\'median\'')

    def _extract_metadata(self):
        """Extracts metadata used for 3D transformation from unstacked dataframe, builds transformation tuple

        Returns:
            None
        """

        # Extract metadata
        self.num_metrics = len(self.metrics)
        self.time_periods = int(self._df_unstacked.shape[1] / self.num_metrics)
        self.pop_size = self._df_unstacked.shape[0]
        logger.debug(f'metrics: {self.num_metrics}, time_periods: {self.time_periods}, pop_size: {self.pop_size}')
        self.splitting_values = self._df_unstacked.index.get_level_values(self.splitting).to_numpy()
        self.reshape_tup = (self.pop_size, self.num_metrics, self.time_periods)

    def filter(self, index):
        """Takes empty dataframe index 'index' and inner joins on unstacked data to return a filtered version.

        Used when splitting out sample and population dataframes from concatenated dataframe when using
        Match class

        Args:
            index (pd.DataFame): Indexes to filter for

        Returns:
            pd.DataFrame
        """
        return self._df_unstacked.stack(self.date_col).join(index, how='inner')

    @property
    def unstacked(self):
        return self._df_unstacked

    @property
    def stacked(self):
        return self._df_stacked

    @property
    def index(self):
        return self._df_index

    @property
    def matrix(self):
        """Returns 3D matrix of shape (no_metrics, no_splitting, no_days)

        Returns:
            np.array
        """

        return self._df_unstacked.values.reshape(self.reshape_tup).transpose(1, 0, 2)

    def assign(self, solution):
        """Assigns genetic algorithm solution (binary numpy array) to dataframe index.

        Returned dataframe is the labelled (with index) solution to the splitting problem.

        Returns:
            pd.DataFrame
        """

        if solution is None:
            solution = np.array([None])
        df_assigned = self.index.copy()
        df_assigned.insert(loc=0, column='bin', value=solution.reshape(-1, 1))
        df_assigned['bin'] = df_assigned['bin'].astype(str)
        return df_assigned
