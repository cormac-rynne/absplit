from absplit.param import ParamMixin
import pandas as pd
import numpy as np
from sklearn import preprocessing
import logging

logger = logging.getLogger(__name__)


class Data(ParamMixin):
    """Takes population data and manages all the transformations necessary to use it in the genetic algorithm.

    Processes:
        1) Set indexes as all non-metric columns
        2) Scale data (if required)
            So that metrics are weighted more fairly against each other if they're of different scales
        3) Unstack data by time periods (if required)
        4) Extract metadata needed for 3D transformation
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

    def __init__(self, df, scaler=None, scale=True, **kwargs):
        """Initializes the class and sets the attributes

        Args:
            df (pd.DataFrame): Dataframe to process
            scaler (object, optional): Scaling function object. Default None
            scale (bool, optional): To apply scaling to data or not. Default True
            **kwargs: Additional keyword arguments
        """

        super().__init__(**kwargs)
        self._df_stacked = df
        self._pre_fit_scaler = scaler
        self._scaler_flag = scale
        self.remainder_cols = [x for x in self._df_stacked.columns if x not in self.all_spec_columns]
        self._df_unstacked = None
        self._df_index = None
        self.num_metrics = None
        self.time_periods = None
        self.pop_size = None
        self.splitting_values = None
        self.reshape_tup = None

        logger.debug(f'Remainder: {self.remainder_cols}, dim_spec: {self.dim_spec_columns}, metric:{self.metrics}')

        self._check_columns_object()

        # Transformations
        self._df_stacked = self._set_indexes()
        self._df_unstacked = self._df_stacked.copy()
        self._df_unstacked = self._scale()
        self._df_unstacked = self._unstack()

        # Extract unstacked index
        self._df_index = pd.DataFrame(index=self._df_unstacked.index)

        self._extract_metadata()

        logger.debug(f'shape: {self._df_unstacked.shape}, reshape dims: {self.reshape_tup}, num_genes: {self.pop_size}')

    def _check_columns_object(self):
        """Prints warning if any remainder columns are not of type object

        Returns:
            None
        """
        for col in self.remainder_cols:
            dtype = self._df_stacked[col].dtype
            if not dtype == 'object':
                print(f'Column \'{col}\' will be treated as a variable to split on. Normally additional splitting'
                      f'columns are of type object, this is type {dtype}, this could cause problems in splitting. '
                      'Please check your kwargs are correct and remove any metrics you dont intend on splitting on.')

    def _extract_metadata(self):
        """Extracts metadata used for 3D transformation from unstacked dataframe, builds transformation tuple

        Returns:
            None
        """

        # Extract metadata
        self.num_metrics = len(self.metrics)
        self.time_periods = int(self._df_unstacked.shape[1] / self.num_metrics)
        self.pop_size = self._df_unstacked.shape[0]
        self.splitting_values = self._df_unstacked.index.get_level_values(self.splitting).to_numpy()
        self.reshape_tup = (self.pop_size, self.num_metrics, self.time_periods)

    def _set_indexes(self):
        """Sets indexes for the dataframe passed.

        If user passes dataframe with a column which is not specified by one of the input
        args (date_col, split_by etc) then it is considered a remainder col.

        This is indexed because it is assumed this is important for unique identification.
        For example if users wants to split all counties in the US, they will also need to pass
        the state name as some county names will be duplicates. Even though this column isn't
        specified as an argument, it is still relevant.

        Return:
            pd.DataFrame
        """

        columns = self.remainder_cols + self.dim_spec_columns  # Specifies order of columns
        logger.debug(f'Indexes set: {columns}')
        return self._df_stacked.set_index(columns).sort_index()

    def _unstack(self):
        """Unstack date column if date columns is passed

        Returns:
            pd.DataFrame
        """

        return self._df_unstacked.unstack(self.date_col, fill_value=0) if self.date_col else self._df_stacked

    def _scale(self):
        """Standardise all metrics so all metrics weighted as equally as possible

        Returns:
            pd.DataFrame
        """

        if self._pre_fit_scaler:
            scaler = self._pre_fit_scaler
        else:
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(self._df_stacked[self.metrics])

        df = self._df_stacked.copy()
        if self._scaler_flag:
            df[self.metrics] = scaler.transform(self._df_stacked[self.metrics])
        return df

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
        return df_assigned
