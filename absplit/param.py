from abc import ABC
import logging

logger = logging.getLogger(__name__)


class ParamMixin(ABC):
    """Mixin class to store and process column names

    Attributes:
        metrics (list): Metric column names
        date_col (str): Date column name, if applicable
        splitting (str): Column to split on
        all_spec_columns (list): All column names specified by user
        dim_spec_columns (list): All dimension columns (non-metrics) specified by user

    Methods:
        get_columns(): Get all columns specified by user in specific order
    """

    def __init__(self, metrics, splitting, date_col=None, **kwargs):
        """Initializes the class and sets the attributes

        Args:
            metrics (list): Metric column names
            splitting (str): Column to split on
            date_col (str): Date column name, if applicable
            **kwargs: Additional keyword arguments
        """
        self.metrics = metrics if isinstance(metrics, list) else [metrics]
        self.date_col = date_col
        self.splitting = splitting
        self.all_spec_columns = self.get_columns()
        self.dim_spec_columns = [x for x in self.all_spec_columns if x not in self.metrics]

    def get_columns(self):
        """Gets list of all column names specified by user, in order of:
            * splitting
            * date_col
            * metrics

        Returns:
            list
        """

        col_type_lst = [
            self.splitting,
            self.date_col,
            self.metrics
        ]
        all_columns = []
        for col in col_type_lst:
            if col is not None:
                if isinstance(col, list):
                    all_columns += col
                else:
                    all_columns.append(col)
        return all_columns
