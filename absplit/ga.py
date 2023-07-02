from absplit.param import ParamMixin
from absplit.data import Data
from abc import ABC, abstractmethod
import pandas as pd
import pygad
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from itertools import combinations


logger = logging.getLogger(__name__)


class GAInstance:
    """Manages initialisation and running of pygad genetic algorithm instance

    Attributes:
        num_genes (int): Number of members in the population that is being searched
        ga (pygad.GA): PyGAD GA module
        solution (np.array): The best solution found in genetic algorith search
        fitness (np.float): The fitness score of the best solution

    Methods:
        run(): Runs the genetic algorithm search
    """

    def __init__(self, num_genes, ga_params):
        """Class init

        Args:
            num_genes (int): Number of solutions (genes) in chromosome
            ga_params (GAParams): GA parameter object
        """

        self.num_genes = num_genes
        self.ga = pygad.GA(
            num_genes=self.num_genes,
            **ga_params.params
        )
        self.solution = None
        self.fitness = None

    def run(self):
        """Runs genetic algorithm and returns best solution

        Sets:
            solution (np.array): Array of 0/1 splits
            fitness (float): Fitness value
        """

        if self.num_genes > 1:
            self.ga.run()
            self.solution, self.fitness, _ = self.ga.best_solution()
        else:
            logger.warning('Population size of < 2, skipping')
            self.solution = None
            self.fitness = None


class GAParams:
    """Manages PyGAD GA module parameters and any parameter updates

    Attributes:
        _default_ga_params (dict): Default parameters for genetic algorithm
        params (dict): PyGAD GA module parameters to be implemented

    """

    def __init__(self, splits, **kwargs):
        """Initializes the class and sets the attributes

        Args:
            **kwargs: Genetic algorithm parameters
        """
        global fitness_func_absplit
        self._splits = splits
        self._splits_num = len(self._splits)

        # Default parameters
        self._default_ga_params = dict(
            # -- General Genetic Algorithm params --
            num_generations=200,
            sol_per_pop=100,
            num_parents_mating=5,
            mutation_type='scramble',
            mutation_probability=0.1,
            fitness_func=fitness_func_absplit,
            keep_elitism=1,

            # -- Binary Genetic Algorithm specific parameters --
            init_range_low=0,
            init_range_high=self._splits_num,
            random_mutation_min_val=0,
            random_mutation_max_val=self._splits_num,
            gene_type=int,
        )

        self._dont_touch = [
            'init_range_low',
            'init_range_high',
            'random_mutation_min_val',
            'random_mutation_max_val',
            'gene_type'
        ]

        self._silence = [
            'initial_population'
        ]

        # Copy default, update
        self.params = dict(self._default_ga_params)
        self.update(**kwargs)

    def update(self, **kwargs):
        """Update GA params if any passed

        Returns:
            None
        """

        # Update if args passed
        if kwargs:
            for key, value in kwargs.items():
                if key in self._dont_touch:
                    print(f'Parameter \'{key}\' is essential to running the genetic algorithm as a '
                          f'discrete genetic algorith, and so you\'re not allowed to modify this')
                    continue
                if callable(value) or key in self._silence:
                    print(f'[Updating] {key}')
                else:
                    print(f'[Updating] {key} to {value}')
                self.params[key] = value


class SplitBase(ParamMixin, ABC):
    """Base class for genetic algorithm orchestration.

    Manages applying different weights to costs, using multiple runs, extracting and visualising results.

    Attributes:
        _runs (int): Number of runs of the genetic algorithm to try
        _ga_params (object): GAParams object
        _best_score (float): Stores best score when using multiple runs
        _metric_weights (dict): Dictionary of {col_name: value} format for applying differing weights to columns
        _df (pd.DataFrame): Input data
        _population (object): Data object
        _df_result (pd.DataFrame): Proposed solution dataframe
        _ga (object): GAInstance object
        _df_vis (pd.DataFrame): DataFrame with solutions used for visualisation
        _best_ga (object): GAInstance object of the best solution
        _solution (np.array): Best solution from search

    Properties:
        results (pd.DataFrame): Accessor for _df_result

    Methods:
        _cost_weighting(): Modifies cost weighting array
        run(): Runs the genetic algorithm
        fitness(): Plot generation fitness graph
        visualise(): Plots metrics using results from genetic algorithm output
    """

    def __init__(self, ga_params={}, metric_weights={}, runs=1, splits=[0.5, 0.5], size_penalty=0, sum_penalty=0,
                 cutoff_date=None, **kwargs):
        """Initializes the class and sets the attributes

        Args:
            ga_params (dict, optional): Genetic algorithm parameters to add/modify
            metric_weights (dict, optional): Cost weightings to apply to metrics
            runs (int, optional): How many runs to try
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        global sum_penalty_global
        global size_penalty_global
        global group_ids_global
        size_penalty_global = size_penalty
        group_ids_global = np.arange(0, len(splits))
        sum_penalty_global = sum_penalty

        if ga_params:
            assert isinstance(ga_params, dict), 'ga_params must be a dictionary'
        self._runs = runs
        self._splits = splits
        self._splits_num = len(self._splits)
        self._ga_params = GAParams(splits=splits, **ga_params)
        self._best_score = -1
        self._metric_weights = metric_weights
        self._cutoff_date = cutoff_date
        self._df = None
        self._population = None
        self._df_result = None  # Final results dataframe
        self._df_dist = None  # Group count distribution
        self._df_agg = None  # Group aggregated data
        self._df_totals = None  # Group metrics, summed over time
        self._df_mape = None  # MAPE between groups
        self._ga = None  # Genetic algorithm instance
        self._df_vis = None  # Visualisation dataframe
        self._df_rmse = None  # RMSE between groups
        self._best_ga = None
        self._solution = None

        self._cost_weighting()

    def _cost_weighting(self):
        """Set relative cost weights for each metric. Defaults to 1 unless specified.

        Used if you want the genetic algorithm to penalise the MSE cost of some metrics
        over others. Particularly useful if one metric is quite volatile, as the MSE cost can
        end up with it prioritising this over others.

        Returns:
            None
        """
        global metric_weights_global
        metric_weights_global = np.ones(len(self.metrics))
        if self._metric_weights:
            for key, value in self._metric_weights.items():
                try:
                    i = self.metrics.index(key)
                    metric_weights_global[i] = value
                    print(f'[Updating] {key} weight updated to {value}')
                except ValueError:
                    print(f'Cant find metric name {key}, weight {value} not applied, defaulting to 1')
        # Reshape to broadcast across metrics dimension of 3d array
        metric_weights_global = metric_weights_global.reshape(-1, 1, 1)

    def run(self):
        """Runs genetic algorithm and returns bin splits into _df_result

        Returns:
            None
        """

        global all_metrics_global
        global splits_global
        splits_global = (np.array(self._splits).sum() / np.array(self._splits))

        logger.debug('Splitting..')
        all_metrics_global = self._population.matrix

        # Run multiple times, save the best solution
        for i in range(self._runs):
            if self._runs > 1:
                print(f'[Run {i+1}]')

            # Initialise and run GA, get solution
            self._ga = GAInstance(
                num_genes=self._population.pop_size,
                ga_params=self._ga_params,
            )
            self._ga.run()

            if self._ga.fitness > self._best_score:
                print(f'Best fitness: {self._ga.fitness:.4}')
                self._solution = self._ga.solution
                self._best_ga = self._ga
                self._best_score = self._ga.fitness

        if self._solution is None:
            raise ValueError('Solution is None')

        # Assign solution to index
        self._df_result = self._population.assign(self._solution)
        self._post_run()
        self._build_data()
        logger.debug('Split complete')

    def _build_data(self):
        self._build_distributions()
        self._build_df_vis()
        self._build_aggregation()
        self._build_totals()
        self._df_mape = self._build_score(
            function=lambda x, y: np.mean(np.abs(((x - y) / x)))
        )
        self._df_rmse = self._build_score(
            function=lambda x, y: np.sqrt(((x - y) ** 2).mean())
        )

    @abstractmethod
    def _build_df_vis(self):
        pass

    @abstractmethod
    def _post_run(self):
        pass

    def fitness(self, title=None):
        """Plots the fitness-generation graph

        Returns:
            None
        """

        if self._best_ga:
            self._best_ga.ga.plot_fitness(title=title)
        else:
            print('No solution available, please use .run() first')

    def _build_distributions(self):
        """Build dataframe that contains data on the population count distributions of each group

        Sets:
            _df_dist
        """
        self._df_dist = self._df_result['bin'].value_counts().to_frame('count')
        self._df_dist.index.name = 'bin'
        self._df_dist['pct'] = (self._df_dist['count'] / self._df_dist['count'].sum()).round(4)
        return self._df_dist.sort_index()

    def _build_totals(self):
        df_agg = self._df_agg.copy()

        df_agg.loc[:, 'period'] = 'total'
        df_total = df_agg.groupby(['period', 'bin'])[self.metrics].sum()

        # If cutoff, use post cut off period for rmse
        if self.date_col and self._cutoff_date:
            df_agg.loc[df_agg[self.date_col] <= self._cutoff_date, 'period'] = 'pre-cutoff'
            df_agg.loc[df_agg[self.date_col] > self._cutoff_date, 'period'] = 'post-cutoff'
            df_total2 = df_agg.groupby(['period', 'bin'])[self.metrics].sum()
            df_total = pd.concat([df_total, df_total2], axis=0)

        self._df_totals = df_total

    def _build_score(self, function):
        """Scores each group, for each metric, based on the fuction passed.

        If cutoff_date is passed in class init, scores are only for the post-cutoff period.

        Args:
            function (callable): Scoring function

        Sets:
            _df_rsme (pd.DataFrame): Datafame of RMSE scores for each group and metric
        """
        df_agg = self._df_agg.copy()

        if self.date_col is None:
            df_agg.index = [0]*df_agg.shape[0]

        # If cutoff, use post cut off period for rmse
        if self.date_col and self._cutoff_date:
            df_agg = df_agg[
                df_agg[self.date_col] > pd.to_datetime(self._cutoff_date)
            ]

        # Pivot on bin groups
        df_ = df_agg.pivot(index=self.date_col, columns='bin', values=self.metrics)
        # Flatten columns
        df_.columns = df_.columns.map('_'.join).str.lower()
        groups = self._df_agg['bin'].unique()
        combinations_lst = list(combinations(groups, 2))

        df_lst = []
        for metric in self.metrics:
            df_metric = pd.DataFrame()
            for a, b in combinations_lst:
                col_a, col_b = f'{metric}_{a}', f'{metric}_{b}'
                score = function(df_[col_a], df_[col_b])
                df_metric.loc[a, b] = score
                df_metric.loc[b, a] = score
            df_metric = df_metric.reindex(sorted(df_metric.columns), axis=1)

            # Build multiindexed columns, top level for metric, bottom for group
            df_metric.columns = pd.MultiIndex.from_tuples([(metric, col) for col in df_metric.columns])
            df_lst.append(df_metric)

        df = pd.concat(df_lst, axis=1)
        df.columns.name = 'bin'
        df.index.name = 'bin'
        return df

    def _build_aggregation(self):
        """Aggregates metric data by bins

        Sets:
            _df_agg
        """

        # Aggregate by bin
        group_cols = ['bin']
        group_cols += [self.date_col] if self.date_col else []
        self._df_agg = self._df_vis.groupby(group_cols)[self.metrics].sum().reset_index()

    def visualise(self):
        """Visualise metrics for both bins and there comparative performances

        Returns:
            None
        """

        sns.set_style('darkgrid')
        figsize = (min(20, len(self.metrics) * 8), 5) if self.date_col else (8, 5)
        fig, ax = plt.subplots(1, len(self.metrics), figsize=figsize)
        if not hasattr(ax, '__iter__'):
            ax = [ax]

        # Plot each metric
        for i, metric in enumerate(self.metrics):

            # If over time, plot line graph, else bar
            if self.date_col:
                sns.lineplot(data=self._df_agg, x=self.date_col, y=metric, hue='bin', ax=ax[i], palette='Dark2')
                if self._cutoff_date:
                    ax[i].axvline(pd.to_datetime(self._cutoff_date), label='Cutoff', linestyle='--')
            else:
                self._df_agg['metric'] = metric
                sns.barplot(data=self._df_agg, x='metric', y=metric, hue='bin', ax=ax[i])

            ax[i].set_title(f'{metric.title()}')
            ax[i].tick_params(axis='x', labelrotation=45)
            ax[i].legend()

        plt.show()

    @property
    def totals(self):
        return self._df_totals

    @property
    def mape(self):
        return self._df_mape

    @property
    def results(self):
        """Returns compiled dataframe of solutions

        Returns:
            pd.DataFrame
        """
        return self._df_result

    @property
    def distributions(self):
        """Returns distributions dataframe

        Returns
            pd.DataFrame
        """
        return self._df_dist

    @property
    def rmse(self):
        return self._df_rmse

    @property
    def aggregations(self):
        return self._df_agg


class ABSplit(SplitBase):
    """Splits data into A/B groups based on specified parameters. All members of the population will be in one bin or
    the other.

    Attributes:
        _runs (int): Number of runs of the genetic algorithm to try
        _ga_params (object): GAParams object
        _best_score (float): Stores best score when using multiple runs
        _metric_weights (dict): Dictionary of {col_name: value} format for applying differing weights to columns
        _df (pd.DataFrame): Input data
        _population (object): Data object
        _df_result (pd.DataFrame): Proposed solution dataframe
        _ga (object): GAInstance object
        _df_vis (pd.DataFrame): DataFrame with solutions used for visualisation
        _best_ga (object): GAInstance object of best solution
        _solution (np.array): Best solution from search

    Properties:
        results (pd.DataFrame): Accessor for _df_result

    Methods:
        _cost_weighting(): Modifies cost weighting array
        run(): Runs the genetic algorithm
        fitness(): Plot generation fitness graph
        visualise(): Plots metrics using results from genetic algorithm output
    """

    def __init__(self, df, ga_params={}, metric_weights={}, cutoff_date=None, **kwargs):
        """Initializes the class and sets the attributes

        Args:
            df (pd.DataFrame): Dataframe to be split
            ga_params (dict): Parameters for the genetic algorithm (default: {})
            metric_weights (dict): Weights for each metric in the data (default: {})
            **kwargs: Additional keyword arguments
        """
        super().__init__(ga_params=ga_params, metric_weights=metric_weights, cutoff_date=cutoff_date, **kwargs)
        self.df = df
        self._population = Data(self.df.copy(), cutoff_date=cutoff_date, **kwargs)

    def _build_df_vis(self):
        """Builds dataframe for use in visualisation of groups.

        For ABSplit class, this is just joining the bins to the original data.

        Sets:
            _df_vis (pd.DataFrame):
        """
        # Merge solution results (_df_results) onto input data (metrics) so that data can be visualised
        self._df_vis = self._population.stacked.merge(
            self._df_result,
            left_index=True,
            right_index=True
        )

    def _post_run(self):
        pass

    def __repr__(self):
        lst_str = [f"'{col}', " for col in self.all_spec_columns]
        return f'ABSplit([{lst_str}])'


class MatchDataProc:
    """Ensures consistent number of dates between sample and population data

    Achieves this by concatenating both dataframes, unstacking by date, then filtering
    by the index of the sample and population data and restacking.

    Without this, if there are date mismatches between sample and population (e.g.
    100 distinct date in sample, and 110 in population) then the 2 matrices will be
    different sizes.

    Defaults missing date values to 0

    Attributes:
        df_pop (pd.DataFrame): Dataframe of population data
        df_samp (pd.DataFrame): Dataframe of sample data
        df_all (pd.DataFrame): Concatenation of df_pop and df_samp
        all (Data): Instance of the Data class for all data
        pop (Data): Instance of the Data class for population data
        samp (Data): Instance of the Data class for sample data

    Properties:
        sample (pd.DataFrame): Sample data
        population (pd.DataFrame): Population data
    """

    def __init__(self, df_pop, df_samp, **kwargs):
        """Initializes the class and sets the attributes

        Args:
            df_pop (pd.DataFrame): Dataframe of population data
            df_samp (pd.DataFrame): Dataframe of sample data
            **kwargs: Additional keyword arguments
        """
        self.df_pop = df_pop
        self.df_samp = df_samp
        self.df_all = pd.concat([df_pop, df_samp], axis=0)
        self.all = Data(self.df_all, scale=False, **kwargs)
        self.pop = Data(self.df_pop, scale=False, **kwargs)
        self.samp = Data(self.df_samp, scale=False, **kwargs)

    @property
    def sample(self):
        """Returns the filtered dataframe of sample data

        Returns:
            pd.DataFrame: Filtered dataframe of sample data
        """
        return self.all.filter(self.samp.index)

    @property
    def population(self):
        """Returns the filtered dataframe of population data

        Returns:
            pd.DataFrame: Filtered dataframe of population data
        """
        return self.all.filter(self.pop.index)


class Match(SplitBase):
    """Manages a genetic algorithm to find a group in a population with similar properties
    as a specified sample group. Sample group must be removed from population prior to running
    Match.

    Attributes:
        _df_pop (pd.DataFrame): Dataframe of population data
        _df_samp (pd.DataFrame): Dataframe of sample data
        _population (Data): Instance of the Data class for population data
        _sample (Data): Instance of the Data class for sample data
        _df_result (pd.DataFrame): Dataframe of population data that matches the sample group
        _df_vis (pd.DataFrame): Dataframe of population data, sample data and matched population data for visualization

    Methods:
        run(): Runs the genetic algorithm
        fitness(): Plot generation fitness graph
        visualise(): Plots metrics using results from genetic algorithm output
    """

    def __init__(self, population, sample, ga_params={}, metric_weights={}, **kwargs):
        """Initializes the class and sets the attributes

        Args:
            population (pd.DataFrame): Dataframe of population data
            sample (pd.DataFrame): Dataframe of sample data
            ga_params (dict): Parameters for the genetic algorithm
            metric_weights (dict): Weights for the metrics being compared
            **kwargs: Additional keyword arguments
        """

        # Specify fitness function
        global fitness_func_match
        ga_params['fitness_func'] = fitness_func_match
        super().__init__(ga_params=ga_params, metric_weights=metric_weights, **kwargs)
        self._splits = [0.5, 0.5]
        self._splits_num = 2
        self._df_pop = population
        self._df_samp = sample

        # Ensure consistent dates between sample and population
        if self.date_col:
            mdp = MatchDataProc(df_pop=self._df_pop, df_samp=self._df_samp, **kwargs)
            self._df_pop = mdp.population.reset_index()
            self._df_samp = mdp.sample.reset_index()

        # Fit scaler on combined data
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(pd.concat([self._df_pop, self._df_samp], axis=0)[self.metrics])

        self._population = Data(self._df_pop, scaler=scaler, **kwargs)
        self._sample = Data(self._df_samp, scaler=scaler, **kwargs)

        # Makes the match matrix available globally, sum along population axis
        global match_metrics_global
        global metric_weights_global
        match_metrics_global = self._sample.matrix.sum(1)
        # Reshape to configure for 2d m
        metric_weights_global = metric_weights_global.reshape(1, -1)

        self._update_initial_population()

    def _update_initial_population(self):
        """Update initial population so frequency of '1's in starting population is sample_size/population_size
        to speed up convergence.
        """
        # Update initial population
        pop_size = self._population.unstacked.shape[0]
        sample_size = self._sample.unstacked.shape[0]
        total = pop_size + sample_size
        init_pop = np.random.choice(
            [0, 1],
            p=[pop_size/total, sample_size/total],
            size=(self._ga_params.params['sol_per_pop'], self._population.unstacked.shape[0])
        )
        self._ga_params.update(initial_population=init_pop)

    def _post_run(self):
        """Extract where solution == '1' from solution, concatenate with sample index and label sample bin '0'

        Sets:
            _df_result
        """
        # Filter on only bin == 1
        self._df_result = self._df_result[self._df_result['bin'] == '1']

        # Get sample index, set as bin 0, concat to _df_results (which are all bin 1)
        index_cols = self._sample.index.index.names
        df_match = self._df_samp[index_cols].drop_duplicates().set_index(index_cols)
        df_match['bin'] = '0'
        self._df_result = pd.concat([self._df_result, df_match], axis=0).sort_index()

    def _build_df_vis(self):
        """Build dataframe for visualisation

        For Match class, this both population and sample data concatenated. All samples from the
        popoulation that are not chosen from the matching process are filtered out by the inner
        join on the results.

        Sets:
            _df_vis (pd.DataFrame): Dataframe of metrics and bin groups
        """
        # Concat to form entire population
        self._df_vis = pd.concat([self._population.stacked, self._sample.stacked], axis=0)

        # Inner join to filter out all but sample and match
        self._df_vis = self._df_vis.merge(
            self._df_result,
            left_index=True,
            right_index=True,
            how='inner'
        )

    def __repr__(self):
        lst_str = [f"'{col}', " for col in self.all_spec_columns]
        return f'Match([{lst_str}])'


def fitness_func_absplit(ga_instance, solution, solution_idx):
    """Compute the fitness value for a given solution in the context of ABSplit class.

    The fitness function is a measure of solution quality. In this case, it's computed
    as the inverse of a sum of penalties related to size, mean squared error (MSE), and
    sum, for different groups in the solution.

    Args:
        ga_instance (pygad.GA): An instance of the GA algorithm from the PyGAD library.
            This argument is required by PyGAD but not used in this function.
        solution (numpy.ndarray): A 1-D array representing a potential solution for the
            GA. The elements in the array represent group identifiers.
        solution_idx (int): The index of the solution in the current population. This
            argument is required by PyGAD but not used in this function.

    Globals:
        all_metrics_global (numpy.ndarray): A 3-D array representing different metrics,
            for different population groups, across different dates.
        metric_weights_global (numpy.ndarray): A 1-D array representing the relative
            importance of each metric in `all_metrics_global`.
        splits_global (numpy.ndarray): A 1-D array representing the proportional split
            for each group in the population.
        size_penalty_global (float): A constant representing the penalty weight for size
            discrepancy among different groups.
        sum_penalty_global (float): A constant representing the penalty weight for sum
            discrepancy among different groups.
        group_ids_global (numpy.ndarray): A 1-D array representing the unique identifiers
            for each group in the population.

    Returns:
        float: The fitness value for the given solution. Higher values represent better
        solutions.

    Notes:
        The fitness is computed as follows:
        1. First, for each group in the solution, a binary array is computed. Dimensions:
            (number of groups, size of population.
        2. Size penalty is calculated based on differences in population count between groups
        3. The mean squared error (MSE) for each metric is calculated between groups
        4. Sum penalty is calculated based on the difference in sum of metrics over time costs between groups.
        5. The total cost is computed as the sum of the MSE, size penalty, and sum penalty.
        6. The fitness value is then computed as the inverse of the absolute total cost
           (plus a small number to prevent division by zero).
    """

    global all_metrics_global     # 3d matrix of metric values (metrics, population, dates)
    global metric_weights_global  # Relative weights for each metric
    global splits_global          # Proportional splits for each group
    global size_penalty_global    # Float weight for size penalty
    global sum_penalty_global     # Float weight for sum penalty
    global group_ids_global       # Array of group IDs

    # Generate binary array, 1 row of 0s and 1s for each group (where solution == 1/2/3 etc)
    groups = (solution == group_ids_global[:, None]).astype(int)

    # == Size penalty == #
    # Calculate all_metrics mean * number of days * number of metrics
    mean = np.mean(all_metrics_global) * all_metrics_global.shape[0] * all_metrics_global.shape[2]
    # Get size cost for each group
    mean_group = (groups * mean).sum(1) * splits_global
    # Calculate group differences, sum
    size_cost = (np.abs(np.roll(mean_group, -1) - mean_group).sum()) * size_penalty_global / (4*len(splits_global)) ** 1.2

    # == MSE == #
    costs = (groups @ all_metrics_global) * splits_global.reshape((1, -1, 1)) * metric_weights_global
    diffs = np.roll(costs, -1, axis=1) - costs
    mse = ((diffs ** 2).mean(axis=1)).sum()

    # == Sum penalty == #
    sum_cost = (np.abs(costs.sum(2) - np.roll(costs.sum(2), shift=-1, axis=1)).sum() * sum_penalty_global) / (4*len(splits_global)) ** 1.2

    # Fitness
    total = mse + size_cost + sum_cost
    fitness = 1.0 / np.abs(total + 1e-5)  # Add small sum to prevent divide by zero
    return fitness


def fitness_func_match(ga_instance, solution, solution_idx):
    """Fitness function for Match
    """
    global all_metrics_global
    global match_metrics_global
    global metric_weights_global

    cost1 = match_metrics_global  # Sum along population axis
    cost2 = solution @ all_metrics_global

    # Average over time axis, sum over metric axis
    mse = (metric_weights_global @ ((cost1 - cost2)**2).mean(1)).sum()

    # Fitness
    fitness = 1.0 / np.abs(mse + 1e-10)  # Add small sum to prevent divide by zero
    return fitness
