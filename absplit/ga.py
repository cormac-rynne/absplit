from absplit.param import ParamMixin
from absplit.data import Data
import pandas as pd
import pygad
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

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

        Returns:
            np.array
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

    def __init__(self, **kwargs):
        """Initializes the class and sets the attributes

        Args:
            **kwargs: Genetic algorithm parameters
        """
        global fitness_func_absplit

        # Default parameters
        self._default_ga_params = dict(
            # -- General Genetic Algorithm params --
            num_generations=200,
            sol_per_pop=100,
            num_parents_mating=5,
            mutation_type='swap',
            mutation_probability=0.1,
            fitness_func=fitness_func_absplit,
            keep_elitism=1,

            # -- Binary Genetic Algorithm specific parameters --
            init_range_low=0,
            init_range_high=2,
            random_mutation_min_val=0,
            random_mutation_max_val=2,
            gene_type=int,
        )

        self._dont_touch = [
            'init_range_low',
            'init_range_high',
            'random_mutation_min_val',
            'random_mutation_max_val',
            'gene_type'
        ]

        # Copy default, update
        self.params = dict(self._default_ga_params)
        self._update_params(**kwargs)

    def _update_params(self, **kwargs):
        """Update GA params if any passed

        Returns:
            None
        """

        # Update if args passed
        if kwargs:
            for key, value in kwargs.items():
                if key in self._dont_touch:
                    print(f'Parameter \'{key}\' is essential to running the genetic algorithm as a'\
                          f'binary genetic algorith, and so youre not allowed to modify this')
                    continue
                print(f'Updating {key} to {value}')
                self.params[key] = value


class SplitBase(ParamMixin):
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

    def __init__(self, ga_params={}, metric_weights={}, runs=1, **kwargs):
        """Initializes the class and sets the attributes

        Args:
            ga_params (dict, optional): Genetic algorithm parameters to add/modify
            metric_weights (dict, optional): Cost weightings to apply to metrics
            runs (int, optional): How many runs to try
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        if ga_params:
            assert isinstance(ga_params, dict), 'ga_params must be a dictionary'
        self._runs = runs
        self._ga_params = GAParams(**ga_params)
        self._best_score = -1
        self._metric_weights = metric_weights
        self._df = None
        self._population = None
        self._df_result = None  # Final results dataframe
        self._ga = None  # genetic algorithm instance
        self._df_vis = None  # Visualisation dataframe
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
                    print(f'{key} weight updated to {value}')
                except ValueError:
                    print(f'Cant find metric name {key}, weight {value} not applied, defaulting to 1')

    def run(self):
        """Runs genetic algorithm and returns bin splits into _df_result

        Returns:
            None
        """

        global all_metrics_global
        logger.debug('Splitting..')
        all_metrics_global = self._population.matrix

        # Run multiple times, save the best solution
        for i in range(self._runs):
            if self._runs > 1:
                print(f'Run {i+1}')

            # Initialise and run GA, get solution
            self._ga = GAInstance(
                num_genes=self._population.pop_size,
                ga_params=self._ga_params,
            )
            self._ga.run()

            if self._ga.fitness > self._best_score:
                print(f'Best fitness: {self._ga.fitness}')
                self._solution = self._ga.solution
                self._best_ga = self._ga
                self._best_score = self._ga.fitness

        if self._solution is None:
            raise ValueError('Solution is None')

        # Assign solution to index
        self._df_result = self._population.assign(self._solution)

        logger.debug('Split complete')

    def fitness(self, title=None):
        """Plots the fitness-generation graph

        Returns:
            None
        """

        if self._best_ga:
            self._best_ga.ga.plot_fitness(title=title)
        else:
            print('No solution available, please use .run() first')

    @property
    def results(self):
        """Returns compiled dataframe of solutions

        Returns:
            pd.DataFrame
        """

        return self._df_result

    def visualise(self, column=None):
        """Visualise metrics for both bins and there comparative performances

        Returns:
            None
        """

        # Default to all metric columns if no columns specified
        if column:
            if isinstance(column, str):
                column = [column]
            vis_metrics = column
        else:
            vis_metrics = self.metrics

        if self.results is None:
            print('Must use .run() before visualising')
            return

        # Aggregate by bin
        group_cols = ['bin']
        group_cols += [self.date_col] if self.date_col else []
        df = self._df_vis.groupby(group_cols)[vis_metrics].sum().reset_index()

        if not len(df):
            logger.warning(f'Failed vis')
            return

        sns.set_style('darkgrid')

        # Chart sizing
        figsize = (min(20, len(vis_metrics)*8), 5) if self.date_col else (8, 5)
        fig, ax = plt.subplots(1, len(vis_metrics), figsize=figsize)
        if not hasattr(ax, '__iter__'):
            ax = [ax]

        # Plot each metric
        for i, metric in enumerate(vis_metrics):

            # If over time, plot line graph, else bar
            if self.date_col:
                sns.lineplot(data=df, x=self.date_col, y=metric, hue='bin', ax=ax[i])
            else:
                df['metric'] = metric
                sns.barplot(data=df, x='metric', y=metric, hue='bin', ax=ax[i])

            ax[i].set_title(f'{metric.title()}')
            ax[i].tick_params(axis='x', labelrotation=45)

        plt.show()


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

    def __init__(self, df, ga_params={}, metric_weights={}, **kwargs):
        """Initializes the class and sets the attributes

        Args:
            df (pd.DataFrame): Dataframe to be split
            ga_params (dict): Parameters for the genetic algorithm (default: {})
            metric_weights (dict): Weights for each metric in the data (default: {})
            **kwargs: Additional keyword arguments
        """
        super().__init__(ga_params=ga_params, metric_weights=metric_weights, **kwargs)
        self.df = df
        self._population = Data(self.df.copy(), **kwargs)

    def visualise(self, column=None):
        """Visualizes the A/B split results.

        Returns:
            None
        """

        # Merge solution results (_df_results) onto input data (metrics) so that data can be visualised
        self._df_vis = self._population.stacked.merge(
            self._df_result,
            left_index=True,
            right_index=True
        )
        super().visualise(column=column)

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
        _broadcast_match(): Makes the match matrix available globally
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
        ga_params['mutation_type'] = 'scramble'
        super().__init__(ga_params=ga_params, metric_weights=metric_weights, **kwargs)

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
        self._broadcast_match()

    def _broadcast_match(self):
        """Makes the match matrix available globally

        Returns:
            None
        """
        global match_metrics_global
        match_metrics_global = self._sample.matrix

    def run(self):
        """Runs the genetic algorithm, filters out all except sample and result

        Returns:
            None
        """
        # Run genetic algorithm
        super().run()

        # Filter on only bin == 1
        self._df_result = self._df_result[self._df_result['bin'] == 1]

        # Get sample index, set as bin 0, concat to _df_results (which are all bin 1)
        index_cols = self._sample.index.index.names
        df_match = self._df_samp[index_cols].drop_duplicates().set_index(index_cols)
        df_match['bin'] = 0
        self._df_result = pd.concat([self._df_result, df_match], axis=0).sort_index()

    def visualise(self, column=None):
        """Visualizes sample and results side by side

        Returns:
            None
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

        super().visualise(column=column)

    def __repr__(self):
        lst_str = [f"'{col}', " for col in self.all_spec_columns]
        return f'Match([{lst_str}])'


def fitness_func_absplit(solution, solution_idx):
    """Fitness function for ABSplit
    """
    global all_metrics_global
    global metric_weights_global

    cost1 = solution @ all_metrics_global
    cost2 = (1 - solution) @ all_metrics_global
    # Average over time axis, sum over metric axis
    mse = (metric_weights_global @ ((cost1 - cost2)**2).mean(1)).sum()

    # Fitness
    fitness = 1.0 / np.abs(mse)
    return fitness


def fitness_func_match(solution, solution_idx):
    """Fitness function for Match
    """
    global all_metrics_global
    global match_metrics_global
    global metric_weights_global

    cost1 = match_metrics_global.sum(1)  # Sum along population axis
    cost2 = solution @ all_metrics_global
    # Average over time axis, sum over metric axis
    mse = (metric_weights_global @ ((cost1 - cost2)**2).mean(1)).sum()

    # Fitness
    fitness = 1.0 / np.abs(mse)
    return fitness
