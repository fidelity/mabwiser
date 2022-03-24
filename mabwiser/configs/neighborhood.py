# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from typing import Dict, List, Optional

from spock import spock
from spock.utils import ge, gt

from mabwiser.configs.constants import DistanceMetrics
from mabwiser.utilities.validators import check_sum_to_unity


@spock
class Clusters:
    """Clusters Neighborhood Policy.

    Clusters is a k-means clustering approach that uses the observations
    from the closest *cluster* with a learning policy.
    Supports ``KMeans`` and ``MiniBatchKMeans``.

    Attributes
    ----------
    n_clusters: Num
        The number of clusters. Integer. Must be at least 2. Default value is 2.
    is_minibatch: bool
        Boolean flag to use ``MiniBatchKMeans`` or not. Default value is False.

    Example
    -------
        >>> from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
        >>> list_of_arms = [1, 2, 3, 4]
        >>> decisions = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3]
        >>> rewards = [0, 1, 1, 0, 0, 0, 0, 1, 1, 1]
        >>> contexts = [[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],[0, 2, 2, 3, 5], [1, 3, 1, 1, 1], \
                        [0, 0, 0, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0]]
        >>> mab = MAB(list_of_arms, LearningPolicy.EpsilonGreedy(epsilon=0), NeighborhoodPolicy.Clusters(3))
        >>> mab.fit(decisions, rewards, contexts)
        >>> mab.predict([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])
        [3, 1]
    """

    n_clusters: int = 2
    is_minibatch: bool = False

    def __post_hook_(self):
        try:
            ge(self.n_clusters, bound=2)
        except Exception as e:
            raise ValueError(
                f"`{self.__class__.__name__}` could not be instantiated -- spock message: {e}"
            )


@spock
class KNearest:
    """KNearest Neighborhood Policy.

    KNearest is a nearest neighbors approach that selects the *k-nearest* observations
    to be used with a learning policy.

    Attributes
    ----------
    k: int
        The number of neighbors to select.
        Integer value. Must be greater than zero.
        Default value is 1.
    metric: str
        The metric used to calculate distance.
        Accepts any of the metrics supported by ``scipy.spatial.distance.cdist``.
        Default value is Euclidean distance.

    Example
    -------
        >>> from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
        >>> list_of_arms = [1, 2, 3, 4]
        >>> decisions = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3]
        >>> rewards = [0, 1, 1, 0, 0, 0, 0, 1, 1, 1]
        >>> contexts = [[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],[0, 2, 2, 3, 5], [1, 3, 1, 1, 1], \
                        [0, 0, 0, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0]]
        >>> mab = MAB(list_of_arms, LearningPolicy.EpsilonGreedy(epsilon=0), \
                      NeighborhoodPolicy.KNearest(2, "euclidean"))
        >>> mab.fit(decisions, rewards, contexts)
        >>> mab.predict([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])
        [1, 1]
    """

    k: int = 1
    metric: DistanceMetrics = DistanceMetrics.euclidean

    def __post_hook__(self):
        try:
            gt(self.k, bound=0)
        except Exception as e:
            raise ValueError(
                f"`{self.__class__.__name__}` could not be instantiated -- spock message: {e}"
            )


@spock
class LSHNearest:
    """Locality-Sensitive Hashing Approximate Nearest Neighbors Policy.

    LSHNearest is a nearest neighbors approach that uses locality sensitive hashing with a simhash to
    select observations to be used with a learning policy.

    For the simhash, contexts are projected onto a hyperplane of n_context_cols x n_dimensions and each
    column of the hyperplane is evaluated for its sign, giving an ordered array of binary values.
    This is converted to a base 10 integer used as the hash code to assign the context to a hash table. This
    process is repeated for a specified number of hash tables, where each has a unique, randomly-generated
    hyperplane. To select the neighbors for a context, the hash code is calculated for each hash table and any
    contexts with the same hashes are selected as the neighbors.

    As with the radius or k value for other nearest neighbors algorithms, selecting the best number of dimensions
    and tables requires tuning. For the dimensions, a good starting point is to use the log of the square root of
    the number of rows in the training data. This will give you sqrt(n_rows) number of hashes.

    The number of dimensions and number of tables have inverse effects from each other on the number of empty
    neighborhoods and average neighborhood size. Increasing the dimensionality decreases the number of collisions,
    which increases the precision of the approximate neighborhood but also potentially increases the number of empty
    neighborhoods. Increasing the number of hash tables increases the likelihood of capturing neighbors the
    other random hyperplanes miss and increases the average neighborhood size. It should be noted that the fit
    operation is O(2**n_dimensions).

    Attributes
    ----------
    n_dimensions: int
        The number of dimensions to use for the hyperplane.
        Integer value. Must be greater than zero.
        Default value is 5.
    n_tables: int
        The number of hash tables.
        Integer value. Must be greater than zero.
        Default value is 3.
    no_nhood_prob_of_arm: None or List
        The probabilities associated with each arm. Used to select random arm if a prediction context has no neighbors.
        If not given, a uniform random distribution over all arms is assumed.
        The probabilities should sum up to 1.

    Example
    -------
        >>> from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
        >>> list_of_arms = [1, 2, 3, 4]
        >>> decisions = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3]
        >>> rewards = [0, 1, 1, 0, 0, 0, 0, 1, 1, 1]
        >>> contexts = [[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],[0, 2, 2, 3, 5], [1, 3, 1, 1, 1], \
                        [0, 0, 0, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0]]
        >>> mab = MAB(list_of_arms, LearningPolicy.EpsilonGreedy(epsilon=0), \
                      NeighborhoodPolicy.LSHNearest(5, 3))
        >>> mab.fit(decisions, rewards, contexts)
        >>> mab.predict([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])
        [3, 1]
    """

    n_dimensions: int = 5
    n_tables: int = 3
    no_nhood_prob_of_arm: Optional[List[float]] = None

    def __post_hook__(self):
        try:
            gt(self.n_dimensions, bound=0)
            gt(self.n_tables, bound=0)
            check_sum_to_unity(self.no_nhood_prob_of_arm)
        except Exception as e:
            raise ValueError(
                f"`{self.__class__.__name__}` could not be instantiated -- spock message: {e}"
            )


@spock
class Radius:
    """Radius Neighborhood Policy.

    Radius is a nearest neighborhood approach that selects the observations
    within a given *radius* to be used with a learning policy.

    Attributes
    ----------
    radius: Num
        The maximum distance within which to select observations.
        Integer or Float. Must be greater than zero.
        Default value is 1.
    metric: str
        The metric used to calculate distance.
        Accepts any of the metrics supported by scipy.spatial.distance.cdist.
        Default value is Euclidean distance.
    no_nhood_prob_of_arm: None or List
        The probabilities associated with each arm. Used to select random arm if a prediction context has no neighbors.
        If not given, a uniform random distribution over all arms is assumed.
        The probabilities should sum up to 1.

    Example
    -------
        >>> from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
        >>> list_of_arms = [1, 2, 3, 4]
        >>> decisions = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3]
        >>> rewards = [0, 1, 1, 0, 0, 0, 0, 1, 1, 1]
        >>> contexts = [[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],[0, 2, 2, 3, 5], [1, 3, 1, 1, 1], \
                        [0, 0, 0, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0]]
        >>> mab = MAB(list_of_arms, LearningPolicy.EpsilonGreedy(epsilon=0), \
                      NeighborhoodPolicy.Radius(2, "euclidean"))
        >>> mab.fit(decisions, rewards, contexts)
        >>> mab.predict([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])
        [3, 1]
    """

    radius: float = 0.05
    metric: DistanceMetrics = DistanceMetrics.euclidean
    no_nhood_prob_of_arm: Optional[List[float]] = None

    def __post_hook__(self):
        try:
            gt(self.radius, bound=0.0)
            check_sum_to_unity(self.no_nhood_prob_of_arm)
        except Exception as e:
            raise ValueError(
                f"`{self.__class__.__name__}` could not be instantiated -- spock message: {e}"
            )


class _DTCCriterion(Enum):
    squared_error = "squared_error"
    friedman_mse = "friedman_mse"
    absolute_error = "absolute_error"
    poisson = "poisson"


class _DTCSplitter(Enum):
    best = "best"
    random = "random"


@spock
class TreeBandit:
    """TreeBandit Neighborhood Policy.

    This policy fits a decision tree for each arm using context history.
    It uses the leaves of these trees to partition the context space into regions
    and keeps a list of rewards for each leaf.
    To predict, it receives a context vector and goes to the corresponding
    leaf at each arm's tree and applies the given context-free MAB learning policy
    to predict expectations and choose an arm.

    The TreeBandit neighborhood policy is compatible with the following
    context-free learning policies only: EpsilonGreedy, ThompsonSampling and UCB1.

    The TreeBandit neighborhood policy is a modified version of
    the TreeHeuristic algorithm presented in:
    Adam N. Elmachtoub, Ryan McNellis, Sechan Oh, Marek Petrik
    A Practical Method for Solving Contextual Bandit Problems Using Decision Trees, UAI 2017

    Attributes
    ----------
    tree_parameters: Dict, **kwarg
        Parameters of the decision tree.
        The keys must match the parameters of sklearn.tree.DecisionTreeClassifier.
        When a parameter is not given, the default parameters from
        sklearn.tree.DecisionTreeClassifier will be chosen.
        Default value is an empty dictionary.

    Example
    -------
        >>> from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
        >>> list_of_arms = ['Arm1', 'Arm2']
        >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
        >>> rewards = [20, 17, 25, 9]
        >>> contexts = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 1, 0], [3, 2, 1, 0]]
        >>> mab = MAB(list_of_arms, LearningPolicy.EpsilonGreedy(epsilon=0), NeighborhoodPolicy.TreeBandit())
        >>> mab.fit(decisions, rewards, contexts)
        >>> mab.predict([[3, 2, 0, 1]])
        'Arm2'

    """

    criterion: Optional[_DTCCriterion] = _DTCCriterion.squared_error
    splitter: Optional[_DTCSplitter] = _DTCSplitter.best
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Optional[int] = None
    random_state: Optional[int] = None
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    ccp_alpha: float = 0.0

    def __post_hook__(self):
        try:
            if self.ccp_alpha is not None:
                ge(self.ccp_alpha, bound=0.0)
        except Exception as e:
            raise ValueError(
                f"`{self.__class__.__name__}` could not be instantiated -- spock message: {e}"
            )


class NeighborhoodPolicy(Enum):
    lsh_nearest = LSHNearest
    clusters = Clusters
    k_nearest = KNearest
    radius = Radius
    tree_bandit = TreeBandit
