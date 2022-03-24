# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

from typing import Callable, Dict, Optional

from spock import spock
from spock.utils import ge, gt, within


# TODO: Use inheritance to clean up dupes... lazy here to match current implementation
@spock
class Popularity:
    """Randomized Popularity Learning Policy.

    Returns a randomized popular arm for each prediction.
    The probability of selection for each arm is weighted by their mean reward.
    It assumes that the rewards are non-negative.

    The probability of selection is calculated as:

    .. math::
        P(arm) = \\frac{ \\mu_i } { \\Sigma{ \\mu }  }

    where :math:`\\mu_i` is the mean reward for that arm.

    Example
    -------
        >>> from mabwiser.mab import MAB, LearningPolicy
        >>> list_of_arms = ['Arm1', 'Arm2']
        >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
        >>> rewards = [20, 17, 25, 9]
        >>> mab = MAB(list_of_arms, LearningPolicy.Popularity())
        >>> mab.fit(decisions, rewards)
        >>> mab.predict()
        'Arm1'
    """

    ...


@spock
class Random:
    """Random Learning Policy.

    Returns a random arm for each prediction.
    The probability of selection for each arm is uniformly at random.

    Example
    -------
        >>> from mabwiser.mab import MAB, LearningPolicy
        >>> list_of_arms = ['Arm1', 'Arm2']
        >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
        >>> rewards = [20, 17, 25, 9]
        >>> mab = MAB(list_of_arms, LearningPolicy.Random())
        >>> mab.fit(decisions, rewards)
        >>> mab.predict()
        'Arm2'
    """

    ...


@spock
class EpsilonGreedy:
    """Epsilon Greedy Learning Policy.

    This policy selects the arm with the highest expected reward with probability 1 - :math:`\\epsilon`,
    and with probability :math:`\\epsilon` it selects an arm at random for exploration.

    Attributes
    ----------
    epsilon: Num
        The probability of selecting a random arm for exploration.
        Integer or float. Must be between 0 and 1.
        Default value is 0.1.

    Example
    -------
        >>> from mabwiser.mab import MAB, LearningPolicy
        >>> arms = ['Arm1', 'Arm2']
        >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
        >>> rewards = [20, 17, 25, 9]
        >>> mab = MAB(arms, LearningPolicy.EpsilonGreedy(epsilon=0.25), seed=123456)
        >>> mab.fit(decisions, rewards)
        >>> mab.predict()
        'Arm1'
    """

    epsilon: float = 0.1

    def __post_hook__(self):
        try:
            within(self.epsilon, 0.0, 1.0, inclusive_lower=True, inclusive_upper=True)
        except Exception as e:
            raise ValueError(
                f"`{self.__class__.__name__}` could not be instantiated -- spock message: {e}"
            )


@spock
class LinGreedy:
    """LinGreedy Learning Policy.

    This policy trains a ridge regression for each arm.
    Then, given a given context, it predicts a regression value.
    This policy selects the arm with the highest regression value with probability 1 - :math:`\\epsilon`,
    and with probability :math:`\\epsilon` it selects an arm at random for exploration.

    Attributes
    ----------
    epsilon: Num
        The probability of selecting a random arm for exploration.
        Integer or float. Must be between 0 and 1.
        Default value is 0.1.
    l2_lambda: Num
        The regularization strength.
        Integer or float. Cannot be negative.
        Default value is 1.0.
    arm_to_scaler: Dict[Arm, Callable]
        Standardize context features by arm.
        Dictionary mapping each arm to a scaler object. It is assumed
        that the scaler objects are already fit and will only be used
        to transform context features.
        Default value is None.

    Example
    -------
        >>> from mabwiser.mab import MAB, LearningPolicy
        >>> list_of_arms = ['Arm1', 'Arm2']
        >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
        >>> rewards = [20, 17, 25, 9]
        >>> contexts = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 1, 0], [3, 2, 1, 0]]
        >>> mab = MAB(list_of_arms, LearningPolicy.LinGreedy(epsilon=0.5))
        >>> mab.fit(decisions, rewards, contexts)
        >>> mab.predict([[3, 2, 0, 1]])
        'Arm2'
    """

    epsilon: float = 0.1
    l2_lambda: float = 1.0
    scale: bool = False

    def __post_hook__(self):
        try:
            within(self.epsilon, 0.0, 1.0, inclusive_lower=True, inclusive_upper=True)
            ge(self.l2_lambda, bound=0.0)
        except Exception as e:
            raise ValueError(
                f"`{self.__class__.__name__}` could not be instantiated -- spock message: {e}"
            )


@spock
class LinTS:
    """LinTS Learning Policy

        For each arm LinTS trains a ridge regression and
        creates a multivariate normal distribution for the coefficients using the
        calculated coefficients as the mean and the covariance as:

        .. math::
            \\alpha^{2} (x_i^{T}x_i + \\lambda * I_d)^{-1}

        The normal distribution is randomly sampled to obtain
        expected coefficients for the ridge regression for each
        prediction.

        :math:`\\alpha` is a factor used to adjust how conservative the estimate is.
        Higher :math:`\\alpha` values promote more exploration.
    mas
        The multivariate normal distribution uses Cholesky decomposition to guarantee deterministic behavior.
        This method requires that the covariance is a positive definite matrix.
        To ensure this is the case, alpha and l2_lambda are required to be greater than zero.

        Attributes
        ----------
        alpha: Num
            The multiplier to determine the degree of exploration.
            Integer or float. Must be greater than zero.
            Default value is 1.0.
        l2_lambda: Num
            The regularization strength.
            Integer or float. Must be greater than zero.
            Default value is 1.0.
        arm_to_scaler: Dict[Arm, Callable]
            Standardize context features by arm.
            Dictionary mapping each arm to a scaler object. It is assumed
            that the scaler objects are already fit and will only be used
            to transform context features.
            Default value is None.

        Example
        -------
            >>> from mabwiser.mab import MAB, LearningPolicy
            >>> list_of_arms = ['Arm1', 'Arm2']
            >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
            >>> rewards = [20, 17, 25, 9]
            >>> contexts = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 1, 0], [3, 2, 1, 0]]
            >>> mab = MAB(list_of_arms, LearningPolicy.LinTS(alpha=0.25))
            >>> mab.fit(decisions, rewards, contexts)
            >>> mab.predict([[3, 2, 0, 1]])
            'Arm2'
    """

    alpha: float = 1.0
    l2_lambda: float = 1.0
    scale: bool = False

    def __post_hook__(self):
        try:
            gt(self.alpha, bound=0.0)
            gt(self.l2_lambda, bound=0.0)
        except Exception as e:
            raise ValueError(
                f"`{self.__class__.__name__}` could not be instantiated -- spock message: {e}"
            )


@spock
class LinUCB:
    """LinUCB Learning Policy.

    This policy trains a ridge regression for each arm.
    Then, given a given context, it predicts a regression value
    and calculates the upper confidence bound of that prediction.
    The arm with the highest highest upper bound is selected.

    The UCB for each arm is calculated as:

    .. math::
        UCB = x_i \\beta + \\alpha \\sqrt{(x_i^{T}x_i + \\lambda * I_d)^{-1}x_i}

    Where :math:`\\beta` is the matrix of the ridge regression coefficients, :math:`\\lambda` is the regularization
    strength, and I_d is a dxd identity matrix where d is the number of features in the context data.

    :math:`\\alpha` is a factor used to adjust how conservative the estimate is.
    Higher :math:`\\alpha` values promote more exploration.

    Attributes
    ----------
    alpha: Num
        The parameter to control the exploration.
        Integer or float. Cannot be negative.
        Default value is 1.0.
    l2_lambda: Num
        The regularization strength.
        Integer or float. Cannot be negative.
        Default value is 1.0.
    arm_to_scaler: Dict[Arm, Callable]
        Standardize context features by arm.
        Dictionary mapping each arm to a scaler object. It is assumed
        that the scaler objects are already fit and will only be used
        to transform context features.
        Default value is None.

    Example
    -------
        >>> from mabwiser.mab import MAB, LearningPolicy
        >>> list_of_arms = ['Arm1', 'Arm2']
        >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
        >>> rewards = [20, 17, 25, 9]
        >>> contexts = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 1, 0], [3, 2, 1, 0]]
        >>> mab = MAB(list_of_arms, LearningPolicy.LinUCB(alpha=1.25))
        >>> mab.fit(decisions, rewards, contexts)
        >>> mab.predict([[3, 2, 0, 1]])
        'Arm2'
    """

    alpha: float = 1.0
    epsilon: float = 0.0
    l2_lambda: float = 1.0
    scale: bool = False

    def __post_hook__(self):
        try:
            ge(self.alpha, bound=0.0)
            ge(self.epsilon, bound=0.0)
            ge(self.l2_lambda, bound=0.0)
        except Exception as e:
            raise ValueError(
                f"`{self.__class__.__name__}` could not be instantiated -- spock message: {e}"
            )


@spock
class Softmax:
    """Softmax Learning Policy.

    This policy selects each arm with a probability proportionate to its average reward.
    The average reward is calculated as a logistic function with each probability as:

    .. math::
        P(arm) = \\frac{ e ^  \\frac{\\mu_i - \\max{\\mu}}{ \\tau } }
        { \\Sigma{e ^  \\frac{\\mu - \\max{\\mu}}{ \\tau }}  }

    where :math:`\\mu_i` is the mean reward for that arm and :math:`\\tau` is the "temperature" to determine the degree of
    exploration.

    Attributes
    ----------
    tau: Num
         The temperature to control the exploration.
         Integer or float. Must be greater than zero.
         Default value is 1.

    Example
    -------
        >>> from mabwiser.mab import MAB, LearningPolicy
        >>> list_of_arms = ['Arm1', 'Arm2']
        >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
        >>> rewards = [20, 17, 25, 9]
        >>> mab = MAB(list_of_arms, LearningPolicy.Softmax(tau=1))
        >>> mab.fit(decisions, rewards)
        >>> mab.predict()
        'Arm2'
    """

    tau: float = 1.0

    def __post_hook__(self):
        try:
            gt(self.tau, bound=0.0)
        except Exception as e:
            raise ValueError(
                f"`{self.__class__.__name__}` could not be instantiated -- spock message: {e}"
            )


@spock
class ThompsonSampling:
    """Thompson Sampling Learning Policy.

    This policy creates a beta distribution for each arm and
    then randomly samples from these distributions.
    The arm with the highest sample value is selected.

    Notice that rewards must be binary to create beta distributions.
    If rewards are not binary, see the ``binarizer`` function.

    Attributes
    ----------
    binarizer: Callable
        If rewards are not binary, a binarizer function is required.
        Given an arm decision and its corresponding reward, the binarizer function
        returns `True/False` or `0/1` to denote whether the decision counts
        as a success, i.e., `True/1` based on the reward or `False/0` otherwise.

        The function signature of the binarizer is:

        ``binarize(arm: Arm, reward: Num) -> True/False or 0/1``

    Example
    -------
        >>> from mabwiser.mab import MAB, LearningPolicy
        >>> list_of_arms = ['Arm1', 'Arm2']
        >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
        >>> rewards = [1, 1, 1, 0]
        >>> mab = MAB(list_of_arms, LearningPolicy.ThompsonSampling())
        >>> mab.fit(decisions, rewards)
        >>> mab.predict()
        'Arm2'

        >>> from mabwiser.mab import MAB, LearningPolicy
        >>> list_of_arms = ['Arm1', 'Arm2']
        >>> arm_to_threshold = {'Arm1':10, 'Arm2':10}
        >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
        >>> rewards = [10, 20, 15, 7]
        >>> def binarize(arm, reward): return reward > arm_to_threshold[arm]
        >>> mab = MAB(list_of_arms, LearningPolicy.ThompsonSampling(binarizer=binarize))
        >>> mab.fit(decisions, rewards)
        >>> mab.predict()
        'Arm2'


    """

    binarizer: Optional[Callable] = None


@spock
class UCB1:
    """Upper Confidence Bound1 Learning Policy.

    This policy calculates an upper confidence bound for the mean reward of each arm.
    It greedily selects the arm with the highest upper confidence bound.

    The UCB for each arm is calculated as:

    .. math::
        UCB = \\mu_i + \\alpha \\times \\sqrt[]{\\frac{2 \\times log(N)}{n_i}}

    Where :math:`\\mu_i` is the mean for that arm,
    :math:`N` is the total number of trials, and
    :math:`n_i` is the number of times the arm has been selected.

    :math:`\\alpha` is a factor used to adjust how conservative the estimate is.
    Higher :math:`\\alpha` values promote more exploration.

    Attributes
    ----------
    alpha: Num
        The parameter to control the exploration.
        Integer of float. Cannot be negative.
        Default value is 1.

    Example
    -------
        >>> from mabwiser.mab import MAB, LearningPolicy
        >>> list_of_arms = ['Arm1', 'Arm2']
        >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
        >>> rewards = [20, 17, 25, 9]
        >>> mab = MAB(list_of_arms, LearningPolicy.UCB1(alpha=1.25))
        >>> mab.fit(decisions, rewards)
        >>> mab.predict()
        'Arm2'
    """

    alpha: float = 1.0

    def __post_hook__(self):
        try:
            ge(self.alpha, bound=0.0)
        except Exception as e:
            raise ValueError(
                f"`{self.__class__.__name__}` could not be instantiated -- spock message: {e}"
            )


ALL_LP = [
    Popularity,
    EpsilonGreedy,
    Random,
    Softmax,
    ThompsonSampling,
    UCB1,
    LinGreedy,
    LinTS,
    LinUCB,
]

ALL_LP_NAMES = [val.__name__ for val in ALL_LP]

LP = namedtuple("LP", ALL_LP_NAMES)

LearningPolicy = LP(
    *ALL_LP
)

