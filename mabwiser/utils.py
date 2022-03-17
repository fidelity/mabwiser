# -*- coding: utf-8 -*-
# SPDX-License-Identifer: Apache-2.0

"""
This module provides a number of constants and helper functions.
"""

import abc
from typing import Dict, Union, Iterable, NamedTuple, Tuple, NewType, NoReturn, List

import numpy as np

Arm = NewType('Arm', Union[int, float, str])
"""Arm type is defined as integer, float, or string."""

Num = Union[int, float]
"""Num type is defined as integer or float."""


class Constants(NamedTuple):
    """
    Constant values used by the modules.
    """

    default_seed = 123456
    """The default random seed."""

    distance_metrics = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice",
                        "euclidean", "hamming", "jaccard", "kulsinski", "mahalanobis", "matching", "minkowski",
                        "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean"]
    """The distance metrics supported by neighborhood policies."""


def argmax(dictionary: Dict[Arm, Num]) -> Arm:
    """
    Returns the first key with the maximum value.
    """
    return max(dictionary, key=dictionary.get)


def argmin(dictionary: Dict) -> Arm:
    """
    Returns the first key that has the minimum value.
    """
    return min(dictionary, key=dictionary.get)


def check_false(expression: bool, exception: Exception) -> NoReturn:
    """
    Checks that given expression is false, otherwise raises the given exception.
    """
    if expression:
        raise exception


def check_true(expression: bool, exception: Exception) -> NoReturn:
    """
    Checks that given expression is true, otherwise raises the given exception.
    """
    if not expression:
        raise exception


def reset(dictionary: Dict, value) -> NoReturn:
    """
    Maps every key to the given value.
    """
    dictionary.update({}.fromkeys(dictionary, value))


class _BaseRNG(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, seed: int):
        """ Random Number Generator (RNG) with the given seed"""
        self.seed = seed
        self.rng = None

    @abc.abstractmethod
    def rand(self, size=None):
        """ Return return values in range [0, 1) with a given shape.

            Parameters
            ----------
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
                ``m * n * k`` samples are drawn.  Default is None, in which case a
                single value is returned.

            Returns
            -------
            out : Array of random floats of shape size (unless size=None, in which case a single float is returned).
        """
        pass

    @abc.abstractmethod
    def randint(self, low: int, high: int = None, size=None):
        """ Return random integers from low (inclusive) to high (exclusive).
            Return random integers from the “discrete uniform” distribution
            in the “half-open” interval [low, high).
            If high is None (the default), then results are from [0, low).

            Parameters
            ----------
            low : int or array-like of ints
                Lowest (signed) integers to be drawn from the distribution (unless
                ``high=None``, in which case this parameter is one above the
                *highest* such integer).
            high : int or array-like of ints, optional
                If provided, one above the largest (signed) integer to be drawn
                from the distribution (see above for behavior if ``high=None``).
                If array-like, must contain integer values
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
                ``m * n * k`` samples are drawn.  Default is None, in which case a
                single value is returned.

            Returns
            -------
            out : int or ndarray of ints
                `size`-shaped array of random integers from the appropriate
                distribution, or a single such random int if `size` not provided.
        """
        pass

    @abc.abstractmethod
    def choice(self, a: Union[int, Iterable[int]], size: Union[int, Tuple[int]] = None, p: Iterable[float] = None):
        """ Return a random sample from a given 1-D array
            Based on the probabilities associated with each entry in the array

            Parameters
            ----------
            a : 1-D array-like or int
                If an ndarray, a random sample is generated from its elements.
                If an int, the random sample is generated as if a were np.arange(a)
            size : int or tuple of ints, optional
                    Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
                    ``m * n * k`` samples are drawn.  Default is None, in which case a
                    single value is returned.
            p : 1-D array-like, optional
                The probabilities associated with each entry in options.

            Returns
            -------
            samples : single item or ndarray
                The generated random samples
        """

        pass

    @abc.abstractmethod
    def beta(self, alpha: int, beta: int, size=None):
        """ Return a sample from a Beta distribution.

            Parameters
            ----------
            alpha : float or array_like of floats
                    Alpha, positive (>0).
            beta : float or array_like of floats
                    Beta, positive (>0).
            size : int or tuple of ints, optional
                    Output shape.
                    If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
                    If size is None (default), a single value is returned if a and b are both scalars.
                    Otherwise, np.broadcast(a, b).size samples are drawn.

            Returns
            -------
            out : scalar or ndarray
               Drawn samples from the parameterized beta distribution.
        """
        pass

    @abc.abstractmethod
    def standard_normal(self, size=None):
        """ Draw samples from a standard Normal distribution (mean=0, stdev=1).

            Parameters
            ----------
            size : int or tuple of ints
                Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
                ``m * n * k`` samples are drawn.

            Returns
            -------
            out : ndarray
                A floating-point array of shape ``size`` of drawn samples
        """
        pass

    def multivariate_normal(self, mean: List[float], covariance: List[List[float]], size=None):
        """ Draw samples from a multivariate Normal distribution with given mean and covariance.

            Parameters
            ----------
            mean : list of floats
                The mean of each random variable, of length ``N``.
            covariance : list of list of floats
                The covariance of each random variable. If the length of parameter ``mean``
                is ``N``, this parameter should contain ``N`` lists, each with ``N`` floats.
            size : int or tuple of ints or None
                Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
                ``m * n * k * N`` samples are drawn, where ``N`` is the length of
                parameter ``mean``. If ``None``, a vector of length ``N`` is returned.

            Returns
            -------
            out : ndarray
                A floating-point array of shape ``m * n * k * N`` of drawn samples based on
                the ``size`` and the length of ``mean`` parameters.
        """
        pass

    @abc.abstractmethod
    def dirichlet(self, alpha: List[float], size=None):
        """ Draw samples from the Dirichlet distribution.

            Parameters
            ----------
            alpha : list of floats
                Parameter of the distribution (length k)
            size: int or tuple of ints or None
                Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
                ``m * n * k * N`` samples are drawn, where ``N`` is the length of
                parameter ``mean``. If ``None``, a vector of length ``N`` is returned.

            Returns
            -------
            samples: ndarray
                The drawn samples, of shape (size, k).
        """
        pass


class _NumpyRNG(_BaseRNG):

    def __init__(self, seed):
        super().__init__(seed)
        self.rng = np.random.default_rng(self.seed)

    def rand(self, size=None):
        return self.rng.random(size)

    def randint(self, low: int, high: int = None, size: int = None):
        return self.rng.integers(low=low, high=high, size=size)

    def choice(self, a: Union[int, Iterable[int]], size: Union[int, Tuple[int]] = None, p: Iterable[float] = None):
        return self.rng.choice(a=a, size=size, p=p)

    def beta(self, num_success: int, num_failure: int, size=None):
        return self.rng.beta(num_success, num_failure, size)

    def standard_normal(self, size=None):
        return self.rng.standard_normal(size)

    def multivariate_normal(self, mean: Union[np.ndarray, List[float]],
                            covariance: Union[np.ndarray, List[List[float]]], size=None):
        return np.squeeze(self.rng.multivariate_normal(mean, covariance, size=size, method='cholesky'))

    def dirichlet(self, alpha: List[float], size=None):
        return self.rng.dirichlet(alpha, size)


def create_rng(seed: int) -> _BaseRNG:
    """ Returns an rng object

        Parameters
        ----------
        seed : int
            the seed of the rng

        Returns
        -------
        out : _BaseRNG
            An rng object that implements the base rng class
    """
    return _NumpyRNG(seed)
