# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""
This module defines the public interface of the **MABWiser Library**
"""

from typing import Dict, List, Optional, Tuple, Union

import attr
import numpy as np
import pandas as pd

from mabwiser._version import __author__, __copyright__, __email__, __version__
from mabwiser.configs.arm import ArmConfig, WarmStartConfig
from mabwiser.configs.calls import LPCall, NPCall
from mabwiser.configs.learning import SpecialLinearPolicy
from mabwiser.configs.mab import MABConfig
from mabwiser.utilities.converters import convert_array, convert_context
from mabwiser.utilities.random import create_rng
from mabwiser.utilities.validators import (
    check_false,
    check_fit_input,
    check_in_arms,
    check_len,
    check_true,
    validate_2d,
)

__author__ = __author__
__email__ = __email__
__version__ = __version__
__copyright__ = __copyright__


class MAB:
    """**MABWiser: Contextual Multi-Armed Bandit Library**

    MABWiser is a research library for fast prototyping of multi-armed bandit algorithms.
    It supports **context-free**, **parametric** and **non-parametric** **contextual** bandit models.

    Attributes
    ----------
    arms : list
        The list of all of the arms available for decisions. Arms can be integers, strings, etc.
    learning_policy : LearningPolicy
        The learning policy.
    neighborhood_policy : NeighborhoodPolicy
        The neighborhood policy.
    is_contextual : bool
        True if contextual policy is given, false otherwise. This is a read-only data field.
    seed : numbers.Rational
        The random seed to initialize the internal random number generator. This is a read-only data field.
    n_jobs: int
        This is used to specify how many concurrent processes/threads should be used for parallelized routines.
        Default value is set to 1.
        If set to -1, all CPUs are used.
        If set to -2, all CPUs but one are used, and so on.
    backend: str, optional
        Specify a parallelization backend implementation supported in the joblib library. Supported options are:
        - “loky” used by default, can induce some communication and memory overhead when exchanging input and
          output data with the worker Python processes.
        - “multiprocessing” previous process-based backend based on multiprocessing.Pool. Less robust than loky.
        - “threading” is a very low-overhead backend but it suffers from the Python Global Interpreter Lock if the
          called function relies a lot on Python objects.
        Default value is None. In this case the default backend selected by joblib will be used.

    Examples
    --------
        >>> from mabwiser.mab import MAB, LearningPolicy
        >>> arms = ['Arm1', 'Arm2']
        >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
        >>> rewards = [20, 17, 25, 9]
        >>> mab = MAB(arms, LearningPolicy.EpsilonGreedy(epsilon=0.25), seed=123456)
        >>> mab.fit(decisions, rewards)
        >>> mab.predict()
        'Arm1'
        >>> mab.add_arm('Arm3')
        >>> mab.partial_fit(['Arm3'], [30])
        >>> mab.predict()
        'Arm3'
        >>> from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
        >>> arms = ['Arm1', 'Arm2']
        >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1', 'Arm2']
        >>> rewards = [20, 17, 25, 9, 11]
        >>> contexts = [[0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 1]]
        >>> contextual_mab = MAB(arms, LearningPolicy.EpsilonGreedy(), NeighborhoodPolicy.KNearest(k=3))
        >>> contextual_mab.fit(decisions, rewards, contexts)
        >>> contextual_mab.predict([[1, 1, 0], [1, 1, 1], [0, 1, 0]])
        ['Arm2', 'Arm2', 'Arm2']
        >>> contextual_mab.add_arm('Arm3')
        >>> contextual_mab.partial_fit(['Arm3'], [30], [[1, 1, 1]])
        >>> contextual_mab.predict([[1, 1, 1]])
        'Arm3'
    """

    def __init__(self, config: MABConfig):
        # Map the config
        self._config = config
        # Create the random number generator
        self._rng = create_rng(config.seed)
        # Use a little enum magic to create the learning policy object
        self._lp = LPCall[self._config.learning_policy].value(
            rng=self._rng,
            arms=self._config.arms,
            n_jobs=self._config.n_jobs,
            backend=self._config.backend,
            **attr.asdict(self._config.learning_policy),
        )
        self._is_initial_fit = False
        self.is_contextual = False
        # Create the mab implementor
        if self._config.neighborhood_policy is not None:
            # Set contextual flag
            self.is_contextual = True
            # Do not use parallel fit or predict for Learning Policy when contextual
            self._lp.n_jobs = 1
            # Use a little enum magic to create the implementor object
            self._imp = NPCall[self._config.neighborhood_policy].value(
                lp=self._lp,
                rng=self._rng,
                arms=self._config.arms,
                n_jobs=self._config.n_jobs,
                backend=self._config.backend,
                **attr.asdict(self._config.neighborhood_policy),
            )
            # Validate that the combinations work
            self._validate_init()
        else:
            # If the learning policy is any of the linear implementations (Greedy, TS, or UCB) then set contextual
            self.is_contextual = SpecialLinearPolicy.has_value(
                self._config.learning_policy
            )
            # Map lp to imp
            self._imp = self._lp

    @property
    def config(self):
        return self._config

    @property
    def learning_policy(self):
        return self._config.learning_policy

    @property
    def neighborhood_policy(self):
        return self._config.neighborhood_policy

    @property
    def lp(self):
        return self._lp

    @property
    def lp_name(self):
        return type(self.lp).__name__

    @property
    def implementor(self):
        return self._imp

    @property
    def imp_name(self):
        return type(self.implementor).__name__

    @property
    def arms(self):
        return self._imp.arms

    def add_arm(self, arm: ArmConfig) -> None:
        """Adds an _arm_ to the list of arms.

        Incorporates the arm into the learning and neighborhood policies with no training data.

        Parameters
        ----------
        arm: ArmConfig
            The new arm to be added

        Returns
        -------
        None

        Raises
        ------
        TypeError:  For ThompsonSampling, binarizer must be a callable function.
        TypeError:  The standard scaler object must have a transform method.
        TypeError:  The standard scaler object must be fit with calculated ``mean_`` and ``var_`` attributes.

        ValueError: A binarizer function was provided but the learning policy is not Thompson Sampling.
        ValueError: The arm already exists.
        ValueError: The arm is ``None``.
        ValueError: The arm is ``NaN``.
        ValueError: The arm is ``Infinity``.
        """
        if arm in self.arms:
            raise ValueError("The arm is already in the list of arms.")
        if arm.binarizer is not None:
            check_true(
                LPCall.isinstance(self._imp, "ThompsonSampling")
                or LPCall.isinstance(self.lp, "ThompsonSampling"),
                TypeError(
                    "Learning policy must be Thompson Sampling to use a binarizer function."
                ),
            )
        self.lp.add_arm(arm)
        if self.lp is not self.implementor:
            self.implementor.add_arm(arm)

    def remove_arm(self, arm: str) -> None:
        """Removes an _arm_ from the list of arms.

        Parameters
        ----------
        arm: str
            The existing arm to be removed.

        Returns
        -------
        No return.

        Raises
        ------
        ValueError: The arm does not exist.
        ValueError: The arm is ``None``.
        ValueError: The arm is ``NaN``.
        ValueError: The arm is ``Infinity``.
        """
        if arm not in self.arms:
            raise ValueError("The arm is not in the list of arms.")
        self._imp.remove_arm(arm)

    def _validate_init(self) -> None:
        if NPCall.isinstance(self.implementor, "TreeBandit"):
            check_true(
                LPCall.isinstance(
                    self.lp, ("EpsilonGreedy", "UCB1", "ThompsonSampling")
                ),
                ValueError(
                    f"Tree-Bandit is not compatible with the learning policy `{self.lp_name}`"
                ),
            )

    def _validate_fit_args(
        self,
        decisions: Union[List[str], np.ndarray, pd.Series],
        rewards: Union[List[float], np.ndarray, pd.Series],
        contexts: Optional[
            Union[List[List[float]], np.ndarray, pd.Series, pd.DataFrame]
        ] = None,
    ) -> None:
        """
        Validates argument types for fit and partial_fit functions.
        """
        # Type check for decisions & rewards

        # Type check for decisions & rewards
        check_fit_input((decisions, rewards))
        # Type check for contexts -- don't use "if contexts" since it's n-dim array
        if contexts is not None:
            validate_2d(contexts, "contexts")
            # Sync contexts data with contextual policy
            check_true(
                self.is_contextual,
                TypeError(
                    "Fitting contexts data requires context policy or parametric learning policy."
                ),
            )
            # Make sure lengths of decisions and contexts match
            check_len(decisions, contexts, "decisions", "contexts")
        else:
            check_false(
                self.is_contextual,
                TypeError(
                    "Fitting contextual policy or parametric learning policy requires contexts data."
                ),
            )

        # Check that the decisions are actually within the arms
        check_in_arms(decisions, self.arms)
        # Length check for decisions and rewards
        check_len(decisions, rewards, "decisions", "rewards")
        # Thompson Sampling: works with binary rewards or requires function to convert non-binary rewards
        if (
            LPCall.isinstance(self.lp, "ThompsonSampling")
            and self.learning_policy.binarizer is None
        ):
            check_false(
                np.setdiff1d(rewards, [0, 0.0, 1, 1.0]).size,
                ValueError(
                    "Thompson Sampling requires binary rewards when binarizer function is not "
                    "provided."
                ),
            )

    def _validate_predict_args(
        self,
        contexts: Optional[
            Union[List[List[float]], np.ndarray, pd.Series, pd.DataFrame]
        ] = None,
    ) -> None:
        """
        Validates argument types for predict and predict_expectation functions.
        """

        # Context policy and context data should match
        if self.is_contextual:  # don't use "if contexts" since it's n-dim array
            check_true(
                contexts is not None,
                ValueError("Prediction with context policy requires context data."),
            )
            validate_2d(contexts, "contexts")
        else:
            check_true(
                contexts is None,
                ValueError(
                    "Prediction with no context policy cannot handle context data."
                ),
            )

    def _pre_fit(
        self,
        decisions: Union[List[str], np.ndarray, pd.Series],  # Decisions that are made
        rewards: Union[List[float], np.ndarray, pd.Series],  # Rewards that are received
        contexts: Optional[
            Union[List[List[float]], np.ndarray, pd.Series, pd.DataFrame]
        ] = None,  # Contexts, optional
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Validate arguments
        self._validate_fit_args(decisions, rewards, contexts)
        # Convert to numpy array for efficiency
        decisions = convert_array(decisions)
        rewards = convert_array(rewards)
        # Check rewards are valid
        check_true(
            np.isfinite(sum(rewards)),
            TypeError("Rewards cannot contain None, nan or infinity."),
        )
        # Convert contexts to numpy array for efficiency
        if contexts is not None:
            contexts = convert_context(
                lp=self.lp, imp=self._imp, contexts=contexts, decisions=decisions
            )
        return decisions, rewards, contexts

    def fit(
        self,
        decisions: Union[List[str], np.ndarray, pd.Series],  # Decisions that are made
        rewards: Union[List[float], np.ndarray, pd.Series],  # Rewards that are received
        contexts: Optional[
            Union[List[List[float]], np.ndarray, pd.Series, pd.DataFrame]
        ] = None,  # Contexts, optional
    ) -> None:
        """Fits the multi-armed bandit to the given *decisions*, their corresponding *rewards*
        and *contexts*, if any.

        Validates arguments and raises exceptions in case there are violations.

        This function makes the following assumptions:
            - each decision corresponds to an arm of the bandit.
            - there are no ``None``, ``Nan``, or ``Infinity`` values in the contexts.

        Parameters
        ----------
         decisions : Union[List[str], np.ndarray, pd.Series]
            The decisions that are made.
         rewards : Union[List[Num], np.ndarray, pd.Series]
            The rewards that are received corresponding to the decisions.
         contexts : Union[None, List[Num], List[List[Num]], np.ndarray, pd.Series, pd.DataFrame]
            The context under which each decision is made. Default value is ``None``, i.e., no contexts.

        Returns
        -------
        No return.

        Raises
        ------
        TypeError:  Decisions and rewards are not given as list, numpy array or pandas series.
        TypeError:  Contexts is not given as ``None``, list, numpy array, pandas series or data frames.

        ValueError: Length mismatch between decisions, rewards, and contexts.
        ValueError: Fitting contexts data when there is no contextual policy.
        ValueError: Contextual policy when fitting no contexts data.
        ValueError: Rewards contain ``None``, ``Nan``, or ``Infinity``.
        """
        # Call pre-fit
        decisions, rewards, contexts = self._pre_fit(
            decisions=decisions, rewards=rewards, contexts=contexts
        )
        # Call the fit method
        self._imp.fit(decisions, rewards, contexts)
        # Turn initial to true
        self._is_initial_fit = True

    def partial_fit(
        self,
        decisions: Union[List[str], np.ndarray, pd.Series],
        rewards: Union[List[float], np.ndarray, pd.Series],
        contexts: Union[
            None, List[List[float]], np.ndarray, pd.Series, pd.DataFrame
        ] = None,
    ) -> None:
        """Updates the multi-armed bandit with the given *decisions*, their corresponding *rewards*
        and *contexts*, if any.

        Validates arguments and raises exceptions in case there are violations.

        This function makes the following assumptions:
            - each decision corresponds to an arm of the bandit.
            - there are no ``None``, ``Nan``, or ``Infinity`` values in the contexts.

        Parameters
        ----------
         decisions : Union[List[str], np.ndarray, pd.Series]
            The decisions that are made.
         rewards : Union[List[Num], np.ndarray, pd.Series]
            The rewards that are received corresponding to the decisions.
         contexts : Union[None, List[Num], List[List[Num]], np.ndarray, pd.Series, pd.DataFrame]
            The context under which each decision is made. Default value is ``None``, i.e., no contexts.

        Returns
        -------
        No return.

        Raises
        ------
        TypeError:  Decisions, rewards are not given as list, numpy array or pandas series.
        TypeError:  Contexts is not given as ``None``, list, numpy array, pandas series or data frames.

        ValueError: Length mismatch between decisions, rewards, and contexts.
        ValueError: Fitting contexts data when there is no contextual policy.
        ValueError: Contextual policy when fitting no contexts data.
        ValueError: Rewards contain ``None``, ``Nan``, or ``Infinity``
        """
        # Call pre-fit
        decisions, rewards, contexts = self._pre_fit(
            decisions=decisions, rewards=rewards, contexts=contexts
        )
        # Call the fit or partial fit method
        if self._is_initial_fit:
            self._imp.partial_fit(decisions, rewards, contexts)
        else:
            self.fit(decisions, rewards, contexts)

    def predict(
        self,
        contexts: Optional[
            Union[List[List[float]], np.ndarray, pd.Series, pd.DataFrame]
        ] = None,
    ) -> Union[str, List[str]]:
        """Returns the "best" arm (or arms list if multiple contexts are given) based on the expected reward.

        The definition of the *best* depends on the specified learning policy.
        Contextual learning policies and neighborhood policies require contexts data in training.
        In testing, they return the best arm given new context(s).

        Parameters
        ----------
        contexts : Optional[Union[List[List[Num]], np.ndarray, pd.Series, pd.DataFrame]]
            The context under which each decision is made. Default value is None.
            Contexts should be ``None`` for context-free bandits and is required for contextual bandits.

        Returns
        -------
        The recommended arm or recommended arms list.

        Raises
        ------
        TypeError:  Contexts is not given as ``None``, list, numpy array, pandas series or data frames.

        ValueError: Predicting with contexts data when there is no contextual policy.
        ValueError: Contextual policy when predicting with no contexts data.
        """

        # Check that fit is called before
        check_true(self._is_initial_fit, Exception("Call fit before prediction"))
        # Validate arguments
        self._validate_predict_args(contexts)
        # Convert contexts to numpy array for efficiency
        if contexts is not None:
            contexts = convert_context(lp=self.lp, imp=self._imp, contexts=contexts)
        # Return the arm with the best expectation
        return self._imp.predict(contexts)

    def warm_start(self, warm_start_config: WarmStartConfig) -> None:
        """Warm-start untrained (cold) arms of the multi-armed bandit.

        Validates arguments and raises exceptions in case there are violations.

        The warm-start procedure depends on the learning and neighborhood policy. Note that for certain neighborhood
        policies (e.g., LSHNearest, KNearest, Radius) warm start can only be performed after the nearest neighbors
        have been determined in the "predict" step. Accordingly, warm start has to be executed for each context being
        predicted which is computationally expensive.

        Parameters
        ----------
        warm_start_config: WarmStartConfig
            Configuration for handling warm start conditions

        Returns
        -------
        No return.

        Raises
        ------
        TypeError:  Arm features are not given as a dictionary.
        TypeError:  Distance quantile is not given as a float.

        ValueError:  Distance quantile is not between 0 and 1.
        ValueError:  The arms in arm_to_features do not match arms.
        """
        check_true(
            set(self.arms) == set(warm_start_config.arm_to_features.keys()),
            ValueError("The arms in arm features do not match arms."),
        )
        self._imp.warm_start(
            warm_start_config.arm_to_features, warm_start_config.distance_quantile
        )

    def predict_expectations(
        self,
        contexts: Union[
            None, List[float], List[List[float]], np.ndarray, pd.Series, pd.DataFrame
        ] = None,  # Contexts, optional
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """Returns a dictionary of arms (key) to their expected rewards (value).

        Contextual learning policies and neighborhood policies require contexts data for expected rewards.

        Parameters
        ----------
        contexts : Union[None, List[Num], List[List[Num]], np.ndarray, pd.Series, pd.DataFrame]
            The context for the expected rewards. Default value is None.
            Contexts should be ``None`` for context-free bandits and is required for contextual bandits.

        Returns
        -------
        The dictionary of arms (key) to their expected rewards (value), or a list of such dictionaries.

        Raises
        ------
        TypeError:  Contexts is not given as ``None``, list, numpy array or pandas data frames.

        ValueError: Predicting with contexts data when there is no contextual policy.
        ValueError: Contextual policy when predicting with no contexts data.
        """

        # Check that fit is called before
        check_true(self._is_initial_fit, Exception("Call fit before prediction"))
        # Validate arguments
        self._validate_predict_args(contexts)
        # Convert contexts to numpy array for efficiency
        if contexts is not None:
            contexts = convert_context(lp=self.lp, imp=self._imp, contexts=contexts)
        # Return a dictionary from arms (key) to expectations (value)
        return self._imp.predict_expectations(contexts)
