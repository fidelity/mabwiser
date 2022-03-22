# -*- coding: utf-8 -*-
from typing import Callable, List, Optional

from spock import spock
from spock.utils import within

from mabwiser.configs.constants import Backend
from mabwiser.configs.learning import LearningPolicy
from mabwiser.configs.neighborhood import NeighborhoodPolicy


@spock
class MABConfig:
    arms: List[str]
    learning_policy: LearningPolicy
    neighborhood_policy: Optional[NeighborhoodPolicy] = None
    seed: int = 123456
    n_jobs: int = 1
    backend: Optional[Backend] = None

    def __post_hook__(self):
        try:
            if len(self.arms) != len(set(self.arms)):
                ValueError("The list of arms cannot contain duplicate values.")
        except Exception as e:
            raise ValueError(
                f"`{self.__class__.__name__}` could not be instantiated -- spock message: {e}"
            )


@spock
class SimulatorConfig:
    """Configuration for the Simulator class

    Attributes:
        scaler: One of the scalers from sklearn.preprocessing. Optional.
        test_size: The fraction of data to use in the test set. Must be in the range (0, 1).
        is_ordered: Whether to divide the data randomly or to use the order given. When set to True, the test data will
            be the final n rows of the data set where n is determined by the split. When set to False, sklearn's
            train_test_split will be used.
        batch_size: The batch size to test before partial fitting during _online learning. Cannot exceed the size of
            the test set. When batch size is 0, the simulation will be offline.
        evaluator: Function for scoring the predictions. Must have the function signature function(arm_to_stats_train:
            dictionary, predictions: list, decisions: np.ndarray, rewards: np.ndarray, stat: str, start_index: int,
            nn: bool).
        seed: The seed for simulation
        is_quick: Flag to omit neighborhood statistics. Default value is False.
        log_file: The logfile to store debug output. Optional.
        log_format: The logger format used
    """
    scaler: Optional[Callable] = None
    test_size: float = 0.3
    is_ordered: bool = False
    batch_size: int = 0
    evaluator: Callable
    seed: int = 123456
    is_quick: bool = False
    log_file: str = None
    log_format: str = "%(asctime)s %(levelname)s %(message)s"

    def __post_hook__(self):
        try:
            within(self.test_size, low_bound=0.0, upper_bound=1.0)
        except Exception as e:
            raise ValueError(
                f"`{self.__class__.__name__}` could not be instantiated -- spock message: {e}"
            )
