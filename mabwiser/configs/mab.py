# -*- coding: utf-8 -*-
from typing import List, Optional

from spock import spock
from spock.utils import gt

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
