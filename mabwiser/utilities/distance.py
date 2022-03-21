# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, List

import numpy as np
from scipy.spatial.distance import cdist


def get_arm_distances(
    from_arm: str,
    arm_to_features: Dict[str, List[float]],
    metric: str = "cosine",
    self_distance: int = 999999,
) -> Dict[str, float]:
    """
    Calculates the distances of the given from_arm to all the arms.

    Distances calculated based on the feature vectors given in arm_to_features using the given distance metric.
    The distance of the arm to itself is set as the given self_distance.

    Parameters
    ---------
    from_arm: Arm
        Distances from this arm.
    arm_to_features: Dict[Arm, list[Num]]
        Features for each arm used to calculate distances.
    metric: str
        Distance metric to use.
        Default value is 'cosine'.
    self_distance: int
        The value to set as the distance to itself.
        Default value is 999999.

    Returns
    -------
    Returns distance from given arm to arm v as arm_to_distance[v].
    """

    # Find the distance of given from_arm to all arms including self
    arm_to_distance = {}
    for to_arm in arm_to_features.keys():
        if from_arm == to_arm:
            arm_to_distance[to_arm] = self_distance
        else:
            arm_to_distance[to_arm] = cdist(
                np.asarray([arm_to_features[from_arm]]),
                np.asarray([arm_to_features[to_arm]]),
                metric=metric,
            )[0][0]

            # Cosine similarity can be nan when a feature vector is all-zeros
            if np.isnan(arm_to_distance[to_arm]):
                arm_to_distance[to_arm] = self_distance

    return arm_to_distance


def get_pairwise_distances(
    arm_to_features: Dict[str, List[float]],
    metric: str = "cosine",
    self_distance: int = 999999,
) -> Dict[str, Dict[str, float]]:
    """
    Calculates the distances between each pair of arms.

    Distances calculated based on the feature vectors given in arm_to_features using the given distance metric.
    The distance of the arm to itself is set as the given self_distance.

    Parameters
    ---------
    arm_to_features: Dict[Arm, list[Num]]
        Features for each arm used to calculate distances.
    metric: str
        Distance metric to use.
        Default value is 'cosine'.
    self_distance: int
        The value to set as the distance to itself.
        Default value is 999999.

    Returns
    -------
    Returns the distance between two arms u and v as distance_from_to[u][v].
    """

    # For every arm, calculate its distance to all arms including itself
    distance_from_to = {}
    for from_arm in arm_to_features.keys():
        distance_from_to[from_arm] = get_arm_distances(
            from_arm, arm_to_features, metric, self_distance
        )
    return distance_from_to


def get_distance_threshold(
    distance_from_to: Dict[str, Dict[str, float]],
    quantile: float,
    self_distance: int = 999999,
) -> float:
    """
    Calculates a threshold for doing warm-start conditioned on minimum pairwise distances of arms.

    Parameters
    ---------
    distance_from_to: Dict[Arm, Dict[Arm, Num]]
        Dictionary of pairwise distances from arms to arms.
    quantile: Num
        Quantile used to compute threshold.
    self_distance: int
        The distance of arm to itself.
        Default value is 999999.

    Returns
    -------
    A threshold on pairwise distance for doing warm-start.
    """

    closest_distances = []
    for arm, arm_to_distance in distance_from_to.items():

        # Get distances from one arm to others
        distances = [distance for distance in arm_to_distance.values()]

        # Get the distance to closest arm (if not equal to self_distance)
        if min(distances) != self_distance:
            closest_distances.append(min(distances))

    # Calculate threshold distance based on quantile
    threshold = np.quantile(closest_distances, q=quantile)

    return threshold
