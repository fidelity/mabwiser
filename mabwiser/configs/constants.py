# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
from enum import Enum


class DistanceMetrics(Enum):
    """Enum for distance metric choices"""

    braycurtis = "braycurtis"
    canberra = "canberra"
    chebyshev = "chebyshev"
    cityblock = "cityblock"
    correlation = "correlation"
    cosine = "cosine"
    dice = "dice"
    euclidean = "euclidean"
    hamming = "hamming"
    jaccard = "jaccard"
    kulsinski = "kulsinski"
    mahalanobis = "mahalanobis"
    matching = "matching"
    minkowski = "minkowski"
    rogerstanimoto = "rogerstanimoto"
    russellrao = "russellrao"
    seuclidean = "seuclidean"
    sokalmichener = "sokalmichener"
    sokalsneath = "sokalsneath"
    sqeuclidean = "sqeuclidean"
    yule = "yule"
