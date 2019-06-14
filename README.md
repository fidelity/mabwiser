# MABWiser: Contextual Multi-Armed Bandits 

MABWiser is a research library for fast prototyping of multi-armed bandit algorithms.
It supports **context-free**, **parametric** and **non-parametric** **contextual** bandit models.
It provides built-in parallelization for both training and testing components and a simulation utility 
for algorithm comparisons and hyper-parameter tuning.
The library follows the scikit-learn style, adheres to [PEP-8 standards](https://www.python.org/dev/peps/pep-0008/)
, and is tested heavily.

MABWiser is released by Fidelity Investments Artificial Intelligence Center of Excellence.

Available Learning Policies:
* Epsilon Greedy
* LinUCB
* Softmax
* Thompson Sampling (TS)
* Upper Confidence Bound (UCB1)


Available Neighborhood Policies: 
* Clusters
* K-Nearest
* Radius

## Documentation

The API Reference and Installation instructions can be found at TBD.

## Quick Start

```python
# An example that shows how to use the UCB1 learning policy
# to make decisions between two arms based on their expected rewards.

# Import MABWiser Library
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy

# Data
arms = ['Arm1', 'Arm2']
decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
rewards = [20, 17, 25, 9]

# Model 
mab = MAB(arms, LearningPolicy.UCB1(alpha=1.25))

# Train
mab.fit(decisions, rewards)

# Test
mab.predict()
```

## Usage Examples

See /examples folder for usage examples 
on context-free, contextual, parametric, non-parametric and customized bandit algorithms.

Conceptually, given a set of historical decisions and their corresponding rewards, 
the high-level idea behind MABWiser is to train a model using the `fit()` method to make predictions 
about next best decisions using the `predict()` method.

It is possible to retrieve the expected reward of each arm using the `predict_expectations()` method
and online training is available using the `partial_fit()` method.
New arms can be added to the bandits using the `add_arm()` method.
Decisions and rewards data support lists, 1D numpy arrays, and pandas series.
Contexts data supports 2D lists, 2D numpy arrays, pandas series and data frames.

## Bug Reports

You can send feedback, bug reports and feature requests to <a href="mailto:mabwiser@fmr.com?subject=%5BMABWiser%5D%20Feedback%20&body=Feedback%20%26%20Feature%20Request%0A%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%0A%0AXXX%0A%0ABug%20Report%20%0A%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%0A%0AOS%3A%20XXX%0APython%20Environment%3A%20XXX%20%20%20%20%20%20%20%20%20%20%20%20%20%20%0ADescription%3A%20XXX%0ASteps%20to%20reproduce%3A%20XXX%0AExpected%20result%3A%20XXX%0AActual%20result%3A%20XXX%0A%0A%0APlease%20allow%20up%20to%201-2%20business%20days%20for%20a%20response.%20%0A%0AAtlas%20Team%0A">mabwiser@fmr.com</a>.

<br>

© Copyright 2018, FMR LLC

