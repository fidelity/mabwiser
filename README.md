# MABWiser: Parallelizable Contextual Multi-Armed Bandits 

MABWiser is a research library written in Python for rapid prototyping of multi-armed bandit algorithms.
It supports **context-free**, **parametric** and **non-parametric** **contextual** bandit models and provides built-in 
parallelization for both training and testing components. The library also provides a simulation utility for comparing 
different policies and performing hyper-parameter tuning. MABWiser follows a scikit-learn style public interface, adheres to 
[PEP-8 standards](https://www.python.org/dev/peps/pep-0008/), and is tested heavily. Full documentation is available at 
[fidelity.github.io/mabwiser](https://fidelity.github.io/mabwiser).

MABWiser is developed by the Artificial Intelligence Center of Excellence at Fidelity Investments.

## Quick Start

```python
# An example that shows how to use the UCB1 learning policy
# to choose between two arms based on their expected rewards.

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

## Available Bandit Policies

Available Learning Policies:
* Epsilon Greedy
* LinTS
* LinUCB
* Popularity
* Random
* Softmax
* Thompson Sampling (TS)
* Upper Confidence Bound (UCB1)

Available Neighborhood Policies: 
* Clusters
* K-Nearest
* Radius

## Installation

MABWiser can be installed from the wheel file or building from source by following the instructions in 
our [documentation](https://fidelity.github.io/mabwiser/installation.html).

## Support

Please submit bug reports and feature requests as [Issues](https://github.com/fidelity/mabwiser/issues).

For additional questions and feedback, please contact us at [mabwiser@fmr.com](mailto:mabwiser@fmr.com?subject=[Github]%20MABWiser%20Feedback).


## Citation

If you use MABWiser in a publication, please cite it as:

E. Strong,  B. Kleynhans, and S. Kadioglu, "MABWiser: A Parallelizable Contextual Multi-Armed Bandit Library for Python," in 2019 IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI 2019) (pp.885-890). IEEE, 2019.


```bibtex

    @inproceedings{mabwiser2019,
      author    = {Strong, Emily and Kleynhans, Bernard and Kadioglu, Serdar},
      title     = {MABWiser: A Parallelizable Contextual Multi-Armed Bandit Library for Python},
      booktitle = {2019 IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI 2019)},
      year      = {2019},
      pages     = {885-890},
      organization = {IEEE},
      url       = {https://github.com/fidelity/mabwiser}
    }
```

## License

MABWiser is licensed under the [Apache License 2.0](LICENSE).

<br>
