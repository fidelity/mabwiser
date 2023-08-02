[![ci](https://github.com/fidelity/mabwiser/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/fidelity/mabwiser/actions/workflows/ci.yml) [![PyPI version fury.io](https://badge.fury.io/py/mabwiser.svg)](https://pypi.python.org/pypi/mabwiser/) [![PyPI license](https://img.shields.io/pypi/l/mabwiser.svg)](https://pypi.python.org/pypi/mabwiser/) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) [![Downloads](https://static.pepy.tech/personalized-badge/mabwiser?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/mabwiser)

# MABWiser: Parallelizable Contextual Multi-Armed Bandits 

MABWiser ([IJAIT 2021](https://www.worldscientific.com/doi/10.1142/S0218213021500214), [ICTAI 2019](https://ieeexplore.ieee.org/document/8995418)) is a research library written in Python for rapid prototyping of multi-armed bandit algorithms. It supports **context-free**, **parametric** and **non-parametric** **contextual** bandit models and provides built-in parallelization for both training and testing components. 

The library also provides a simulation utility for comparing different policies and performing hyper-parameter tuning. MABWiser follows a scikit-learn style public interface, adheres to [PEP-8 standards](https://www.python.org/dev/peps/pep-0008/), and is tested heavily. 

MABWiser is developed by the Artificial Intelligence Center of Excellence at Fidelity Investments. Documentation is available at [fidelity.github.io/mabwiser](https://fidelity.github.io/mabwiser).

## Bandit-based Recommender Systems
To solve personalized recommendation problems, MABWiser is integrated into our [Mab2Rec library](https://github.com/fidelity/mab2rec). Mab2Rec enables building content- and context-aware recommender systems, whereby MABWiser helps selecting the next best item (arm).

## Bandit-based Large-Neighborhood Search
To solve combinatorial optimization problems, MABWiser is integrated into [Adaptive Large Neighborhood Search](https://github.com/N-Wouda/ALNS). The ALNS library enables building metaheuristics for complex optimization problems, whereby MABWiser helps selecting the next best destroy, repair operation (arm).  

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
* Epsilon Greedy [1, 2]
* LinGreedy [1, 2]
* LinTS [3]. See [11] for a formal treatment of reproducibility in LinTS
* LinUCB [4]
* Popularity [2]
* Random [2]
* Softmax [2]
* Thompson Sampling (TS) [5]
* Upper Confidence Bound (UCB1) [2]

Available Neighborhood Policies: 
* Clusters [6]
* K-Nearest [7, 8]
* LSH Nearest [9]
* Radius [7, 8]
* TreeBandit [10]

## Installation

MABWiser requires **Python 3.8+** and can be installed from PyPI using ``pip install mabwiser`` or by building from source as shown in [installation instructions](https://fidelity.github.io/mabwiser/installation.html).

## Support

Please submit bug reports and feature requests as [Issues](https://github.com/fidelity/mabwiser/issues).

## Citation

If you use MABWiser in a publication, please cite it as:

* **[IJAIT 2021]** [E. Strong,  B. Kleynhans, and S. Kadioglu, "MABWiser: Parallelizable Contextual Multi-Armed Bandits"](https://www.worldscientific.com/doi/abs/10.1142/S0218213021500214)
* **[ICTAI 2019]** [E. Strong,  B. Kleynhans, and S. Kadioglu, "MABWiser: A Parallelizable Contextual Multi-Armed Bandit Library for Python"](https://ieeexplore.ieee.org/document/8995418)

```bibtex
    @article{DBLP:journals/ijait/StrongKK21,
      author    = {Emily Strong and Bernard Kleynhans and Serdar Kadioglu},
      title     = {{MABWiser:} Parallelizable Contextual Multi-armed Bandits},
      journal   = {Int. J. Artif. Intell. Tools},
      volume    = {30},
      number    = {4},
      pages     = {2150021:1--2150021:19},
      year      = {2021},
      url       = {https://doi.org/10.1142/S0218213021500214},
      doi       = {10.1142/S0218213021500214},
    }

    @inproceedings{DBLP:conf/ictai/StrongKK19,
    author    = {Emily Strong and Bernard Kleynhans and Serdar Kadioglu},
    title     = {MABWiser: {A} Parallelizable Contextual Multi-Armed Bandit Library for Python},
    booktitle = {31st {IEEE} International Conference on Tools with Artificial Intelligence, {ICTAI} 2019, Portland, OR, USA, November 4-6, 2019},
    pages     = {909--914},
    publisher = {{IEEE}},
    year      = {2019},
    url       = {https://doi.org/10.1109/ICTAI.2019.00129},
    doi       = {10.1109/ICTAI.2019.00129},
    }
```

## License

MABWiser is licensed under the [Apache License 2.0](LICENSE).

## References

1. John Langford and Tong Zhang. The epoch-greedy algorithm for contextual multi-armed bandits
2. Volodymyr Kuleshov and Doina Precup. Algorithms for multi-armed bandit problems
3. Agrawal, Shipra and Navin Goyal. Thompson sampling for contextual bandits with linear payoffs
4. Chu, Wei, Li, Lihong, Reyzin Lev, and Schapire Robert. Contextual bandits with linear payoff functions
5. Osband, Ian, Daniel Russo, and Benjamin Van Roy. More efficient reinforcement learning via posterior sampling
6. Nguyen, Trong T. and Hady W. Lauw. Dynamic clustering of contextual multi-armed bandits
7. Melody Y. Guan and Heinrich Jiang, Nonparametric stochastic contextual bandits
8. Philippe Rigollet and Assaf Zeevi. Nonparametric bandits with covariates 
9. Indyk, Piotr, Motwani, Rajeev, Raghavan, Prabhakar, Vempala, Santosh. Locality-preserving hashing in multidimensional spaces
10. Adam N. Elmachtoub, Ryan McNellis, Sechan Oh, Marek Petrik, A practical method for solving contextual bandit problems using decision trees
11. Doruk Kilitcioglu, Serdar Kadioglu, Non-deterministic behavior of thompson sampling with linear payoffs and how to avoid it

<br>
