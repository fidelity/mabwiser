MABWiser Contextual Multi-Armed Bandits
=======================================

MABWiser is a research library for fast prototyping of multi-armed bandit algorithms.
It supports **context-free**, **parametric** and **non-parametric** **contextual** bandit models.
It provides built-in parallelization for both training and testing components and
a simulation utility for algorithm comparisons and hyper-parameter tuning.
The library follows the scikit-learn style, adheres to `PEP-8 standards`_, and is tested heavily. 
MABWiser is released by Fidelity Investments Artificial Intelligence Center of Excellence.

Quick Start 
===========

.. code-block:: python

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

.. sidebar:: Contents

   .. toctree::
    :maxdepth: 2

    about
    installation
    quick
    examples
    contributing
    new_bandit
    api

Source Code
===========
The source code is hosted on `GitHub`_.

Available Bandit Policies
=========================

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

Bug Reports
===========

You can send feedback to mabwiser@fmr.com. Please use the GitHub `Issues`_ tracking for bug reports and feature requests.

Citation
========
You can cite MABWiser as:

E. Strong,  B. Kleynhans, and S. Kadioglu, "MABWiser: A Parallelizable Contextual Multi-Armed Bandit Library for Python," in 2019 IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI 2019) (pp.885-890). IEEE, 2019.

.. code-block:: bibtex

    @inproceedings{mabwiser2019,
      author    = {Strong, Emily and Kleynhans, Bernard and Kadioglu, Serdar},
      title     = {MABWiser: A Parallelizable Contextual Multi-Armed Bandit Library for Python},
      booktitle = {2019 IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI 2019)},
      year      = {2019},
      pages     = {885-890},
      organization = {IEEE},
      url       = {https://github.com/fmr-llc/mabwiser}
    }


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

.. _GitHub: https://github.com/fidelity/mabwiser
.. _PEP-8 standards: https://www.python.org/dev/peps/pep-0008/
.. _Issues: https://github.com/fidelity/mabwiser/issues
