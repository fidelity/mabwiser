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
* LinGreedy
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
* LSH Nearest
* Radius
* TreeBandit

Bug Reports
===========

Please use the GitHub `Issues`_ tracking for bug reports and feature requests.

Citation
========
You can cite MABWiser as:

* **[IJAIT 2021]** `E. Strong,  B. Kleynhans, and S. Kadioglu, "MABWiser: Parallelizable Contextual Multi-Armed Bandits" <https://www.worldscientific.com/doi/abs/10.1142/S0218213021500214>`_
* **[ICTAI 2019]** `E. Strong,  B. Kleynhans, and S. Kadioglu, "MABWiser: A Parallelizable Contextual Multi-Armed Bandit Library for Python" <https://ieeexplore.ieee.org/document/8995418>`_

.. code-block:: bibtex

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


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

.. _GitHub: https://github.com/fidelity/mabwiser
.. _PEP-8 standards: https://www.python.org/dev/peps/pep-0008/
.. _Issues: https://github.com/fidelity/mabwiser/issues
