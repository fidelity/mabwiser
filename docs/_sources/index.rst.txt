MABWiser Contextual Multi-Armed Bandits
=======================================

MABWiser is a research library for fast prototyping of multi-armed bandit algorithms.
It supports **context-free**, **parametric** and **non-parametric** **contextual** bandit models.
It provides built-in parallelization for both training and testing components and a simulation utility for algorithm comparisons and hyper-parameter tuning.
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
    api

Source Code
===========
The source code is hosted on `GitHub`_.

Bug Reports
===========

You can send feedback, bug reports and feature requests to mabwiser@fmr.com.

	   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

.. _GitHub: https://github.com/fmr-llc/mabwiser
.. _PEP-8 standards: https://www.python.org/dev/peps/pep-0008/
