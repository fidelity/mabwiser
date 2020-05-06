.. _quick:

Quick Start 
===========

.. code-block:: python

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
