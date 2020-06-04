.. _examples:

Usage Examples
==============

The examples below show how to use different bandit policies to make decisions among multiple arms based on their expected rewards.

Conceptually, given a set of historical decisions and their corresponding rewards,
the high-level idea behind MABWiser is to train a model using the ``fit()`` method to make predictions
about next best decisions using the ``predict()`` method.

It is possible to retrieve the expected reward of each arm using the ``predict_expectations()`` method and
online training is available using the ``partial_fit()`` method.
New arms can be added to the bandits using the ``add_arm()`` method.
Decisions and rewards data support lists, 1D numpy arrays, and pandas series.
Contexts data supports 2D lists, 2D numpy arrays, pandas series and data frames.

.. seealso:: Additional examples are available in the `examples folder`_ in the repo.

Context-Free MAB
----------------

.. code-block:: python
   
	from mabwiser.mab import MAB, LearningPolicy

    ######################################################################################
    #
    # MABWiser
    # Scenario: A/B Testing for Website Layout Design
    #
    # An e-commerce website experiments with 2 different layouts options for their homepage
    # Each layouts decision leads to generating different revenues
    #
    # What should the choice of layouts be based on historical data?
    #
    ######################################################################################

    # Arms
    options = [1, 2]

    # Historical data of layouts decisions and corresponding rewards
    layouts = [1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1]
    revenues = [10, 17, 22, 9, 4, 0, 7, 8, 20, 9, 50, 5, 7, 12, 10]

    ###################################
    # Epsilon Greedy Learning Policy
    ###################################

    # Epsilon Greedy learning policy with random exploration set to 15%
    greedy = MAB(arms=options,
                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15),
                 seed=123456)

    # Learn from previous layouts decisions and revenues generated
    greedy.fit(decisions=layouts, rewards=revenues)

    # Predict the next best layouts decision
    prediction = greedy.predict()

    # Expected revenues of each layouts learnt from historical data based on epsilon greedy policy
    expectations = greedy.predict_expectations()

    # Results
    print("Epsilon Greedy: ", prediction, " ", expectations)
    assert(prediction == 1)

    # Additional historical data becomes available which allows online learning
    additional_layouts = [1, 2, 1, 2]
    additional_revenues = [0, 12, 7, 19]

    # Online updating of the model
    greedy.partial_fit(additional_layouts, additional_revenues)

    # Adding a new layout option
    greedy.add_arm(3)


Parametric Contextual MAB
-------------------------

.. code-block:: python

	import pandas as pd
	from sklearn.preprocessing import StandardScaler

	from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy

	######################################################################################
	#
	# MABWiser
	# Scenario: Advertisement Optimization
	#
	# An e-commerce website needs to solve the problem of which ad to display to online users
	# Each advertisement decision leads to generating different revenues
	#
	# What should the choice of advertisement be given the context of an online user
	# based on customer data such as age, click rate, subscriber?
	#
	######################################################################################

	# Arms
	ads = [1, 2, 3, 4, 5]

	# Historical data of ad decisions with corresponding revenues and context information
	train_df = pd.DataFrame({'ad': [1, 1, 1, 2, 4, 5, 3, 3, 2, 1, 4, 5, 3, 2, 5],
							 'revenues': [10, 17, 22, 9, 4, 20, 7, 8, 20, 9, 50, 5, 7, 12, 10],
							 'age': [22, 27, 39, 48, 21, 20, 19, 37, 52, 26, 18, 42, 55, 57, 38],
							 'click_rate': [0.2, 0.6, 0.99, 0.68, 0.15, 0.23, 0.75, 0.17,
											0.33, 0.65, 0.56, 0.22, 0.19, 0.11, 0.83],
							 'subscriber': [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0]}
							 )

	# Test data to for new prediction
	test_df = pd.DataFrame({'age': [37, 52], 'click_rate': [0.5, 0.6], 'subscriber': [0, 1]})
	test_df_revenue = pd.Series([7, 13])

	# Scale the training and test data
	scaler = StandardScaler()
	train = scaler.fit_transform(train_df[['age', 'click_rate', 'subscriber']])
	test = scaler.transform(test_df)

	##################################################
	# Linear Upper Confidence Bound Learning Policy
	##################################################

	# LinUCB learning policy with alpha 1.25 and l2_lambda 1
	linucb = MAB(arms=ads, 
				 learning_policy=LearningPolicy.LinUCB(alpha=1.25, l2_lambda=1))

	# Learn from previous ads shown and revenues generated
	linucb.fit(decisions=train_df['ad'], rewards=train_df['revenues'], contexts=train)

	# Predict the next best ad to show
	prediction = linucb.predict(test)

	# Expectation of each ad based on learning from past ad revenues
	expectations = linucb.predict_expectations(test)

	# Results
	print("LinUCB: ", prediction, " ", expectations)
	assert(prediction == [5, 2])

	# Online update of model
	linucb.partial_fit(decisions=prediction, rewards=test_df_revenue, contexts=test)

	# Update the model with new arm
	linucb.add_arm(6)



Non-Parametric Contextual MAB
-----------------------------

.. code-block:: python

	import pandas as pd
	from sklearn.preprocessing import StandardScaler

	from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy

	######################################################################################
	#
	# MABWiser
	# Scenario: Advertisement Optimization
	#
	# An e-commerce website needs to solve the problem of which ad to display to online users
	# Each advertisement decision leads to generating different revenues
	#
	# What should the choice of advertisement be given the context of an online user
	# based on customer data such as age, click rate, subscriber?
	#
	######################################################################################

	# Arms
	ads = [1, 2, 3, 4, 5]

	# Historical data of ad decisions with corresponding revenues and context information
	train_df = pd.DataFrame({'ad': [1, 1, 1, 2, 4, 5, 3, 3, 2, 1, 4, 5, 3, 2, 5],
							 'revenues': [10, 17, 22, 9, 4, 20, 7, 8, 20, 9, 50, 5, 7, 12, 10],
							 'age': [22, 27, 39, 48, 21, 20, 19, 37, 52, 26, 18, 42, 55, 57, 38],
							 'click_rate': [0.2, 0.6, 0.99, 0.68, 0.15, 0.23, 0.75, 0.17,
											0.33, 0.65, 0.56, 0.22, 0.19, 0.11, 0.83],
							'subscriber': [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0]}
							)

	# Test data to for new prediction
	test_df = pd.DataFrame({'age': [37, 52], 'click_rate': [0.5, 0.6], 'subscriber': [0, 1]})
	test_df_revenue = pd.Series([7, 13])

	# Scale the training and test data
	scaler = StandardScaler()
	train = scaler.fit_transform(train_df[['age', 'click_rate', 'subscriber']])
	test = scaler.transform(test_df)

	########################################################
	# Radius Neighborhood Policy with UCB1 Learning Policy
	########################################################

	# Radius contextual policy with radius equals to 5 and ucb1 learning with alpha 1.25
	radius = MAB(arms=ads,
				 learning_policy=LearningPolicy.UCB1(alpha=1.25),
				 neighborhood_policy=NeighborhoodPolicy.Radius(radius=5))

	# Learn from previous ads shown and revenues generated
	radius.fit(decisions=train_df['ad'], rewards=train_df['revenues'], contexts=train)

	# Predict the next best ad to show
	prediction = radius.predict(test)

	# Expectation of each ad based on learning from past ad revenues
	expectations = radius.predict_expectations(test)

	# Results
	print("Radius: ", prediction, " ", expectations)
	assert(prediction == [4, 4])

	# Online update of model
	radius.partial_fit(decisions=prediction, rewards=test_df_revenue, contexts=test)

	# Updating of the model with new arm
	radius.add_arm(6)



Parallel MAB 
------------

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    from mabwiser.mab import MAB, LearningPolicy

    ######################################################################################
    #
    # MABWiser
    # Scenario: Playlist recommendation for music streaming service
    #
    # An online music streaming service wants to recommend a playlist to a user
    # based on a user's listening history and user features. There is a large amount
    # of data available to train this recommender model, which means the parallel
    # functionality in MABWiser can be useful.
    #
    #
    ######################################################################################

    # Seed
    seed = 111

    # Arms
    arms = list(np.arange(100))

    # Historical on user contexts and rewards (i.e. whether a user clicked
    # on the recommended playlist or not)
    contexts, rewards = make_classification(n_samples=100000, n_features=200,
                                            n_informative=20, weights=[0.01], scale=None)

    # Independently simulate the recommended playlist for each event
    decisions = np.random.choice(arms, size=100000)

    # Split data into train and test data sets
    contexts_train, contexts_test = train_test_split(contexts, test_size=0.3, random_state=seed)
    rewards_train, rewards_test = train_test_split(rewards, test_size=0.3, random_state=seed)
    decisions_train, decisions_test = train_test_split(decisions, test_size=0.3, random_state=seed)

    #############################################################################
    # Parallel Radius Neighborhood Policy with UCB1 Learning Policy using 8 Cores
    #############################################################################

    # Radius contextual policy with radius equals to 5 and ucb1 learning with alpha 1.25
    radius = MAB(arms=ads,
				 learning_policy=LearningPolicy.UCB1(alpha=1.25),
				 neighborhood_policy=NeighborhoodPolicy.Radius(radius=5),
				 n_jobs=8)
				 
    # Parallel Training
    # Learn from playlists shown and observed click rewards for each arm
    # In reality, we can scale the data --skipping this step in the toy example here
    radius.fit(decisions=decisions_train, rewards=rewards_train, contexts=contexts_train)

    # Parallel Testing
    # Predict the next best playlist to recommend
    prediction = radius.predict(contexts_test)

    # Results
    print("radius: ", prediction[:10])


Simulator
---------

.. code-block:: python

    import random
    from sklearn.preprocessing import StandardScaler
    from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
    from mabwiser.simulator import Simulator

    ######################################################################################
    #
    # MABWiser
    # Scenario: Hyper-Parameter Tuning using the built-in Simulator capability
    #
    ######################################################################################

    # Data
    size = 1000
    decisions = [random.randint(0, 2) for _ in range(size)]
    rewards = [random.randint(0, 1000) for _ in range(size)]
    contexts = [[random.random() for _ in range(50)] for _ in range(size)]

    # Bandits to simulate
    n_jobs = 2
    hyper_parameter_tuning = []
    for radius in range(6, 10):
        hyper_parameter_tuning.append(('Radius'+str(radius),
                                      MAB([0, 1], LearningPolicy.UCB1(1),
                                          NeighborhoodPolicy.Radius(radius),
                                          n_jobs=n_jobs)))

    # Simulator with given bandits and data
    # The parameters uses standard scaler,
    # Test split size set to 0.5
    # The split is not order dependent, i.e., random split
    # Online training with batch size 10, i.e., bandits will re-train at each batch
    # Offline training can be run with batch_size 0, i.e., no re-training during test phase
    sim = Simulator(hyper_parameter_tuning, decisions, rewards, contexts,
                    scaler=StandardScaler(), test_size=0.5, is_ordered=False, batch_size=10, seed=123456)

    # Run the simulator
    sim.run()

    # Save the results with a prefix
    sim.save_results("my_results_")

    # You can probe the fields of the simulator for other statisics
    for mab_name, mab in sim.bandits:
        print(mab_name + "\n")

        # Since the simulation is online, print the 'total' stats
        print('Worst Case Scenario:', sim.bandit_to_arm_to_stats_min[mab_name]['total'])
        print('Average Case Scenario:', sim.bandit_to_arm_to_stats_avg[mab_name]['total'])
        print('Best Case Scenario:', sim.bandit_to_arm_to_stats_max[mab_name]['total'], "\n\n")

    # Plot the average case results per every arm for each bandit
    sim.plot(metric='avg', is_per_arm=True)


.. seealso:: Additional examples are available in the `examples folder`_ in the repo. 

.. _examples folder: https://github.com/fidelity/mabwiser/tree/master/examples
