.. _new_bandit:

Adding a New Bandit
===================

In this section, we provide high-level guidelines on how to introduce a new bandit algorithm in MABWiser.

.. admonition:: High-Level Overview

	Adding a new bandit algorithm to MABWiser consists of three main steps:

	1. Exposing the new bandit policy within the Public API
	2. Developing the underlying bandit algorithm 
	3. Testing the behavior of the bandit policy
	
	These steps can be followed by a Pull Request to include your new policy in the MABWiser library. In the following, the details of each of step are provided.

1. Exposing the Public API
--------------------------

Imagine you would like to introduce a new bandit algorithm, called ``MyCoolPolicy``, with an hyper-parameter ``my_parameter``.

First and foremost, the users of MABWiser need to be able to access your cool bandit algorithm.
This is how it would look like in a usage example. Notice how the ``mab`` model is created with your new policy.

.. code-block:: python

	# Import MABWiser Library
	from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy

	# Data
	arms = ['Arm1', 'Arm2']
	decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
	rewards = [20, 17, 25, 9]

	# Model 
	mab = MAB(arms, LearningPolicy.MyCoolPolicy(my_parameter=42))

	# Train
	mab.fit(decisions, rewards)

	# Test
	mab.predict()

To enable public access to your bandit policy, you need the the following changes in ``mab.py``:

.. important:: Make sure to complete the following steps in a new feature branch created via
 ```git checkout -b mycoolpolicy```. Later on, you can create a pull request from this branch
 to make your contributions part of the library.

a. Introduce the new bandit algorithm as an inner namedtuple class as part of the ``LearningPolicy`` class. See existing bandit policies as an example.
b. The parameter ``my_parameter`` will be a class member of this new inner class. Make sure to add type hinting. If possible, set a default value.
c. Implement the ``_validate()`` function for error checking on input values. If this raises errors, please document those in the ``mab`` constructor.
d. Add pydoc string documentation within this inner class to provide users a description of your new bandit policy. You can even use ``.. math::`` to express formulas.
e. Create a doctest based on a usage example with you new policy, like the one above. See other doctests in existing policies.
f. The same idea applies if you were introducing a new ``NeighborhoodPolicy``.

You now have an entry point to your new bandit algorithm.
The next step is to connect this bandit with an *implementor* object in the constructor of the ``MAB`` class:

a. Go to the constructor of the ``MAB`` class.
b. Set the value of the ``lp`` to your internal implementor class, in this case ``_MyCoolPolicy``.
c. Pass down the parameter ``my_parameter`` to the internal implementor object.
d. Make sure to update the decorator ``@property`` so that we can return the ``learning_policy`` back to the user.
e. In the ``_validate_mab_args`` function, register your new policy as a valid bandit to pass input validation.
f. The same idea applies if you were introducing a new ``NeighborhoodPolicy``.

**Congratulations!!** You can now increment the version of library ``__version__``.
Next, let's move on to the implementation phase of your cool bandit algorithm!

2. Implementing the Bandit Algorithm
------------------------------------

The previous section allowed users to access your new cool bandit policy.
What remains is to implement the learning algorithm behind your bandit policy.

Start by creating a new Python file named ``mycoolpolicy.py`` under the /mabwiser folder
to implement a class called ``_MyCoolPolicy``. This is where the bandit implementation will live.

.. important:: In Python, the prefix ``_`` in the class name denotes a private class. That is, the users do not need to directly access
this implementor class. Instead, they work with an immutable namedtuple object as shown in the usage example above.

Here is, at a high-level, what you need to implement in your bandit policy:

.. code-block:: python

	class _MyCoolPolicy(BaseMAB):
        # Your new bandit class will most likely inherit from the abstract BaseMAB class.
        # The BaseMAB is an abstract meta class which defines the public interface for all bandit algorithms.

        # The BaseMAB dictates the function signatures of core bandit operations such as:
        # fit(), partial_fit(), fit_arm() -- these are used during Training
        # predict() and predict_expectations(), _predict_contexts() -- these are used during Testing
        # and _uptake_new_arm() -- this is used as the system evolves and new arms emerge.

        # In case your new bandit policy is similar to an existing algorithm
        # it can take advantage of its implementation.
        # See for example how the Popularity bandit inherits
        # from the Greedy bandit and leverages from its training methods.

        def __init__(self, rng: np.random.RandomState, arms: List[Arm], n_jobs: int, backend: Optional[str]):
            # The BaseMAB provides every bandit policy with:
            #   - rng: a random number generator, in case it is needed
            #   - arms: the list of arms
            #   - arm_to_expectation: the dictionary that stores the expectation of each arm
            super().__init__(rng, arms, n_jobs, backend)

            # TODO:
            # Decide what other fields your new policy might need to calculate its expectations.
            # Declare those fields here as class members in your constructor.
            # For example, the greedy policy needs a counter and the total sum for each arm.
            # These fields are declared here and initialized to zero.
            self.my_value_to_arm = dict.fromkeys(self.arms, 0)

        def fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:
            # TODO:
            # This method trains your algorithm from scratch each time its called.
            # You might need to reset the internal fields
            # So that we can train from scratch with new data.
            reset(self.my_value_to_arm, 0)

            # Call _parallel_fit() here from the base class.
            # This automatically activates parallelization in the training phase.
            self._parallel_fit(decisions, rewards, contexts)

        def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:
            # This method trains your algorithm in a continuous fashion.
            # Unlike fit() operation, the partial_fit() does not reset internal fields typically.
            # This allows us to continue learning online

            # Call _parallel_fit() here from the base class.
            # This automatically activates parallelization in the training phase.
            self._parallel_fit(decisions, rewards, contexts)

        def predict(self, contexts: np.ndarray = None) -> Arm:
            # TODO:
            # This method returns the best arm to the user according to your policy.
            # It bases its decision on arm_to_expectation which is calculated in the _fit_arm() method.
            best_arm = ... # magic goes here

            return best_arm

        def predict_expectations(self, contexts: np.ndarray = None) -> Dict[Arm, Num]:
            # This method returns a copy of the expectations dictionary.
            # Make sure to return a copy of the internal object,
            # so that the user cannot accidentally break your policy.
            return self.arm_to_expectation.copy()

        def _fit_arm(self, arm: Arm, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None):
            # TODO:
            # This is the MOST IMPORTANT function to implement.
            # This method is for the algorithm behind your bandit policy on how it trains for each arm.
            # Based on the given input decisions and rewards,
            # This function calculates arm_to_expectation
            self.arm_to_expectation = ... # magic goes here

        def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                              seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:
            pass

        def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None):
            # TODO:
            # This method is called when add_arm() method is used to introduce new arms.
            # if you have declared additional fields in the constructor
            # Make sure that the new arms has these fields too
            self.my_value_to_arm[arm] = 0

**Congratulations!!** You have now implemented your cool new bandit policy. Next, let's move onto running this for real!

3. Testing the Bandit Algorithm
-------------------------------

The previous sections introduced the new bandit algorithm to the public API and implemented the underlying policy. 
What remains to be seen is to use this new algorithm and assess how it performs in action.

Start by creating a new Python file called ``test_mycoolbandit.py`` under the /tests folder to implement a class called ``MyCoolBanditTest``.
This class inherits from the ``BaseTest`` class which extends the ``unittest`` framework.

This is where we will implement unit tests to make sure our new bandit policy performs as expected.
Every test starts with the ``test_`` prefix followed by some descriptive name.

.. code-block:: python

    from tests.test_base import BaseTest

    class PopularityTest(BaseTest):

        # First, implement a simple case using the Public API you created in the first section
        # Utilize the self.predict() utility wrapper method from base test to create test cases quickly
        # When is_predict flag is set to True it returns the predicted arm
        def test_simple_usecase_arm(self):
            arm, mab = self.predict(arms=[1, 2],
                                    decisions=[1, 1, 1, 2, 2, 2],
                                    rewards=[1, 1, 1, 1, 1, 1],
                                    learning_policy=LearningPolicy._MyCoolPolicy(),
                                    seed=123456,
                                    num_run=1,
                                    is_predict=True)

            # Assert the predicted arm
            self.assertEqual(arm, 1)

        # When is_predict flag is set to False it returns the arm_to_prediction
        def test_simple_usecase_expectation(self):
            exp, mab = self.predict(arms=[1, 2],
                                    decisions=[1, 1, 1, 2, 2, 2],
                                    rewards=[1, 1, 1, 1, 1, 1],
                                    learning_policy=LearningPolicy._MyCoolPolicy(),
                                    seed=123456,
                                    num_run=1,
                                    is_predict=False)

            # Assert the arm expectations
            self.assertDictEqual({1: 0, 2:0}, exp)

        def test_zero_rewards(self):
            # Test zero/negative rewards

        def test_my_parameter(self):
            # Test how you parameters such as my_parameter
            # effect the behavior of your policy

        def test_within_neighborhood_policy(self):
            # Test your new learning policy within a
            # neighborhood policy when contexts are available.

        def test_fit_twice(self):
            # Test for two successive fit operation
            # Assert that training from scratch is done properly

        def test_partial_fit(self):
            # Test for one fit operation followed by partial_fit operation
            # Assert that online training is done properly

        def test_unused_arm(self):
            # Test the case when an arm remains unused
            # Or when an arm has no corresponding decision or reward

        def test_add_new_arm(self):
            # Test adding a new arm and assert that it is handled properly

        def test_parallelization(self):
            # Test how parallelization behaves for your new bandit using the n_jobs param

        def test_input_types(self):
            # Test different input types such as
            # strings for arms, data series or numpy arrays for decisions and rewards

To strengthen your test suite, consider other unittests with different number of arms,
decisions and rewards to assert that your bandit behaves correctly.

**Congratulations!!** You are now ready to share your new cool policy with everyone. Next, let's send a pull request for code review.

4. Sending a Pull Request
-------------------------

The previous sections finalized the implementation of your cool new policy.
This is no small step and deserves its own victory dance!
Now it is time to share it with everyone in the world by sending a pull request so that your code can be merged to the master branch.

Preparing a pull request typically involves the following steps:

a. Add a note about your changes in the CHANGELOG.txt.
b. Update the library version. You can use a keyword search for "version" to make sure you cover all version fields.
c. Update the README.md, in necessary.
d. Update the documentation rst files under the /docsrc folder , if necessary.
e. If you update any documentation, make sure to recompile the docs by running ``make github`` under the /docsrc folder.
f. Build a new wheel package and remove the old one in /dist folder. See installation page on how to build a package.

**Congratulations!!** You are ready to send a Pull Request and include your changes in the MABWiser library.
How cool is that? :)

.. _GitHub: https://github.com/fidelity/mabwiser
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/
