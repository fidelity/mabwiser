# MABWiser: Parallelizable Contextual Multi-Armed Bandits 

MABWiser is a research library written in Python for rapid prototyping of multi-armed bandit algorithms.
It supports **context-free**, **parametric** and **non-parametric** **contextual** bandit models and provides built-in 
parallelization for both training and testing components. The library also provides a simulation utility for comparing 
different policies and performing hyper-parameter tuning. MABWiser follows a scikit-learn style public interface, adheres to 
[PEP-8 standards](https://www.python.org/dev/peps/pep-0008/), and is tested heavily.

MABWiser is developed by the Artificial Intelligence Center of Excellence at Fidelity Investments.

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

## Available Bandit Policies

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

## Installation

There are two alternatives to install the library: 

1. Install from the provided wheel package
2. Build from the source code 
	
### Requirements

The library requires Python **3.6+**. The ``requirements.txt`` lists the necessary
packages. The following packages are used currently:

```python
joblib
numpy
pandas
scikit-learn
scipy
seaborn>=0.9.0
```

### Install from wheel package

You can install the library from the provided wheel package using the following commands:

```bash
git clone https://github.com/fmr-llc/mabwiser.git 
cd mabwiser
pip install dist/mabwiser-X.X.X-py3-none-any.whl
```
Note: Don't forget to replace ``X.X.X`` with the current version number. 

### Install from source code

Alternatively, you can build a wheel package on your platform from scratch using the source code:

```bash
git clone https://github.com/fmr-llc/mabwiser.git
cd mabwiser
pip install setuptools wheel # if wheel is not installed
python setup.py bdist_wheel 
pip install dist/mabwiser-X.X.X-py3-none-any.whl
```

### Test Your Setup
To confirm that cloning the repo was successful, run the tests and all should pass.

```bash
git clone https://github.com/fmr-llc/mabwiser.git
cd mabwiser
python -m unittest discover -v tests
```

To confirm that installation was successful, import the library in Python shell or notebook. 

```python
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
```

### Upgrade the Library

To upgrade to the latest version of the library, run ``git pull origin master`` in the repo folder, 
and then run ``pip install --upgrade --no-cache-dir dist/mabwiser-X.X.X-py3-none-any.whl``.


## Usage Examples

See the [/examples](https://github.com/fmr-llc/mabwiser/tree/master/examples) folder for self-contained usage examples:

* [Content-Free Bandits](https://github.com/fmr-llc/mabwiser/blob/master/examples/context_free_mab.py)
* [Contextual Parametric Bandits](https://github.com/fmr-llc/mabwiser/blob/master/examples/parametric_mab.py)
* [Contextual Non-Parametric Bandits](https://github.com/fmr-llc/mabwiser/blob/master/examples/contextual_mab.py)
* [Customizing Bandits](https://github.com/fmr-llc/mabwiser/blob/master/examples/customized_mab.py)
* [Parallelization](https://github.com/fmr-llc/mabwiser/blob/master/examples/parallel_mab.py)
* [Hyper-Parameter Tuning](https://github.com/fmr-llc/mabwiser/blob/master/examples/simulator.py)

Conceptually, given a set of historical decisions and their corresponding rewards, 
the high-level idea behind MABWiser is to train a model using the `fit()` method, and then, predict the 
next best decisions using the `predict()` method.

Apart from the actual predictions, it is also possible to retrieve the expected reward of each arm using 
the `predict_expectations()` method. As time progresses and new decision and reward history becomes available, online 
training can be performed using the `partial_fit()` method. New arms can be introduced to the system dynamically using 
the `add_arm()` method. 

In terms of input data types, decisions and rewards data support lists, 1D numpy arrays, and pandas series while 
contexts data supports 2D lists, 2D numpy arrays, pandas series and data frames.

## Multi-Armed Bandits In a Nutshell

There are many real-world situations in which we have to decide between multiple options yet we are
only able to learn the best course of action by testing each option sequentially. 

[**Multi-Armed Bandit (MAB)** algorithms](https://en.wikipedia.org/wiki/Multi-armed_bandit) are suitable for such 
sequential, online decision making problems under uncertainty.
As such, they play an important role in many machine learning applications in internet advertising, recommendation 
engines, and clinical trials among many others.
In this setting, for each and every renewed decision we face an underlying question: 
Do we stick to what we know and receive an expected result ("**_exploit_**") or choose an option we do not know much 
about and potentially learn something new ("**_explore_**")?

**Problem Definition:** In a multi-armed bandits problem, the model of outcomes is unknown, and the outcomes can be 
deterministic
or stochastic. The agent needs to make a sequence of decisions in time *1, 2, ..., T*.
At each time *t* the agent is given a set of *K* arms, and it has to decide which arm to pull. 
After pulling an arm, it receives a *reward* of that arm, and the rewards of other arms are unknown. 
In a stochastic setting the reward of an arm is sampled from some unknown distribution. Situations exist in which we 
also observe side information at each time *t*. This side information is referred to as *context*. The arm that has the 
highest expected reward may be different given different contexts.
This variant is called **contextual multi-armed bandits**. Overall, the objective is to minimize _regret_, or 
equivalently, maximize the cumulative expected reward in the long run.

## Support
Please submit bug reports and feature requests as [Issues](https://github.com/fmr-llc/mabwiser/issues).

For additional questions and feedback, please contact us at [mabwiser@fmr.com](mailto:mabwiser@fmr.com?subject=[Github]%20MABWiser%20Feedback).



## License

MABWiser is licensed under the [Apache License 2.0](LICENSE.md).

<br>