.. _installation:

Installation
============

.. admonition:: Installation Options

	There are two alternatives to install the library:

	1. pip install mabwiser
	2. Build from the source code
	
Requirements
------------

The library requires Python **3.6+**. The ``requirements.txt`` lists the necessary
packages. The following packages are used currently:

.. code-block:: python

	joblib
	numpy
	pandas
	scikit-learn>=0.22.0
	scipy
	seaborn>=0.9.0


Source Code
-----------

Alternatively, you can build a wheel package on your platform from scratch using the source code:

.. code-block:: python

	git clone https://github.com/fidelity/mabwiser.git
	cd mabwiser
	pip install setuptools wheel # if wheel is not installed
	python setup.py sdist bdist_wheel
	pip install dist/mabwiser-X.X.X-py3-none-any.whl

Test Your Setup
---------------
To confirm that cloning was successful, run the tests included in the project. 

All tests should pass.

.. code-block:: python

	git clone https://github.com/fidelity/mabwiser.git
	cd mabwiser
	python -m unittest discover tests

To confirm that installation was successful, try importing MABWiser in a Python shell or notebook. 

.. code-block:: python

	from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy

For examples of how to use the library, refer to :ref:`Usage Examples<examples>`.

Upgrade the Library
-------------------

To upgrade to the latest version of the library, run pip install --upgrade mabwiser.

If you are a custom build:
1. run ``git pull origin master`` in the repo folder
2. run python setup.py sdist bdist_wheel
3. run ``pip install --upgrade --no-cache-dir dist/mabwiser-X.X.X-py3-none-any.whl``.
