.. _contributing:

Contributing
============

We are to happy to have you contribute! The project is hosted on `GitHub`_ and the easiest way to get started is to check out the "Issues" sections of the repo.

In general, our main requirements for contribution are:

* **Clean public API** New functionality should be coherent with the existing API design.
* **Contributions do not affect the default functionality of existing methods.** We want to make sure any new features or modifications still work with older code samples. All previous tests should pass.
* **Contributions should be accompanied by unit tests to ensure basic functionality and adhere to PEP-8 standards if applicable.** Tests are contained in the ``./tests/`` directory. 
* **Contributions are accompanied by documentation.** MABWiser uses the `numpydoc`_ standard.

.. _GitHub: https://github.com/fidelity/mabwiser
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/