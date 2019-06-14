# Building the Package
To create a new wheel:
cd to the top level directory where setup.py is
python setup.py bdist_wheel
This will create a Python 3 wheel.

# Installing the Package on a New Machine
Download the wheel
cd to the directory with the wheel file
pip install mabwiser-<version>-py3-none-any.whl
The current version is 1.5.6

# About Sphinx Documentation
The Sphinx documentation is in the mabwiser/docs directory. Sphinx converts files written in reStructuredText to html, and can also extract docstrings from python code that are written in that format. In MABWiser we use the Napoleon extension to convert numpy docstrings to reStructuredText, but Napoleon will also convert the Google format.

# Updating the Documentation
Documentation is in reStructuredText format in the rst files in the docs directory. The files are:

index.rst: This contains the homepage and the table of contents.

To add a new page to the navigation, in index.rst under ..toctree:: add the file name without the .rst extension in the order in which you want it to appear in the menu. All top-level headers on the page will be added.

To add a new link to the list at the top of index.rst:

In the enumerated list format the link as :ref:`Title To Display<anchor>` where anchor is the keyword or phrase used to identify the link target
Insert a line above the header of the new page or section and add .. _anchor::

# Compiling the Documentation
The package dependencies for compiling the documentation are: sphinx, sphinx_rtd_theme, sphinx-autodoc-typehints (not used currently)

pip install sphinx

pip install sphinx.rtd_theme

When Sphinx extracts docstrings from the code, it looks at the local code only if the library is not in site packages and you don't have a wheel feel. If you have a wheel for MABWiser, you will need to rebuild the wheel every time you make changes. If you have installed it, you will also need to reinstall the wheel each time.

To compile the documentation:

cd to mabwiser/docs
If you have added or removed classes or modules you can manually update mabwiser.rst following the examples of the existing entries, OR:
rm mabwiser.rst 
sphinx-apidoc -o . ../mabwiser
In the new mabwiser.rst
Insert as the first line: .. _MABWiser API:
Change the page header to "MABWiser API" from "mabwiserpackage"
rm modules.rst
To compile run make html
The files will be in _build/html