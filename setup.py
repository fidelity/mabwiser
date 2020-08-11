import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    required = fh.read().splitlines()

setuptools.setup(
    name="mabwiser",
    description="MABWiser: Parallelizable Contextual Multi-Armed Bandits Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="1.10.0",
    author="FMR LLC",
    url="https://github.com/fidelity/mabwiser",
    packages=setuptools.find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Documentation": "https://fidelity.github.io/mabwiser/",
        "Source": "https://github.com/fidelity/mabwiser"
    }
)
