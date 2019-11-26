import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mabwiser",
    description="MABWiser: Parallelizable Contextual Multi-Armed Bandits Library",
    long_description=long_description,
    version="1.7.0",
    author="FMR LLC",
    url="https://github.com/fmr-llc/mabwiser",
    packages=setuptools.find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Documentation": "https://fmr-llc.github.io/mabwiser/",
        "Source": "https://github.com/fmr-llc/mabwiser"
    }
)
