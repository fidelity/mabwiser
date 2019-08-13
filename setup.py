import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mabwiser",
    description="MABWiser: Parallelizable Contextual Multi-Armed Bandits Library",
    long_description=long_description,
    version="1.6.0",
    author="FMR LLC",
    url="https://github.com/fmr-llc/mabwiser",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Source": "https://github.com/fmr-llc/mabwiser"
    }
)
