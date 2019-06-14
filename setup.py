import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mabwiser",
    description="MABWiser: Contextual Multi-Armed Bandits Library",
    long_description=long_description,
    version="1.5.6",
    author="FMR LLC",
    author_email="mabwiser@fmr.com",
    url="https://github.com/fmr-llc/mabwiser",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Source": "https://github.com/fmr-llc/mabwiser",
        "Bug Reports": "mailto:mabwiser@fmr.com?subject=%5BMABWiser%5D%20Feedback%20&body=Feedback%20%26%20Feature%20Request%0A%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%0A%0AXXX%0A%0ABug%20Report%20%0A%3D%3D%3D%3D%3D%3D%3D%3D%3D%3D%0A%0AOS%3A%20XXX%0APython%20Environment%3A%20XXX%20%20%20%20%20%20%20%20%20%20%20%20%20%20%0ADescription%3A%20XXX%0ASteps%20to%20reproduce%3A%20XXX%0AExpected%20result%3A%20XXX%0AActual%20result%3A%20XXX%0A%0A%0APlease%20allow%20up%20to%201-2%20business%20days%20for%20a%20response.%20%0A%0AAtlas%20Team%0A"
    }
)
