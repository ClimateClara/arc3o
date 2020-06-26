"""
WORK IN PROGRESS
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="arc3o-ClimateClara", 
    version="0.0.1",
    author="Clara Burgard",
    author_email="clara.burgard@gmail.com",
    description="The Arctic Ocean Observation Operator for 6.9 GHz",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ClimateClara/arc3o",
    packages=setuptools.find_packages(),
    license='GPL-3.0',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)