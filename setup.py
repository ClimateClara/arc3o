"""
Created on 03.07.2020

THIS DOCUMENT IS WORK IN PROGRESS

Based on: https://github.com/pypa/sampleproject
@author: Clara Burgard, clara.burgard@gmail.com
    Copyright (C) {2020}  {Clara Burgard}
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import setuptools

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setuptools.setup(
    name="arc3o-ClimateClara", 
    version="0.1",
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