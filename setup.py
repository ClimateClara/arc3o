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
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')


setuptools.setup(
	
	#The project's name
    name='arc3o',
    
    #The project's version 
    version='0.1',
    
    #The project's metadata
    author='Clara Burgard',
    author_email='clara.burgard@gmail.com',
    description='An observation operator for the Arctic Ocean for 6.9 GHz',
    long_description=long_description,
    
    #The project's main homepage.
    url='https://github.com/ClimateClara/arc3o',
    
    #The project's license
    license='GPL-3.0',
    
    packages=find_packages(exclude=['docs', 'tests*', 'examples']),
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Climate scientists',
        'License :: OSI Approved :: GNU General Public License v3 (GPL-3.0)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    
    install_requires=[
          'numpy',
          'xarray',
          'pandas',
          'datetime',
          'time',
          'timeit',
          'itertools',
          'tqdm',
          'subprocess',
          'pathos.multiprocessing'
      ],
      
    project_urls={
    	'Bug Reports': 'https://github.com/ClimateClara/arc3o/issues',
    #    'Documentation': 'https://arc3o.readthedocs.io',
      },
    
    keywords='earth-sciences climate-modeling sea-ice arctic oceanography remote-sensing',
    
    python_requires='>=3.5',
)