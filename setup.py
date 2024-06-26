"""A module which goes through the setup of files"""
from setuptools import setup, find_packages

with open("README.md", encoding="utf8") as f:
    long_description = f.read()

setup(
    name='hvsrprocpy',
    version='0.1.1',
    description='A package for performing horizontal-to-vertical spectral ratio (HVSR) processing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/fjornelas/hvsrprocpy',
    license='GNU General Public License v3 or later (GPLv3+)',
    author='Francisco Javier Ornelas',
    author_email='jornela1@g.ucla.edu',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='HVSR, FAS, Time Series, microtremors  ',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=['numpy>=1.20', 'pandas>1.4', 'matplotlib>3.8', 'os', 'obspy>1.3', 'mplcursors>0.4'],
    project_urls={
        'Bug Reports': 'https://github.com/fjornelas/hvsrprocpy/issues',
        'Source': 'https://github.com/fjornelas/hvsrprocpy/',
    },
)

