from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.org'), encoding='utf-8') as f:
    long_description = f.read()

config = {
    'name': 'resources',
    'description': 'Tool for full-specified resources description',
    'long_description': long_description,

    'author': 'krvkir',
    'author_email': 'krvkir@gmail.com',

    'url': '',
    'download_url': '',

    'version': '0.1.dev1',
    'install_requires': [
        'pandas', 'geopandas', 'bcolz',
        ],
    'packages': find_packages(exclude=['docs', 'contrib', 'tests']),
    'scripts': [],
}

setup(**config)