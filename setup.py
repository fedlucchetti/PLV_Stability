import os
from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name = "federated_gmcc",
    version = "3.0.1",
    author = "Federico Lucchetti",
    author_email = "federico.lucchetti@uni.lu",
    description = ("Federated Geometric Monte Carlo Clustering "),
    license = "BSD",
    keywords = "example documentation tutorial",
    url = "https://github.com/fedlucchetti/PlantHivian",
    # package_dir = {'bin/modules/'},include_package_data=True,
    packages=find_packages(include=['federated']),
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
