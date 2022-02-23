"""
Setup script
"""

from setuptools import setup, find_packages

setup(
    name="nsl",
    version="1.0",
    # include all packages in src
    packages=find_packages(),
    include_package_data=True)
