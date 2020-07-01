# -*- coding: utf-8 -*-
# Author: Waldmann Frédéric
"""
setup.py
Installer for lmlib
"""

import codecs
import os
from setuptools import setup

# Get the requirements from requirements.txt and environment
with open("requirements.txt", "r") as fid:
    install_requires = [line.strip() for line in fid]

setup(
    name="lmlib",
    version="1.0",
    description="signal processing library using local model approximation",
    author="Waldmann Frédéric, Wildhaber Reto",
    author_email="waf1@bfh.ch",
    packages=["lmlib"],
    install_requires=install_requires,  # external packages defined requirements.txt
)
