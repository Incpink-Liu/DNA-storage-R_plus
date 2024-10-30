# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/29 15:46
# @Author  : incpink Liu
# @File    : setup.py
from setuptools import setup


setup(
    name="R_plus",
    version="0.1",
    author="Incpink Liu, Zhi Ping",
    author_email="liuderuilin@genomics.cn, liuderuilin22@mails.ucas.ac.cn",
    maintainer="BGI-research",
    url="https:/github.com/incpink-Liu/DNA-storage-R+",
    description="R+ implementation",
    long_description="R+ is a DNA storage transcoding strategy developed by BGI-research. "
                     "Briefly, it can provide a direct mapping refence between expanded molecular alphabet and "
                     "N-nary digits in the absence of high-performance transcoding algorithm at present.",
    packages=["R_plus", "R_plus/utils"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ]
)