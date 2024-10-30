# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/28 15:15
# @Author  : incpink Liu
# @File    : log.py
from loguru import logger


SUCCESS = 0
WARNING = 1
ERROR = 2


def output(level, cls_name: str, meth_name: str, msg: str) -> None:

    string = f"Class: {cls_name} -- Method: {meth_name} -- Message: {msg}"

    if level == SUCCESS:
        logger.success(string)
    elif level == WARNING:
        logger.warning(string)
    elif level == ERROR:
        logger.error(string)
    else:
        print("Log level wrong!")
        exit(1)
