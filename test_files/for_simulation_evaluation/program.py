# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/10 15:47
# @Author  : incpink Liu
# @File    : scheme.py
import sys

import numpy as np
from tqdm import tqdm

from utils import log


class QuaternaryRotate(object):

    rotate_rule = {
        "A": ["T", "C", "G", "M"],          # "M" is 5-methylcytosine
        "T": ["C", "G", "M", "A"],
        "C": ["G", "M", "A", "T"],
        "G": ["M", "A", "T", "C"],
        "M": ["A", "T", "C", "G"]
    }

    quaternary_code = {number: np.base_repr(number, base=4).zfill(4) for number in range(256)}

    def __init__(self, virtual_base: str = "A", need_logs: bool = True, need_monitor: bool = True):
        self.virtual_base = virtual_base

        if self.virtual_base not in ["A", "C", "G", "T", "M"]:
            raise ValueError("Please ensure the virtual base must be in ['A', 'C', 'G', 'T', 'M']")

        self.quaternary_code_length = 4
        self.need_logs = need_logs
        self.need_monitor = need_monitor

    def encode(self, digital_segments: list, pattern: str = "binary", with_std: bool = True, std_length: int = 100) -> list:

        base_segments = []

        if pattern == "binary":
            bit_segments = tqdm(digital_segments) if self.need_monitor else digital_segments

            for seg_index, bit_segment in enumerate(bit_segments):
                if len(bit_segment) % 8 != 0:
                    raise ValueError("Please ensure the length of bit sequence can be divided by 8!")

                if len(bit_segment) < 32:
                    raise ValueError("Please ensure the length of bit sequence is greater than or equal to 32!")

                base_segment = []
                quaternary_values = []

                for pos in range(0, len(bit_segment), 8):
                    denary_value = int("".join(list(map(str, bit_segment[pos: pos + 8]))), 2)
                    quaternary_code = self.quaternary_code.get(denary_value)
                    quaternary_values += list(map(int, list(quaternary_code)))

                if with_std:
                    quaternary_values = self.standardize(quaternary_values, std_length=std_length)

                previous_base = self.virtual_base

                for quaternary_value in quaternary_values:
                    current_base = self.rotate_rule.get(previous_base)[quaternary_value]
                    base_segment.append(current_base)
                    previous_base = current_base

                base_segments.append("".join(base_segment))

        elif pattern == "denary":
            denary_segments = tqdm(digital_segments) if self.need_monitor else digital_segments

            for value_index, denary_segment in enumerate(denary_segments):

                base_segment = []
                quaternary_values = []

                for denary_value in denary_segment:
                    if denary_value >= 256:
                        raise ValueError("Please ensure the value of denary number is between 0 and 255!")

                    quaternary_code = self.quaternary_code.get(denary_value)
                    quaternary_values += list(map(int, list(quaternary_code)))

                if with_std:
                    quaternary_values = self.standardize(quaternary_values, std_length=std_length)

                previous_base = self.virtual_base

                for quaternary_value in quaternary_values:
                    current_base = self.rotate_rule.get(previous_base)[quaternary_value]
                    base_segment.append(current_base)
                    previous_base = current_base

                base_segments.append("".join(base_segment))

        else:
            raise ValueError("Please ensure the value of pattern is 'binary' or 'denary'!")

        if self.need_logs:
            log.output(level=log.SUCCESS,
                       cls_name=self.__class__.__name__,
                       meth_name=sys._getframe().f_code.co_name,
                       msg=f"Quaternary Rotation algo encoding successful!")

        return base_segments

    def decode(self, base_segments: list, pattern: str = "binary", without_std: bool = True) -> list:
        digital_segments = []

        base_segments = tqdm(base_segments) if self.need_monitor else base_segments

        for seg_index, base_segment in enumerate(base_segments):
            previous_base, quaternary_values = self.virtual_base, []

            for base in base_segment:
                quaternary_values.append(self.rotate_rule.get(previous_base).index(base))
                previous_base = base

            if without_std:
                quaternary_values = self.de_standardize(quaternary_values)

            quaternary_code, digital_segment = "", []

            for quaternary_value in quaternary_values:
                quaternary_code += str(quaternary_value)

                if quaternary_code in self.quaternary_code.values():
                    denary_value = list(self.quaternary_code.keys())[list(self.quaternary_code.values()).index(quaternary_code)]

                    if pattern == "binary":
                        digital_segment.append(bin(denary_value)[2:].zfill(8))
                    elif pattern == "denary":
                        digital_segment.append(denary_value)
                    else:
                        raise ValueError("Please ensure the value of pattern is 'binary' or 'denary'!")

                    quaternary_code = ""

                else:
                    if len(quaternary_code) == self.quaternary_code_length:
                        digital_segment.append(None)
                        quaternary_code = ""

            digital_segments.append(digital_segment)

        if self.need_logs:
            log.output(level=log.SUCCESS,
                       cls_name=self.__class__.__name__,
                       meth_name=sys._getframe().f_code.co_name,
                       msg=f"Quaternary Rotation algo decoding successful!")

        return digital_segments

    @staticmethod
    def standardize(quaternary_segment: list, std_length: int) -> list:
        quaternary_length = np.base_repr(len(quaternary_segment), base=4).zfill(4)
        pad_length = std_length - len(quaternary_segment) - len(quaternary_length)

        if pad_length < 0:
            raise ValueError(f"Please ensure the value of std_length: {std_length} "
                             f"is greater than or equal to the sum of "
                             f"len(quaternary_segment): {len(quaternary_segment)} and "
                             f"len(quaternary_length): {len(quaternary_length)}!")

        std_segment = quaternary_segment + [0] * pad_length + list(map(int, quaternary_length))

        return std_segment

    @staticmethod
    def de_standardize(std_segment: list) -> list:
        quaternary_length = int("".join(map(str, std_segment[-4:])), 4)

        return std_segment[:quaternary_length]
