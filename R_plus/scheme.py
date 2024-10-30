# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/28 15:23
# @Author  : incpink Liu
# @File    : scheme.py
import sys

import numpy as np
from tqdm import tqdm

from R_plus.utils import log


class RotationPlus(object):

    SMA = ["A", "T", "C", "G"]              # SMA: Standard Molecular Alphabet

    name_of_N_nary = {                      # base-n to N-nary
        2: "binary", 3: "ternary", 4: "quaternary",
        5: "quinary", 6: "senary", 7: "septenary",
        8: "octonary", 9: "novenary", 10: "denary"
    }

    def __init__(self, virtual_letter: str = "A", N_value: int = 4, extra_letters: str | list[str] = "M", need_logs: bool = True, need_monitor: bool = True):
        """
        Introduction:
        ------------
        The initialization method of RotationPlus.

        Parameters:
        -----------
        virtual_letter: str
            The virtual letter to start the rotation mapping.
        N_value: int
            The value of N, or the base of digits.
        extra_letters: str | list[str]
            Additional letters to expand the molecular alphabet.
        need_logs: bool
            Whether to output logs.
        need_monitor: bool
            Whether to monitor the progress bar.
        """
        self.virtual_letter = virtual_letter
        self.N_nary = N_value
        self.letters_cnt = N_value + 1
        self.extra_letters = extra_letters
        self.limit_of_info_density = np.log2(N_value + 1)
        self.N_nary_code_length = len(np.base_repr(255, base=N_value))

        self.N_nary_code = {number: np.base_repr(number, base=N_value).zfill(self.N_nary_code_length) for number in range(256)}
        #  default = {0: "0000", 1: "0001", ..., 255: "3333"}

        self.EMA = self.SMA + [extra_letters] if isinstance(extra_letters, str) else self.SMA + extra_letters
        # EMA: Expanded Molecular Alphabet
        # default = ["A", "T", "C", "G", "M"]

        self.rotation_mapping = {}
        self.get_rotation_mapping()
        # default = {"A": ["T", "C", "G", "M"],
        #            "T": ["C", "G", "M", "A"],
        #            "C": ["G", "M", "A", "T"],
        #            "G": ["M", "A", "T", "C"],
        #            "M": ["A", "T", "C", "G"]}

        self.need_logs = need_logs
        self.need_monitor = need_monitor

    def get_rotation_mapping(self) -> None:
        sequential_mapping = self.EMA * 2
        for letter in self.EMA:
            letter_index = sequential_mapping.index(letter)
            self.rotation_mapping[letter] = sequential_mapping[letter_index + 1: letter_index + self.letters_cnt]

    def encode(self, digital_segments: list, ins_redundancy: bool = True) -> list:
        """
        Introduction:
        ------------
        Encode the digital segments into DNA segments.

        Parameters:
        -----------
        digital_segments: list
            The digital segments to be encoded.
        ins_redundancy: bool
            Whether to insert redundancy into the digital segments.

        Returns:
        --------
        DNA_segments: list
            The DNA segments after encoding.
        """
        DNA_segments = []
        denary_segments = tqdm(digital_segments, desc="R_plus encode") if self.need_monitor else digital_segments

        for value_index, denary_segment in enumerate(denary_segments):
            DNA_segment = []
            N_nary_values = []

            for denary_value in denary_segment:
                if denary_value >= 256:
                    raise ValueError("Please ensure the value of denary number is between 0 and 255!")
                N_nary_code = self.N_nary_code.get(denary_value)
                N_nary_values += list(map(int, list(N_nary_code)))

            if ins_redundancy:
                N_nary_values = self.ins_redundancy(N_nary_values, total_length=100)

            previous_letter = self.virtual_letter

            for N_nary_value in N_nary_values:
                current_letter = self.rotation_mapping.get(previous_letter)[N_nary_value]
                DNA_segment.append(current_letter)
                previous_letter = current_letter

            DNA_segments.append("".join(DNA_segment))

        if self.need_logs:
            log.output(level=log.SUCCESS,
                       cls_name=self.__class__.__name__,
                       meth_name=sys._getframe().f_code.co_name,
                       msg=f"Encode using {self.name_of_N_nary[self.N_nary]} form of R_plus successfully!")

        return DNA_segments

    def decode(self, DNA_segments: list, del_redundancy: bool = True) -> list:
        """
        Introduction:
        ------------
        Decode the DNA segments into digital segments.

        Parameters:
        -----------
        DNA_segments: list
            The DNA segments to be decoded.
        del_redundancy: bool
            Whether to delete redundancy from the digital segments.

        Returns:
        --------
        digital_segments: list
            The digital segments after decoding.
        """
        digital_segments = []
        DNA_segments = tqdm(DNA_segments, desc="R_plus decode") if self.need_monitor else DNA_segments

        for seg_index, DNA_segment in enumerate(DNA_segments):
            DNA_segment = list(DNA_segment)
            previous_letter, N_nary_values = self.virtual_letter, []

            for b in range(len(DNA_segment)):
                if DNA_segment[b] == previous_letter:    # Mutations occur.
                    if previous_letter == "M":
                        DNA_segment[b] = "C"
                    elif previous_letter == "C":
                        DNA_segment[b] = "M"
                    else:
                        EMA = self.EMA.copy()
                        EMA.remove(previous_letter)
                        try:
                            EMA.remove(DNA_segment[b + 1])
                        except (IndexError, ValueError):
                            pass
                        finally:
                            DNA_segment[b] = np.random.choice(EMA)

                N_nary_values.append(self.rotation_mapping.get(previous_letter).index(DNA_segment[b]))
                previous_letter = DNA_segment[b]

            if del_redundancy:
                # if len(DNA_segment) == 100:
                N_nary_values = self.del_redundancy(N_nary_values)

            N_nary_code, digital_segment = "", []

            for N_nary_value in N_nary_values:
                N_nary_code += str(N_nary_value)

                if len(N_nary_code) == self.N_nary_code_length:
                    denary_value = list(self.N_nary_code.keys())[list(self.N_nary_code.values()).index(N_nary_code)]
                    digital_segment.append(denary_value)
                    N_nary_code = ""

            digital_segments.append(digital_segment)

        if self.need_logs:
            log.output(level=log.SUCCESS,
                       cls_name=self.__class__.__name__,
                       meth_name=sys._getframe().f_code.co_name,
                       msg=f"Decode using {self.name_of_N_nary[self.N_nary]} form of R_plus successfully!")

        return digital_segments

    def ins_redundancy(self, N_nary_segment: list, total_length: int) -> list:
        """
        Introduction:
        ------------
        Insert redundancy into the N-nary segment.

        Parameters:
        -----------
        N_nary_segment: list
            The N-nary(digital) segment to be inserted redundancy (not XOR redundancy).
        total_length: int
            The total length of the N-nary segment after inserting redundancy.

        Returns:
        --------
        redundant_segment: list
            The N-nary segment after inserting redundancy.
        """
        N_nary_length = np.base_repr(len(N_nary_segment), base=self.N_nary).zfill(self.N_nary_code_length)
        pad_length = total_length - len(N_nary_segment) - len(N_nary_length)
        if pad_length < 0:
            raise ValueError(f"Please ensure the value of total_length: {total_length} "
                             f"should be greater than or equal to the sum of "
                             f"len(N_nary_segment): {len(N_nary_segment)} and "
                             f"len(N_nary_length): {len(N_nary_length)}!")
        redundant_segment = N_nary_segment + [0] * pad_length + list(map(int, N_nary_length))
        return redundant_segment

    def del_redundancy(self, redundant_segment: list) -> list:
        """
        Introduction:
        ------------
        Delete redundancy from the N-nary segments.

        Parameters:
        -----------
        redundant_segment: list
            The N-nary(digital) segment contained redundancy.

        Returns:
        --------
        N_nary_segment: list
            The N-nary segment after deleting redundancy.
        """
        N_nary_length = int("".join(map(str, redundant_segment[-self.N_nary_code_length:])), base=self.N_nary)
        return redundant_segment[:N_nary_length]