# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/28 15:25
# @Author  : incpink Liu
# @File    : pipeline.py
import struct as st

from R_plus.scheme import RotationPlus
from R_plus.utils.tools import SequenceTools, ErrorTools


def encode(algorithm: RotationPlus, input_file: str, output_file: str, segment_bytes: int = 22, ins_xor: bool = True) -> None:
    """
    Introduction:
    ------------
    Using the seleted algorithm to encode the digital data into DNA data and output DNA data.

    Parameters:
    -----------
    algorithm: RotationPlus
        The selected form of Rotation plus.
    input_file: str
        The path of the digital file.
    output_file: str
        The path of the file contained DNA data.
    segment_bytes: int
        The number of bytes in each digital segment.
    ins_xor: bool
        Whether to insert the XOR sequence.
    """
    with open(input_file, mode="rb") as f_in:
        bytestream = f_in.read()
        denary_list = list(st.unpack(f"<{len(bytestream)}B", bytestream))

    if ins_xor:
        denary_list = SequenceTools.insert_xor_sequence(denary_list, need_group=True, group_members=segment_bytes)
    else:
        denary_list = [denary_list[s: s + segment_bytes] for s in range(0, len(denary_list), segment_bytes)]

    short_segments = algorithm.encode(denary_list, ins_redundancy=True)
    with open(output_file, mode="w") as f_out:
        for seg_index, short_seg in enumerate(short_segments):
            f_out.write(f"segment {seg_index}: {short_seg}\n")


def decode(algorithm: RotationPlus, input_file: str, output_file: str, del_xor: bool = True) -> None:
    """
    Introduction:
    ------------
    Using the seleted algorithm to decode the DNA data into digital data and output the corresponding digital file.

    Parameters:
    -----------
    algorithm: RotationPlus
        The selected form of Rotation plus. Note the same as the form selected in the encoding process.
    input_file: str
        The path of the file contained DNA data.
    output_file: str
        The path of the digital file.
    del_xor: bool
        Whether to delete the XOR sequence.
    """
    short_segments = []
    with open(input_file, mode="r") as f_in:
        while True:
            info = f_in.readline()

            if not info:
                break

            short_segment = info.rsplit(" ")[-1].rstrip()
            short_segments.append(short_segment)

    denary_list = algorithm.decode(short_segments, del_redundancy=True)
    if del_xor:
        denary_list = SequenceTools.delete_xor_sequence(denary_list, need_ungroup=True)

    denary_list = [number == 0 if number is None else number for number in denary_list]
    bytestream = st.pack(f"<{len(denary_list)}B", *tuple(denary_list))
    with open(output_file, mode="wb") as f_out:
        f_out.write(bytestream)