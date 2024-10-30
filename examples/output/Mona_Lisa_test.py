# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/28 15:42
# @Author  : incpink Liu
# @File    : Mona_Lisa_test.py
from R_plus import pipeline, scheme

read_file_path = "../files/Mona_Lisa.bmp"
dna_path = "../files/Mona_Lisa-DNA-segment.txt"
write_file_path = "../files/Mona_Lisa_recovered.bmp"


if __name__ == "__main__":
    pipeline.encode(
        algorithm=scheme.RotationPlus(virtual_letter="A", N_value=4, extra_letters="M", need_logs=True, need_monitor=True),
        input_file=read_file_path,
        output_file=dna_path,
        ins_xor=True
    )

    pipeline.decode(
        algorithm=scheme.RotationPlus(virtual_letter="A", N_value=4, extra_letters="M", need_logs=True, need_monitor=True),
        input_file=dna_path,
        output_file=write_file_path,
        del_xor=True
    )
