# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/28 15:16
# @Author  : incpink Liu
# @File    : tools.py
import re
import subprocess
from itertools import product

import numpy as np
from tqdm import tqdm
from Bio import motifs, pairwise2
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner

from R_plus.utils import log


class HammingDistanceTools(object):
    """
    References
    ----------
    Statistic method hamming_group and hamming_matrix are cited from
    https://github.com/HaolingZHANG/FasterHammingCalculator/blob/main/calculator.py
    """
    @staticmethod
    def hamming_group(observed_sample: np.ndarray, sample_group: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """
        Introduction
        ------------
        Calculate the Hamming distances between the observed sample and all the samples in sample group.

        Parameters
        ----------
        observed_sample: one-dimensional variable array, the shape of which could be (variable number,).
        sample_group: two-dimensional variable array, the shape of which could be (sample number, variable number).
        threshold: threshold for real number system.

        Returns
        -------
        hamming distance array between the observed sample and sample group.

        Examples
        --------
        >>> from numpy import array
        >>> from hamming_distance_calculate import hamming_group
        >>> sample = array([0, 1, 0])
        >>> sample_group = array([[1, 1, 0], [0, 1, 1], [0, 1, 0]])
        >>> hamming_group(observed_sample=sample, sample_group=sample_group, threshold=None)
        array([1, 1, 0])
        >>> sample = array([0.2, 0.5, 0.8])
        >>> sample_group = array([[0.7, 0.1, 0.2], [0.3, 0.4, 0.7], [0.9, 0.2, 0.4]])
        >>> hamming_group(observed_sample=sample, sample_group=sample_group, threshold=0.4)
        array([2, 0, 1])

        Notes
        -----
        The variable number of parameter 'observed_sample' and 'sample_group' should be equal.
        """

        sample = np.expand_dims(observed_sample, axis=0)

        matrix = np.abs(sample_group - sample)

        if threshold is None:
            matrix = matrix.astype(bool)  # set bool for illustrating the difference flag rapidly.
        else:
            matrix = np.where(matrix > threshold, 1, 0)  # use 'where' function for real number system with threshold.

        distances = np.sum(matrix, axis=1)

        return distances

    @staticmethod
    def hamming_matrix(samples: np.ndarray, other_samples: np.ndarray | None = None, threshold: float | None = None) -> np.ndarray:
        """
        Introduction
        ------------
        Calculate hamming matrix between samples.

        Parameters
        ----------
        samples: two-dimensional sample array, the shape of which could be (sample number, variable number).
        other_samples: another two-dimensional sample array with above-mentioned shape.
        threshold: threshold for real number system.

        Returns
        -------
        hamming distance matrix of samples.

        Examples
        --------
        >>> from numpy import array
        >>> from hamming_distance_calculate import hamming_matrix
        >>> samples_1 = array([[1, 1, 0], [0, 1, 1], [0, 1, 0]])
        >>> hamming_matrix(samples=samples_1)
        array([[0, 2, 1],
               [2, 0, 1],
               [1, 1, 0]])
        >>> samples_2 = array([[0, 0, 1], [1, 0, 1]])
        >>> hamming_matrix(samples=samples_1, other_samples=samples_2)
        array([[3, 2],
               [1, 2],
               [2, 3]])
        >>> samples_1 = array([[0.7, 0.1, 0.2], [0.3, 0.4, 0.7], [0.9, 0.2, 0.4]])
        >>> hamming_matrix(samples=samples_1, threshold=0.4)
        array([[0, 1, 0],
               [1, 0, 1],
               [0, 1, 0]])
        >>> samples_2 = array([[0.6, 0.7, 0.8], [0.5, 0.9, 0.1]])
        >>> hamming_matrix(samples=samples_1, other_samples=samples_2, threshold=0.4)
        array([[2, 1],
               [0, 2],
               [1, 1]])

        Notes
        -----
        The variable number of parameter 'samples' and 'other_samples' should be equal.

        If 'other_samples' is None, the shape of outputted matrix is (sample number, sample number).
        Otherwise, that of outputted matrix is (sample number, other sample number).
        """

        # prepare for the cross subtraction for samples or two sample groups.
        if other_samples is None:
            former = np.repeat(np.expand_dims(samples, 0), len(samples), 0)
            latter = np.swapaxes(former.copy(), 0, 1)
        else:
            former = np.repeat(np.expand_dims(other_samples, 0), len(samples), 0)
            latter = np.swapaxes(np.repeat(np.expand_dims(samples, 0), len(other_samples), 0), 0, 1)

        # do cross subtraction to define the actual difference between any two samples in the same position.
        matrix = np.abs(former - latter)

        if threshold is None:
            matrix = matrix.astype(bool)  # set bool for illustrating the difference flag rapidly.
        else:
            matrix = np.where(matrix > threshold, 1, 0)  # use 'where' function for real number system with threshold.

        matrix = np.sum(matrix, axis=2, dtype=int)  # calculate hamming distances.

        return matrix

    @staticmethod
    def hamming_filter(sequences: list[str], min_hamming_distance: int) -> list[str]:
        """
        Introduction
        ------------
        Filter out a list of sequences with a hamming distance between any two sequences
        that is not less than the minimum hamming distance.

        Parameters
        ----------
        sequences: a list of sequences that need to be filtered.
        min_hamming_distance: minimum hamming distance.

        Returns
        -------
        a list of sequences where the hamming distance between any two sequences is not less than
        the minimum hamming distance.
        """
        group_sequences = [list(map(ord, list(sequence))) for sequence in sequences]
        filtered_sequences = []

        while True:
            if len(group_sequences) == 0:
                break

            align_sequence = group_sequences[0]

            hamming_distances = HammingDistanceTools.hamming_group(np.asarray(align_sequence), np.asarray(group_sequences))
            hamming_distances_indices = np.where(hamming_distances < min_hamming_distance)[0]

            for hamming_distances_index in hamming_distances_indices[:: -1]:
                del group_sequences[hamming_distances_index]

            filtered_sequences.append(align_sequence)

        if len(filtered_sequences) == 1:
            raise ValueError("No solution!")

        for filtered_sequence in filtered_sequences:
            verify_hamming_distances = HammingDistanceTools.hamming_group(filtered_sequence, filtered_sequences)
            if np.where(verify_hamming_distances < min_hamming_distance)[0].size == 1:
                print(verify_hamming_distances)
                continue
            else:
                raise ValueError(
                    "Filter failed, please check if there is a problem with the logic of the filter algorithm!"
                )

        sequences = ["".join(list(map(chr, sequence))) for sequence in filtered_sequences]

        return sequences

    @staticmethod
    def hamming_distance_in_seqs(ref_seq: str, qry_seq: str) -> int:
        if len(ref_seq) == len(qry_seq):
            return sum([1 for i in range(len(ref_seq)) if ref_seq[i] != qry_seq[i]])
        else:
            return len(ref_seq)


class ValidityTools(object):
    """
    References
    ----------
    Statistic method check_homopolymer, check_GC_content and check_fold are cited from
    https://github.com/ntpz870817/DNA-storage-YYC/blob/master/yyc/utils/validity.py
    """
    @staticmethod
    def check_homopolymer(sequence: str, max_homopolymer: int) -> bool:
        """
        Introduction
        ------------
        Check the max homopolymer of requested DNA sequence.

        Parameters
        ----------
        sequence: DNA sequence needs detecting.
        max_homopolymer: maximum length of homopolymer.

        Returns
        -------
        Whether the DNA sequence can be considered as valid for DNA synthesis and sequencing.
        """
        void_homopolymers = [
            "A" * (max_homopolymer + 1),
            "C" * (max_homopolymer + 1),
            "G" * (max_homopolymer + 1),
            "T" * (max_homopolymer + 1),
        ]

        for void_homopolymer in void_homopolymers:
            if void_homopolymer in sequence:
                return False

        return True

    @staticmethod
    def check_GC_content(sequence: str, max_content: int = 60) -> bool:
        """
        Introduction
        ------------
        Check the C and G content of requested DNA sequence.

        Parameters
        ----------
        sequence: requested DNA sequence.
        max_content: maximum content of C and G, which means GC content is in [1 - max_content, max_content].

        Returns
        -------
        Whether the DNA sequence can be considered as valid for DNA synthesis and sequencing.
        """
        if max_content < 50 or max_content > 70:
            raise ValueError("Please ensure the value of max_content is between 50 and 70!")

        GC_content = (sequence.count("C") + sequence.count("G")) * 100 / len(sequence)

        return (100 - max_content) <= GC_content <= max_content

    @staticmethod
    def check_fold(sequence: str, min_free_energy: float | None = None) -> bool:
        """
        Introduction
        ------------
        Call RNA fold to calculate hairpin MFE of a motif

        Parameters
        ----------
        sequence: requested DNA sequence.
        min_free_energy: min free energy.

        Returns
        -------
        Whether the free energy of DNA sequence is lower than required min free energy.
        """
        if min_free_energy is None:
            return True

        process = subprocess.Popen('echo "%s" | RNAfold --noPS --noGU --noconv -T 59.1' % sequence,
                                   stdout=subprocess.PIPE, shell=True)
        process.wait()
        if process.returncode == 0:
            line = process.stdout.read().decode().split('\n')[1]
            m = re.search("(\\S+)\\s+\\(\\s*(\\S+)\\)", line)
            if m:
                if min_free_energy > float(m.group(2)):
                    return True

        return False

    @staticmethod
    def check_hamming_distance(align_sequence: str, group_sequence: list[str], min_hamming_distance: int) -> bool:
        """
        Introduction
        ------------
        Check the hamming distances between align sequence and group sequence.

        Parameters
        ----------
        align_sequence: align sequence.
        group_sequence: sequences need to be aligned.
        min_hamming_distance: minimum of hamming distance.

        Returns
        -------
        Whether the hamming distance between the sequence in group sequence and align sequence is greater than or equal
        to the minimum hamming distance.
        """
        align_sequence = list(map(ord, list(align_sequence)))
        group_sequence = [list(map(ord, list(seq))) for seq in group_sequence]

        hamming_distances = HammingDistanceTools.hamming_group(np.asarray(align_sequence), np.asarray(group_sequence))
        hamming_distances_indices = np.where(hamming_distances < min_hamming_distance)[0]

        if hamming_distances_indices.size != 0:
            return False

        return True


class SequenceTools(object):

    @staticmethod
    def get_repeating_indices(sequence: str, repeating_element: str) -> list:
        """
        Introduction
        ------------
        Get the indices of repeating elements in the sequence.

        Parameters
        ----------
        sequence: input sequence.
        repeating_element: repeating element.

        Returns
        -------
        Indices of repeating elements in the sequence.
        """
        return [indices for (indices, element) in enumerate(list(sequence)) if element == repeating_element]

    @staticmethod
    def generate_sequence(length: int, max_homopolymer: int | None = None, max_gc_content: int | None = None) -> list[str]:
        """
        Introduction
        ------------
        Generate DNA sequences based on parameters.

        Parameters
        ----------
        length: DNA sequences length.
        max_homopolymer: maximum length of DNA sequences homopolymer.
        max_gc_content: maximum GC content of DNA sequences

        Returns
        -------
        sequences that meet the requirements.
        """
        all_sequences = ["".join(nucleotides) for nucleotides in list(product(["A", "C", "G", "T"], repeat=length))]
        valid_sequences = []

        if max_homopolymer is None and max_gc_content is None:
            return all_sequences

        elif max_homopolymer is None and max_gc_content is not None:
            for sequence in all_sequences:
                if not ValidityTools.check_GC_content(sequence, max_gc_content):
                    continue
                else:
                    valid_sequences.append(sequence)

            return valid_sequences

        elif max_homopolymer is not None and max_gc_content is None:
            for sequence in all_sequences:
                if not ValidityTools.check_homopolymer(sequence, max_homopolymer):
                    continue
                else:
                    valid_sequences.append(sequence)

            return valid_sequences

        elif max_homopolymer is not None and max_gc_content is not None:
            for sequence in all_sequences:
                if not ValidityTools.check_homopolymer(sequence, max_homopolymer):
                    continue
                else:
                    if not ValidityTools.check_GC_content(sequence, max_gc_content):
                        continue
                    else:
                        valid_sequences.append(sequence)

            return valid_sequences

        else:
            raise ValueError

    @staticmethod
    def insert_xor_sequence(sequences: list, intro_xor: int = 2, need_group: bool = False, group_members: int | None = None) -> list:
        """
        Introduction
        ------------
        Generate XOR decimal sequences based on parameters and insert it into the original sequence.

        Parameters
        ----------
        sequences: input sequences.
        intro_xor: number of sequences involved in the XOR operation, the default value is 2.
        need_group: sequence grouping, if True, input sequences will be grouped based on group_members,
                    and the dimension of input sequences must be 1.
        group_members: number of decimals per group.

        Returns
        -------
        XOR sequences, original sequences (grouped, if paras need_group == True), and complete sequences of both in a list.

        Examples
        --------
        >>> sequences_1 = [1, 2, 3, 4]
        >>> complete_sequences_1 = insert_xor_sequence(sequences_1, need_group=True, group_members=2)
        >>> complete_sequences_1
        [[1, 2], [3, 4], [2, 6]]

        >>> sequences_2 = [[77, 25, 64, 255], [190, 78, 16, 11], [67, 134, 231, 162]]
        >>> complete_sequences_2 = insert_xor_sequence(sequences_2, intro_xor=3)
        >>> complete_sequences_2
        [[77, 23, 64, 255], [190, 78, 16, 11], [67, 134, 231, 162], [176, 209, 183, 86]]

        >>> sequences_3 = [1, 2, 3, 4, 5, 6, 7, 8]
        >>> complete_sequences_3 = insert_xor_sequence(sequences_3, intro_xor=3, need_group=True, group_members=3)
        >>> complete_sequences_3
        [[1, 2, 3], [4, 5, 6], [7, 8], [2, 15, 5]]
        """
        if intro_xor < 2:
            raise ValueError("Please ensure the value of intro_xor is greater than 2!")

        original_sequences = sequences[:]

        if need_group:
            if type(sequences[0]) == list and type(sequences[0][0]) == int:
                raise ValueError("If need group, please ensure the dimension of sequences is 1!")
            if len(sequences) % group_members != 0:
                redundancy = [0] * (group_members - len(sequences) % group_members)
                sequences += redundancy

            sequences = [sequences[s: s + group_members] for s in range(0, len(sequences), group_members)]
            original_sequences = [original_sequences[s: s + group_members] for s in range(0, len(original_sequences), group_members)]

        if len(original_sequences) % intro_xor != 0:
            original_sequences = original_sequences + [original_sequences[-1]]
            sequences = sequences + [sequences[-1]]
            print("It is detected that the length of sequences is not divided by intro_xor, "
                  "so the last sequences will be repeated once!")

        if type(sequences[0]) == list and type(sequences[0][0]) == int:
            xor_sequences = []

            for s in range(0, len(sequences), intro_xor):
                xor_groups = list(zip(*sequences[s: s + intro_xor]))
                xor_sequence = []

                for xor_group in xor_groups:
                    xor_value = XorTools.xor_compute(xor_group)
                    xor_sequence.append(xor_value)

                xor_sequences.append(xor_sequence)

            complete_sequences = []

            for x in range(len(xor_sequences)):
                complete_sequences.extend(original_sequences[x * intro_xor: (x + 1) * intro_xor])
                if len(original_sequences[x]) == len(xor_sequences[x]):
                    complete_sequences.append(xor_sequences[x])
                else:
                    complete_sequences.append(xor_sequences[x][: len(original_sequences[x])])

            return complete_sequences

        else:
            raise ValueError("Please ensure the dimensions of sequences is 2!")

    @staticmethod
    def delete_xor_sequence(sequences: list, intro_xor: int = 2, need_ungroup: bool = False) -> list:
        """
        Introduction
        ------------
        Delete XOR decimal sequences based on parameters and restore the original sequence.

        Parameters
        ----------
        sequences: input sequences.
        intro_xor: number of sequences involved in the XOR operation, the default value is 2.
        need_ungroup: sequence ungrouping, if True, original sequences will be ungrouped (two-dimension to one-dimension).

        Returns
        -------
        Original sequences.
        """
        if intro_xor < 2:
            raise ValueError("Please ensure the value of intro_xor is greater than 2!")

        if type(sequences[0]) == list and (type(sequences[0][0]) == int or type(sequences[0][0]) is None):
            original_sequences = []
            for i in range(0, len(sequences), intro_xor + 1):
                original_sequences.extend(sequences[i: i + intro_xor])

            if need_ungroup:
                original_sequences = [digit for data in original_sequences for digit in data]

            return original_sequences


class XorTools(object):

    @staticmethod
    def xor_compute(integer_list: list | tuple) -> int | list:
        """
        Introduction
        ------------
        Calculate the XOR value of a one-dimension list of integers.
        If the dimension of the list is 2, the XOR value of each one-dimension list will be calculated.

        Parameters
        ----------
        integer_list: a one-dimension or two-dimension list of integers.

        Returns
        -------
        Single or multiple XOR values of a list of integers.

        Examples
        --------
        >>> list_xor_value_1d = xor_compute([1, 2, 3, 4])
        >>> list_xor_value_1d
        4
        >>> list_xor_value_2d = xor_compute([[1, 2, 3, 4], [5, 6, 7, 8]])
        >>> list_xor_value_2d
        [4, 12]
        """
        if type(integer_list[0]) == int:
            xor_value = 0

            for x in range(len(integer_list)):
                xor_value ^= integer_list[x]

            return xor_value

        elif type(integer_list[0]) == list or tuple and type(integer_list[0][0]) == int:
            xor_values = []

            for x in range(len(integer_list)):
                xor_value = 0

                for y in range(len(integer_list[x])):
                    xor_value ^= integer_list[x][y]

                xor_values.append(xor_value)

            return xor_values

        else:
            raise ValueError("Please ensure the dimensions of integer_list is 1 or 2!")


class ErrorTools(object):

    @staticmethod
    def introduce_errors(right_sequence: str, error_count: int = 0, error_rate: float = 0) -> str:
        """
        Introduction
        ------------
        Generate a wrong DNA string from the right DNA string randomly.

        Parameters
        ----------
        right_sequence: right DNA sequence.
        error_count: count of errors introduced, 0 < error_count <= len(right_sequence).
        error_rate: rate of errors introduced, 0 < error_rate <= 1.

        Returns
        -------
        Wrong DNA string.
        """
        wrong_types = np.array([0, 1, 2])  # substitution, deletion, and insertion.
        wrong_types_prob = np.array([0.7, 0.2, 0.1])

        if error_rate != 0:
            error_count = ceil(len(right_sequence) * error_rate)

        wrong_dna_string = list(right_sequence)

        for _ in range(error_count):
            wrong_type = np.random.choice(wrong_types, p=wrong_types_prob)
            error_location = np.random.randint(0, len(wrong_dna_string))

            if wrong_type == 0:  # substitution
                nucleotide = np.random.choice(
                    list(filter(lambda n: n != wrong_dna_string[error_location], ["A", "C", "G", "T"]))
                )
                wrong_dna_string[error_location] = nucleotide

            elif wrong_type == 1:  # deletion
                del wrong_dna_string[error_location]

            else:  # insertion
                nucleotide = np.random.choice(["A", "C", "G", "T"])
                wrong_dna_string.insert(error_location, nucleotide)

        return "".join(wrong_dna_string)

    @staticmethod
    def error_correct(reference: str, digital_segments: list, need_logs: bool = False) -> list:
        """
        Introduction
        ------------
        Correct the errors in the digital segments based on the reference and XOR redundancy.

        Parameters
        ----------
        reference: str
            The path of the reference file.
        digital_segments: list
            The digital segments to be corrected.
        need_logs: bool
            Whether to output logs.
        """
        with open(reference, mode="r") as ref:
            ref_sequences = ref.readlines()

            for r in range(len(ref_sequences)):
                ref_sequences[r] = list(map(int, ref_sequences[r].strip().split(":")[-1].split(" ")))

        for n in range(0, len(digital_segments), 3):
            original_sequence_1, original_sequence_2, xor_sequence = digital_segments[n: n + 3]
            original_reference_1, original_reference_2, xor_reference = ref_sequences[n: n + 3]

            if xor_sequence == xor_reference:
                if original_sequence_1 == original_reference_1:
                    if original_sequence_2 == original_reference_2:
                        log.output(
                            level=log.SUCCESS,
                            cls_name=ErrorTools.__name__,
                            meth_name=ErrorTools.error_correct.__name__,
                            msg=f"Sequences {n}-{n + 2} have no errors!"
                        ) if need_logs else None

                    else:
                        original_sequence_2 = XorTools.xor_compute(list(zip(original_sequence_1, xor_sequence)))
                        digital_segments[n + 1] = original_sequence_2
                        log.output(
                            level=log.SUCCESS,
                            cls_name=ErrorTools.__name__,
                            meth_name=ErrorTools.error_correct.__name__,
                            msg=f"Sequences {n}&{n + 2} have no errors, sequence {n + 1} was corrected!"
                        ) if need_logs else None

                else:
                    if original_sequence_2 == original_reference_2:
                        original_sequence_1 = XorTools.xor_compute(list(zip(original_sequence_2, xor_sequence)))
                        digital_segments[n] = original_sequence_1
                        log.output(
                            level=log.SUCCESS,
                            cls_name=ErrorTools.__name__,
                            meth_name=ErrorTools.error_correct.__name__,
                            msg=f"Sequences {n + 1}&{n + 2} have no errors, sequence {n} was corrected!"
                        ) if need_logs else None
                    else:
                        log.output(
                            level=log.ERROR,
                            cls_name=ErrorTools.__name__,
                            meth_name=ErrorTools.error_correct.__name__,
                            msg=f"Sequences {n}&{n + 1} have errors, correction failed!"
                        ) if need_logs else None
            else:
                if original_sequence_1 == original_reference_1:
                    if original_sequence_2 == original_reference_2:
                        xor_sequence = XorTools.xor_compute(list(zip(original_sequence_1, original_sequence_2)))
                        digital_segments[n + 2] = xor_sequence
                        log.output(
                            level=log.SUCCESS,
                            cls_name=ErrorTools.__name__,
                            meth_name=ErrorTools.error_correct.__name__,
                            msg=f"Sequences {n}&{n + 1} have no errors, sequence {n + 2} was corrected!"
                        ) if need_logs else None
                    else:
                        log.output(
                            level=log.ERROR,
                            cls_name=ErrorTools.__name__,
                            meth_name=ErrorTools.error_correct.__name__,
                            msg=f"Sequences {n}&{n + 2} have errors, correction failed!"
                        ) if need_logs else None
                else:
                    log.output(
                        level=log.ERROR,
                        cls_name=ErrorTools.__name__,
                        meth_name=ErrorTools.error_correct.__name__,
                        msg=f"Sequences {n + 1}&{n + 2} have errors, correction failed!"
                    ) if need_logs else None

        return digital_segments


class ConsensusTools(object):

    @staticmethod
    def del_ont_barcode(ont_read: str, head: str = "CAGCACCT", tail: str = "GGTGCTGT", threshold: int = 150) -> str:
        """
        Introduction
        ------------
        Delete the barcode sequence from the ONT read.

        Parameters
        ----------
        ont_read: str
            The DNA sequence sequenced by ONT.
        head: str
            5' barcode sequence.
        tail: str
            3' barcode sequence.
        threshold: int
            The interval length of retrieving the barcode sequence starting from the head or tail.
               |<-------------------------------------sequence------------------------------------>|
            5' ===================================================================================== 3'
               |<- barcode ->|                                                       |<- barcode ->|
               |<-- threshold -->|                                               |<-- threshold -->|

        Returns
        -------
        The DNA sequence without the barcode sequence.
        """
        aligner = PairwiseAligner()
        original_read = ont_read[:]

        if head != "":
            position_head = ont_read.find(head)
            head_length = len(head)
            if position_head != -1:
                if position_head <= threshold:
                    ont_read = ont_read[position_head + head_length:]
                else:
                    raise DeleteHeadError(
                        f"The head: \n{head}\n"
                        f"is not found in the first {threshold} bases in original read: \n{original_read}\n "
                        f"and please check if ONT read has other different 5'-adapters"
                    )
            else:
                score = 0
                for i in range(threshold):
                    alignment = aligner.align(head, ont_read[i: i + len(head)])[0]

                    if alignment.score >= score:
                        score = alignment.score
                    else:
                        if head_length - score <= 2:
                            ont_read = ont_read[i + head_length:]
                            break

                if len(ont_read) == len(original_read):
                    raise DeleteHeadError(
                        f"The adapter sequence is not found in the first {threshold} bases, "
                        f"and please check if ONT read has other different 5'-adapters"
                    )

        if tail != "":
            without_head = ont_read[:]
            position_tail = ont_read.find(tail)
            tail_length = len(tail)
            if position_tail != -1:
                if position_tail >= len(ont_read) - threshold:
                    ont_read = ont_read[: position_tail]
                else:
                    raise DeleteTailError(
                        f"The tail: \n{tail}\n"
                        f"is not found in the last {threshold} bases in original read: \n{original_read}\n "
                        f"and please check if ONT read has other different 3'-adapters"
                    )
            else:
                score = 0
                for i in range(len(ont_read), len(ont_read) - threshold, -1):
                    alignment = aligner.align(tail, ont_read[i - len(tail): i])[0]

                    if alignment.score >= score:
                        score = alignment.score
                    else:
                        if tail_length - score <= 2:
                            ont_read = ont_read[: i - tail_length]
                            break

                if len(ont_read) == len(without_head):
                    raise DeleteTailError(
                        f"The adapter sequence is not found in the last {threshold} bases, "
                        f"and please check if ONT read has other different 3'-adapters"
                    )
        return ont_read

    @staticmethod
    def del_assembly_adaptor(sequence: str, assembly_adaptors: list, threshold: int = 20) -> str:
        """
        Introduction
        ------------
        Delete the assembly adaptor sequence from the DNA sequence.

        Parameters
        ----------
        sequence: str
            DNA sequence.
        assembly_adaptors: list
            Assembly adaptor sequences.
        threshold: int
            The interval length of retrieving the assembly adaptor sequence starting at nodes along two directions(5' and 3').
               |<--------------------------------part of sequence--------------------------------->|
            5' ===================================================================================== 3'
                             |<- adaptor ->|                               |<- adaptor ->|
               |<-threshold->|||<-threshold->|                       |<-threshold->|||<-threshold->|
                            node1                                                 node2

            Here, the length of oligos and the length between two nodes are both 100 nt.


        Returns
        -------
        The short DNA oligos.
        """
        adapter_used = 0
        disassembly_seqs = []

        aligner = PairwiseAligner()

        for i in range(len(assembly_adaptors)):
            adaptor = assembly_adaptors[i]
            position = sequence.find(adaptor)

            if position != -1 and position in range(80, 120):
                disassembly_seqs.append(sequence[: position])
                sequence = sequence[position + 8:]
                adapter_used += 1

            else:
                scores = []
                start = 100 - threshold
                end = 100 + threshold if len(sequence) >= 100 + threshold else len(sequence)

                if start >= end:
                    disassembly_seqs.append(sequence)
                    break

                else:
                    for j in range(start, end):
                        alignment = aligner.align(adaptor, sequence[j: j + 8])[0]
                        scores.append(alignment.score)

                    max_score = max(scores)
                    if max_score < 6 and end == len(sequence):
                        disassembly_seqs.append(sequence)
                        break

                    max_score_indices = np.where(np.array(scores) == max_score)[0]

                    if len(max_score_indices) == 1:
                        target_index = max_score_indices[0] + 100 - threshold
                    else:
                        hamming_distances = [
                            HammingDistanceTools.hamming_distance_in_seqs(
                                ref_seq=adaptor,
                                qry_seq=sequence[max_score_indices[x] + start: max_score_indices[x] + start + 8]
                            )
                            for x in range(len(max_score_indices))
                        ]
                        target_index = max_score_indices[hamming_distances.index(min(hamming_distances))] + 100 - threshold

                    disassembly_seqs.append(sequence[: target_index])
                    sequence = sequence[target_index + 8:]

                    adapter_used += 1

            if len(disassembly_seqs) != i + 1:
                warnings.warn(f"Adapter: {adaptor}-{i} and its similar sequences are not detected "
                              f"between index 80-120, which may indicate a greater degree of mutation "
                              f"in {adaptor}-{i}. Please check it separately")

        if len(assembly_adaptors) == 11 and sequence != "":
            disassembly_seqs.append(sequence[: 115])

        print(len(disassembly_seqs), [len(x) for x in disassembly_seqs])

        return disassembly_seqs[: 15] if len(assembly_adaptors) == 15 else disassembly_seqs[: 12]

    @staticmethod
    def get_consensus_seq(file: str | None = None, sequences: list[str] | None = None) -> str:
        """
        Introduction
        ------------
        Get the consensus sequence from the input sequences.

        Parameters
        ----------
        file: str
            The path of the file containing equal-length sequences.
        sequences: list
            Equal-length sequences.

        Select one of the parameters 'file' and 'sequences' to input.

        Returns
        -------
        Consensus sequence.
        """
        seqs = []
        if file is not None:
            with open(file, "r") as fa:
                if file.endswith(".fasta"):
                    seq = ""
                    while True:
                        line = fa.readline()
                        if not line:
                            break
                        if line.startswith(">"):
                            if seq != "":
                                seqs.append(seq)
                                seq = ""
                            else:
                                continue
                        else:
                            seq += line.strip()

                elif file.endswith(".txt"):
                    for seq in fa.readlines():
                        seqs.append(seq.strip())

        elif sequences is not None:
            seqs = sequences

        motif = motifs.create(seqs, alphabet="ACGTM-")
        consensus_seq = motif.consensus
        consensus_seq = consensus_seq.replace("-", "")

        return consensus_seq

    @staticmethod
    def compute_hamming_distance(ref_seq: str, qry_seq: str) -> int:
        """
        Introduction
        ------------
        Calculate the hamming distance between the reference sequence and the query sequence.

        Parameters
        ----------
        ref_seq: str
            Reference sequence.
        qry_seq: str
            Query sequence.

        Returns
        -------
        Hamming distance.
        """
        if len(ref_seq) == len(qry_seq):
            return sum([1 for i in range(len(ref_seq)) if ref_seq[i] != qry_seq[i]])
        else:
            return len(ref_seq)

    @staticmethod
    def del_duplicate_bases(sequence: str) -> str:
        """
        Introduction
        ------------
        Delete duplicate bases in the sequence.

        Parameters
        ----------
        sequence: str
            DNA sequence.

        Returns
        -------
        DNA sequence without duplicate bases.
        """
        pattern = r"([A, T, C, G, M])(\1+)"
        result = re.sub(pattern, r"\1", sequence)
        return result


class DeleteHeadError(Exception):

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class DeleteTailError(Exception):

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg
