#!/usr/bin/env python3

# builtins
import os
import pickle
import shutil
import multiprocessing
import warnings
from itertools import combinations
from random import randrange, choice
from types import SimpleNamespace
# dependencies
import numpy as np
from Bio import pairwise2
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from Bio.codonalign import CodonSeq
    from Bio.codonalign.codonseq import cal_dn_ds
try:
    import pysam
except ImportError:
    pysam = False
# included
try:
    import simanneal
except ImportError:
    from vlstools import simanneal
pairwise2.MAX_ALIGNMENTS = 10000


class Alignment(object):
    """
    Corresponds to one way of aligning a read to a reference.
    """

    def __init__(self, gappy_r, gappy_q):
        self.query = str(gappy_q)
        self.reference = str(gappy_r)
        self.transform = get_mapping(gappy_q=gappy_q, gappy_r=gappy_r)
        self.cigar, self.start, self.end = get_cigar(gappy_r=gappy_r, gappy_q=gappy_q)


def get_cigar(gappy_q, gappy_r):
    """

    :param gappy_q: gapped query sequence
    :param gappy_r: gapped reference sequence
    :return: returns a tuple, including a list of [operation, length] CIGAR, and the [start,end) 0-based coordinates of
    the alignment.
    """
    assert len(gappy_q) == len(gappy_r)
    cigar = []
    for q, r in zip(gappy_q, gappy_r):
        if q == "-":  # deletion
            if cigar and cigar[-1][0] == 2:
                cigar[-1][1] += 1
            else:
                cigar.append([2, 1])
        elif r == "-":  # insertion
            if cigar and cigar[-1][0] == 1:
                cigar[-1][1] += 1
            else:
                cigar.append([1, 1])
        else:
            if cigar and cigar[-1][0] == 0:
                cigar[-1][1] += 1
            else:
                cigar.append([0, 1])
    start, end = 0, 0
    if cigar[0][0] == 2:
        start = cigar[0][1]
        cigar.pop(0)
    else:
        start = 0
    if cigar[-1][0] == 2:
        cigar.pop()
        end = start + \
                   sum(length for op, length in cigar if op == 0) + \
                   sum(length for op, length in cigar if op == 2)
    return cigar, start, end


def transform(reference, mapping, start=0, stop=None):
    """
    Transforms a reference using a mapping object
    :param reference: a string
    :param mapping: a list of [{position},{"I"|"D"|"S"},{None|length_of_deletion|sequence_of_insertion}] operations
    :param start: option to specify a reference start position
    :param stop:  option to specify a reference stop position
    :return: a sequence as a string
    """
    if stop is None:
        stop = len(reference)
    query = list(reference[start:stop])
    window_mapping = [[op[0] - start] + list(op[1:]) for op in sorted(mapping, key=lambda x: x[0])
                      if start <= op[0] < stop]
    offset = 0
    for op in window_mapping:
        pos = op[0] + offset
        if op[1] == "S":
            query[pos] = op[2]
        elif op[1] == "I":
            query[pos:pos] = list(op[2])
            offset += len(op[2])
        elif op[1] == "D":
            del query[pos:pos+op[2]]
            offset -= op[2]
    # Effects deletions that precede but spill over into the window
    pre_deletions = [op for op in mapping if op[1] == "D" and op[0] < start < op[0] + op[2]]
    assert len(pre_deletions) in (0, 1)
    if pre_deletions:
        pre = pre_deletions[0]
        trim_length = pre[0] + pre[2] - start
        query = query[trim_length:]
    return "".join(query)


def score_left_justification(mapping: tuple, length: int):
    """
    Gives an integer score which can be used to sort alignments of the same sequence by their justification, especially
    for choosing the minimum scoring (most leftward) alignment. The score is the sum of the distances of all insertions
    from the start of the sequence and the sum of the distances of all deletions from the end of the sequence.
    :param mapping: a mapping (tuple of tuples)
    :param length: an integer
    :return:
    """
    score = 0
    for op in mapping:
        if op[1] == "I":
            score += len(op[2]) * op[0]  # insertions are measured from start
        elif op[1] == "D":
            for x in range(op[2]):
                score += length - op[0] + x
    return score


def nontemp_mask(reference, transformer, cassette_ops):
    """

    :param reference:
    :param transformer:
    :param cassette_ops:
    :return: returns a mask where for each base in a query sequence, there is a True at the position of all the bases
    that are either identical to the reference or found in the cassettes.
    """
    query = list(reference)
    mask = [False for x in query]
    transformer = sorted(transformer, key=lambda x: x[0])
    offset = 0
    for op in transformer:
        pos = op[0] + offset
        if op[1] == "S":
            query[pos] = op[2]
            mask[pos] = True if op in cassette_ops else False
        elif op[1] == "I":
            query[pos:pos] = list(op[2])
            mask[pos:pos] = [True]*len(op[2]) if op in cassette_ops else [False]*len(op[2])
            offset += len(op[2])
        elif op[1] == "D":
            del query[pos:pos+op[2]]
            del mask[pos:pos+op[2]]
            offset -= op[2]
    return mask


def count_snps(transform, count_indels_once=False):
    """
    Counts the SNPs (technically, the number of variant bases) from a reference by a read's "transform".
    :param transform:
    :return: the number of bases different from a reference sequence as represented in the provided transform.
    """

    if count_indels_once:
        return len(transform)
    else:
        count = 0
        for op in transform:
            if op[1] in "S":
                count += 1
            elif op[1] == "I":
                count += len(op[2])
            elif op[1] == "D":
                count += op[2]
            else:
                raise ValueError("Transform contained invalid operation: %s" % op[1])
        return count


def find_tandem_repeats(seq, minhomology=3, minunitsize=1):
    """
    Finds all tandem repeats including those partial repeats (those with at least <minhomology> number of bases
    contiguous with the last unit). These include overlapping or equivalent repeats (2-mers that are also 4-mers, for
    example).
    :param sequence:
    :return: a list of (start, stop, unitlength) tuples.
    """
    results = []
    for length in range(minunitsize, len(seq)):
        window = length + minhomology
        for x in range(len(seq)-window+1):
            unitseq = seq[x:x+length]
            nextseq = seq[x+length:x+window]
            if (unitseq+nextseq).startswith(nextseq):
                if results and results[-1][2] == length and results[-1][1] == x+window-1:
                    results[-1][1] += 1
                else:
                    results.append([x, x+length+minhomology, length])
    return results


def is_tandem_repeat(start, stop, unitlength, sequence):
    """

    :param start: the start (inclusive) of a tandem repeat sequence
    :param stop: the stop (exclusive) of a tandem repeat sequence
    :param unitlength: the unit length of the repeat sequence
    :param sequence: the sequence to which the coordinates refer
    :return: boolean True or False if the tandem repeat is in the sequence.
    """
    subseq = sequence[start:stop]
    first_unit = subseq[:unitlength]
    subseq = subseq[unitlength:]
    while subseq:
        thisunit = subseq[:unitlength]
        subseq = subseq[unitlength:]
        if not first_unit.startswith(thisunit):
            return False
    return True


def get_mapping(gappy_q, gappy_r):
    """
    Produces a mapping from a gapped pairwise alignment. A mapping represents a mapped read, and contains all the
    information to reconstruct a read sequence based on a reference sequence.
    :return: tuple of tuples (basically a list of operations to perform on the reference to get the read).
    """
    transform = []
    rgaps = 0  # counts number of reference gaps
    for x, r in enumerate(str(gappy_r)):
        x_ref = x-rgaps
        q = str(gappy_q)[x]
        if q == r:  # match
            continue
        elif q == "-":  # deletion
            if transform and transform[-1][1] == "D" and transform[-1][0] + transform[-1][2] == x_ref:
                # follows right after a D
                transform[-1][2] += 1
            else:  # start of new deletion
                transform.append([x_ref, "D", 1])
        elif r == "-":  # insertion
            if transform and transform[-1][1] == "I" and transform[-1][0] == x_ref:  # follows right after an I
                transform[-1][2] += q
            else:  # start of new insertion
                transform.append([x_ref, "I", q])
            rgaps += 1
        else:  # mismatch
            transform.append([x_ref, "S", q])
    return tuple(tuple(x) for x in transform)  # convert to tuple of tuples


def trim_transform(mapping, ref_length: int):
    """
    Trims mapped sequences that have a deletion operation on the ends arising from aligning a shorter sequence to a
    longer one.
    :param mapping: an iterable of (position, {"S", "D", "I"}, int for deletion or string for insertion)
    :param ref_length:
    :return: a transform that has been trimmed to remove "D" (deletion) operations from the ends.
    """
    mapping = list(mapping)
    if mapping:
        if mapping[0][1] == "D" and mapping[0][0] == 0:
            del mapping[0]
        if mapping[-1][1] == "D" and mapping[-1][0] == ref_length - mapping[-1][2]:
            del mapping[-1]
    return tuple(mapping)


def trim_alignment(alignment: Alignment, ref_length: int):
    """
    Same as trim_transform, except it works on Alignment objects and returns the start and stop coordinates of the new
    alignment.
    :param alignment:
    :param ref_length:
    :return:
    """
    mapping = list(alignment.transform)
    start = 0
    stop = ref_length
    if mapping:
        if mapping[0][1] == "D" and mapping[0][0] == 0:
            start = alignment.start + mapping.pop(0)[2]
        if mapping[-1][1] == "D" and mapping[-1][0] == ref_length - mapping[-1][2]:
            stop = alignment.end - mapping.pop(-1)[2]
    alignment.transform = tuple(mapping)
    return start, stop, alignment


def align(reference, query):
    """
    do a pairwise alignment of the query to the reference, outputting up to 10000 of the highest-scoring alignments.
    :param reference: a STRING of the reference sequence
    :param query: a STRING of the query sequence
    :return: a list of up to 10000 Alignment objects
    """
    alns = pairwise2.align.localms(reference, query, 1, -1, -2, -1)  # match, mismatch, gap-open, gap-extension
    alignments = []
    for aln in alns:
        al1, al2, score, begin, end = aln
        alignments.append(Alignment(gappy_r=al1, gappy_q=al2))
    return alignments


def align_proteins(reference, query):
    """
    do a pairwise alignment of the query to the reference, outputting up to 10000 of the highest-scoring alignments.
    :param reference: a STRING of the reference sequence
    :param query: a STRING of the query sequence
    :return: a list of up to 10000 Alignment objects
    """
    alns = pairwise2.align.globalms(reference, query, 1, -1, -2, -1)  # match, mismatch, gap-open, gap-extension
    alignments = []
    for aln in alns:
        al1, al2, score, begin, end = aln
        alignments.append(Alignment(gappy_r=al1, gappy_q=al2))
    return alignments

def map_distance(mapping1, mapping2):
    """
    Measures the distance of any two mapped reads based on a mapping to a common reference
    :param mapping1: A mapping of a read to a reference
    :param mapping2: A mapping of another read to the same reference
    :return: an integer score corresponding to the number of different bases that differ between the two mappings.
    """
    score = 0
    # substitutions are per base, so we just count the number of different entries in the transforms
    aln1_s = {x for x in mapping1 if x[1] == "S"}
    aln2_s = {x for x in mapping2 if x[1] == "S"}
    score += len(aln1_s ^ aln2_s)

    # deletions are calculated as the difference between the lengths
    aln1_d = {x for x in mapping1 if x[1] == "D"}
    aln2_d = {x for x in mapping2 if x[1] == "D"}
    dels = aln1_d ^ aln2_d
    while dels:
        position1, _, length1 = dels.pop()
        others = {x for x in dels if x[0] == position1}
        if others:
            assert len(others) == 1
            other = others.pop()
            dels.discard(other)
            position2, _, length2 = other
            score += abs(length2 - length1)
        else:
            score += length1

    # insertions that are non-identical must be compared at the subsequence level in order to count the number of
    # differences
    aln1_i = {x for x in mapping1 if x[1] == "I"}
    aln2_i = {x for x in mapping2 if x[1] == "I"}
    inserts = aln1_i ^ aln2_i
    while inserts:
        position1, _, inseq1 = inserts.pop()
        others = {x for x in inserts if x[0] == position1}
        if others:
            assert len(others) == 1
            other = others.pop()
            inserts.discard(other)
            position2, _, inseq2 = other
            if inseq1.endswith(inseq2) or inseq1.startswith(inseq2):
                score += len(inseq1) - len(inseq2)
            elif inseq2.endswith(inseq1) or inseq2.startswith(inseq1):
                score += len(inseq2) - len(inseq1)
            else:
                score += max(len(inseq1), len(inseq2))
        else:
            score += len(inseq1)
    return score


def snp_histogram(reads, reference, templated_ops=None, templated=True):
    """
    Returns a histogram of the frequency of base-changes at each position.
    :param reads: a list of reads
    :param reference: the reference object
    :param templated_ops: a list of templated ops; if None, returns all SNPs, irrespective of the templated parameter
    :param templated: If templated_ops are specified, then templated=True will return the templated histogram, and
    False will return the non-templated histogram.
    :return: a list of floats
    """
    hist = [0] * len(reference.seq)
    for read in reads:
        for aln in read.alns:
            for op in aln.transform:
                # determines whether to include this op in histogram
                if templated_ops:
                    if (op in templated_ops) == templated:
                        include = True
                    else:
                        include = False
                else:
                    include = True

                # adds to histogram the frequency at each position, normalizing by the number of possible alignments of
                # the read.
                if op[1] in "S":
                    if include:
                        hist[op[0]] += 1 / len(reads) / len(read.alns)
                elif op[1] == "D":
                    for x in range(op[2]):
                        if include:
                            hist[op[0] + x] += 1 / len(reads) / len(read.alns)
                elif op[1] == "I":
                    if include:
                        hist[op[0]] += len(op[2]) / len(reads) / len(read.alns)
    return hist


class CassetteChoiceProblem(simanneal.Annealer):

    """
    Annealer class for the multiple alignment problem. This version saves computed map distances so
    that energy calculations become faster over time.
    """

    # pass extra data (the alignments) into the constructor
    def __init__(self, state, alns):
        self.alignments = alns
        self.lengths = [len(x) for x in self.alignments]
        self.known_map_distances = {}
        super(CassetteChoiceProblem, self).__init__(state)  # important!

    def move(self):
        """Randomly selects a different alignment of one of the cassettes"""
        parameter_to_change = randrange(len(self.state))
        self.state[parameter_to_change] = randrange(0, self.lengths[parameter_to_change])

    def energy(self):
        """Calculates the sum of all pairwise distances."""
        e = 0
        for pair in combinations([(cas, aln) for cas, aln in enumerate(self.state)], r=2):
            if pair in self.known_map_distances:
                e += self.known_map_distances[pair]
            else:
                pair_distance = map_distance(self.alignments[pair[0][0]][pair[0][1]],
                                             self.alignments[pair[1][0]][pair[1][1]])
                e += pair_distance
                self.known_map_distances[pair] = pair_distance
        return e


def anneal(arg_tuple):
    init_state, maxtemp, mintemp, aligned_reads = arg_tuple
    problem = CassetteChoiceProblem(init_state, aligned_reads)
    problem.copy_strategy = "slice"
    problem.steps = 10000
    problem.Tmax, problem.Tmin = maxtemp, mintemp
    problem.updates = 0
    return problem.anneal()


def align_cassettes(reference, outputdir, cpus=multiprocessing.cpu_count()):
    """Aligns a set of cassettes to a reference, minimizing the collective edit distance.
    :param reference: a string containing the reference sequence
    :returns a dictionary {name: aln_object for each cassette}
    """
    # Maps each cassette to the reference
    worker_args = ((reference.seq, cassette_name, cassette_seq)
                   for cassette_name, cassette_seq in reference.cassettes.items())
    aligned_reads = []
    aligned_read_names = []
    aligned_reads_for_bam = []
    with multiprocessing.Pool(cpus) as p:
        for x in p.map(cassette_align_worker, worker_args):
            aligned_read_names.append(x[0])
            aligned_reads.append(x[1])
            aligned_reads_for_bam.append(x[2])

    max_temps = [2500, 1500, 1000, 750, 500, 400]
    min_temps = [0.5, 0.1, 0.05, 0.01]
    pool_args = ([[randrange(0, len(alns)) for alns in aligned_reads], choice(max_temps), choice(min_temps), aligned_reads]
                 for _ in range(1000))

    with multiprocessing.Pool(cpus) as p:
        results = p.map(anneal, pool_args)  # a list of state, energy pairs
    min_energy = min([energy for state, energy in results])
    reduced_results = []
    for state, energy in results:
        if energy == min_energy:
            if (state, energy) not in reduced_results:
                reduced_results.append((state, energy))

    # text file results
    with open(os.path.join(outputdir, reference.name + ".txt"), "w") as f:
        for x, (state, energy) in enumerate(reduced_results):
            f.write("Result %d: Energy = %d\n" % (x, energy))
            for cas, aln in enumerate(state):
                f.write(str(aligned_read_names[cas]) + ": " + str(aligned_reads[cas][aln]) + "\n")
            f.write("\n")

    # pyc output of all equivalent results
    for x, (state, energy) in enumerate(reduced_results):
        msa = [(aligned_read_names[cas],
                transform(reference=reference.seq, mapping=aligned_reads_for_bam[cas][aln].transform),
                aligned_reads_for_bam[cas][aln]) for cas, aln in enumerate(state)]
        with open(os.path.join(outputdir, "%d.pyc" % x), "wb") as h:
            pickle.dump(msa, h, protocol=4)

    # bam file results
    if pysam:  # doesn't output bam files if pysam isn't found.
        for x, (state, energy) in enumerate(reduced_results):
            header = {'HD': {'VN': '1.0'}, 'SQ': [{'LN': len(reference.seq), 'SN': reference.name}]}
            with pysam.AlignmentFile(os.path.join(outputdir, "%d.bam" % x), "wb", header=header) as outf:
                for cas, aln in enumerate(state):
                    alignment = aligned_reads_for_bam[cas][aln]
                    a = pysam.AlignedSegment()
                    a.query_name = aligned_read_names[cas]
                    a.query_sequence = transform(reference=reference.seq, mapping=alignment.transform)
                    a.reference_id = 0
                    a.reference_start = alignment.start
                    a.cigar = alignment.cigar
                    outf.write(a)
            pysam.sort("-o", os.path.join(outputdir, "%d-sorted.bam" % x), os.path.join(outputdir, "%d.bam" % x))
            shutil.move(os.path.join(outputdir, "%d-sorted.bam" % x), os.path.join(outputdir, "%d.bam" % x))
            pysam.index(os.path.join(outputdir, "%d.bam" % x))


    # return the first result
    msa_zero = [(aligned_read_names[cas],
                transform(reference=reference.seq, mapping=aligned_reads_for_bam[cas][aln].transform),
                aligned_reads_for_bam[cas][aln]) for cas, aln in enumerate(reduced_results[0][0])]

    return {cassette[0]: cassette[2] for cassette in msa_zero}


def cassette_align_worker(arg_tuple):
    """
    wraps the align function to pass through the query name and yields both a simple mapping as well as the full
    alignment objects.
    :param arg_tuple: a reference, query tuple
    :return: a list of up to 10000 Alignment objects
    """
    reference, query_name, query_seq = arg_tuple
    alns = align(reference=reference, query=query_seq)
    trimmed_transforms = []
    for aln in alns:
        trimmed_transforms.append(trim_transform(mapping=aln.transform, ref_length=len(reference)))
    return query_name, trimmed_transforms, alns


def read_align_worker(arg_tuple):
    (seq, reference) = arg_tuple
    alignments = align(reference=reference.seq, query=seq)
    if reference.cassettes_aln:

        alignment_distances = [sum(map_distance(aln.transform, cas_aln.transform)
                                   for cas_aln in reference.cassettes_aln.values())
                               for aln in alignments]
        min_distance = min(alignment_distances)
        best_alignments = [alignments[x] for x, score in enumerate(alignment_distances) if score == min_distance]
    else:
        best_alignments = alignments  # if no cassettes are provided, all equally scoring alignments are returned
    return seq, reference.name, best_alignments


def write_bam(filename, refid, refseq, reads_dict):
    """

    :param filename:
    :param refid:
    :param refseq:
    :param reads:
    :return:
    """

    header = {'HD': {'VN': '1.0'}, 'SQ': [{'LN': len(refseq), 'SN': refid}]}
    with pysam.AlignmentFile(filename, "wb", header=header) as outf:
        for readname, alignment in reads_dict.items():
            a = pysam.AlignedSegment()
            a.query_name = readname
            a.query_sequence = transform(reference=refseq, mapping=alignment.transform,
                                         start=alignment.start, stop=alignment.end)
            a.reference_id = 0
            a.reference_start = alignment.start
            a.cigar = alignment.cigar
            outf.write(a)
    pysam.sort("-o", filename[:-4] + ".sorted.bam", filename)
    shutil.move(filename[:-4] + ".sorted.bam", filename)  # overwrites original output with sorted bam file.
    pysam.index(filename)


def is_subset(smaller, larger):
    """
    checks if a switch event is a subset of a larger one
    :param smaller: a (int, int, frozenset(str))
    :param larger: a (int, int, frozenset(str))
    :return: True if smaller in larger else False
    """
    if smaller[2] != larger[2]:
        return False
    larger_range = range(larger[0], larger[1] + 1)
    if smaller[0] in larger_range and smaller[1] in larger_range:
        return True
    else:
        return False


def templated(mapping, reference):
    """

    :param mapping:
    :param reference:
    :return:
    """
    # make a list of all the SNPs that are templated
    templated_ops = [op for aln in reference.cassettes_aln.values() for op in
                     trim_transform(aln.transform, len(reference.seq))]
    templated_mapping = tuple(op for op in mapping if op in templated_ops)
    return templated_mapping


def nontemplated(mapping, reference):
    """

    :param mapping:
    :param reference:
    :return:
    """
    # make a list of all the SNPs that are templated
    templated_ops = [op for aln in reference.cassettes_aln.values() for op in
                     trim_transform(aln.transform, len(reference.seq))]
    nontemplated_mapping = tuple(op for op in mapping if op not in templated_ops)
    return nontemplated_mapping


def find_poly_g(sequence, size):
    assert all(b in "actgACTG" for b in sequence)
    array = [False for x in sequence]
    for x, b in enumerate(sequence):
        if sequence[x:x+size] == "G"*size:
            array[x] = True
    return array


def get_codon_positions(reference):
    if reference.offset % 3 == 0:
        start = 0
    else:
        start = 3 - reference.offset % 3
    stop = start + (len(reference.seq) - start) // 3 * 3
    return list(range(start, stop, 3))


def translate(dna, offset=0):
    """
    Translates a DNA sequence, with a given offset.
    :param dna: a string with a DNA sequence.
    :return: protein sequence. Underscores denote stop codons.
    """
    dna = dna.upper()
    assert all(d in "ATCG" for d in dna)
    codontable = {
        'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
        'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
        'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
        'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
        'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
        'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
        'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
        'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
        'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
        'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
        'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
        'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_',
        'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W',
    }
    codons = [dna[x:x+3] for x in range(3 - offset % 3, len(dna), 3)]
    protein = ''.join([codontable[c] for c in codons if len(c) == 3])
    return protein


def templated_variants(iterable_of_reads, reference):
    """
    Get a set of variants from a bin of reads.
    :param iterable_of_reads: iterable containing reads.
    :param reference: the reference that the reads are aligned to.
    :return: a list of variants; each variant is a set of mappings.
    """
    variants = []
    for read in iterable_of_reads:
        variant = frozenset(templated(aln.transform, reference) for aln in read.alns)
        variants.append(variant)
    return variants


def get_frame_shift(op):
    """

    :param op:
    :return:
    """
    if op[1] == "I":
        return len(op[2])
    elif op[1] == "S":
        return 0
    elif op[1] == "D":
        return -op[2]


def error_scrub(mapping, window_length=20):
    """
    Detects erroneous bases in a mapped read by detecting frameshifts that last longer than a specified window_length.
    :param mapping: a list of lists
    :param reference: a reference object
    :param window_length: an integer of bases, defaults to 20bp.
    :return: a new mapping with errors removed; should be the length of the reference plus or minus multiples of three.
    """
    scrubbed = []
    diffs = [(op_idx, op[0], get_frame_shift(op)) for op_idx, op in enumerate(mapping)]
    while len(diffs) > 0:
        for diff_idx, (op_idx, x, shift) in enumerate(diffs):
            subdiffs = diffs[:diff_idx+1]
            if len(subdiffs) == 1:  # start the window if there is only one diff in the subdiff
                window_start = x
            if x > window_start + window_length or len(subdiffs) == len(diffs):  # reached the end of window
                diffs.pop(0)
                break
            elif sum([x[2] for x in subdiffs]) % 3 == 0:
                for _ in range(len(subdiffs)):
                    scrubbed.append(mapping[diffs.pop(0)[0]])
                break
    return scrubbed


def translate_mapping(mapping: list, reference: SimpleNamespace, templ: bool=True, nontempl: bool=True,
                      correctframe: bool=True, filterframe: bool=True, filternonsense: bool=True):
    """
    creates a protein mapping from a dna mapping.
    :param mapping: a list/tuple of ops.
    :param reference: the reference object to which the mapping is relative.
    :param templ: include templated ops
    :param nontempl: include nontemplated ops
    :param correctframe: removes isolated ops that disrupt the frame
    :param filterframe: don't return a mapping if there are remaining frameshifts.
    :param filternonsense: don't return a mapping if contains a stop codon
    :return:
    """

    # create a mapping with the appropriate SNPs
    base_mapping = []
    if templ:
        base_mapping.extend(templated(mapping, reference))
    if nontempl:
        base_mapping.extend(nontemplated(mapping, reference))
    base_mapping.sort(key=lambda x: x[0])

    # correct errors
    if correctframe:
        base_mapping = error_scrub(base_mapping)

    # filter for whether it is in frame or not.
    if filterframe and not len(transform(reference.seq, base_mapping)) % 3 == len(reference.seq) % 3:
        return []

    protein = translate(transform(reference.seq, base_mapping), offset=reference.offset)
    if filternonsense and "_" in protein:
        return []

    protein_alns = align_proteins(reference.protein, protein)
    return protein_alns


def is_templated(op, reference):
    # make a list of all the SNPs that are templated
    templated_ops = {tuple(op) for aln in reference.cassettes_aln.values() for op in
                     trim_transform(aln.transform, len(reference.seq))}
    return tuple(op) in templated_ops


def get_dnds(alignment, reference):
    """
    Gets the dN (frequency of nonsynonymous mutations) and dS (frequency of synonymous mutations) from an alignment.
    :param mapping:
    :param reference:
    :return: Namespace(dN, dS)
    """
    # patch a bug in sequence data in the database TODO
    if alignment.end == 0:
        alignment.end = len(reference.seq)
    codon_starts = [x for x in get_codon_positions(reference=reference) if alignment.start <= x <= alignment.end - 3]
    reference_codons = "".join([reference.seq[x:x+3] for x in codon_starts])
    ref_cs = CodonSeq(reference_codons)  # biopython object
    substitutions = [op for op in alignment.transform if op[1] == "S"]
    variant_seq = transform(reference=reference.seq, mapping=substitutions)
    variant_codons = "".join([variant_seq[x:x+3] for x in codon_starts])
    var_cs = CodonSeq(variant_codons) # biopython object
    try:
        dn, ds = cal_dn_ds(ref_cs, var_cs, method="NG86")
    except KeyError:
        dn, ds = None, None
    return SimpleNamespace(dn=dn, ds=ds)

def get_burs(ref_seq, ref_cassettes_aln):
    """
    Assemble left and right boundary uncertainty regions (BURs) around each SNP in the cassettes
    :param reference:
    :return:
    """
    left_burs = {casname: {} for casname in ref_cassettes_aln}
    right_burs = {casname: {} for casname in ref_cassettes_aln}
    for casname, casaln in ref_cassettes_aln.items():
        casaln.transform = trim_transform(casaln.transform, len(ref_seq))
        for idx, op in enumerate(casaln.transform):
            if idx == 0:  # first op in transform
                left_burs[casname][op] = casaln.start
            else:
                left_burs[casname][op] = casaln.transform[idx - 1][0]
            if idx == len(casaln.transform) - 1:  # last op in transform
                right_burs[casname][op] = casaln.end
            else:
                right_burs[casname][op] = casaln.transform[idx + 1][0]

    return SimpleNamespace(left=left_burs, right=right_burs)


def get_slips(read, slips, reference):
    """
    Return a list of slips from the alignment with the most non-templated SNPs explained by Polymerase slippage events.
    :param read:
    :param slips:
    :param reference:
    :return:
    """
    results_per_aln = []
    for aln in read.alns:
        nontemplated_tf = nontemplated(aln.transform, reference)
        if (nontemplated_tf, reference.name) not in slips:
            raise ValueError(
                "Read %s has not been analysed for slippage. Run \"vls label_slippage\" to "
                "calculate polymerase slippage before exporting report.")
        slips_by_origin = slips[(nontemplated_tf, reference.name)]
        result = set(slip for origin in slips_by_origin.values() for slip in origin)
        nt_idxs_explained = set()
        for slip in result:
            for nt_idx in range(slip[0], slip[1] + 1):
                nt_idxs_explained.add(nt_idx)
        nontemplated_tf_explained = tuple(op for idx, op in enumerate(nontemplated_tf)
                                          if idx in nt_idxs_explained)
        score = count_snps(transform=nontemplated_tf_explained)
        results_per_aln.append((score, aln, result))

    results_per_aln.sort(key=lambda x: x[0])
    best_slipset = results_per_aln[0][2]
    best_alignment = results_per_aln[0][1]
    return best_slipset, best_alignment


def get_aa_frequencies(argtuple):
    read, methods, coords, reference = argtuple
    frequency = {method: np.zeros([len(coords)]) for method in methods}
    for method in methods:
        for aln in read.alns:
            protein_alns = translate_mapping(mapping=trim_transform(aln.transform, len(reference.seq)),
                                             reference=reference,
                                             templ=("_all" in method or "_templated" in method),
                                             nontempl=("_all" in method or "_nontemplated" in method),
                                             correctframe=("corrected" in method),
                                             filterframe=("inframe" in method),
                                             filternonsense=("inframe" in method)
                                             )
            for protein_aln in protein_alns:
                for op in protein_aln.transform:
                    if op[1] == "S":
                        frequency[method][op[0]] += 1 / len(read.alns) / len(protein_alns)
                    if op[1] == "D":
                        for x in range(op[2]):
                            frequency[method][op[0] + x] += 1 / len(read.alns) / len(protein_alns)
                    elif op[1] == "I":
                        frequency[method][op[0]] += len(op[2]) / len(read.alns) / len(protein_alns)
    return frequency
