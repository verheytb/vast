"""
A bunch of tools for multiprocessing, UI formatting,
"""
from multiprocessing import Value
from itertools import combinations
from datetime import datetime
from math import exp
import re
import numpy as np


class Counter(object):
    """
    A multiprocessing-safe counter. Use increment() to increase, or the value property to read.
    """
    def __init__(self, initval=0):
        self.val = Value('i', initval)

    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    def reset(self):
        with self.val.get_lock():
            self.val.value = 0

    @property
    def value(self):
        return self.val.value


class VennTable(object):
    """
    Given a dictionary of sets, computes all the overlapping values, and stores.
    """
    def __init__(self, dict_of_sets):
        assert isinstance(dict_of_sets, dict)  # must be dict
        assert all(isinstance(s, (set, frozenset)) for s in dict_of_sets.values())  # dict values must be sets
        self.data = dict_of_sets

    def get_labels(self):
        return set(self.data.keys())

    def number_of_sets(self):
        return len(self.data)

    def get_overlap(self, labels):
        """
        Get the value for one segment of the venn diagram.
        :param labels: a list, tuple, set, or frozenset of labels
        :return: an integer of the number of sequences shared between the supplied labels and not shared with those not
        supplied.
        """
        assert isinstance(labels, (list, tuple, set, frozenset))
        include = [self.data[label] for label in labels]
        exclude = [self.data[label] for label in self.data if label not in labels]
        segment = set.intersection(*include) - (set.union(*exclude) if exclude else set())
        return len(segment)


def tprint(contents, ontop=False):
    """

    :param contents: a string to print as a message
    :param ontop: if True, does not end with a newline
    :return:
    """
    if not ontop:
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "   " + contents)
    else:
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "   " + contents, end="\r")


def quiet_exp(x):
    """

    :param x:
    :return:
    """

    try:
        result = exp(x)
    except OverflowError:
        result = float("inf")
    return result


def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None


def chunk_combinations(iterable, r, start_indices=None, stop_indices=None):
    # modified from python's itertools.combinations, to support splitting the combinatorial generator into chunks for
    # multiprocessing.
    #
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations('ABCD', 2, [0,1], [1,2]) --> AB AC AD
    # combinations('ABCD', 2, [1,2], [2,3]) --> BC BD
    # combinations('ABCD', 2, [2,3], [3,4]) --> CD
    pool = tuple(iterable)
    n = len(pool)
    assert start_indices is None or len(start_indices) == r
    assert stop_indices is None or len(stop_indices) == r
    if r > n:
        return
    if start_indices:
        indices = start_indices
    else:
        indices = list(range(r))
    yield tuple(pool[k] for k in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        if indices == stop_indices:
            return
        else:
            yield tuple(pool[k] for k in indices)


def interval_start_iterator(l, n, k):
    """
    Generates lists of start positions for all the arrangements of n intervals with length k in total sequence length l.
    This algorithm using a bijection from each result to a bit array to simplify finding the positions, an idea from
    Arron Kau at https://brilliant.org/discussions/thread/stars-and-bars/.
    :param l: length of total sequence
    :param n: number of intervals
    :param k: length of intervals
    :return:
    """
    if n == 0:
        return
    sumtotal = l - n * k
    zero_position_options = list(range(sumtotal + n))
    for zero_bits in combinations(zero_position_options, n):
        # Note: assumes zero_bits is sorted, which it should be from itertools.combinations
        starts = [zero_bits[0]]
        for x, zero_pos in enumerate(zero_bits[1:], start=1):
            starts.append(starts[-1] + k + (zero_pos - zero_bits[x-1] - 1))
        yield starts


def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


def normalize(vector):
    """
    calculates the unit-length vector of a given vector
    :param vector: an iterable of integers
    :return: a list of integers
    """
    return [int/sum(vector) for int in vector]


def round_up(x):
    """rounds positive numbers up to one significant figure"""
    pos = np.int(np.floor(np.log10(np.abs(x))))
    return np.ceil(x/(10**pos))*10**pos


def is_sublist(smaller, larger):
    """
    Tests if one list is found as a slice of a larger list
    :param smaller: a list to be tested
    :param larger: a list that might contain a smaller list
    :return: True or False
    """
    for i in range(len(larger) - len(smaller) + 1):
        if smaller == larger[i:i+len(smaller)]:
            return True
    return False


def get_tagstring(refid, tags):
    """Creates a string from a reference ID and a sequence of tags, for use in report filenames."""
    return "_".join([refid] + [t[0] + "." + (t[1] if t[1] else "None") for t in tags])


def qq_values(sample1, sample2):
    """
    Computes the qqplot comparing two lists of samples
    :param sample1: a list of bootstrapping 1D datasets (a list of lists of ints)
    :param sample2: a list of bootstrapping 1D datasets (a list of lists of ints)
    :return: a list of sample1, sample2 duples for plotting in a qqplot.
    """
    s1 = []
    s2 = []
    for q in np.arange(start=0.1, stop=100, step=0.1):
        s1.append(np.mean([np.percentile(sample_set, q) for sample_set in sample1]))
        s2.append(np.mean([np.percentile(sample_set, q) for sample_set in sample2]))
    return s1, s2