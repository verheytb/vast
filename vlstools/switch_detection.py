
import multiprocessing
from itertools import combinations, product
from scipy.misc import comb
from .alignments import is_subset
from .utils import Counter, tprint, chunk_combinations, is_sublist


def switches_worker(inqueue: multiprocessing.Queue, outqueue: multiprocessing.Queue, counter: Counter,
                    references: dict):
    """
    Optimized for speed, and implemented as a worker process that continues until queues empty.
    """
    while not inqueue.empty():
        transform, refid = inqueue.get()
        reference = references[refid]
        transform = list(transform)
        # Find all possible switches that could explain any part of the read transform.
        switches = []
        for size_idx in range(1, len(transform) + 1):  # switch size is here measured in number of ops
            for start_idx in range(0, len(transform) - size_idx + 1):
                stop_idx = start_idx + size_idx
                origins = set()
                for casname, casaln in reference.cassettes_aln.items():
                    if is_sublist(transform[start_idx:stop_idx], list(casaln.transform)):
                        origins.add(casname)
                if origins:
                    switches.append((start_idx, stop_idx, frozenset(origins)))

        # simplify all the switches to exclude those that are merely subsets of larger switches. Without this optimization,
        # identifying more than 5 switches becomes too computationally intensive.
        switches.sort(key=lambda s: (s[1]-s[0]), reverse=True)
        nonredundant_switches = []
        for switch in switches:
            if not any(is_subset(switch, nr) for nr in nonredundant_switches):
                nonredundant_switches.append(switch)
        switches = nonredundant_switches
        # starting with 0, see how many switches are actually needed to explain.
        idxs_to_match = set(range(len(transform)))
        switch_sets = []

        switches_by_idx = [set() for x in range(len(transform))]
        for x in range(len(transform)):
            for switch in switches:
                if x in range(switch[0], switch[1]):
                    switches_by_idx[x].add(switch)

        # find positions where there is only one possible switch
        necessary = set(next(iter(switches)) for switches in switches_by_idx if len(switches) == 1)

        # break up into subproblems
        subproblems = []
        for idx, options in enumerate(switches_by_idx):
            # if this position is already covered by a necessary switch event, any other options will only increase
            # the number of switches in the final outcomes. Therefore, skip.
            if len(options & necessary) > 0:
                continue
            # start new subproblem
            if all(opt[0] == idx for opt in options) or len(switches_by_idx[idx-1] & necessary) > 0:
                subproblem = {"start": idx, "stop": idx + 1, "switches": options, "solutions": []}
                subproblems.append(subproblem)
            else:  # extend prior subproblem
                subproblems[-1]["switches"] |= options
                subproblems[-1]["stop"] = idx + 1

        # solve the subproblems
        for subproblem_id, subproblem in enumerate(subproblems):
            idxs_to_match = set(range(subproblem["start"], subproblem["stop"]))
            subproblem_switches = subproblem["switches"]
            for number_of_switches in range(0, len(subproblem["switches"]) + 1):
                for subset in combinations(iterable=subproblem_switches, r=number_of_switches):
                    if set(pos for switch in subset for pos in range(switch[0], switch[1])) >= idxs_to_match:
                        subproblem["solutions"].append(list(subset))
                if subproblem["solutions"]:
                    break

        # combine subproblems
        solutions_by_subproblem = [subproblem["solutions"] for subproblem in subproblems]
        for combined_solutions in product(*solutions_by_subproblem):
            switch_set = [switch for subproblem in combined_solutions for switch in subproblem] + list(necessary)
            switch_set.sort(key=lambda s: s[0])
            switch_sets.append(switch_set)

        # For large, non-redundant switch sets, find all the subsets that also explain
        expanded_switch_sets = []
        for switch_set in switch_sets:
            breakpoints = [[] for _ in switch_set]  # each sublist X_N in this list-of-lists represents the options for
                                                    # the start position of the Nth switch in the switch_set.
            for N, switch in enumerate(switch_set):
                start, stop, origins = switch
                if N == 0:
                    breakpoints[N].append(0)
                else:
                    breakpoints[N].extend(list(range(start, switch_set[N-1][1] + 1)))
            for break_list in product(*breakpoints):
                new_switch_set = []
                for N, breakpoint in enumerate(break_list):
                    if N + 1 == len(switch_set):
                        end = switch_set[-1][1]
                    else:
                        end = break_list[N+1]
                    new_switch_set.append((breakpoint, end, switch_set[N][2]))
                expanded_switch_sets.append(new_switch_set)
        switch_sets = expanded_switch_sets

        # score each switch set by total length, and select those with the minimum.
        lengths = []
        for switch_set in switch_sets:
            length = 0
            for switch in switch_set:
                start = transform[switch[0]][0]
                stop_idx = transform[switch[1]-1][0]
                length += stop_idx - start
            lengths.append(length)
        minlength = min(lengths)
        minimal_switch_sets = [switch_set for i, switch_set in enumerate(switch_sets) if lengths[i] == minlength]
        outqueue.put((tuple(transform), reference.name, minimal_switch_sets))
        counter.increment()
