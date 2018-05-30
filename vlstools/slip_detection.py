
import multiprocessing
import vlstools.alignments as al
from itertools import combinations, product
from .utils import Counter


def slippage_worker(inqueue: multiprocessing.Queue, outqueue: multiprocessing.Queue, counter: Counter,
                    references: dict, min_homology=1):
    """
    Optimized for speed, and implemented as a worker process that continues until queues empty.
    """
    while not inqueue.empty():
        transform, refid = inqueue.get()
        reference = references[refid]
        transform = list(transform)

        # Find all possible slippage events starting from the largest
        slippages = {x: [] for x in [None] + list(reference.cassettes_aln.keys())}
        for size_idx in reversed(range(1, len(transform) + 1)):  # slip size is here measured in number of ops
            for start_idx in range(0, len(transform) - size_idx + 1):
                stop_idx = start_idx + size_idx - 1

                # Search range goes up until the next non-templated SNP (or the edge of the reference) on both sides
                # of the window.
                if start_idx == 0:  # start at beginning
                    start_coord = 0
                else:
                    start_coord = transform[start_idx-1][0] + 1  # start at previous non-templated SNP
                if stop_idx == len(transform) - 1:
                    stop_coord = len(reference.seq)  # end at the end of the reference
                else:
                    stop_coord = transform[stop_idx+1][0]  # end at next non-templated SNP

                # compare the non-templated mapping in the window with the reference and cassettes
                readwin = al.transform(reference.seq, transform, start_coord, stop_coord)  # sequence in read
                refwins = {None: reference.seq[start_coord:stop_coord]}
                for casname, casaln in reference.cassettes_aln.items():  # cassette transforms
                    # trim end "deletions" from cassettes, which are due to cassettes being
                    # different lengths from the reference.
                    casaln = al.trim_transform(casaln.transform, len(reference.seq))

                    # do not include cassettes with a deletion that spans across the edges of the
                    # window
                    if not any(border_pos in range(op[0], op[0] + op[2]) for border_pos in (start_coord, stop_coord)
                               for op in casaln if op[1] == "D"):
                        refwins[casname] = al.transform(reference.seq, casaln, start_coord, stop_coord)

                for refname, refwin in refwins.items():
                    insertion_size = len(readwin) - len(refwin)  # inferred insertion size
                    # find the larger sequence
                    if insertion_size > 0:  # insertion
                        factor = 1
                        larger, smaller = readwin, refwin
                    elif insertion_size < 0:  # deletion
                        factor = -1
                        larger, smaller = refwin, readwin
                    else:  # no indel, so skip computing the effect of tandem repeats
                        continue

                    # find tandem repeats that explain the difference
                    window_results = []
                    for start, stop, unitlength in al.find_tandem_repeats(larger, minhomology=min_homology):
                        if insertion_size % unitlength == 0:
                            allowed_deletion = min((stop - start - min_homology, abs(insertion_size)))
                            units_deleted = allowed_deletion // unitlength
                            unitseq = larger[start:start + unitlength]
                            deletedseq = larger[:start] + larger[start + units_deleted * unitlength:]
                            if deletedseq == smaller:  # deletion is explanatory!
                                # collect results
                                window_results.append(
                                    [start + start_coord, stop - start, unitlength, factor * units_deleted,
                                     factor * units_deleted * unitlength, unitseq, size_idx])

                    # if there are slippages, narrow down the list to the largest and simplest.
                    if window_results:
                        start_length_of_results = len(window_results)
                        # select those with the smallest unit size
                        smallest_unit_size = min(r[2] for r in window_results)
                        window_results = [res for res in window_results if res[2] == smallest_unit_size]
                        # select those with the largest total repeat size:
                        largest_total_repeat = max(r[1] for r in window_results)
                        window_results = [res for res in window_results if res[1] == largest_total_repeat]
                        slippages[refname].append(tuple([start_idx, stop_idx] + window_results[0]))

        outqueue.put((tuple(transform), reference.name, slippages))
        counter.increment()
