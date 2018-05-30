"""
Contains simulations that are necessary for some reports
"""


from .utils import tprint
from .alignments import trim_transform
import random
import numpy as np


def simulate_switch_length(db, num_trials, recompute=False):
    references = db.get("references")
    aligned_refs = [r for r in references.values() if r.cassettes_aln]
    tprint("Running simulations for the following references: " + ", ".join(r.name for r in aligned_refs))
    for ref_num, ref in enumerate((r for r in references.values() if r.cassettes_aln)):
        if not hasattr(ref, "sim_switch") or ref.sim_switch is None or recompute:
            alns = list(ref.cassettes_aln.values())
            results = []
            for actual in range(1, len(ref.seq)):
                trials = []
                while len(trials) < num_trials:
                    transform = random.choice(alns).transform
                    start = random.randrange(0, len(ref.seq) - actual)
                    stop = start + actual
                    op_positions = [op[0] for op in transform if op[0] in range(start, stop)]
                    if len(op_positions) > 0:  # note that trials with no observable change are discarded because they
                                               # would never be detected as a switch event for which we are using this
                                               # correction factor.
                        measured = max(op_positions) - min(op_positions) + 1
                        trials.append(measured)
                        if len(trials) % 1000 == 0:
                            tprint("Completed %d of %d references, %d of %d switch lengths, %d of %d trials." %
                                   (ref_num, len(aligned_refs), actual, len(ref.seq), len(trials), num_trials),
                                   ontop=True)
                if trials:
                    results.append((actual, np.mean(trials), np.std(trials), len(trials)))
            ref.sim_switch = results  # saves the results as an attribute of the references
    db.save(references, "references")


def simulate_switch_endpoints(reference, num_trials: int, switch_size: int, g_len: int):
    alns = list(reference.cassettes_aln.values())
    left_array = []
    right_array = []
    while len(left_array) < num_trials:
        transform = sorted(trim_transform(random.choice(alns).transform, len(reference.seq)), key=lambda x: x[0])
        start = random.randrange(0, len(reference.seq) - switch_size)
        stop = start + switch_size
        switch_ops = [(idx, op) for idx, op in enumerate(transform) if op[0] in range(start, stop)]
        if len(switch_ops) > 0:  # note that trials with no observable change are discarded because they
                                 # would never be detected
            if switch_ops[0][0] > 0:
             start_range = [transform[switch_ops[0][0] - 1][0], transform[switch_ops[0][0]][0]]
            else:  # no previous snps; start from beginning of read
             start_range = [None, transform[switch_ops[0][0]][0]]
            if switch_ops[-1][0] < len(transform) - 1:
             stop_range = [transform[switch_ops[-1][0]][0], transform[switch_ops[-1][0] + 1][0]]
            else:  # no following snps; go to end of read
             stop_range = [transform[switch_ops[-1][0]][0], None]
            if "G" * g_len in reference.seq[start_range[0]:start_range[1]]:
                left_array.append(1)
            else:
                left_array.append(0)
            if "G" * g_len in reference.seq[stop_range[0]:stop_range[1]]:
                right_array.append(1)
            else:
                right_array.append(0)
            if len(left_array) % 1000 == 0:
                tprint("Completed %d of %d simulated switches of length %d from reference %s." %
                       (len(left_array), num_trials, switch_size, reference.name),
                       ontop=True)
    return {"left": {"mean": np.mean(left_array), "std": np.std(left_array), "nobs": len(left_array)},
            "right": {"mean": np.mean(right_array), "std": np.std(right_array), "nobs": len(right_array)}}


def simulate_double_switch_distance(reference, length, num_trials=1000000):
    alns = list(reference.cassettes_aln.values())
    results = {"length": length, "num_trials": num_trials, "min": [], "mid": [], "max": []}
    x = 0
    while x < num_trials:
        switches = []
        while len(switches) < 2:
            cassette_tf = random.choice(alns).transform
            start = random.randrange(0, len(reference.seq) - length)
            stop = start + length
            switch_tf = [op[0] for op in cassette_tf if op[0] in range(start, stop)]
            if switch_tf:
                min_start, min_stop = min(switch_tf), max(switch_tf)
                pre_switch_tf = [op[0] for op in cassette_tf if op[0] in range(0, start)]
                if pre_switch_tf:
                    max_start = max(pre_switch_tf)
                else:
                    max_start = 0
                post_switch_tf = [op[0] for op in cassette_tf if op[0] in range(stop, len(reference.seq))]
                if post_switch_tf:
                    max_stop = min(post_switch_tf)
                else:
                    max_stop = len(reference.seq)
                positions = {"min_start": min_start, "min_stop": min_stop, "mid_start": min_start - max_start,
                             "mid_stop": max_stop - min_stop, "max_start": max_start, "max_stop": max_stop}
                switches.append(positions)
        switches.sort(key=lambda x: x["min_start"])
        if switches[1]["min_start"] <= switches[0]["min_stop"]:
            continue  # no overlapping switches
        # truncate maxes to next switch if they go beyond
        if switches[0]["max_stop"] > switches[1]["min_start"]:
            switches[0]["max_stop"] = switches[1]["min_start"]
        if switches[1]["max_start"] < switches[0]["min_stop"]:
            switches[1]["max_start"] = switches[0]["min_stop"]

        results["min"].append(switches[1]["min_start"] - switches[0]["min_stop"])
        results["mid"].append(
            max(np.mean((switches[1]["min_start"], switches[1]["max_start"])) -
                np.mean((switches[0]["max_stop"], switches[0]["min_stop"])),
                0),  # zero if overlapping.
        )
        results["max"].append(
            max(switches[1]["max_start"] - switches[0]["max_stop"], 0)  # zero if overlapping
        )
        x += 1
        if x % 1000 == 0:
            tprint("Completed %d of %d trials." % (x, num_trials), ontop=True)
    return results

