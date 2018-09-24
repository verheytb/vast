#!/usr/bin/env python3

# builtin
import os
import csv
import math
import string
import random
import shutil
import multiprocessing
from itertools import product, combinations, chain
from functools import reduce, partial
from operator import mul, add

# dependencies
import numpy as np
import pandas as pd
from scipy import stats, signal, ndimage
from statsmodels.stats.weightstats import DescrStatsW
from matplotlib import use, rcParams
use("Agg")  # for non-interactive environments like Windows Subsystem for Linux (WSL)
rcParams['pdf.fonttype'] = 42  # Decrease font size to prevent clashing in larger plots.
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import get_cmap, colors
try:  # pysam is optional
    import pysam
except ImportError:
    pysam = False

# included
import vlstools.alignments as al
import vlstools.zip as zi
import vlstools.simulations as si
import vlstools.utils as ut
math.exp = ut.quiet_exp  # overwrite the exponential function to prevent overflow errors in stats.anderson_ksamp


def simple_functions(metric, data, reportdir, database, count_indels_once=False, min_switch_size=1):
    references = database.get("references")
    switches = database.get("switches")
    try:
        database.get("slips")
    except FileNotFoundError:
        slips = None
    tag_labels = list(set(t[0] for refid, tags, read_subset in data for t in tags))
    tsvpath = os.path.join(reportdir, "report.tsv")
    with open(tsvpath, "w") as handle:
        tsv_writer = csv.writer(handle, delimiter="\t")

        headers = {"distinct_variants": ["Distinct Variants", "Distinct Variants (Templated Only)"],
                   "snp_frequency": ["Number of Bases",
                                     "All SNPs (Mean)", "All SNPs (SD)",
                                     "Insertions (Mean)", "Insertions (SD)",
                                     "Deletions (Mean)", "Deletions (SD)",
                                     "Substitutions (Mean)", "Substitutions (SD)",
                                     "All Templated (Mean)", "All Templated (SD)",
                                     "Templated Insertions (Mean)", "Templated Insertions (SD)",
                                     "Templated Deletions (Mean)", "Templated Deletions (SD)",
                                     "Templated Substitutions (Mean)", "Templated Substitutions (SD)",
                                     "All Non-Templated (Mean)", "All Non-Templated (SD)",
                                     "Non-Templated Insertions (Mean)", "Non-Templated Insertions (SD)",
                                     "Non-Templated Deletions (Mean)", "Non-Templated Deletions (SD)",
                                     "Non-Templated Substitutions (Mean)", "Non-Templated Substitutions (SD)"],
                   "parentals": ["Parental Reads", "Parental Reads (Templated Only)"],
                   "nontemp_indel_sizes": list(range(1, 16)) + ["16+"],
                   "snp_frequency_vr_cr": ["CR Size", "VR Size", "CR SNPs (Mean)", "CR SNPs (SD)", "CR SNPs (N)",
                                           "VR SNPs (Mean)", "VR SNPs (SD)", "VR SNPs (N)"],
                   "nontemplated_snp_frequency_vr_cr": ["CR Size", "VR Size", "CR SNPs (Mean)", "CR SNPs (SD)",
                                                        "CR SNPs (N)", "VR SNPs (Mean)", "VR SNPs (SD)", "VR SNPs (N)"],
                   "annotated_vr_snp_frequency": ["CR Size", "VR Size", "CR SNPs (Mean)", "CR SNPs (N)",
                                                  "VR SNPs (Mean)", "VR SNPs (N)",
                                                  "Templated CR SNPs (Mean)", "Templated CR SNPs (N)",
                                                  "Templated VR SNPs (Mean)", "Templated VR SNPs (N)"],
                   "switch_length": ["Number of Switches N (single-switch reads only)",
                                     "Switch Tract Length Mean (single-switch reads only)",
                                     "Switch Tract Length SD (single-switch reads only)",
                                     "Number of Switches N (all reads)",
                                     "Switch Tract Length Mean (all reads)",
                                     "Switch Tract Length SD (all reads)",
                                     ],
                   "switches_per_read": ["Mean", "SD", "N"],
                   "frameshifts": ["Frameshifted Reads", "Reads with Nonsense Mutations"],
                   "unique_variants": ["Variants in subset", "Variants in other subsets", "Unique Variants",
                                       "Shared Variants"],
                   "slips_per_read": ["Mean", "SD", "N"],
                   "slipped_snps_per_nontemplated_snp": ["All (Mean)", "All (SD)", "All (N)",
                                                    "Insertions (Mean)", "Insertions (SD)", "Insertions (N)",
                                                    "Deletions (Mean)", "Deletions (SD)", "Deletions (N)",
                                                    "Substitutions (Mean)", "Substitutions (SD)", "Substitutions (N)"],
                   "dn_ds": ["Cassettes dN/dS (Mean)", "Cassettes dN/dS (SD)", "Cassettes dS=0 Fraction",
                             "Sample dN/dS (Mean)", "Sample dN/dS (SD)", "Sample dS=0 Fraction"]
                   }

        tsv_writer.writerow(["Reference", "Number of Reads in Subset"] + tag_labels + headers[metric])
        # filter for non-empty bins, and require references with aligned cassettes
        data = [(refid, tags, read_subset) for refid, tags, read_subset in data
                if references[refid].cassettes_aln is not None and read_subset]
        for refid, tags, read_subset in data:
            reference = references[refid]
            if metric == "snp_frequency":
                outputs = [len([x for read in read_subset for x in read.seq])]
                for opset in ("IDS", "I", "D", "S"):
                    snp_array = []
                    for read in read_subset:
                        for x in range(len(read.seq)):
                            snps_per_aln_per_base = [al.count_snps([op for op in aln.transform
                                                                    if (op[0] == x and op[1] in opset)],
                                                                   count_indels_once=count_indels_once)
                                                     for aln in read.alns]
                            snp_array.append(np.mean(snps_per_aln_per_base))
                    outputs.extend([np.mean(snp_array), np.std(snp_array)])

                if reference.cassettes_aln is not None:  # add templated/nontemplated snp frequencies
                    templated_ops = [op for aln in reference.cassettes_aln.values() for op in
                                     al.trim_transform(aln.transform, len(reference.seq))]
                    for t in (True, False):
                        for opset in ("IDS", "I", "D", "S"):
                            snp_array = []
                            for read in read_subset:
                                for x in range(len(read.seq)):
                                    snps_per_aln_per_base = [al.count_snps([op for op in aln.transform if op[0] == x
                                                                            and op[1] in opset
                                                                            and (op in templated_ops) == t],
                                                                           count_indels_once=count_indels_once)
                                                             for aln in read.alns]
                                    snp_array.append(np.mean(snps_per_aln_per_base))
                            outputs.extend([np.mean(snp_array), np.std(snp_array)])

            elif metric == "snp_frequency_vr_cr":
                if reference.cassettes_aln is not None:
                    templated_ops = [op for aln in reference.cassettes_aln.values() for op in
                                     al.trim_transform(aln.transform, len(reference.seq))]
                    vr_start = min(op[0] for op in templated_ops)
                    vr_stop = max(op[0] for op in templated_ops)
                    ir_len = vr_start + len(reference.seq) - vr_stop
                    vr_len = vr_stop - vr_start
                    ir_array = []
                    vr_array = []
                    for read in read_subset:
                        for x in range(len(reference.seq)):
                            snps_at_x = np.mean([al.count_snps([op for op in aln.transform if op[0] == x])
                                                 for aln in read.alns])
                            if x in range(vr_start, vr_stop):
                                vr_array.append(snps_at_x)
                            else:
                                ir_array.append(snps_at_x)
                    outputs = [ir_len, vr_len,  # VR definition
                               np.mean(ir_array), np.std(ir_array), len(ir_array),  # invariable region stats
                               np.mean(vr_array), np.std(vr_array), len(vr_array)]  # variable region stats
                else:
                    outputs = []

            elif metric == "nontemplated_snp_frequency_vr_cr":
                if reference.cassettes_aln is not None:
                    templated_ops = [op for aln in reference.cassettes_aln.values() for op in
                                     al.trim_transform(aln.transform, len(reference.seq))]
                    vr_start = min(op[0] for op in templated_ops)
                    vr_stop = max(op[0] for op in templated_ops)
                    ir_len = vr_start + len(reference.seq) - vr_stop
                    vr_len = vr_stop - vr_start
                    ir_array = []
                    vr_array = []
                    for read in read_subset:
                        for x in range(len(reference.seq)):
                            snps_at_x = np.mean([al.count_snps([op for op in aln.transform
                                                                if op[0] == x and op not in templated_ops])
                                                 for aln in read.alns])
                            if x in range(vr_start, vr_stop):
                                vr_array.append(snps_at_x)
                            else:
                                ir_array.append(snps_at_x)
                    outputs = [ir_len, vr_len,  # VR definition
                               np.mean(ir_array), np.std(ir_array), len(ir_array),  # invariable region stats
                               np.mean(vr_array), np.std(vr_array), len(vr_array)]  # variable region stats
                else:
                    outputs = []

            elif metric == "annotated_vr_snp_frequency":
                if reference.variable_regions is not None and reference.cassettes_aln is not None:

                    templated_ops = [op for aln in reference.cassettes_aln.values() for op in
                                     al.trim_transform(aln.transform, len(reference.seq))]

                    # number of bases in the IR and VRs
                    vr_len = sum(r[1] - r[0] for r in reference.variable_regions)
                    ir_len = len(reference.seq) - vr_len

                    # all SNPs counted
                    all_hist = al.snp_histogram(reads=read_subset, reference=reference)
                    all_vr_hist = [y for x, y in enumerate(all_hist)
                                   if any(x in range(r[0], r[1]) for r in reference.variable_regions)]
                    all_ir_hist = [y for x, y in enumerate(all_hist)
                                   if all(x not in range(r[0], r[1]) for r in reference.variable_regions)]

                    # templated SNPs counted
                    templated_hist = al.snp_histogram(reads=read_subset, reference=reference,
                                                      templated_ops=templated_ops, templated=True)
                    templated_vr_hist = [y for x, y in enumerate(templated_hist)
                                         if any(x in range(r[0], r[1]) for r in reference.variable_regions)]
                    templated_ir_hist = [y for x, y in enumerate(templated_hist)
                                         if all(x not in range(r[0], r[1]) for r in reference.variable_regions)]

                    outputs = [ir_len, vr_len,
                               sum(all_ir_hist) * len(read_subset), len(all_ir_hist) * len(read_subset),
                               sum(all_vr_hist) * len(read_subset), len(all_vr_hist) * len(read_subset),
                               sum(templated_ir_hist) * len(read_subset), len(templated_ir_hist) * len(read_subset),
                               sum(templated_vr_hist) * len(read_subset), len(templated_vr_hist) * len(read_subset)]
                else:
                    outputs = []

            elif metric == "distinct_variants":
                templated_variants = set(al.templated_variants(read_subset, reference))
                all_variants = {}
                for read in read_subset:
                    if read.seq in all_variants:
                        all_variants[read.seq] += 1
                    else:
                        all_variants[read.seq] = 1
                outputs = [len(all_variants), len(templated_variants)]

            elif metric == "parentals":
                parentals = 0
                parentals_templated = 0
                for read in read_subset:
                    if read.seq == reference.seq:
                        parentals += 1
                if reference.cassettes_aln is not None:
                    for read in read_subset:
                        if any((al.templated(aln.transform, reference) == tuple()) for aln in read.alns):
                            parentals_templated += 1
                outputs = [parentals, parentals_templated]

            elif metric == "nontemp_indel_sizes":
                if reference.cassettes_aln is not None:
                    templated_ops = [op for aln in reference.cassettes_aln.values() for op in
                                     al.trim_transform(aln.transform, len(reference.seq))]
                    outputs = [0] * 16
                    for read in read_subset:
                        for aln in read.alns:
                            for op in aln.transform:
                                if op[1] == "I" and op not in templated_ops:
                                    if len(op[2]) > 15:
                                        outputs[15] += 1 / len(read.alns)
                                    else:
                                        outputs[len(op[2]) - 1] += 1 / len(read.alns)
                                elif op[1] == "D" and op not in templated_ops:
                                    if op[2] > 15:
                                        outputs[15] += 1 / len(read.alns)
                                    else:
                                        outputs[op[2] - 1] += 1 / len(read.alns)
                else:
                    outputs = []

            elif metric == "switches_per_read":
                if not switches:
                    raise ValueError("switches.p not found. Run \"vls label_switches\" to calculate switches "
                                     "before exporting report.")
                if reference.cassettes_aln is not None:
                    num_switches = []
                    for read in read_subset:
                        switches_per_aln = []
                        for aln in read.alns:
                            templated_tf = al.templated(aln.transform, reference)  # for switch lookup
                            if (templated_tf, refid) in switches:  # switches must be computed
                                switch_sets = switches[(templated_tf, refid)]
                                if not isinstance(switch_sets, int):  # excludes alignments that have too many switches
                                    count_per_switch_set = []
                                    for switch_set in switch_sets:
                                        count = 0
                                        for s in switch_set:
                                            snp_count = s[1] - s[0]
                                            if snp_count >= min_switch_size:
                                                count += 1
                                        count_per_switch_set.append(count)
                                    switches_per_aln.append(np.mean(count_per_switch_set))
                        if switches_per_aln:
                            num_switches.append(np.mean(switches_per_aln))
                    outputs = [np.mean(num_switches), np.std(num_switches), len(num_switches)]
                else:
                    outputs = []

            elif metric == "slips_per_read":
                if not slips:
                    raise ValueError("slips.p not found. Run \"vls label_slippage\" to calculate polymerase slippage "
                                     "before exporting report.")
                outputs = []
                if reference.cassettes_aln is not None:
                    slips_per_read = [len(al.get_slips(read, slips, reference)[0]) for read in read_subset]
                    outputs.extend([np.mean(slips_per_read), np.std(slips_per_read), len(slips_per_read)])

            elif metric == "slipped_snps_per_nontemplated_snp":
                if not slips:
                    raise ValueError("slips.p not found. Run \"vls label_slippage\" to calculate polymerase slippage "
                                     "before exporting report.")
                outputs = []
                if reference.cassettes_aln is not None:
                    results = {"IDS": [[], []], "I": [[], []], "D": [[], []], "S": [[], []]}  # count separately for
                                                                                              # each column
                    for read in read_subset:
                        best_slipset, best_aln = al.get_slips(read, slips, reference)
                        slipped_idxs = set(x for slip in best_slipset for x in range(slip[0], slip[1] + 1))
                        nontemplated_tf = al.nontemplated(best_aln.transform, reference)
                        for idx, op in enumerate(nontemplated_tf):
                            for optype in results:
                                if op[1] in optype:
                                    # values
                                    results[optype][0].append(1 if idx in slipped_idxs else 0)
                                    # weights
                                    results[optype][1].append(1 / len(read.alns))

                    for optype in ["IDS", "I", "D", "S"]:
                        stats = DescrStatsW(data=results[optype][0], weights=results[optype][1])
                        outputs.extend([stats.mean, stats.std, stats.nobs])

            elif metric == "frameshifts":
                outputs = []
                if reference.cassettes_aln is not None:
                    persistent_frameshifts = 0
                    nonsense_mutations = 0
                    for read in read_subset:
                        for aln in read.alns:
                            nonshifted_read = al.translate_mapping(mapping=aln.transform, reference=reference,
                                                                   templ=True, nontempl=False, correctframe=False,
                                                                   filternonsense=False, filterframe=True)
                            if not nonshifted_read:
                                persistent_frameshifts += 1 / len(read.alns)
                            else:
                                persistent_frameshifts += 0

                            nonstop_read = al.translate_mapping(mapping=aln.transform, reference=reference,
                                                                templ=True, nontempl=False, correctframe=False,
                                                                filternonsense=True, filterframe=False)
                            if not nonstop_read:
                                nonsense_mutations += 1 / len(read.alns)
                            else:
                                nonsense_mutations += 0

                    outputs.extend([persistent_frameshifts, nonsense_mutations])


            elif metric == "switch_length":
                if not switches:
                    raise ValueError("switches.p not found. Run \"vls label_switches\" to calculate switches "
                                     "before exporting report.")
                if reference.cassettes_aln is not None:
                    singles_lengths = []
                    singles_weights = []
                    all_lengths = []
                    all_weights = []
                    for read in read_subset:
                        switch_sets_by_aln = []
                        for aln in read.alns:
                            templated_tf = al.templated(aln.transform, reference)
                            if (templated_tf, reference.name) in switches and not isinstance(
                                    switches[(templated_tf, reference.name)], int):
                                switch_sets_lengths = []  # each switch set is a list of lengths of switches
                                for switch_set in switches[(templated_tf, reference.name)]:
                                    list_of_lengths = [templated_tf[s[1] - 1][0] - templated_tf[s[0]][0]
                                                       for s in switch_set]
                                    switch_sets_lengths.append(list_of_lengths)
                                if switch_sets_lengths:
                                    shortest_sum_length = min(sum(sset) for sset in switch_sets_lengths)
                                    switch_sets_lengths = [sset for sset in switch_sets_lengths
                                                           if sum(s for s in sset) == shortest_sum_length]
                                    # choose one at random since all the minima will have identical lengths
                                    switch_sets_by_aln.append(random.choice(switch_sets_lengths))
                        # All switches
                        for list_of_lengths in switch_sets_by_aln:
                            all_lengths.extend(list_of_lengths)
                            all_weights.extend([1 / len(read.alns)] * len(list_of_lengths))
                        # single switches only
                        switch_sets_by_aln = [ss for ss in switch_sets_by_aln if len(ss) == 1]
                        for list_of_lengths in switch_sets_by_aln:
                            singles_lengths.append(list_of_lengths[0])
                            singles_weights.append(1 / len(read.alns))

                    # calculate weighted average and standard deviation
                    single_switches = DescrStatsW(data=singles_lengths, weights=singles_weights)
                    all_switches = DescrStatsW(data=all_lengths, weights=all_weights)
                    outputs = [single_switches.nobs, single_switches.mean, single_switches.std,
                               all_switches.nobs, all_switches.mean, all_switches.std]
                else:
                    outputs = []

            elif metric == "unique_variants":
                if reference.cassettes_aln is not None:
                    # collect set of other variants
                    other_variants = set()
                    for ri, t, rs in data:
                        if ri != refid or t != tags:
                            for read in rs:
                                read_vars = frozenset(al.templated(aln.transform, reference) for aln in read.alns)
                                other_variants.add(read_vars)

                    variants = set()
                    for read in read_subset:
                        read_vars = frozenset(al.templated(aln.transform, reference) for aln in read.alns)
                        variants.add(read_vars)

                    outputs = [len(variants), len(other_variants), len(variants - other_variants),
                               len(variants & other_variants)]
                else:
                    outputs = []

            elif metric == "dn_ds":
                if reference.cassettes_aln is not None:
                    # calculate the dN/dS in the cassettes
                    cassettes_dn_ds = [al.get_dnds(alignment=cassette_aln, reference=reference)
                                       for cassette_aln in reference.cassettes_aln.values()]
                    if cassettes_dn_ds:
                        stats = DescrStatsW(data=[stat.dn/stat.ds for stat in cassettes_dn_ds
                                                  if stat.ds is not None and stat.ds > 0])
                        ds0_count = len([stat for stat in cassettes_dn_ds if stat.ds == 0]) / stats.nobs
                        outputs = [stats.mean, stats.std, ds0_count]
                    else:
                        outputs = [None, None, None]

                    # calculate the dN/dS in the read group
                    values = []
                    weights = []
                    ds0_count = 0
                    for read in read_subset:
                        read_dn_ds = [al.get_dnds(alignment=aln, reference=reference)
                                      for aln in read.alns]
                        if read_dn_ds:
                            read_values = [stat.dn/stat.ds for stat in read_dn_ds
                                           if stat.ds is not None and stat.ds > 0]
                            values.extend(read_values)
                            weights.extend([1/len(read_values) for _ in read_values])
                            ds0_count += len([stat for stat in read_dn_ds if stat.ds == 0]) / len(read.alns)
                    stats = DescrStatsW(data=values, weights=weights)
                    outputs.extend([stats.mean, stats.std, ds0_count / len(read_subset)])

                else:
                    outputs = []

            else:
                raise NotImplementedError

            tagdict = dict(tags)
            tsv_writer.writerow([refid, len(read_subset)] + [str(tagdict[l]) for l in tag_labels] + outputs)


def snp_positions(data, reportdir, database, numtrials=10):
    references = database.get("references")

    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]

    for refid, tags, read_subset in data:
        reference = references[refid]
        # writes TSV report
        with open(os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags) + ".tsv"), "w") as handle:
            tsv_writer = csv.writer(handle, delimiter="\t")
            coords = list(range(reference.offset + 1, reference.offset + 2 + len(reference.seq)))
            tsv_writer.writerow([""] + coords)
            all_snps = [0] * (1 + len(reference.seq))
            # count the snp frequencies per position
            for read in read_subset:
                for aln in read.alns:
                    for op in aln.transform:
                        if op[1] == "S":
                            all_snps[op[0]] += 1 / len(read_subset) / len(read.alns)
                        if op[1] == "D":
                            for x in range(op[2]):
                                all_snps[op[0] + x] += 1 / len(read_subset) / len(read.alns)
                        elif op[1] == "I":
                            all_snps[op[0]] += len(op[2]) / len(read_subset) / len(read.alns)
            tsv_writer.writerow(["SNP frequency from reads"] + all_snps)

            # count the SNP frequency for actual templated/non-templated SNPs, and the number of SNPs in the
            # cassettes
            if reference.cassettes_aln is not None:
                # make a list of all the SNPs that are templated
                templated_ops = [op for aln in reference.cassettes_aln.values() for op in
                                 al.trim_transform(aln.transform, len(reference.seq))]

                # for both templated and nontemplated, make many lists of SNP positions (randomly choosing from
                # equivalent alignments). This will be used for bootstrapped KS tests.
                nontemplated_positions_trials = []
                templated_positions_trials = []
                for trial in range(numtrials):
                    nontemplated_positions = []
                    templated_positions = []
                    for read in read_subset:
                        randomaln = random.choice(read.alns)
                        for op in randomaln.transform:
                            if op[1] == "I":
                                positions = [op[0]] * len(op[2])
                            elif op[1] == "D":
                                positions = [op[0] + x for x in range(op[2])]
                            elif op[1] == "S":
                                positions = [op[0]]
                            else:
                                raise ValueError
                            for pos in positions:
                                if op in templated_ops:
                                    templated_positions.append(pos)
                                else:
                                    nontemplated_positions.append(pos)
                    templated_positions_trials.append(templated_positions)
                    nontemplated_positions_trials.append(nontemplated_positions)

                # bin the data by position
                nontemplated_hist = [0] * (1 + len(reference.seq))
                templated_hist = [0] * (1 + len(reference.seq))

                for read in read_subset:
                    for aln in read.alns:
                        for op in aln.transform:
                            if op[1] in "S":
                                if op in templated_ops:
                                    templated_hist[op[0]] += 1 / len(read_subset) / len(read.alns)
                                else:
                                    nontemplated_hist[op[0]] += 1 / len(read_subset) / len(read.alns)
                            elif op[1] == "D":
                                for x in range(op[2]):
                                    if op in templated_ops:
                                        templated_hist[op[0] + x] += 1 / len(read_subset) / len(read.alns)
                                    else:
                                        nontemplated_hist[op[0] + x] += 1 / len(read_subset) / len(read.alns)
                            elif op[1] == "I":
                                if op in templated_ops:
                                    templated_hist[op[0]] += len(op[2]) / len(read_subset) / len(read.alns)
                                else:
                                    nontemplated_hist[op[0]] += len(op[2]) / len(read_subset) / len(read.alns)

                tsv_writer.writerow(["Templated SNP frequency"] + templated_hist)
                tsv_writer.writerow(["Nontemplated SNP frequency"] + nontemplated_hist)

                # number of SNPs at each position in the cassettes
                cassette_snps = [0] * (1 + len(reference.seq))
                for cassette_name, aln in reference.cassettes_aln.items():
                    for op in al.trim_transform(aln.transform, len(reference.seq)):
                        if op[1] == "S":
                            cassette_snps[op[0]] += 1 / len(reference.cassettes_aln)
                        elif op[1] == "D":
                            for x in range(op[2]):
                                cassette_snps[op[0] + x] += 1 / len(reference.cassettes_aln)
                        elif op[1] == "I":
                            cassette_snps[op[0]] += len(op[2]) / len(reference.cassettes_aln)
                tsv_writer.writerow(["Frequency of SNPs in silent cassettes"] + [str(x) for x in cassette_snps])

                # number of SNPs at each position in the cassettes (within 30bp of cassette end or not)
                cassette_snps_30 = [0] * (1 + len(reference.seq))
                cassette_snps_not30 = [0] * (1 + len(reference.seq))
                for cassette_name, aln in reference.cassettes_aln.items():
                    for op in al.trim_transform(aln.transform, len(reference.seq)):
                        if aln.start <= op[0] <= aln.start + 30 or aln.end - 30 <= op[0] <= aln.end:
                            if op[1] == "S":
                                cassette_snps_30[op[0]] += 1 / len(reference.cassettes_aln)
                            elif op[1] == "D":
                                for x in range(op[2]):
                                    cassette_snps_30[op[0] + x] += 1 / len(reference.cassettes_aln)
                            elif op[1] == "I":
                                cassette_snps_30[op[0]] += len(op[2]) / len(reference.cassettes_aln)
                        else:
                            if op[1] == "S":
                                cassette_snps_not30[op[0]] += 1 / len(reference.cassettes_aln)
                            elif op[1] == "D":
                                for x in range(op[2]):
                                    cassette_snps_not30[op[0] + x] += 1 / len(reference.cassettes_aln)
                            elif op[1] == "I":
                                cassette_snps_not30[op[0]] += len(op[2]) / len(reference.cassettes_aln)


                # Kolmogorov-Smirnov test
                ks_values = [stats.ks_2samp(temp, nontemp) for temp, nontemp
                             in zip(templated_positions_trials, nontemplated_positions_trials)]
                tsv_writer.writerow([])
                tsv_writer.writerow(["Kolmogorov-Smirnov Test (2-sample)"])
                tsv_writer.writerow(["Sample 1:", "Position of Templated SNPs"])
                tsv_writer.writerow(["Sample 2:", "Position of Nontemplated SNPs"])
                tsv_writer.writerow(["Number of bootstrapping trials:", numtrials])
                tsv_writer.writerow(["Mean KS Statistic:", np.mean([val[0] for val in ks_values])])
                tsv_writer.writerow(["P-Value:", np.mean([val[1] for val in ks_values])])

                # Cosine similarity test
                similarity = np.dot(templated_hist, nontemplated_hist) \
                             / (np.linalg.norm(templated_hist) * np.linalg.norm(nontemplated_hist))
                tsv_writer.writerow(["Cosine Similarity:", similarity])

        if reference.variable_regions is None:
            reference.variable_regions = []
        # Plot the data and send to PDF files.
        with PdfPages(os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags) + ".pdf")) as pdf:
            # Page 1: All SNPs vs position.
            fig, (ax_all_snps, ax_cassettes) = plt.subplots(2, figsize=(8, 8), sharex=True)
            ax_all_snps.bar(height=all_snps, x=coords, width=1, linewidth=0, color="black")
            ax_all_snps.set_title("Observed Variants")
            ax_all_snps.set_ylabel("SNP Frequency")
            ax_all_snps.set_xlabel("vlsE position (bp)")
            if reference.cassettes_aln is not None:
                ax_cassettes.bar(height=cassette_snps, x=coords, width=1, linewidth=0, color="black")
                ax_cassettes.set_title("Reference Cassettes")
                ax_cassettes.set_ylabel("Frequency of SNPs in silent cassettes")
                ax_cassettes.set_xlabel("vlsE position (bp)")
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            # Page 2: Templated and Nontemplated SNPs vs position.
            if reference.cassettes_aln is not None:
                fig, (ax_temp_snps, ax_nontemp_snps) = plt.subplots(2, figsize=(8, 8), sharex=True)
                ax_temp_snps.bar(height=templated_hist, x=coords, width=1, linewidth=0, color="black")
                ax_temp_snps.set_ylabel("Frequency of Templated SNPs")
                ax_temp_snps.set_xlabel("vlsE position (bp)")
                ax_nontemp_snps.bar(height=nontemplated_hist, x=coords, width=1, linewidth=0, color="black")
                ax_nontemp_snps.set_ylabel("Frequency of Non-Templated SNPs")
                ax_nontemp_snps.set_xlabel("vlsE position (bp)")
                plt.tight_layout()
                pdf.savefig()
                plt.close()

                # Page 3: Templated SNPs and G-runs
                fig, ax = plt.subplots(1, figsize=(8, 6), sharex=True)
                ax.bar(height=templated_hist, x=coords, width=1, linewidth=0, color="black")
                grun_size = 3
                for x, is_g in enumerate(al.find_poly_g(reference.seq, grun_size)):
                    pos = x + reference.offset
                    if is_g == 1:
                        ax.axvspan(xmin=pos, xmax=pos + grun_size, facecolor='blue', alpha=0.5, linewidth=0)
                ax.set_ylabel("Frequency of Templated SNPs")
                ax.set_xlabel("vlsE position (bp)")
                plt.tight_layout()
                pdf.savefig()
                plt.close()

                # Page 4: Mirror plot comparing the distribution of Templated to Nontemplated
                fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 6), sharex=True)
                plt.subplots_adjust(hspace=0)
                for x, vr in enumerate(reference.variable_regions, start=1):  # show VRs in the background
                    ax2.axvspan(vr[0] + reference.offset, vr[1] + reference.offset,
                                ymax=0.05, facecolor='#8B9476', alpha=0.75, linewidth=0)
                ax1.bar(height=templated_hist, x=coords, width=1, linewidth=0, color="darkblue")
                ax1.set_ylabel("Frequency of Templated SNPs", color="darkblue")
                ax1.set_xlabel("vlsE position (bp)")
                ax1.set_xlim(min(coords), max(coords))
                ax1.spines['bottom'].set_visible(False)
                ax1.xaxis.set_ticks_position('none')
                ax1.yaxis.tick_left()
                for tl in ax1.get_yticklabels():
                    tl.set_color("darkblue")

                ax2.bar(height=nontemplated_hist, x=coords,
                        width=1, linewidth=0, color="firebrick")
                ax2.set_ylabel("Frequency of Non-Templated SNPs", color="firebrick")
                ax2.set_xlabel("vlsE position (bp)")
                ax2.set_ylim(ax2.get_ylim()[::-1])
                ax2.yaxis.set_label_position("right")
                ax2.spines['top'].set_visible(False)
                ax2.xaxis.tick_bottom()
                ax2.yaxis.tick_right()
                for tl in ax2.get_yticklabels():
                    tl.set_color("firebrick")
                pdf.savefig()
                plt.close()

                # Page 5: Mirror plot comparing the distribution of Templated to Cassettes
                fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 8), sharex=True)
                plt.subplots_adjust(hspace=0)
                ax1.bar(height=templated_hist, x=coords, width=1, linewidth=0, color="green")
                ax1.set_ylabel("Frequency of Templated SNPs", color="green")
                ax1.set_xlabel("vlsE position (bp)")
                ax1.set_xlim(min(coords), max(coords))
                ax1.spines['bottom'].set_visible(False)
                ax1.xaxis.set_ticks_position('none')
                ax1.yaxis.tick_left()
                for tl in ax1.get_yticklabels():
                    tl.set_color("green")

                ax2.bar(height=cassette_snps, x=coords, width=1, linewidth=0, color="red")
                ax2.set_ylabel("Frequency of SNPs in Silent Cassettes", color="red")
                ax2.set_xlabel("vlsE position (bp)")
                ax2.set_ylim(ax2.get_ylim()[::-1])
                ax2.yaxis.set_label_position("right")
                ax2.spines['top'].set_visible(False)
                ax2.xaxis.tick_bottom()
                ax2.yaxis.tick_right()
                for tl in ax2.get_yticklabels():
                    tl.set_color("red")
                for x, vr in enumerate(reference.variable_regions, start=1):  # show VRs in the background
                    ax2.axvspan(vr[0] + reference.offset, vr[1] + reference.offset,
                                ymax=0.1, facecolor='black', alpha=0.5, linewidth=0)
                pdf.savefig()
                plt.close()

                # Page 6: Mirror plot comparing the distribution of Templated to Cassettes, distinguishing templated
                # events by whether they are at the end of silent cassettes.
                fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 8), sharex=True)
                plt.subplots_adjust(hspace=0)
                ax1.bar(height=templated_hist, x=coords, width=1, linewidth=0, color="green")
                ax1.set_ylabel("Frequency of Templated SNPs", color="green")
                ax1.set_xlabel("vlsE position (bp)")
                ax1.set_xlim(min(coords), max(coords))
                ax1.spines['bottom'].set_visible(False)
                ax1.xaxis.set_ticks_position('none')
                ax1.yaxis.tick_left()
                for tl in ax1.get_yticklabels():
                    tl.set_color("green")

                ax2.bar(height=cassette_snps_30, x=coords, width=1, linewidth=0, color="black")
                ax2.bar(height=cassette_snps_not30, bottom=cassette_snps_30,
                        x=coords, width=1, linewidth=0, color="red")
                ax2.set_ylabel("Frequency of SNPs in Silent Cassettes", color="red")
                ax2.set_xlabel("vlsE position (bp)")
                ax2.set_ylim(ax2.get_ylim()[::-1])
                ax2.yaxis.set_label_position("right")
                ax2.spines['top'].set_visible(False)
                ax2.xaxis.tick_bottom()
                ax2.yaxis.tick_right()
                for tl in ax2.get_yticklabels():
                    tl.set_color("red")
                for x, vr in enumerate(reference.variable_regions, start=1):  # show VRs in the background
                    ax2.axvspan(vr[0] + reference.offset, vr[1] + reference.offset,
                                ymax=0.1, facecolor='black', alpha=0.5, linewidth=0)
                pdf.savefig()
                plt.close()

                # Page 7: Difference of Normalized Actual Templated Changes to Theoretical Changes
                if sum(templated_hist) != 0:
                    ratios = [(a / sum(templated_hist) - t / sum(cassette_snps))
                              for a, t in zip(templated_hist, cassette_snps)]
                    fig, ax = plt.subplots(1, figsize=(14, 6))
                    ax.bar(x=coords, height=ratios, width=1, linewidth=0, color="darkblue")
                    ax.set_ylabel("Fold Difference")
                    ax.set_xlabel("vlsE position (bp)")
                    ax.set_xlim(min(coords), max(coords))
                    pdf.savefig()
                    plt.close()

                    # Page 8: Cross-correlation of all/cassettes and templated/nontemplated snp frequencies.
                    fig, (ax_nontemp, ax_all) = plt.subplots(2, figsize=(8, 8))
                    crosscorr = signal.correlate(templated_hist, nontemplated_hist, mode="same")
                    ax_nontemp.bar(height=crosscorr, x=[x - coords[0] - len(coords) / 2 for x in coords],
                                   width=1, linewidth=0, color="black")
                    ax_nontemp.set_title("Cross-Correlation of Nontemplated and Templated SNP Frequencies")
                    ax_nontemp.set_xlabel("offset")
                    crosscorr = signal.correlate(all_snps, cassette_snps, mode="same")
                    ax_all.bar(height=crosscorr, x=[x - coords[0] - len(coords) / 2 for x in coords],
                               width=1, linewidth=0, color="black")
                    ax_all.set_title("Cross-Correlation of Observed and Theoretical SNP Frequencies")
                    ax_all.set_xlabel("offset")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()

                    # Page 9: QQ-Plot of templated/nontemplated SNP positions.
                    fig, ax_qq = plt.subplots(figsize=(8, 8))
                    q_t, q_nt = ut.qq_values(templated_positions_trials, nontemplated_positions_trials)
                    ax_qq.plot(q_t, q_nt, color="firebrick", linewidth=2, antialiased=True)
                    ax_qq.set_title("Quantile-Quantile plot of Nontemplated and Templated SNP Positions")
                    ax_qq.set_ylabel("Non-templated SNP quantile")
                    ax_qq.set_xlabel("Templated SNP quantile")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()

            plt.clf()


def snp_positions_cassettes(data, reportdir, database):
    """

    :param data:
    :param reportdir:
    :param database:
    :return:
    """
    references = database.get("references")
    switches = database.get("switches")

    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]

    for refid, tags, read_subset in data:
        reference = references[refid]

        # compute cassette coordinates for start/stop positions
        cassette_coords = {casname: None for casname in reference.cassettes_aln}  # reference-based
        for casname, casaln in reference.cassettes_aln.items():
            if casaln.transform[0][1] == "D" and casaln.transform[0][0] == 0:
                start = casaln.transform[0][2]
            else:
                start = 0
            if casaln.transform[-1][1] == "D" and casaln.transform[-1][0] == len(reference.seq) - casaln.transform[-1][2]:
                stop = casaln.transform[-1][0]
            else:
                stop = len(reference.seq) + 1
            cassette_coords[casname] = (start, stop)

        # compute weights for actual data
        templated_ops = {op for casaln in reference.cassettes_aln.values()
                         for op in al.templated(al.trim_transform(casaln.transform, len(reference.seq)), reference)}
        weights_by_op = {op: 1 / len([name for name, aln in reference.cassettes_aln.items() if op in aln.transform])
                         for op in templated_ops}

        # compute the actual frequencies of all templated SNPs
        op_frequency = {op: 0 for op in templated_ops}
        for read in read_subset:
            for aln in read.alns:
                for op in al.templated(aln.transform, reference):
                    op_frequency[op] += 1 / len(read_subset) / len(read.alns)
        # plot SNPs to line graph
        with PdfPages(os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags) + ".snps.pdf")) as pdf:
            # One page per cassette
            for casname, casaln in sorted(sorted(list(reference.cassettes_aln.items()), key=lambda x: x[0]),
                                          key=lambda x: len(x[0])):
                templated_casaln = al.templated(al.trim_transform(casaln.transform, len(reference.seq)), reference)
                fig, ax = plt.subplots(1, figsize=(6, 4), sharex=True)
                fig.suptitle("%s (%d SNPs)" % (casname, len(templated_casaln)))
                actuals_x = {ids: [op[0] for op in templated_casaln if op[1] in ids] for ids in ["I", "D", "S","IDS"]}
                actuals_y = {ids: [op_frequency[op] * weights_by_op[op]
                                   for op in templated_casaln if op[1] in ids] for ids in ["I", "D", "S","IDS"]}

                ax.set_xlim(cassette_coords[casname])
                ax.set_ylim(0, max(f * weights_by_op[op] for op, f in op_frequency.items()))
                ax.plot(actuals_x["I"], actuals_y["I"], "r+")
                ax.plot(actuals_x["D"], actuals_y["D"], "gx")
                ax.plot(actuals_x["S"], actuals_y["S"], "b_")
                ax.plot(actuals_x["IDS"], actuals_y["IDS"], "k-")
                pdf.savefig()
                plt.close()

        # plot switch density
        switch_density = {casname: [0] * (cassette_coords[casname][1] - cassette_coords[casname][0])
                          for casname in reference.cassettes_aln}
        for read in read_subset:
            for aln in read.alns:
                templated_aln = al.templated(aln.transform, reference)
                switch_sets = switches[(templated_aln, refid)]
                for switch_set in switch_sets:
                    for switch in switch_set:
                        start = templated_aln[switch[0]][0]
                        stop = templated_aln[switch[1] - 1][0]
                        if templated_aln[switch[1]-1][1] == "D":       # if switch ends in deletion,
                            stop += templated_aln[switch[1] - 1][2]  # extend by length of deletion
                        for switch_origin in switch[2]:
                            start_in_cassette = start - cassette_coords[switch_origin][0]
                            stop_in_cassette = stop - cassette_coords[switch_origin][0]
                            for x in range(start_in_cassette, stop_in_cassette):
                                switch_density[switch_origin][x] +=\
                                    1 / len(switch[2]) / len(switch_sets) / len(read.alns) / len(read_subset)

        with PdfPages(os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags) + ".switches.pdf")) as pdf:
            max_y = max(f for density_vector in switch_density.values() for f in density_vector)
            casnames = sorted(sorted(list(reference.cassettes_aln)), key=lambda x: len(x))
            # One page per cassette
            for casname in casnames:
                fig, ax = plt.subplots(1, figsize=(6, 4), sharex=True)
                fig.suptitle(casname)
                ax.set_ylim(0, max_y)
                ax.plot(list(range(len(switch_density[casname]))), switch_density[casname], "k-")
                pdf.savefig()
                plt.close()
                plt.clf()
        with PdfPages(os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags) + ".switches.1panel.pdf")) as pdf:
            # Consecutive on one page
            fig, ax_list = plt.subplots(1, len(casnames), figsize=(36, 6), sharey=True)
            fig.suptitle("All Cassettes")
            for ax, casname in zip(ax_list, casnames):
                ax.plot(range(len(switch_density[casname])), switch_density[casname], "k-")
                ax.set_xlim(0, len(switch_density[casname]))
                plt.subplots_adjust(wspace=.001)
            pdf.savefig()
            plt.close()
            plt.clf()


def ids_colocation(data, reportdir, database, numtrials=10):
    """
    This generates comparisons of whether insertions, deletions, and substitutions in the templated switched bases
    colocate with those in the nontemplated switched bases.
    :param data:
    :param reportdir:
    :param database:
    :param numtrials:
    :return:
    """
    references = database.get("references")

    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]

    for refid, tags, read_subset in data:
        reference = references[refid]
        base_report_path = os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags))
        coords = list(range(reference.offset + 1, reference.offset + 2 + len(reference.seq)))

        # 2-Sample Anderson-Darling, Kolmogorov-Smirnov, and Cosine Similarity Statistics
        opsets = ("I", "D", "S", "ID", "IDS")
        adresults = np.zeros((5, 5))  # a 5x5 array with the AD test results
        ksresults = np.zeros((5, 5))  # a 5x5 array with the KS test results
        csresults = np.zeros((5, 5))  # a 5x5 array with the CS test results
        cassette_ops = [op for aln in reference.cassettes_aln.values() for op in
                        al.trim_transform(aln.transform, len(reference.seq))]
        hists = []

        for x, nontemp_ops in enumerate(opsets):
            for y, temp_ops in enumerate(opsets):
                nontemplated_positions_trials = []
                templated_positions_trials = []
                for trial in range(numtrials):
                    nontemplated_positions = []
                    templated_positions = []
                    for read in read_subset:
                        randomaln = random.choice(read.alns)
                        for op in randomaln.transform:
                            if op[1] in nontemp_ops and op not in cassette_ops:
                                if op[1] == "S":
                                    nontemplated_positions.append(op[0])
                                elif op[1] == "D":
                                    for i in range(op[2]):
                                        nontemplated_positions.append(op[0] + i)
                                elif op[1] == "I":
                                    for i in range(len(op[2])):
                                        nontemplated_positions.append(op[0])
                            if op[1] in temp_ops and op in cassette_ops:
                                if op[1] == "S":
                                    templated_positions.append(op[0])
                                elif op[1] == "D":
                                    for i in range(op[2]):
                                        templated_positions.append(op[0] + i)
                                elif op[1] == "I":
                                    for i in range(len(op[2])):
                                        templated_positions.append(op[0])
                    if templated_positions:
                        templated_positions_trials.append(templated_positions)
                    if nontemplated_positions:
                        nontemplated_positions_trials.append(nontemplated_positions)
                if nontemplated_positions_trials and templated_positions_trials:
                    adresults[x, y] = np.mean([stats.anderson_ksamp([temp, nontemp])[0] for temp, nontemp in
                                               zip(templated_positions_trials, nontemplated_positions_trials)])
                    ksresults[x, y] = np.mean([stats.ks_2samp(temp, nontemp) for temp, nontemp in
                                               zip(templated_positions_trials, nontemplated_positions_trials)])
                else:
                    adresults[x, y] = None
                    ksresults[x, y] = None

                # bin the data by position
                nontemplated_hist = [0] * (1 + len(reference.seq))
                templated_hist = [0] * (1 + len(reference.seq))
                for read in read_subset:
                    for aln in read.alns:
                        for op in aln.transform:
                            if op[1] == "S":
                                if op[1] in temp_ops and op in cassette_ops:
                                    templated_hist[op[0]] += 1 / len(read_subset) / len(read.alns)
                                elif op[1] in nontemp_ops and op not in cassette_ops:
                                    nontemplated_hist[op[0]] += 1 / len(read_subset) / len(read.alns)
                            elif op[1] == "D":
                                for i in range(op[2]):
                                    if op[1] in temp_ops and op in cassette_ops:
                                        templated_hist[op[0] + i] += 1 / len(read_subset) / len(read.alns)
                                    elif op[1] in nontemp_ops and op not in cassette_ops:
                                        nontemplated_hist[op[0] + i] += 1 / len(read_subset) / len(read.alns)
                            elif op[1] == "I":
                                if op[1] in temp_ops and op in cassette_ops:
                                    templated_hist[op[0]] += len(op[2]) / len(read_subset) / len(read.alns)
                                elif op[1] in nontemp_ops and op not in cassette_ops:
                                    nontemplated_hist[op[0]] += len(op[2]) / len(read_subset) / len(read.alns)
                # calculate the cosine similarity of the histograms; each position is a dimension of a vector; the
                # cosine distance is the cosine of the angle between the two vectors.
                csresults[x, y] = np.dot(templated_hist, nontemplated_hist) / (np.linalg.norm(templated_hist) *
                                                                               np.linalg.norm(nontemplated_hist))
                # store the histograms for PDF output
                hists.append({"temp_ops": temp_ops, "nontemp_ops": nontemp_ops, "temp_hist": templated_hist,
                              "nontemp_hist": nontemplated_hist})

        with PdfPages(base_report_path + ".pdf") as pdf:
            # Page 1: heatmap of Anderson-Darling statistics
            fig, ax = plt.subplots()
            heatmap = ax.pcolor(np.transpose(adresults), cmap=get_cmap("YlOrRd_r"))
            # put the major ticks at the middle of each cell
            ax.set_xticks(np.arange(adresults.shape[0]) + 0.5, minor=False)
            ax.set_yticks(np.arange(adresults.shape[1]) + 0.5, minor=False)
            # want a more natural, table-like display
            ax.invert_yaxis()
            ax.xaxis.tick_top()
            # label rows and columns
            ax.set_xticklabels(opsets, minor=False)
            ax.set_yticklabels(opsets, minor=False)
            ax.set_xlabel("Nontemplated")
            ax.set_ylabel("Templated")
            fig.suptitle("Anderson-Darling")
            plt.colorbar(heatmap)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Page 2: heatmap of Kolgomorov-Smirnov statistics
            fig, ax = plt.subplots()
            heatmap = ax.pcolor(np.transpose(ksresults), cmap=get_cmap("YlOrRd_r"))
            # put the major ticks at the middle of each cell
            ax.set_xticks(np.arange(ksresults.shape[0]) + 0.5, minor=False)
            ax.set_yticks(np.arange(ksresults.shape[1]) + 0.5, minor=False)
            # want a more natural, table-like display
            ax.invert_yaxis()
            ax.xaxis.tick_top()
            # label rows and columns
            ax.set_xticklabels(opsets, minor=False)
            ax.set_yticklabels(opsets, minor=False)
            ax.set_xlabel("Nontemplated")
            ax.set_ylabel("Templated")
            fig.suptitle("Kolmogorov-Smirnov")
            plt.colorbar(heatmap)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Page 3: heatmap of Cosine distance statistics
            fig, ax = plt.subplots()
            heatmap = ax.pcolor(np.transpose(csresults), cmap=get_cmap("YlOrRd"))
            # put the major ticks at the middle of each cell
            ax.set_xticks(np.arange(ksresults.shape[0]) + 0.5, minor=False)
            ax.set_yticks(np.arange(ksresults.shape[1]) + 0.5, minor=False)
            # want a more natural, table-like display
            ax.invert_yaxis()
            ax.xaxis.tick_top()
            # label rows and columns
            ax.set_xticklabels(opsets, minor=False)
            ax.set_yticklabels(opsets, minor=False)
            ax.set_xlabel("Nontemplated")
            ax.set_ylabel("Templated")
            fig.suptitle("Cosine Similarity")
            plt.colorbar(heatmap)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Following pages: Mirror Plots of each pair.
            friendly_names = {"I": "Insertions", "D": "Deletions", "S": "Substitutions",
                              "ID": "Indels", "IS": "Insubs", "DS": "Subdels", "IDS": "All SNPs"}
            for hist in hists:
                fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 8), sharex=True)
                fig.suptitle("")
                plt.subplots_adjust(hspace=0)
                ax1.bar(height=hist["temp_hist"], x=coords, width=1, linewidth=0, color="green")
                ax1.set_ylabel("Frequency of Templated %s" % friendly_names[hist["temp_ops"]], color="green")
                ax1.set_xlabel("vlsE position (bp)")
                ax1.set_xlim(min(coords), max(coords))
                ax1.spines['bottom'].set_visible(False)
                ax1.xaxis.set_ticks_position('none')
                ax1.yaxis.tick_left()
                for tl in ax1.get_yticklabels():
                    tl.set_color("green")

                ax2.bar(height=hist["nontemp_hist"], x=coords, width=1, linewidth=0, color="firebrick")
                ax2.set_ylabel("Frequency of Nontemplated %s" % friendly_names[hist["nontemp_ops"]],
                               color="firebrick")
                ax2.set_xlabel("vlsE position (bp)")
                ax2.set_ylim(ax2.get_ylim()[::-1])
                ax2.yaxis.set_label_position("right")
                ax2.spines['top'].set_visible(False)
                ax2.xaxis.tick_bottom()
                ax2.yaxis.tick_right()
                for tl in ax2.get_yticklabels():
                    tl.set_color("firebrick")
                pdf.savefig()
                plt.tight_layout()
                plt.close()


def nontemplated_reads_bam(data, reportdir, database, reads_with_type="IDS"):
    """
    exports all reads with nontemplated mutations for viewing in IGV or other viewers.
    :param data:
    :param reportdir:
    :param database:
    :return:
    """
    if not pysam:
        raise ImportError("The \"nontemp_indels_bam\" report requires pysam.")
    reads_with_type = reads_with_type.upper()
    references = database.get("references")

    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]

    for refid, tags, read_subset in data:
        reference = references[refid]
        templated_ops = [op for aln in reference.cassettes_aln.values() for op in
                         al.trim_transform(aln.transform, len(reference.seq))]
        filename = os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags) + ".bam")
        header = {'HD': {'VN': '1.0'}, 'SQ': [{'LN': len(reference.seq), 'SN': reference.name}]}
        with pysam.AlignmentFile(filename, "wb", header=header) as outf:
            for read in read_subset:
                if any((op not in templated_ops and op[1] in reads_with_type) for aln in read.alns for op in aln.transform):
                    # alignment = read.alns.pop()
                    for i, alignment in enumerate(read.alns):
                        a = pysam.AlignedSegment()
                        a.query_name = read.name + "/aln" + str(i)
                        a.query_sequence = al.transform(reference=reference.seq, mapping=alignment.transform)
                        a.query_qualities = [255 if x else 64 for x in
                                             al.nontemp_mask(reference.seq, alignment.transform, templated_ops)]
                        a.reference_id = 0
                        a.reference_start = alignment.start
                        a.cigar = alignment.cigar
                        outf.write(a)
        pysam.sort("-o", filename[:-4] + ".sorted.bam", filename)
        shutil.move(filename[:-4] + ".sorted.bam", filename)  # overwrites original output with sorted bam file.
        pysam.index(filename)


def two_subset_comparison(data, reportdir, database, reverse_order=False):
    """

    :param data:
    :param reportdir:
    :param database:
    :return:
    """
    references = database.get("references")
    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]
    if reverse_order:
        data.reverse()
    if not len(data) == 2:
        if len(data) > 2:
            errorhelp = "narrow down"
        else:
            errorhelp = "expand"
        raise ValueError("\"two_subset_comparison\" requires exactly two different subsets to be compared. Use the "
                         "--where and --groupby arguments to %s the number of data subsets from %d to 2."
                         % (errorhelp, len(data)))

    refid1, tags1, reads1 = data[0]
    refid2, tags2, reads2 = data[1]

    # Check compatibility
    if refid1 != refid2:
        raise ValueError("The two data subsets provided to \"two_subset_comparison\" have mismatched references: "
                         "%s, %s. Re-run with matching samples." % (refid1, refid2))
    else:
        refid = refid1

    reference = references[refid]
    name1 = ut.get_tagstring(refid=refid, tags=tags1)
    name2 = ut.get_tagstring(refid=refid, tags=tags2)
    pdfpath = os.path.join(reportdir, "%s_vs_%s.pdf" % (name1, name2))

    # make a list of all the SNPs that are templated
    templated_ops = [op for aln in reference.cassettes_aln.values() for op in
                     al.trim_transform(aln.transform, len(reference.seq))]
    # make histograms
    hist1 = al.snp_histogram(reads1, reference, templated_ops, templated=True)
    hist2 = al.snp_histogram(reads2, reference, templated_ops, templated=True)

    # make normalized histograms for each base
    (base_normhist1, base_normhist2) = ({b: al.snp_histogram(readset, reference,
                                                             [t for t in templated_ops
                                                              if t[1] == "S" and t[2] == b], templated=True)
                                         for b in "ACGT"}
                                        for readset in [reads1, reads2])

    coords = list(range(reference.offset + 1, reference.offset + 1 + len(reference.seq)))

    with PdfPages(pdfpath) as pdf:
        # data1 vs data2 Templated SNP Frequencies
        fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 8), sharex=True)
        plt.subplots_adjust(hspace=0)
        ax1.bar(height=hist1, x=coords, width=1, linewidth=0, color="green")
        ax1.set_ylabel("Templated SNP density (%s)" % name1, color="green")
        ax1.set_xlabel("vlsE position (bp)")
        ax1.set_xlim(min(coords), max(coords))
        ax1.spines['bottom'].set_visible(False)
        ax1.xaxis.set_ticks_position('none')
        ax1.yaxis.tick_left()
        for tl in ax1.get_yticklabels():
            tl.set_color("green")

        ax2.bar(height=hist2, x=coords, width=1, linewidth=0, color="red")
        ax2.set_ylabel("Templated SNP density (%s)" % name2, color="red")
        ax2.set_xlabel("vlsE position (bp)")
        ax2.set_ylim(ax2.get_ylim()[::-1])
        ax2.yaxis.set_label_position("right")
        ax2.spines['top'].set_visible(False)
        ax2.xaxis.tick_bottom()
        ax2.yaxis.tick_right()
        for tl in ax2.get_yticklabels():
            tl.set_color("red")
        pdf.savefig()
        plt.close()

        # data1 and data2 base-preference for substitutions
        colours = {"A": "darkgreen", "C": "darkblue", "G": "black", "T": "firebrick"}
        fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 8), sharex=True)
        plt.subplots_adjust(hspace=0)
        past_histograms = [0] * len(reference.seq)
        for b in "ACTG":
            normhist = ut.normalize(base_normhist1[b])
            ax1.bar(height=normhist, bottom=past_histograms, x=coords, width=1, linewidth=0,
                    color=colours[b])
            past_histograms = np.add(past_histograms, normhist)
        ax1.set_ylabel("Normalized per-base enrichment (Templated Substitutions, %s)" % name1)
        ax1.set_xlabel("vlsE position (bp)")
        ax1.set_xlim(min(coords), max(coords))
        ax1.spines['bottom'].set_visible(False)
        ax1.xaxis.set_ticks_position('none')
        ax1.yaxis.tick_left()
        past_histograms = [0] * len(reference.seq)
        for b in "ACTG":
            normhist = ut.normalize(base_normhist2[b])
            ax2.bar(height=normhist, bottom=past_histograms, x=coords, width=1, linewidth=0,
                    color=colours[b])
            past_histograms = np.add(past_histograms, normhist)
        ax2.set_ylabel("Normalized per-base enrichment (Templated Substitutions, %s)" % name2)
        ax2.set_xlabel("vlsE position (bp)")
        ax2.set_ylim(ax2.get_ylim()[::-1])
        ax2.yaxis.set_label_position("right")
        ax2.spines['top'].set_visible(False)
        ax2.xaxis.tick_bottom()
        ax2.yaxis.tick_right()
        pdf.savefig()
        plt.close()

        # difference plot
        colours = {"A": "darkgreen", "C": "darkblue", "G": "black", "T": "firebrick"}
        fig, ax = plt.subplots(1, figsize=(8, 8))
        for is_positive in (True, False):
            past_histograms = [0] * len(reference.seq)
            for b in "ACTG":
                norm1 = ut.normalize(base_normhist1[b])
                norm2 = ut.normalize(base_normhist2[b])
                histogram = [s - w if ((s > w) == is_positive) else 0 for s, w in zip(norm1, norm2)]
                ax.bar(height=histogram, bottom=past_histograms, x=coords, width=1, linewidth=0, color=colours[b])
                past_histograms = np.add(past_histograms, histogram)
        ax.set_ylabel("Templated substitution frequency (%s minus %s)" % (name1, name2))
        ax.set_xlabel("vlsE position (bp)")
        plt.suptitle("Sample 1 minus Sample 2 per-base normalized enrichment")
        pdf.savefig()
        plt.close()


def slippage(data, reportdir, database):
    """

    :param data:
    :param reportdir:
    :param database:
    :return:
    """
    references = database.get("references")
    slips = database.get("slips")
    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]
    for refid, tags, read_subset in data:
        reference = references[refid]
        base_report_path = os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags))

        # convert to Pandas DataFrame for plotting
        columns = ["Start Index", "Stop Index", "Start Position", "Total Repeat Length", "Unit Length", "Unit Shift",
                   "Total Shift", "Unit Sequence", "Nontemplated SNPs explained"]

        sliptable = [slip for read in read_subset for slip in al.get_slips(read, slips, reference)[0]]
        sliptable = pd.DataFrame(data=sliptable, columns=columns)
        if len(sliptable.index) > 0:
            with PdfPages(base_report_path + ".pdf") as pdf:
                # Page 1: Location of slippage events in the vlsE amplicon
                fig, ax = plt.subplots()
                ax.scatter(x=sliptable["Start Position"] + reference.offset, y=sliptable["Total Shift"],
                           s=20, alpha=0.03, linewidths=0)
                for x, vr in enumerate(reference.variable_regions, start=1):  # show VRs in the background
                    ax.axvspan(vr[0] + reference.offset, vr[1] + reference.offset,
                               ymax=0.02, facecolor='green', alpha=0.5, linewidth=0)
                ax.set_xlabel("vlsE Position (bp)")
                ax.set_ylabel("Bases inserted (+) or deleted (-)")
                ax.axhline(y=0, linewidth=1, color="k")
                ax.set_xlim([reference.offset + 1, reference.offset + 1 + len(reference.seq)])
                ax.set_ylim([-20, 20])
                plt.tight_layout()
                pdf.savefig()
                plt.close()

                # Page 2: Histogram of slippage event size
                ax = sliptable["Total Shift"].hist(bins=39, range=(-19.5, 19.5))
                ax.set_xlabel("Bases inserted (+) or deleted (-)")
                ax.set_ylabel("Frequency")
                start, end = ax.get_xlim()
                ax.xaxis.set_ticks(list(range(math.ceil(int(start) / 3) * 3, int(end), 3)))
                plt.tight_layout()
                pdf.savefig()
                plt.close()

                # Page 3: Compare (Mirror plot) actual slippage events with theoretical switching events
                fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(14, 14), sharex=True)
                plt.subplots_adjust(hspace=0)

                coords = list(range(reference.offset + 1, reference.offset + 1 + len(reference.seq)))
                observed_hist = [0] * len(reference.seq)
                for idx, r in sliptable.iterrows():
                    for x in range(r["Start Position"], r["Start Position"] + r["Total Repeat Length"]):
                        observed_hist[x] += abs(r["Total Shift"]) / r["Total Repeat Length"] / len(read_subset)
                ax1.bar(height=observed_hist, x=coords, width=1, linewidth=0, color="darkblue")
                ax1.set_ylabel("Frequency of slipped bases per read", color="darkblue")
                ax1.set_xlim(min(coords), max(coords))
                ax1.spines['bottom'].set_visible(False)
                ax1.xaxis.set_ticks_position('none')
                ax1.yaxis.tick_left()
                for tl in ax1.get_yticklabels():
                    tl.set_color("darkblue")

                cassette_snps = [0] * len(reference.seq)
                for cassette_name, aln in reference.cassettes_aln.items():
                    for op in al.trim_transform(aln.transform, len(reference.seq)):
                        if op[1] == "S":
                            cassette_snps[op[0]] += 1
                        elif op[1] == "D":
                            for x in range(op[2]):
                                cassette_snps[op[0] + x] += 1
                        elif op[1] == "I":
                            cassette_snps[op[0]] += len(op[2])
                cassette_snps = [x / len(reference.cassettes_aln) for x in cassette_snps]
                ax2.bar(height=cassette_snps, x=coords,
                        width=1, linewidth=0, color="firebrick")
                ax2.set_ylabel("Frequency of SNPs in silent cassettes", color="firebrick")
                ax2.set_xlabel("vlsE position (bp)")
                ax2.set_ylim(ax2.get_ylim()[::-1])
                ax2.yaxis.set_label_position("right")
                ax2.spines['top'].set_visible(False)
                ax2.xaxis.tick_bottom()
                ax2.yaxis.tick_right()
                for tl in ax2.get_yticklabels():
                    tl.set_color("firebrick")

                ref_tandem_repeats = [tr for tr in al.find_tandem_repeats(reference.seq, minhomology=1)
                                      if ((tr[1] - tr[0]) // tr[2]) > 1]
                ref_hist = [0] * len(reference.seq)
                for tr in ref_tandem_repeats:
                    for pos in range(tr[0], tr[1]):
                        ref_hist[pos] += 1
                movingaverage = np.convolve(ref_hist, np.ones(20) / float(20), mode="valid")  # smooth
                ax3.plot(coords[10:-9], movingaverage, 'k-')
                ax3.set_xlabel("vlsE Position (bp)")
                ax3.set_ylabel("Number of repeats\n(≥ 2 complete units, 20bp moving average)")
                pdf.savefig()
                plt.close()

                # Page 4 Find repeat sequences in reference
                fig, (ax1, ax2) = plt.subplots(2)
                ref_tandem_repeats = [tr for tr in al.find_tandem_repeats(reference.seq, minhomology=3)
                                      if ((tr[1] - tr[0]) / tr[2]) > 1]  # all repeats with a
                ref_hist = [0] * len(reference.seq)
                for tr in ref_tandem_repeats:
                    for pos in range(tr[0], tr[1]):
                        ref_hist[pos] += 1
                maxul = max(tr[2] for tr in ref_tandem_repeats)
                unitlength_hist_bins = list(range(maxul))
                unitlength_hist = [0] * maxul
                for tr in ref_tandem_repeats:
                    unitlength_hist[tr[2] - 1] += 1
                ax1.bar(x=coords, height=ref_hist, width=1, linewidth=0, color="firebrick")
                ax2.bar(x=unitlength_hist_bins, height=unitlength_hist, width=1, linewidth=0, color="darkblue")
                pdf.savefig()
                plt.close()


def list_of_slips(data, reportdir, database):
    """

    :param data:
    :param reportdir:
    :param database:
    :return:
    """

    references = database.get("references")
    switches = database.get("switches")
    slips = database.get("slips")
    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]
    for refid, tags, read_subset in data:
        reference = references[refid]
        base_report_path = os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags))
        rlen = len(reference.seq)

        # find observed tandem repeats
        templated_ops = [op for aln in reference.cassettes_aln.values() for op in
                         al.trim_transform(aln.transform, len(reference.seq))]

        total_nontemplated_bases = al.count_snps([op for r in read_subset for aln in r.alns for op in aln.transform
                                                  if op not in templated_ops])

        # write the TSV report
        with open(base_report_path + ".slips.tsv", "w") as handle:
            tsv_writer = csv.writer(handle, delimiter="\t")
            tsv_writer.writerow(["Total Nontemplated Bases:", total_nontemplated_bases,
                                 "Number of reads:", len(read_subset)])
            columns = ["Read Name", "Start Position", "Total Repeat Length", "Unit Length", "Unit Shift",
                       "Total Shift", "Unit Sequence", "Nontemplated SNPs explained"]
            tsv_writer.writerow(columns)
            for read in read_subset:
                best_slips, best_aln = al.get_slips(read, slips, reference)
                for slip in best_slips:
                    tsv_writer.writerow([read.name] + list(slip[2:]))


def long_switches(data, reportdir, database, minimum_length=40):
    """

    :param data:
    :param reportdir:
    :param database:
    :param minimum_length:
    :return:
    """
    switches = database.get("switches")
    references = database.get("references")
    if not switches:
        raise ValueError("switches.p is empty. Run \"vls label_switches\" to calculate switches "
                         "before exporting report.")
    if not all((r.cassettes_aln is None) == (r.sim_switch is None) for r in references.values()):
        raise ValueError("Not all references with cassettes have switch simulations computed. Run \"vast "
                         "simulate_switch_lengths\".")

    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]

    for refid, tags, read_subset in data:
        reference = references[refid]
        base_report_path = os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags))
        switch_density = [0 for x in range(len(reference.seq))]
        for read in read_subset:
            for aln in read.alns:
                templated_tf = al.templated(aln.transform, reference)
                switch_sets = switches[(templated_tf, reference.name)]
                if not isinstance(switch_sets, int):
                    for switch_set in switches[(templated_tf, reference.name)]:
                        for switch in switch_set:
                            start_idx = templated_tf[switch[0]][0]
                            stop_idx = templated_tf[switch[1] - 1][0]
                            length = stop_idx - start_idx + 1
                            if length >= minimum_length:
                                for x in range(start_idx, stop_idx):
                                    switch_density[x] += 1 / len(read.alns) / len(switch_sets)

        # Plot the data and send to PDF files.
        coords = list(range(reference.offset + 1, reference.offset + 1 + len(reference.seq)))
        with PdfPages(os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags) + ".pdf")) as pdf:
            # Page 1: Density
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.bar(height=switch_density, x=coords, width=1, linewidth=0, color="black")
            ax.set_title("Density of switches longer than %d bp" % minimum_length)
            ax.set_ylabel("Number of Switches")
            ax.set_xlabel("vlsE position (bp)")
            plt.tight_layout()
            pdf.savefig()
            plt.close()


def detailed_switch_length(data, reportdir, database, use_length_correction=False):
    """

    :param data:
    :param reportdir:
    :param database:
    :return:
    """
    switches = database.get("switches")
    references = database.get("references")

    max_switches = max(len(s) for ss in switches.values() for s in ss)
    if not switches:
        raise ValueError("switches.p is empty. Run \"vls label_switches\" to calculate switches "
                         "before exporting report.")
    if not all((r.cassettes_aln is None) == (r.sim_switch is None) for r in references.values()):
        raise ValueError("Not all references with cassettes have switch simulations computed. Run \"vls "
                         "simulate_switching\".")

    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]

    for refid, tags, read_subset in data:
        reference = references[refid]
        base_report_path = os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags))
        # find all switches from reads with 1-MAX switches
        switchhist = [[] for _ in range(max_switches)]
        weights = [[] for _ in range(max_switches)]
        for read in read_subset:
            weight = 1 / len(read.alns)
            for aln in read.alns:
                templated_tf = al.templated(aln.transform, reference)
                if not isinstance(switches[(templated_tf, reference.name)], int):
                    switch_set = random.choice(switches[(templated_tf, reference.name)])
                    num_switches = len(switch_set)
                    for switch in switch_set:
                        start_idx = templated_tf[switch[0]][0]
                        stop_idx = templated_tf[switch[1] - 1][0]
                        length = stop_idx - start_idx + 1
                        switchhist[num_switches - 1].append(length)
                        weights[num_switches - 1].append(weight)
        with open(base_report_path + ".meanlengths.tsv", 'w') as handle:
            tsv_writer = csv.writer(handle, delimiter="\t")
            tsv_writer.writerow(["From Reads containing exactly N Switches", "Total number of switches",
                                 "Switch Tract Length (Mean)", "Switch Tract Length (SD)"])
            for num_switches in range(1, max_switches + 1):
                switches_for_n = np.array(switchhist[num_switches - 1])
                weights_for_n = np.array(weights[num_switches - 1])
                wstats = DescrStatsW(data=switches_for_n, weights=weights_for_n)
                tsv_writer.writerow([num_switches, wstats.nobs, wstats.mean, wstats.std])
            all_switches = np.array([value for sublist in switchhist for value in sublist])
            all_weights = np.array([value for sublist in weights for value in sublist])
            wstats = DescrStatsW(data=all_switches, weights=all_weights, ddof=0)
            tsv_writer.writerow(["All Reads", wstats.nobs, wstats.mean, wstats.std])

        if len(all_switches) > 0:
            histbins = np.arange(min(all_switches), max(all_switches) + 1)
            hist, bin_edges = np.histogram(a=all_switches, weights=all_weights, bins=histbins)
            # TSV with histogram of switch lengths
            with open(base_report_path + ".lengthhist.tsv", "w") as handle:
                tsv_writer = csv.writer(handle, delimiter="\t")
                tsv_writer.writerow(["Bin [x, x+1)", "Absolute Frequency", "Frequency (Fraction of Total)"])
                for x, y in zip(histbins, hist):
                    tsv_writer.writerow([x, y, y / sum(hist)])
            # PDF with histogram of switch lengths
            with PdfPages(base_report_path + ".lengthhist.pdf") as pdf:
                # Page 1: Pretty version, with split axis.
                fig = plt.figure(figsize=(5, 3))
                ax1 = plt.subplot2grid((3, 1), (0, 0))
                ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=2)
                for ax in (ax1, ax2):
                    ax.bar(x=histbins[:-1], height=hist * 100 / sum(hist), width=1, linewidth=0, color="black")
                    ax.set_xlim((0, 150))
                    ax.axvline(wstats.mean)
                ax2.set_xlabel("Measured Length (bp)")
                ax2.set_ylabel("% of Total")
                ax1.set_ylim((6, 45))
                ax1.set_yticks([15, 30, 45])
                ax2.set_ylim((0, 6))
                ax2.set_yticks([0, 2, 4, 6])
                ax2.spines['top'].set_visible(False)
                ax1.spines['bottom'].set_visible(False)
                ax2.xaxis.tick_bottom()
                ax1.xaxis.tick_top()
                ax1.set_xticklabels([])
                # # adds secondary x-axis with predicted switch size
                # measured_sizes = [x[1] for x in reference.sim_switch]
                # actual_sizes = [x[0] for x in reference.sim_switch]
                # measured_size = interpolate.interp1d(actual_sizes, measured_sizes)
                # ax1.set_xticks([int(measured_size(x)) for x in range(20, 220, 20)])
                # ax1.set_xticklabels(list(range(20, 220, 20)))
                # ax1.xaxis.set_label_position('top')
                # ax1.set_xlabel("Predicted Length (bp)")
                plt.tight_layout()
                pdf.savefig()
                plt.close()

                # Page 2: Distribution of all switch sizes
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.bar(x=histbins[:-1], height=hist, width=1, linewidth=0, color="black")
                ax.set_xlabel("Measured Length (bp)")
                ax.set_ylabel("Frequency")
                pdf.savefig()
                plt.close()

                # Page 3+: Distributions for reads with 1, 2, 3, 4, ... switches per read. Just a sanity check.
                for num_switches in range(1, max_switches + 1):
                    switches_for_n = np.array(switchhist[num_switches - 1])
                    weights_for_n = np.array(weights[num_switches - 1])
                    if len(switches_for_n) > 0:
                        histbins = np.arange(min(switches_for_n) - 0.5, max(switches_for_n) + 1.5, 1)
                        hist, bin_edges = np.histogram(a=switches_for_n, weights=weights_for_n, bins=histbins)
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.bar(x=histbins[:-1], height=hist, width=1, linewidth=0, color="black")
                        ax.set_xlabel("Measured Length (bp)")
                        ax.set_ylabel("Frequency")
                        fig.suptitle("Reads with %d switches" % num_switches)
                        pdf.savefig()
                        plt.close()


def switch_length_simulation(data, reportdir, database, iterations=10000):
    """

    :param data:
    :param reportdir:
    :param database:
    :return:
    """
    references = database.get("references")
    # check simulation data exists first
    missing_simulations = [ref.name for ref in references.values() if
                           ref.cassettes_aln is not None and ref.sim_switch is None]
    if missing_simulations:
        ut.tprint("Simulation data is missing for some references. Running simulation(s) now.")
        si.simulate_switch_length(db=database, num_trials=iterations, recompute=False)
        references = database.get("references")

    for ref in references.values():
        if ref.cassettes_aln is not None:  # output analyzes each set of aligned cassettes
            base_report_path = os.path.join(reportdir, "%s" % ref.name)
            results = ref.sim_switch
            # export a TSV file.
            with open(base_report_path + ".tsv", "w") as handle:
                tsv_writer = csv.writer(handle, delimiter="\t")
                tsv_writer.writerow(["Actual length", "Measured length (mean)", "Measured length (stdev)"])
                for result in results:
                    tsv_writer.writerow(result)
            # export a plot
            with PdfPages(base_report_path + ".pdf") as pdf:
                # Page 1: Location of slippage events in the vlsE amplicon
                fig, ax = plt.subplots(figsize=(5, 5))
                x = np.array([r[0] for r in results])
                y = np.array([r[1] for r in results])
                y_error = np.array([r[2] for r in results])
                ax.plot(x, y, "k-")
                ax.fill_between(x, y - y_error, y + y_error, linewidths=0, facecolor="0.5")
                ax.set_ylim((0, 200))
                ax.set_xlim((0, 200))
                ax.set_xlabel("Actual Switch Length (bp)")
                ax.set_ylabel("Measured Switch Length (bp)")
                pdf.savefig()
                plt.close()


def cassette_usage(data, reportdir, database):
    """

    :param data:
    :param reportdir:
    :param database:
    :return:
    """
    # only do analysis for those bins where cassettes are aligned.
    references = database.get("references")
    switches = database.get("switches")

    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]

    for refid, tags, read_subset in data:
        reference = references[refid]
        if reference.cassettes_aln is not None:
            base_report_path = os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags))
            casnames = [cas for cas in reference.cassettes]
            casnames.sort(key=lambda x: int(x[x.rfind("_") + 1:]))  # sort the cassette names by the integer following
                                                                    # the underscore

            class UniqueSite(object):
                def __init__(self, cassette_name: str, op: tuple):
                    self.cassette_name = cassette_name
                    self.op = op
                    self.frequency = 0

                def increment(self, weight: float):
                    self.frequency += weight

            unique_sites = dict()
            for casname in casnames:
                for operation in al.trim_transform(reference.cassettes_aln[casname].transform, len(reference.seq)):
                    if len([op for casaln in reference.cassettes_aln.values()
                            for op in casaln.transform if op == operation]) == 1:
                        unique_sites[operation] = UniqueSite(casname, operation)

            unique_ops = {us.op for us in unique_sites.values()}
            for read in read_subset:
                for aln in read.alns:
                    for op in aln.transform:
                        if op in unique_ops:
                            unique_sites[op].increment(1 / len(read.alns))

            # report unique sites only.
            with open(base_report_path + ".uniquesites.tsv", "w") as handle:
                tsv_writer = csv.writer(handle, delimiter="\t")
                tsv_writer.writerow(["Cassette", "Position", "Operation", "Deletion Length or Insertion Sequence",
                                     "Frequency"])
                unique_sites_by_position = list(unique_sites.values())
                unique_sites_by_position.sort(key=lambda x: x.op[0])
                unique_sites_by_position.sort(key=lambda x: int(x.cassette_name[x.cassette_name.rfind("_") + 1:]))
                for us in unique_sites_by_position:
                    tsv_writer.writerow([us.cassette_name] + list(us.op) + [us.frequency])

            # report an average of unique sites for each cassette.
            with open(base_report_path + ".uniquesites_grouped_by_cassette.tsv", "w") as handle:
                tsv_writer = csv.writer(handle, delimiter="\t")
                tsv_writer.writerow(["Cassette", "Frequency (Mean)", "Frequency (STD)", "Frequency (Nobs)"])
                for casname in casnames:
                    sites = [us.frequency for us in unique_sites_by_position if us.cassette_name == casname]
                    mean = np.mean(sites)
                    std = np.std(sites)
                    nobs = len(sites)
                    tsv_writer.writerow([casname, mean, std, nobs])

            # report the total number of sites (indels are treated as single events) per cassette, divided by the
            # number of cassettes those sites could have originated from.
            sites_per_cassette = {casname: 0 for casname in casnames}
            for read in read_subset:
                for aln in read.alns:
                    for op in aln.transform:
                        casnames_with_op = set()
                        for casname, casaln in reference.cassettes_aln.items():
                            if op in casaln.transform:
                                casnames_with_op.add(casname)
                        for casname in casnames_with_op:
                            sites_per_cassette[casname] += 1 / (len(casnames_with_op) * len(read.alns))

            with open(base_report_path + ".allsites.tsv", "w") as handle:
                tsv_writer = csv.writer(handle, delimiter="\t")
                tsv_writer.writerow(["Cassette", "Frequency (number of sites)"])
                for casname in casnames:
                    tsv_writer.writerow([casname, sites_per_cassette[casname]])

            # report the number of switch events per cassette, where each switch event is divided by the number of
            # cassettes it could come from
            switches_per_cassette = {casname: 0 for casname in casnames}
            for read in read_subset:
                for aln in read.alns:
                    templated_tf = al.templated(aln.transform, reference)
                    if (templated_tf, refid) in switches:
                        switch_sets = switches[(templated_tf, refid)]
                        if not isinstance(switch_sets, int):
                            for ss in switch_sets:
                                for switch in ss:
                                    start, stop, origins = switch
                                    weight = 1 / (len(origins) * len(switch_sets) * len(read.alns))
                                    for origin in origins:
                                        switches_per_cassette[origin] += weight

            with open(base_report_path + ".switches.tsv", "w") as handle:
                tsv_writer = csv.writer(handle, delimiter="\t")
                tsv_writer.writerow(["Cassette", "Frequency (number of switches)"])
                for casname in casnames:
                    tsv_writer.writerow([casname, switches_per_cassette[casname]])


def detailed_switches_per_read(data, reportdir, database):
    """

    :param data:
    :param reportdir:
    :param database:
    :return:
    """
    references = database.get("references")
    switches = database.get("switches")

    if not switches:
        raise ValueError("switches.p not found or empty. Run \"vls label_switches\" to calculate switches "
                         "before exporting report.")

    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]

    for refid, tags, read_subset in data:
        base_report_path = os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags))
        reference = references[refid]

        num_switches = []
        tmtc = []
        for read in read_subset:
            switches_per_aln = []
            impossibles_per_aln = []
            for aln in read.alns:
                templated_tf = al.templated(aln.transform, reference)
                if (templated_tf, refid) in switches:
                    switch_sets = switches[(templated_tf, refid)]
                    if not isinstance(switch_sets, int):
                        switches_per_aln.append(min(len(s) for s in switch_sets))
                    else:
                        impossibles_per_aln.append(switch_sets)
                else:
                    raise ValueError("Switch not found in label_switches database; please run \"vls "
                                     "label_switches\" to calculate switches before exporting report.")
            if switches_per_aln:
                num_switches.append(min(switches_per_aln))
            else:
                tmtc.append(min(impossibles_per_aln))
        mean = np.mean(num_switches)
        model = zi.ZeroInflatedPoisson(num_switches)
        results = model.fit()
        boot_mean, boot_std, boot_samples = results.bootstrap(nrep=1000, store=True)
        boot_pis = boot_samples[:, 0]
        boot_lambdas = boot_samples[:, 1]

        # writes TSV report
        with open(base_report_path + ".tsv", "w") as handle:
            tsv_writer = csv.writer(handle, delimiter="\t")
            max_num_switches = max(len(s) for ss in switches.values() if not isinstance(ss, int) for s in ss)
            coords = list(range(0, max_num_switches + 1))
            tsv_writer.writerow(["Arithmetic Mean:", np.mean(num_switches)])
            tsv_writer.writerow(["Number of Reads:", len(num_switches)])
            tsv_writer.writerow(["Number of Switches:", sum(num_switches)])
            tsv_writer.writerow(["Min length of TMTC (Too many to count) reads:"] + tmtc)
            tsv_writer.writerow([])
            tsv_writer.writerow(["Zero-Inflated Poisson Maximum Likelihood Estimates", "Mean", "SD",
                                 "N (number of bootstraps)"])
            tsv_writer.writerow(["Excess Zeros, \u03C0:", np.mean(boot_pis), np.std(boot_pis), len(boot_pis)])
            tsv_writer.writerow(["Mean Corrected for Excess Zeros, \u03bb:",
                                 np.mean(boot_lambdas), np.std(boot_lambdas), len(boot_lambdas)])
            tsv_writer.writerow([])
            tsv_writer.writerow(["Switches per Read"] + coords + ["Too many to compute"])
            tsv_writer.writerow(["Actual Frequency"] + [num_switches.count(x) for x in coords] + [len(tmtc)])
            poisson_dist = [stats.poisson.pmf(k=a, mu=mean) * len(num_switches) for a in coords]
            tsv_writer.writerow(["Poisson"] + poisson_dist)
            zip_dist = [zi.zip_pmf(x=a, pi=np.mean(boot_pis), lambda_=np.mean(boot_lambdas)) * len(num_switches)
                        for a in coords]
            tsv_writer.writerow(["Zero-inflated Poisson"] + zip_dist)


def variant_frequency(data, reportdir, database, max_venn=4):
    """

    :param data:
    :param reportdir:
    :param database:
    :param max_venn:
    :param sortorder: a list of the fields in the groupby option by which to sort the results. Defaults to alphabetical
    :return:
    """
    references = database.get("references")
    reads = database.get("reads")
    switches = database.get("switches")

    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]
    bins = [(refid, tags) for refid, tags, read_subset in data]
    datadict = {(refid, tags): read_subset for refid, tags, read_subset in data}

    # group tagsets by refid
    refids = {refid: [] for refid, tags in bins}
    for refid, tags in bins:
        refids[refid].append(tuple(sorted(list(tags))))  # hashable

    # analyze each reference independently
    for refid, tagsets in refids.items():
        # create a dict of dicts so that variants[variant][tags] yields a frequency
        # variants are referenced by a transform (itself a tuple of templated ops only, to avoid interference from
        # errors.
        reference = references[refid]
        variants = {}
        for variant in set(al.templated_variants(reads, reference)):
            variants[variant] = {tags: 0 for tags in tagsets}

        readcounts = []

        # populate the table
        for tags in tagsets:
            read_subset = datadict[(refid, frozenset(tags))]
            readcounts.append(len(read_subset))
            for variant in al.templated_variants(read_subset, references[refid]):
                variants[variant][tags] += 1 / len(read_subset)

        # Output every single variant and its frequency in every bin
        basename = os.path.join(reportdir, refid)
        tag_names = sorted(list({tag[0] for refid, tags in bins for tag in tags}))
        # orders the variants for output from most to least shared (between all the bins). Variants present in more bins
        # are earlier in the sequence.
        outputs = list(variants)
        outputs.sort(key=lambda variant: len([v for v in variants[variant].values() if v > 0]))
        outputs.reverse()
        with open(basename + ".frequency.tsv", "w") as handle:
            tsv_writer = csv.writer(handle, delimiter="\t")
            for tagname in tag_names:
                tsv_writer.writerow([tagname, ""] + [dict(tags)[tagname] for tags in tagsets])
            tsv_writer.writerow([])
            tsv_writer.writerow(["Number of Reads", ""] + readcounts)
            tsv_writer.writerow([])
            tsv_writer.writerow(["Variant", "Number of switches"])
            for variant in outputs:
                variant_switches = set()
                for v in variant:
                    # Some variants have multiple templated alignments where one of them is (); I don't know why.
                    # However, they are also not found in the switches db, so there must be something wrong with them.
                    if (v, refid) not in switches:  # workaround
                        variant_switches.add("")
                    elif isinstance(switches[(v, refid)], int):
                        variant_switches.add(">" + str(switches[(v, refid)]))
                    else:
                        variant_switches.add(str(min(len(switch_set) for switch_set in switches[(v, refid)])))
                tsv_writer.writerow([repr(variant), ",".join(variant_switches)]
                                    + [variants[variant][tags] for tags in tagsets])

        # hotspots in frequently re-emerging variants.

        coords = list(range(reference.offset + 1, reference.offset + 1 + len(reference.seq)))
        hist = [0] * len(reference.seq)
        # get names of variants that have emerged independently
        variants_to_plot = [o for o in outputs if len([v for v in variants[o].values() if v > 0]) > 1]

        # # this filter step is to select those only that are found in multiple bclones, and so could not be preexisting
        # variants_to_plot = [o for o in variants_to_plot
        #                     if len(set(dict(k)["bclone"] for k, v in variants[o].items() if v > 0)) > 1]

        for variant in variants_to_plot:
            variant_switches = set()
            for v in variant:
                # Some variants have multiple templated alignments where one of them is (); I don't know why.
                # However, they are also not found in the switches db, so there must be something wrong with them.
                if (v, refid) in switches:  # bypasses this bug
                    variant_switches.add(str(min(len(switch_set) for switch_set in switches[(v, refid)])))

        for variant in variants_to_plot:
            for mapping in variant:
                for op in mapping:
                    if op[1] == "S":
                        hist[op[0]] += 1 / len(read_subset) / len(variant)
                    if op[1] == "D":
                        for x in range(op[2]):
                            hist[op[0] + x] += 1 / len(read_subset) / len(variant)
                    elif op[1] == "I":
                        hist[op[0]] += len(op[2]) / len(read_subset) / len(variant)
        # get cassette SNPs for reference
        cassette_snps = [0] * len(reference.seq)
        for cassette_name, aln in reference.cassettes_aln.items():
            for op in al.trim_transform(aln.transform, len(reference.seq)):
                if op[1] == "S":
                    cassette_snps[op[0]] += 1
                elif op[1] == "D":
                    for x in range(op[2]):
                        cassette_snps[op[0] + x] += 1
                elif op[1] == "I":
                    cassette_snps[op[0]] += len(op[2])
        cassette_snps = [x / len(reference.cassettes_aln) for x in cassette_snps]

        with PdfPages(basename + ".reemergence_hotspots.pdf") as pdf:
            # Page 5: Mirror plot comparing the distribution of Templated to Cassettes
            fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 8), sharex=True)
            plt.subplots_adjust(hspace=0)
            ax1.bar(height=hist, x=coords, width=1, linewidth=0, color="darkblue")
            ax1.set_ylabel("Frequency of Templated SNPs", color="darkblue")
            ax1.set_xlabel("vlsE position (bp)")
            ax1.set_xlim(min(coords), max(coords))
            ax1.spines['bottom'].set_visible(False)
            ax1.xaxis.set_ticks_position('none')
            ax1.yaxis.tick_left()
            for tl in ax1.get_yticklabels():
                tl.set_color("darkblue")

            ax2.bar(height=cassette_snps, x=coords, width=1, linewidth=0, color="firebrick")
            ax2.set_ylabel("Frequency of SNPs in Silent Cassettes", color="firebrick")
            ax2.set_xlabel("vlsE position (bp)")
            ax2.set_ylim(ax2.get_ylim()[::-1])
            ax2.yaxis.set_label_position("right")
            ax2.spines['top'].set_visible(False)
            ax2.xaxis.tick_bottom()
            ax2.yaxis.tick_right()
            for tl in ax2.get_yticklabels():
                tl.set_color("firebrick")
            pdf.savefig()
            plt.close()


        # count the percentage of total variants shared between two samples
        shared = np.zeros((len(tagsets), len(tagsets)))
        unique = np.zeros((len(tagsets), len(tagsets)))
        for x, x_tags in enumerate(tagsets):
            for y, y_tags in enumerate(tagsets):
                x_variants = set(v for v in variants
                                 if variants[v][x_tags]
                                 and v != frozenset({()}))  # excludes parentals
                y_variants = set(v for v in variants if variants[v][y_tags] and v != frozenset({()}))
                try:
                    shared[x, y] = len(x_variants & y_variants) / (len(x_variants | y_variants))
                    unique[x, y] = len(x_variants ^ y_variants) / (len(x_variants | y_variants))
                except ZeroDivisionError:
                    shared[x, y] = 0
                    unique[x, y] = np.NaN
        labels = ["_".join([t[0] + "." + (t[1] if t[1] else "None") for t in tags]) for tags in tagsets]

        # export shared heatmap as TSV
        with open(basename + ".commonvariants.tsv", "w") as handle:
            tsv_writer = csv.writer(handle, delimiter="\t")
            tsv_writer.writerow([""] + labels)
            for label, row in zip(labels, shared):
                tsv_writer.writerow([label] + list(row))

        # export unique heatmap as TSV
        with open(basename + ".uniquevariants.tsv", "w") as handle:
            tsv_writer = csv.writer(handle, delimiter="\t")
            tsv_writer.writerow([""] + labels)
            for label, row in zip(labels, unique):
                tsv_writer.writerow([label] + list(row))

        # Plot heatmaps to PDF
        with PdfPages(basename + ".common_unique_variants.pdf") as pdf:
            # Page 1: heatmap of shared (templated) variants
            fig, ax = plt.subplots()
            heatmap = ax.pcolor(shared, cmap=get_cmap("inferno"), norm=colors.SymLogNorm(0.005))
            # put the major ticks at the middle of each cell
            ax.set_xticks(np.arange(shared.shape[0]) + 0.5, minor=False)
            ax.set_yticks(np.arange(shared.shape[1]) + 0.5, minor=False)
            # want a more natural, table-like display
            ax.invert_yaxis()
            ax.xaxis.tick_top()
            # label rows and columns
            ax.set_xticklabels(labels, minor=False, rotation="vertical")
            ax.set_yticklabels(labels, minor=False)
            plt.colorbar(heatmap, ticks=[1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0])
            plt.tick_params(axis='both', which='major', labelsize=4)
            pdf.savefig(bbox_inches='tight')
            plt.close()

            # Page 2: heatmap of unique (templated) variants
            fig, ax = plt.subplots()
            heatmap = ax.pcolor(unique, cmap=get_cmap("inferno"),
                                norm=colors.SymLogNorm(linthresh=0.99, linscale=0.001))
            # put the major ticks at the middle of each cell
            ax.set_xticks(np.arange(unique.shape[0]) + 0.5, minor=False)
            ax.set_yticks(np.arange(unique.shape[1]) + 0.5, minor=False)
            # want a more natural, table-like display
            ax.invert_yaxis()
            ax.xaxis.tick_top()
            # label rows and columns
            ax.set_xticklabels(labels, minor=False, rotation="vertical")
            ax.set_yticklabels(labels, minor=False, )
            plt.colorbar(heatmap, ticks=[1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0])
            plt.tick_params(axis='both', which='major', labelsize=4)
            pdf.savefig(bbox_inches='tight')
            plt.close()

        # Generates tables with Venn diagram data up to the maximum venn table size.
        for set_count in range(1, min(max_venn + 1, len(tagsets) + 1)):
            with open(basename + ".venn%02.0f-shared.tsv" % set_count, "w") as handle:
                tsv_writer = csv.writer(handle, delimiter="\t")
                metavariables = string.ascii_uppercase[:set_count]
                metavariables_choose_n = [list(combinations(metavariables, n)) for n in range(1, set_count + 1)]
                metavariable_combinations = [", ".join(l) for l in chain.from_iterable(metavariables_choose_n)]
                header = ["Set %s" % m for m in metavariables] + metavariable_combinations
                tsv_writer.writerow(header)
                for tagsubsets in combinations(tagsets, set_count):
                    venn_dict = {tags: set(v for v in variants if variants[v][tags] and v != tuple())
                                 for tags in tagsubsets}
                    venn_table = ut.VennTable(venn_dict)
                    set_names = [", ".join(["=".join(t) for t in tags]) for tags in tagsubsets]
                    results = []
                    for n in range(1, set_count + 1):
                        for combination in combinations(tagsubsets, n):
                            results.append(venn_table.get_overlap(combination))
                    tsv_writer.writerow(set_names + results)


def distance_between_switches(data, reportdir, database, switchlength, iterations=1000000):
    """

    :param data:
    :param reportdir:
    :param database:
    :param switchlength:
    :param iterations:
    :return:
    """
    references = database.get("references")
    switches = database.get("switches")

    # run simulations
    simulations = {}
    for reference in references.values():
        ut.tprint("Running simulations for reference %s..." % reference.name)
        if reference.cassettes_aln is not None:
            simulations[reference.name] = si.simulate_double_switch_distance(reference,
                                                                             length=switchlength, num_trials=iterations)


    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]

    for refid, tags, read_subset in data:
        reference = references[refid]
        tagstring = "_".join([refid] + [t[0] + "." + (t[1] if t[1] else "None") for t in tags])
        base_report_path = os.path.join(reportdir, tagstring)
        mins = []
        mids = []
        maxes = []
        weights = []
        for read in read_subset:
            alns = [{"templated_tf": al.templated(aln.transform, reference)} for aln in read.alns]
            for aln in alns:
                if not (aln["templated_tf"], refid) in switches:
                    raise ValueError("Switch not found in label_switches database; please run \"vls "
                                     "label_switches\" to calculate switches before exporting report.")

                tf = aln["templated_tf"]
                switch_sets = switches[(tf, refid)]
                if not isinstance(switch_sets, int) and all(len(ss) == 2 for ss in switch_sets):  # doubles only
                    aln["is_double"] = True
                    aln["min"] = []
                    aln["max"] = []
                    aln["mid"] = []
                    for ss in switch_sets:
                        min_start = tf[ss[0][1] - 1][0]
                        min_stop = tf[ss[1][0]][0]
                        aln["min"].append(min_stop - min_start)
                        max_per_cassette_pair = []
                        mid_per_cassette_pair = []
                        for cas1 in ss[0][2]:
                            for cas2 in ss[1][2]:
                                post_cas1_tf = [op[0] for op in reference.cassettes_aln[cas1].transform
                                                if op[0] in range(min_start, len(reference.seq))]
                                if post_cas1_tf:
                                    max_start = min(min(post_cas1_tf), min_stop)
                                else:
                                    max_start = min_stop
                                pre_cas2_tf = [op[0] for op in reference.cassettes_aln[cas2].transform
                                               if op[0] in range(0, min_stop)]
                                if pre_cas2_tf:
                                    max_stop = max(max(pre_cas2_tf), min_start)
                                else:
                                    max_stop = min_start
                                max_per_cassette_pair.append(max(max_stop - max_start, 0))
                                mid_per_cassette_pair.append(max(np.mean([max_stop, min_stop]) -
                                                                 np.mean([max_start, min_start]), 0))
                        aln["max"].append(np.mean(max_per_cassette_pair))
                        aln["mid"].append(np.mean(mid_per_cassette_pair))

                else:  # marks False to avoid using this aln.
                    aln["is_double"] = False

            # use double-switch alignments only.
            alns = [aln for aln in alns if aln["is_double"]]
            if alns:
                for aln in alns:
                    maxes.append(np.mean(aln["max"]))
                    mids.append(np.mean(aln["mid"]))
                    mins.append(np.mean(aln["min"]))
                    weights.append(1 / len(alns))

        min_stats = DescrStatsW(data=mins, weights=weights)
        mid_stats = DescrStatsW(data=mids, weights=weights)
        max_stats = DescrStatsW(data=maxes, weights=weights)
        sim_mins = simulations[reference.name]["min"]
        sim_mids = simulations[reference.name]["mid"]
        sim_maxes = simulations[reference.name]["max"]
        with open(base_report_path + ".tsv", "w") as handle:
            tsvwriter = csv.writer(handle, delimiter="\t")
            tsvwriter.writerow(["", "Actual (Mean)", "Actual (SD)", "Actual (N)",
                                "Simulated (Mean)", "Simulated (SD)", "Simulated (N)"])
            tsvwriter.writerow(["Minimal Distance", min_stats.mean, min_stats.std, min_stats.nobs,
                                np.mean(sim_mins), np.std(sim_mins), len(sim_mins)])
            tsvwriter.writerow(["Midpoint Distance", mid_stats.mean, mid_stats.std, mid_stats.nobs,
                                np.mean(sim_mids), np.std(sim_mids), len(sim_mids)])
            tsvwriter.writerow(["Maximal Distance", max_stats.mean, max_stats.std, max_stats.nobs,
                                np.mean(sim_maxes), np.std(sim_maxes), len(sim_maxes)])

        # PDF with histogram of switch distances
        with PdfPages(base_report_path + ".pdf") as pdf:
            # Page 1: Smoothed Minimal
            fig, (ax_actual, ax_simulated) = plt.subplots(2, figsize=(8, 8), sharex=True)
            fig.suptitle("Minimal Distances (binned in 50)")
            all_data = sim_mins + sim_mids + sim_maxes + mins + mids + maxes
            histbins = np.arange(min(all_data), max(all_data) + 1, 50)
            actual_hist, _ = np.histogram(a=mins, weights=weights, bins=histbins)
            ax_actual.bar(x=histbins[:-1], height=actual_hist, width=50, linewidth=0, color="black")
            simulated_hist, _ = np.histogram(a=sim_mins, bins=histbins)
            ax_simulated.bar(x=histbins[:-1], height=simulated_hist, width=50, linewidth=0, color="black")
            ax_simulated.set_xlabel("Distance between switches (bp)")
            ax_actual.set_ylabel("Frequency (Actual)")
            ax_simulated.set_ylabel("Frequency (Simulated)")
            pdf.savefig()
            plt.close()

            # Page 1: Minimal
            fig, (ax_actual, ax_simulated) = plt.subplots(2, figsize=(8, 8), sharex=True)
            fig.suptitle("Minimal Distances")
            all_data = sim_mins + sim_mids + sim_maxes + mins + mids + maxes
            histbins = np.arange(min(all_data), max(all_data) + 1)
            actual_hist, _ = np.histogram(a=mins, weights=weights, bins=histbins)
            ax_actual.bar(x=histbins[:-1], height=actual_hist, width=1, linewidth=0, color="black")
            simulated_hist, _ = np.histogram(a=sim_mins, bins=histbins)
            ax_simulated.bar(x=histbins[:-1], height=simulated_hist, width=1, linewidth=0, color="black")
            ax_simulated.set_xlabel("Distance between switches (bp)")
            ax_actual.set_ylabel("Frequency (Actual)")
            ax_simulated.set_ylabel("Frequency (Simulated)")
            pdf.savefig()
            plt.close()
            # Page 2: Midpoints
            fig, (ax_actual, ax_simulated) = plt.subplots(2, figsize=(8, 8), sharex=True)
            fig.suptitle("Midpoint Distances")
            all_data = sim_mins + sim_mids + sim_maxes + mins + mids + maxes
            histbins = np.arange(min(all_data), max(all_data) + 1)
            actual_hist, _ = np.histogram(a=mids, weights=weights, bins=histbins)
            ax_actual.bar(x=histbins[:-1], height=actual_hist, width=1, linewidth=0, color="black")
            simulated_hist, _ = np.histogram(a=sim_mids, bins=histbins)
            ax_simulated.bar(x=histbins[:-1], height=simulated_hist, width=1, linewidth=0, color="black")
            ax_simulated.set_xlabel("Distance between switches (bp)")
            ax_actual.set_ylabel("Frequency (Actual)")
            ax_simulated.set_ylabel("Frequency (Simulated)")
            pdf.savefig()
            plt.close()
            # Page 3: Maximal
            fig, (ax_actual, ax_simulated) = plt.subplots(2, figsize=(8, 8), sharex=True)
            fig.suptitle("Maximal Distances")
            all_data = sim_mins + sim_mids + sim_maxes + mins + mids + maxes
            histbins = np.arange(min(all_data), max(all_data) + 1)
            actual_hist, _ = np.histogram(a=maxes, weights=weights, bins=histbins)
            ax_actual.bar(x=histbins[:-1], height=actual_hist, width=1, linewidth=0, color="black")
            simulated_hist, _ = np.histogram(a=sim_maxes, bins=histbins)
            ax_simulated.bar(x=histbins[:-1], height=simulated_hist, width=1, linewidth=0, color="black")
            ax_simulated.set_xlabel("Distance between switches (bp)")
            ax_actual.set_ylabel("Frequency (Actual)")
            ax_simulated.set_ylabel("Frequency (Simulated)")
            pdf.savefig()
            plt.close()


def cassette_similarity(data, reportdir, database):
    """

    :param data:
    :param reportdir:
    :param database:
    :return:
    """
    references = database.get("references")
    for reference in references.values():
        if reference.cassettes_aln is not None:
            with PdfPages(os.path.join(reportdir, reference.name + ".pdf")) as pdf:
                cassette_names = [casname for casname in reference.cassettes_aln]
                cassette_names.sort(key=lambda s: int(ut.get_trailing_number(s)))
                similarity_matrix = np.zeros((len(cassette_names), len(cassette_names)))
                for x, c1 in enumerate(cassette_names):
                    for y, c2 in enumerate(cassette_names):
                        cas1 = reference.cassettes_aln[c1].transform
                        cas2 = reference.cassettes_aln[c2].transform
                        similarity_matrix[x, y] = al.map_distance(cas1, cas2)

                # plots a heatmap
                fig, ax = plt.subplots()
                heatmap = ax.pcolor(similarity_matrix, cmap=get_cmap("inferno"))

                # put the major ticks at the middle of each cell
                ax.set_xticks(np.arange(len(cassette_names)) + 0.5, minor=False)
                ax.set_yticks(np.arange(len(cassette_names)) + 0.5, minor=False)

                # want a more natural, table-like display
                ax.invert_yaxis()
                ax.xaxis.tick_top()

                # label rows and columns
                ax.set_xticklabels(cassette_names, minor=False, rotation="vertical")
                ax.set_yticklabels(cassette_names, minor=False)
                plt.colorbar(heatmap)
                pdf.savefig(bbox_inches='tight')
                plt.close()

            with open(os.path.join(reportdir, reference.name + ".tsv"), "w") as handle:
                tsv_writer = csv.writer(handle, delimiter="\t")
                tsv_writer.writerow([None] + cassette_names)
                for x, row in enumerate(similarity_matrix):
                    tsv_writer.writerow([cassette_names[x]] + [str(cell) for cell in row])


def max_variants(data, reportdir, database, limit_max_num_switches=2):
    """
    For each reference, outputs a table with the max number of variants it can be produced for a given switch
    :param data:
    :param reportdir:
    :param database:
    :param limit_max_num_switches: The maximum number of switches to compute the maximum variant count for.
    Computation complexity gets high beyond 3 switches.
    :return:
    """

    references = database.get("references")
    for reference in (r for r in references.values() if r.cassettes_aln):
        # count the total number of possible nucleotide variants
        insubs = {op for aln in reference.cassettes_aln.values() for op in
                  al.trim_transform(aln.transform, len(reference.seq)) if op[1] != "D"}
        variants_at_position = []
        for position in range(len(reference.seq)):
            insertions = len([i for i in insubs if i[0] == position and i[1] == "I"])
            substitutions = len([i for i in insubs if i[0] == position and i[1] == "S"])
            variants_at_position.append((1 + insertions) * (1 + substitutions))
        dnas = np.product(variants_at_position, dtype=np.float64)

        # break it down by number of switches and switch size
        with open(os.path.join(reportdir, "%s.nucleotides.tsv" % reference.name), "w") as handle:
            tsv_writer = csv.writer(handle, delimiter="\t")
            tsv_writer.writerow(["WARNING: Deletions were ignored to simplify computation, so numbers are "
                                 "underestimates."])
            tsv_writer.writerow(["Maximum number of DNA variants:", dnas])
            tsv_writer.writerow([])
            tsv_writer.writerow(["Length of Switch↓   Number of Switches→"] +
                                ["%d" % i for i in range(1, limit_max_num_switches + 1)])
            for switch_length in range(1, len(reference.seq) + 1):  # row in tsv for each simulated switch length
                # calculate the number of possible variants for one switch starting at a given position
                variants_by_switch_start = []
                for start_x in range(len(reference.seq) - switch_length):
                    variant_switches = set(tuple(op for op in casaln.transform
                                                 if start_x <= op[0] <= (start_x + switch_length))
                                           for casaln in reference.cassettes_aln.values())
                    variants_by_switch_start.append(len(variant_switches))
                max_num_switches = min(len(reference.seq) // switch_length, limit_max_num_switches)
                maxvariants = []
                for num_switches in range(1, max_num_switches + 1):
                    ut.tprint("Switch Length: %d; Number of switches: %d" % (switch_length, num_switches), ontop=True)
                    total = 0
                    for starts in ut.interval_start_iterator(l=len(reference.seq) - 1, n=num_switches, k=switch_length):
                        variantlist = [variants_by_switch_start[start] for start in starts]
                        total += reduce(mul, variantlist, 1)
                    maxvariants.append(total)
                tsv_writer.writerow([switch_length] + maxvariants)

        # count the total number of possible protein variants
        codon_positions = al.get_codon_positions(reference)  # positions of all full-length codons in frame
        variants_at_position = []
        for codon_start in codon_positions:
            insubs_in_codon = []
            for x, (optype, position) in enumerate(((o, p) for o in "IS" for p in range(0, 3))):
                dependent_ops = [i for i in insubs if
                                 (codon_start + position == i[0]) and i[1] == optype]
                dependent_ops.append(None)  # no ops is an option and should be counted as a possibility
                insubs_in_codon.append(dependent_ops)
            peptides = set(al.translate(al.transform(reference.seq, [i for i in independent_ops if i],
                                                     start=codon_start, stop=codon_start + 3))
                           for independent_ops in product(*insubs_in_codon))
            variants_at_position.append(len(peptides))
        proteins = np.product(variants_at_position, dtype=np.float64)

        with open(os.path.join(reportdir, "%s.proteins.tsv" % reference.name), "w") as handle:
            tsv_writer = csv.writer(handle, delimiter="\t")
            tsv_writer.writerow(["WARNING: Deletions were ignored to simplify computation, so numbers are "
                                 "underestimates."])
            tsv_writer.writerow(["Maximum number of Protein variants:", proteins])


def aa_positions(data, reportdir, database):
    """

    :param data:
    :param reportdir:
    :param database:
    :return:
    """

    # Calculate the theoretical aa_position frequencies
    references = database.get("references")
    references_aa_positions = {refid: {"inframe": [], "corrected": []} for refid in references}
    for reference in references.values():
        if reference.cassettes_aln:
            for casname, alignment in reference.cassettes_aln.items():
                # 1. Proteins from all in-frame alns, no error-correction.
                protein_alns = al.translate_mapping(mapping=al.trim_transform(alignment.transform, len(reference.seq)),
                                                    reference=reference, templ=True,
                                                    nontempl=True, correctframe=False, filterframe=True,
                                                    filternonsense=False)
                references_aa_positions[reference.name]["inframe"].append(protein_alns)
                # 4. Proteins from all in-frame alns, with error-correction.
                protein_alns = al.translate_mapping(mapping=al.trim_transform(alignment.transform, len(reference.seq)),
                                                    reference=reference, templ=True,
                                                    nontempl=True, correctframe=True, filterframe=True,
                                                    filternonsense=False)
                references_aa_positions[reference.name]["corrected"].append(protein_alns)

    # Compute from data
    for refid, tags, read_subset in data:
        ut.tprint("Computing: %s" % ut.get_tagstring(refid=refid, tags=tags), ontop=False)
        reference = references[refid]
        methods = {"inframe_native_all", "inframe_native_templated", "inframe_native_nontemplated",
                   "inframe_corrected_all", "inframe_corrected_templated", "inframe_corrected_nontemplated"}
        refmethods = set(method for method in references_aa_positions[refid].keys())
        if not read_subset:
            continue

        coords = list(range(int(math.ceil(reference.offset / 3) + 1),  # start at first full amino acid
                            (reference.offset + len(reference.seq)) // 3 + 1))  # end at last full amino acid

        # compute frequency and read counts
        counts = {method: np.zeros([len(coords)]) for method in methods}
        num_reads = len(read_subset)

        P = multiprocessing.Pool(multiprocessing.cpu_count())
        arg_generator = ((read, methods, coords, reference) for read in read_subset)
        for x, results in enumerate(P.imap_unordered(al.get_aa_frequencies, arg_generator), 1):
            for method, vector in results.items():
                counts[method] += vector
            ut.tprint("Computing protein alignments: %d of %d reads completed." % (x, num_reads), ontop=True)
        print()  # newline

        # compute theoretical frequencies
        reference_aa_freq_by_method = {}
        for method in refmethods:
            freq = [0] * len(coords)
            for cassette in references_aa_positions[refid][method]:
                for alignment in cassette:
                    for op in alignment.transform:
                        if op[1] == "S":
                            freq[op[0]] += 1 / len(references_aa_positions[refid][method]) / len(cassette)
                        if op[1] == "D":
                            for x in range(op[2]):
                                freq[op[0] + x] += (1 / len(references_aa_positions[refid][method])
                                                       / len(cassette))
                        elif op[1] == "I":
                            freq[op[0]] += (len(op[2]) / len(references_aa_positions[refid][method])
                                               / len(cassette))
            reference_aa_freq_by_method[method] = freq

        # writes TSV report
        with open(os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags) + ".tsv"), "w") as handle:
            tsv_writer = csv.writer(handle, delimiter="\t")

            # Part 1: actual frequencies
            tsv_writer.writerow([])
            tsv_writer.writerow(["Actual frequencies"])
            tsv_writer.writerow([])
            tsv_writer.writerow(["Position:"] + coords)
            tsv_writer.writerow(["Amino Acid:"] + list(reference.protein))
            for method in methods:
                tsv_writer.writerow(["Frequency of variant amino acid (Method: %s)" % method]
                                    + list(counts[method]/num_reads))

            # Part 2: theoretical frequencies
            tsv_writer.writerow([])
            tsv_writer.writerow(["Theoretical frequencies"])
            tsv_writer.writerow([])
            tsv_writer.writerow(["Position:"] + coords)
            tsv_writer.writerow(["Amino Acid:"] + list(reference.protein))
            for method in refmethods:
                tsv_writer.writerow(["Frequency of variant amino acid (Method: %s)" % method]
                                    + reference_aa_freq_by_method[method])

            # Part 3: read counts
            tsv_writer.writerow([])
            tsv_writer.writerow(["Read Counts"])
            tsv_writer.writerow([])
            tsv_writer.writerow(["Number of Reads:", len(read_subset)])
            tsv_writer.writerow(["Position:"] + coords)
            tsv_writer.writerow(["Amino Acid:"] + list(reference.protein))
            for method in methods:
                tsv_writer.writerow(["Counts of variant amino acid (Method: %s)" % method]
                                    + list(counts[method]))

        # write files that can be used by data2bfactor.py to colour 3D structures
        for method in methods:
            with open(os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags) + "_data2b_%s.txt" % method),
                      "w") as d2bhandle:
                d2b_writer = csv.writer(d2bhandle, delimiter=" ")
                for coord in range(coords[0]):  # prefix with zeros to overwrite b-factors in pdb
                    d2b_writer.writerow([coord, 0])
                for coord, one_letter, freq in zip(coords, reference.protein, counts[method]/num_reads):
                    d2b_writer.writerow([coord, freq])
                for coord in range(coords[-1] + 1, 10000):  # suffix with zeros to overwrite b-factors in pdb
                    d2b_writer.writerow([coord, 0])

        # write a file that can be used by data2bfactor.py to colour 3D structures
        for method in refmethods:
            with open(os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags)
                      + "_data2b_theoretical_%s.txt" % method), "w") as d2bhandle:
                d2b_writer = csv.writer(d2bhandle, delimiter=" ")
                for coord in range(coords[0]):  # prefix with zeros to overwrite b-factors in pdb
                    d2b_writer.writerow([coord, 0])
                for coord, one_letter, freq in zip(coords, reference.protein, reference_aa_freq_by_method[method]):
                    d2b_writer.writerow([coord, freq])
                for coord in range(coords[-1]+1, 10000):  # suffix with zeros to overwrite b-factors in pdb
                    d2b_writer.writerow([coord, 0])

def dna_op_frequency(data, reportdir, database):
    """

    :param data:
    :param reportdir:
    :param database:
    :return:
    """
    references = database.get("references")
    reads = database.get("reads")

    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]
    bins = {(refid, tags): read_subset for refid, tags, read_subset in data}

    # group tagsets by refid
    refids = {refid: [] for refid, tags in bins}
    for refid, tags in bins:
        refids[refid].append(tuple(sorted(list(tags))))  # hashable

    # analyze each reference independently
    for refid, tagsets in refids.items():
        # create a dict of dicts so that op_frequencies[op][tags] yields a frequency
        reference = references[refid]
        op_frequencies = {op: {tags: 0 for tags in tagsets}
                          for r in reads for aln in r.alns for op in aln.transform if r.refid == refid}

        # add ops that are only found in cassettes
        for casaln in reference.cassettes_aln.values():
            for op in al.trim_transform(casaln.transform, len(reference.seq)):
                if op not in op_frequencies:
                    op_frequencies[op] = {tags: 0 for tags in tagsets}

        readcounts = []
        # populate the table
        for tags in tagsets:
            read_subset = bins[(refid, frozenset(tags))]
            readcounts.append(len(read_subset))
            for read in read_subset:
                for aln in read.alns:
                    for op in aln.transform:
                        op_frequencies[op][tags] += 1 / len(read.alns) / len(read_subset)

        # Output every single op and its frequency in every bin
        basename = os.path.join(reportdir, refid)
        tag_names = sorted(list({tag[0] for refid, tags in bins for tag in tags}))
        # orders the variants for output from most to least shared (between all the bins). Ops present in more bins are
        # earlier in the sequence.
        outputs = list(op_frequencies)
        outputs.sort(key=lambda op: len([x for x in op_frequencies[op].values() if x > 0]))
        outputs.reverse()

        dist_from_edge = []
        with open(basename + ".frequency.tsv", "w") as handle:
            tsv_writer = csv.writer(handle, delimiter="\t")
            for tagname in tag_names:
                tsv_writer.writerow([tagname, "", ""] + [dict(tags)[tagname] for tags in tagsets])
            tsv_writer.writerow([])
            tsv_writer.writerow(["Number of Reads", "", ""] + readcounts)
            tsv_writer.writerow([])
            tsv_writer.writerow(["Op (ie. SNP)", "Position", "Average Distance from Cassette Start", "Average Distance from Cassette End", "Frequency in Cassettes"])
            for op in outputs:
                freq_in_cassettes = 0
                distances_to_start = []
                distances_to_end = []
                for casaln in reference.cassettes_aln.values():
                    if op in casaln.transform:
                        freq_in_cassettes += 1 / len(reference.cassettes_aln)
                        distances_to_start.append(op[0] - casaln.start)
                        distances_to_end.append(casaln.end - op[0])

                # output empty columns if not in cassettes
                if distances_to_start and distances_to_end:
                    d_start = np.mean(distances_to_start)
                    d_end = np.mean(distances_to_end)
                    dist_from_edge.append([min(d_start, d_end),
                                           {tags: op_frequencies[op][tags]/freq_in_cassettes for tags in tagsets}
                                           ])
                else:
                    d_start = ""
                    d_end = ""

                tsv_writer.writerow([repr(op), op[0] + reference.offset, d_start, d_end, freq_in_cassettes]
                                    + [op_frequencies[op][tags] for tags in tagsets])

        with PdfPages(basename + ".relativefreq.pdf") as pdf:
            for tags in tagsets:
                fig, ax = plt.subplots(1, figsize=(8, 6), sharex=True)
                x = [x for x, y in dist_from_edge]
                y = [y[tags] for x, y in dist_from_edge]
                ax.plot(x, y, color="red", marker=".", linestyle="None", clip_on=False)
                plt.suptitle("%s" % ut.get_tagstring(refid=refid, tags=tags))
                ax.set_ylabel("Relative Abundance")
                ax.set_xlabel("Average distance to nearest cassette edge")
                plt.tight_layout()
                pdf.savefig()
                plt.close()

            # cropped to 100bp
            for tags in tagsets:
                fig, ax = plt.subplots(1, figsize=(8, 6), sharex=True)
                x = [x for x, y in dist_from_edge if x <= 100]
                y = [y[tags] for x, y in dist_from_edge if x <= 100]
                ax.plot(x, y, color="red", marker=".", linestyle="None", clip_on=False)
                plt.suptitle("%s (max 100bp from edge)" % ut.get_tagstring(refid=refid, tags=tags))
                ax.set_ylabel("Relative Abundance")
                ax.set_xlabel("Average distance to nearest cassette edge")
                plt.tight_layout()
                pdf.savefig()
                plt.close()


def aa_op_frequency(data, reportdir, database):
    """

        :param data:
        :param reportdir:
        :param database:
        :return:
        """
    references = database.get("references")
    reads = database.get("reads")

    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]
    bins = [(refid, tags) for refid, tags, read_subset in data]
    datadict = {(refid, tags): read_subset for refid, tags, read_subset in data}

    # group tagsets by refid
    refids = {refid: [] for refid, tags in bins}
    for refid, tags in bins:
        refids[refid].append(tuple(sorted(list(tags))))  # hashable

    # analyze each reference independently
    for refid, tagsets in refids.items():
        # create a dict of dicts so that aa_frequencies[(pos, aa)][tags] yields a frequency
        reference = references[refid]
        op_frequencies = {op: {tags: 0 for tags in tagsets}
                          for r in reads for aln in r.protein_aln["inframe_corrected_all"] for op in aln.transform
                          if r.refid == refid}

        protein_offset = math.ceil(reference.offset / 3) + 1  # ceil is for first full codon; +1 is for 1-based coords
        readcounts = []
        # populate the table
        for tags in tagsets:
            read_subset = datadict[(refid, frozenset(tags))]
            readcounts.append(len(read_subset))
            for read in read_subset:
                for aln in read.protein_aln["inframe_corrected_all"]:
                    for op in aln.transform:
                        op_frequencies[op][tags] += 1 / len(read.protein_aln["inframe_corrected_all"]) / len(read_subset)

        # create a dict of aa_frequencies for the silent cassettes
        theoretical_op_frequencies = {op: 0 for op in op_frequencies}
        for casaln in reference.cassettes_aln.values():
            # get all possible protein alignments from casaln; retain stop codons in case switching happens in a
            # cassette containing a stop codon in a different region.
            protein_alns = al.translate_mapping(mapping=casaln.transform, reference=reference, filternonsense=False)
            for protein_aln in protein_alns:
                for op in protein_aln.transform:
                    if not op in theoretical_op_frequencies:
                        theoretical_op_frequencies[op] = 0
                    if not op in op_frequencies:  # also adds to experimental, so that all possible ops are present
                        op_frequencies[op] = {tags: 0 for tags in tagsets}  # adds also to
                    theoretical_op_frequencies[op] += 1 / len(reference.cassettes_aln) / len(protein_alns)

        # Output every single op and its frequency in every bin
        basename = os.path.join(reportdir, refid)
        tag_names = sorted(list({tag[0] for refid, tags in bins for tag in tags}))
        # orders the variants for output from most to least shared (between all the bins). Ops present in more bins are
        # earlier in the sequence.
        outputs = list(op_frequencies)
        outputs.sort(key=lambda op: len([x for x in op_frequencies[op].values() if x > 0]))
        outputs.reverse()
        with open(basename + ".frequency.tsv", "w") as handle:
            tsv_writer = csv.writer(handle, delimiter="\t")
            for tagname in tag_names:
                tsv_writer.writerow([tagname] + [""]*4 + [dict(tags)[tagname] for tags in tagsets])
            tsv_writer.writerow([])
            tsv_writer.writerow(["Number of Reads"] + [""]*4 + readcounts)
            tsv_writer.writerow([])
            tsv_writer.writerow(["Position", "Reference", "Op (ie. SNP)", "Shorthand", "Frequency in Cassettes"])
            for op in outputs:
                mutstring = str(reference.protein[op[0]]) + str(op[0] + protein_offset)
                if op[1] == "D":
                    mutstring += "d" + str(op[2])
                elif op[1] == "S":
                    mutstring += op[2]
                elif op[1] == "I":
                    mutstring += "^" + op[2]
                tsv_writer.writerow([protein_offset + op[0], reference.protein[op[0]], repr(op), mutstring,
                                     theoretical_op_frequencies[op]] + [op_frequencies[op][tags] for tags in tagsets])


def switch_boundary_g_run_coincidence(data, reportdir, database):
    """

    :param data:
    :param reportdir:
    :param database:
    :return:
    """
    references = database.get("references")
    switches = database.get("switches")

    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]
    bins = [(refid, tags) for refid, tags, read_subset in data]
    datadict = {(refid, tags): read_subset for refid, tags, read_subset in data}

    # group tagsets by refid
    refids = {refid: [] for refid, tags in bins}
    for refid, tags in bins:
        refids[refid].append(tuple(sorted(list(tags))))  # hashable


    # analyze each reference independently
    for refid, tagsets in refids.items():
        reference = references[refid]

        # Output every single op and its frequency in every bin
        basename = os.path.join(reportdir, refid)
        tag_names = sorted(list({tag[0] for refid, tags in bins for tag in tags}))
        with open(basename + ".tsv", "w") as handle:
            tsv_writer = csv.writer(handle, delimiter="\t")
            # Simulation Data
            tsv_writer.writerow(["Simulated switch length",
                                 ">G3, left (Mean)", ">G3, left (SD)", ">G3, left (N)",
                                 ">G3, right (Mean)", ">G3, right (SD)", ">G3, right (N)",
                                 ">G4, left (Mean)", ">G4, left (SD)", ">G4, left (N)",
                                 ">G4, right (Mean)", ">G4, right (SD)", ">G4, right (N)",
                                 ">G5, left (Mean)", ">G5, left (SD)", ">G5, left (N)",
                                 ">G5, right (Mean)", ">G5, right (SD)", ">G5, right (N)",
                                 ">G6, left (Mean)", ">G6, left (SD)", ">G6, left (N)",
                                 ">G6, right (Mean)", ">G6, right (SD)", ">G6, right (N)"
                                 ])
            for switch_length in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
                row = [switch_length]
                for g_len in (3, 4, 5, 6):
                    sim = si.simulate_switch_endpoints(reference, 1000000, switch_length, g_len)
                    row.extend([sim["left"]["mean"], sim["left"]["std"], sim["left"]["nobs"]])
                    row.extend([sim["right"]["mean"], sim["right"]["std"], sim["right"]["nobs"]])
                tsv_writer.writerow(row)

            tsv_writer.writerow([])
            # Actual Data
            tsv_writer.writerow(tag_names +
                                ["Number of Reads",
                                 ">G3, left (Mean)", ">G3, left (SD)", ">G3, left (N)",
                                 ">G3, right (Mean)", ">G3, right (SD)", ">G3, right (N)",
                                 ">G4, left (Mean)", ">G4, left (SD)", ">G4, left (N)",
                                 ">G4, right (Mean)", ">G4, right (SD)", ">G4, right (N)",
                                 ">G5, left (Mean)", ">G5, left (SD)", ">G5, left (N)",
                                 ">G5, right (Mean)", ">G5, right (SD)", ">G5, right (N)",
                                 ">G6, left (Mean)", ">G6, left (SD)", ">G6, left (N)",
                                 ">G6, right (Mean)", ">G6, right (SD)", ">G6, right (N)"
                                 ])
            for tags in tagsets:
                readset = datadict[(refid, frozenset(tags))]
                row = [dict(tags)[tag_name] for tag_name in tag_names]  # tag values
                row.append(len(readset))  # read count
                for g_len in (3, 4, 5, 6):
                    weights = []
                    left_array = []
                    right_array = []
                    for read in readset:
                        for aln in read.alns:
                            templated_tf = al.templated(aln.transform, reference)
                            if (templated_tf, refid) in switches:
                                switch_sets = switches[(templated_tf, refid)]
                                for ss in switch_sets:
                                    for switch in ss:
                                        if switch[0] > 0:
                                            start_range = [templated_tf[switch[0]-1][0], templated_tf[switch[0]][0]]
                                        else:  # no previous snps; start from beginning of read
                                            start_range = [None, templated_tf[switch[0]][0]]
                                        if switch[1] < len(templated_tf):
                                            stop_range = [templated_tf[switch[1]-1][0], templated_tf[switch[1]][0]]
                                        else:  # no following snps; go to end of read
                                            stop_range = [templated_tf[switch[1]-1][0], None]

                                        weights.append(1 / len(switch_sets) / len(read.alns))
                                        if "G"*g_len in reference.seq[start_range[0]:start_range[1]]:
                                            left_array.append(1)
                                        else:
                                            left_array.append(0)
                                        if "G"*g_len in reference.seq[stop_range[0]:stop_range[1]]:
                                            right_array.append(1)
                                        else:
                                            right_array.append(0)
                    weighted_left = DescrStatsW(data=left_array, weights=weights)
                    weighted_right = DescrStatsW(data=right_array, weights=weights)
                    row.extend([weighted_left.mean, weighted_left.std, weighted_left.nobs])
                    row.extend([weighted_right.mean, weighted_right.std, weighted_right.nobs])
                tsv_writer.writerow(row)


def switch_boundaries(data, reportdir, database, unambiguous_switches_only=False, window_size=50,
                      minimum_g_run_length=4):
    """

    :param data:
    :param reportdir:
    :param database:
    :param unambiguous_switches_only: If True, only uses switches that are unambiguously located.
    :return:
    """
    references = database.get("references")
    switches = database.get("switches")

    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]
    bins = [(refid, tags) for refid, tags, read_subset in data]
    datadict = {(refid, tags): read_subset for refid, tags, read_subset in data}

    # group tagsets by refid
    refids = {refid: [] for refid, tags in bins}
    for refid, tags in bins:
        refids[refid].append(tuple(sorted(list(tags))))  # hashable


    # analyze each reference independently
    for refid, tagsets in refids.items():
        reference = references[refid]

        # Output every single op and its frequency in every bin
        basename = os.path.join(reportdir, refid)
        tag_names = sorted(list({tag[0] for refid, tags in bins for tag in tags}))

        # trim cassette transforms
        cassettes = {casname: casaln for casname, casaln in reference.cassettes_aln.items()}
        for casname, casaln in cassettes.items():
            casaln.transform = al.trim_transform(casaln.transform, len(reference.seq))

        # build array of zeros
        data = {tags: {} for tags in tagsets}
        for casname, casaln in cassettes.items():
            # First window
            for windows in data.values():
                # first window
                windows[(casname, casaln.start, casaln.transform[0][0])] = {'left': 0, 'right': 0}
                # middle windows
                for x, op in enumerate(casaln.transform[:-1]):
                    next_op = casaln.transform[x+1]
                    windows[(casname, op[0], next_op[0])] = {'left': 0, 'right': 0}
                # last window
                windows[(casname, casaln.transform[-1][0], casaln.end)] = {'left': 0, 'right': 0}

        # populate array from data by counting the frequency of boundary usage
        for tags in tagsets:
            readset = datadict[(refid, frozenset(tags))]
            for read in readset:
                for aln in read.alns:
                    templated_tf = al.templated(aln.transform, reference)
                    if (templated_tf, refid) in switches:
                        switch_sets = switches[(templated_tf, refid)]
                        if isinstance(switch_sets, int):
                            continue
                        for ss in switch_sets:
                            for switch in ss:
                                is_unambiguous = all(switch in ss for ss in switch_sets)
                                for source in switch[2]:
                                    cas_tf = cassettes[source].transform
                                    # left side
                                    left_op = templated_tf[switch[0]]
                                    left_boundary = left_op[0]
                                    max_left_boundary_idx = cas_tf.index(left_op)
                                    if max_left_boundary_idx > 0:
                                        max_left_boundary = cas_tf[cas_tf.index(left_op) - 1][0]
                                    else:
                                        max_left_boundary = cassettes[source].start
                                    if not unambiguous_switches_only or is_unambiguous:
                                        data[tags][(source, max_left_boundary, left_boundary)]['left'] \
                                            += 1 / len(switch_sets) / len(read.alns)
                                    # right side
                                    right_op = templated_tf[switch[1] - 1]
                                    right_boundary = right_op[0]
                                    max_right_boundary_idx = cas_tf.index(right_op)
                                    if max_right_boundary_idx < len(cas_tf) - 1:
                                        max_right_boundary = cas_tf[cas_tf.index(right_op) + 1][0]
                                    else:
                                        max_right_boundary = cassettes[source].end
                                    if not unambiguous_switches_only or is_unambiguous:
                                        data[tags][(source, right_boundary, max_right_boundary)]['right']\
                                            += 1 / len(switch_sets) / len(read.alns)

        # output file
        all_cassette_regions = list({c_reg for subset in data.values() for c_reg in subset})
        all_cassette_regions.sort(key=lambda r: r[1])
        all_cassette_regions.sort(key=lambda r: r[0])

        # convolve G-run density
        g_run_densities = {}
        for casname, casaln in cassettes.items():
            g_run_density = []
            current_length = 0
            for b in reference.seq:
                if b == "G":
                    current_length += 1
                else:
                    if current_length > 0:
                        if current_length >= minimum_g_run_length:
                            g_run_density.extend([1/current_length] * current_length)
                        else:
                            g_run_density.extend([0] * current_length)
                        current_length = 0
                    g_run_density.append(0)
            g_run_densities[casname] = list(ndimage.filters.convolve(g_run_density, [1/window_size] * window_size))

        # convolve sequence identity
        sequence_identities = {}
        for casname, casaln in cassettes.items():
            sequence_identity = []
            for x in range(len(reference.seq)):
                snps_at_base = al.count_snps([op for op in casaln.transform if op[0] == x])
                sequence_identity.append(1 - float(snps_at_base))
            # set sequence identity at ends of cassettes to zero
            for x in range(casaln.start):
                sequence_identity[x] = 0
            for x in range(casaln.end, len(reference.seq)):
                sequence_identity[x] = 0
            sequence_identities[casname] = ndimage.filters.convolve(sequence_identity, [1/window_size] * window_size)

        with open(basename + ".all_boundaries.tsv", "w") as handle:
            tsv_writer = csv.writer(handle, delimiter="\t")
            for tagname in tag_names:
                tsv_writer.writerow([tagname] + [""]*5 + [dict(tags)[tagname] for tags in tagsets for _ in range(3)])
            tsv_writer.writerow([])
            readcounts = [len(datadict[(refid, frozenset(tags))]) for tags in tagsets]
            tsv_writer.writerow(["Number of Reads"] + [""]*5 + [str(r) for r in readcounts for _ in range(3)])
            tsv_writer.writerow([])

            header = ["Originating Cassette", "Start", "End", "Size",
                      "G run density (G-runs per bp, %d bp smoothing)" % window_size,
                      "Sequence Identity (%d bp smoothing)" % window_size]
            header.extend(["Left", "Right", "All"] * len(tagsets))
            tsv_writer.writerow(header)

            for source, start, stop in all_cassette_regions:
                row = [source, start, stop, stop-start]
                if start == stop:
                    nearest_stop = stop + 1
                else:
                    nearest_stop = stop
                row.append(np.mean(g_run_densities[source][start:nearest_stop]))
                row.append(np.mean(sequence_identities[source][start:nearest_stop]))

                # count data
                for tags in tagsets:
                    left = data[tags][(source, start, stop)]["left"]
                    right = data[tags][(source, start, stop)]["right"]
                    row.extend([left, right, left + right])
                tsv_writer.writerow(row)


def cassette_sequence_identity(data, reportdir, database):
    """

    :param data:
    :param reportdir:
    :param database:
    :return:
    """
    references = database.get("references")

    for refid, reference in references.items():
        if reference.cassettes_aln:
            basename = os.path.join(reportdir, refid)
            seq_ids = [reference.name]
            seq_transforms = {tuple(): len(reference.seq)}
            for casname, casaln in sorted(list(reference.cassettes_aln.items()), key=lambda x:x[0]):
                seq_ids.append(casname)
                seq_transforms[al.trim_transform(casaln.transform, len(reference.seq))] = casaln.end - casaln.start

            # method 1: map distance
            data = [[1 - al.map_distance(tf1, tf2) / min((size1, size2))
                     for tf1, size1 in seq_transforms.items()]
                    for tf2, size2 in seq_transforms.items()]
            with open(basename + ".map_distance.tsv", "w") as handle:
                tsv_writer = csv.writer(handle, delimiter="\t")
                tsv_writer.writerow([''] + seq_ids)
                for seq_id, row in zip(seq_ids, data):
                    tsv_writer.writerow([seq_id] + row)

            # method 2: exclusive ops
            data = [[1 - len(set(tf1) ^ set(tf2)) / min((size1, size2))
                     for tf1, size1 in seq_transforms.items()]
                    for tf2, size2 in seq_transforms.items()]
            with open(basename + ".exclusive_ops.tsv", "w") as handle:
                tsv_writer = csv.writer(handle, delimiter="\t")
                tsv_writer.writerow([''] + seq_ids)
                for seq_id, row in zip(seq_ids, data):
                    tsv_writer.writerow([seq_id] + row)


def switching_and_nontemplated(data, reportdir, database, bootstrap=10, changetypes="IDS", minimum_switch_length=0,
                               unambiguous_switches_only=False, nonoverlapping_switches_only=False):
    """

        :param data:
        :param reportdir:
        :param database:
        :param changetypes: The types of nontemplated mutations to look for. "IDS" is all, but subsets will also work.
        :param unambiguous_switches_only: If True, only uses switches that are unambiguously located.
        :return:
        """
    switches = database.get("switches")
    references = database.get("references")

    max_switches = max(len(s) for ss in switches.values() for s in ss)
    if not switches:
        raise ValueError("switches.p is empty. Run \"vls label_switches\" to calculate switches "
                         "before exporting report.")

    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]

    for refid, tags, read_subset in data:
        reference = references[refid]
        base_report_path = os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags))
        # count non-templated SNPs for reads with 0, 1, 2... switch events
        nt_snps_per_read = [{optype: [] for optype in "IDS"} for _ in range(max_switches + 1)]
        weights = [[] for _ in range(max_switches + 1)]
        for read in read_subset:
            for aln in read.alns:
                templated_tf = al.templated(aln.transform, reference)
                nontemplated_tf = al.nontemplated(aln.transform, reference)
                num_switches = len(switches[(templated_tf, reference.name)][0])
                for optype in "IDS":
                    op_count = len([op for op in nontemplated_tf if op[1] == optype])
                    nt_snps_per_read[num_switches][optype].append(op_count)
                weights[num_switches].append(1/len(read.alns))

        # output report
        with open(base_report_path + ".tsv", 'w') as handle:
            tsv_writer = csv.writer(handle, delimiter="\t")
            tsv_writer.writerow(["Number of switches", "Number of reads",
                                 "Non-templated insertions per read (mean)", "Non-templated insertions (SD)",
                                 "Non-templated deletions (mean)", "Non-templated deletions (SD)",
                                 "Non-templated substitutions (mean)", "Non-templated substitutions (SD)",
                                 "Non-templated SNPs (mean)", "Non-templated SNPs (SD)"])
            for num_switches in range(max_switches + 1):
                row = [num_switches, sum(weights[num_switches])]
                if weights[num_switches]:
                    for optype in "IDS":
                        op_counts = nt_snps_per_read[num_switches][optype]
                        opstats = DescrStatsW(data=op_counts, weights=weights[num_switches])
                        row.extend([opstats.mean, opstats.std])
                    op_counts = [sum(ops) for ops in zip(nt_snps_per_read[num_switches]["I"],
                                                         nt_snps_per_read[num_switches]["D"],
                                                         nt_snps_per_read[num_switches]["S"])]
                    opstats = DescrStatsW(data=op_counts, weights=weights[num_switches])
                    row.extend([opstats.mean, opstats.std])
                tsv_writer.writerow(row)
        # assemble left and right boundary uncertainty regions (BURs) around each SNP in the cassettes
        burs = al.get_burs(ref_seq=reference.seq, ref_cassettes_aln=reference.cassettes_aln)

        # rather than computing all possible switch combinations and source cassettes (complex), this randomly chooses
        # a switch set as well as a source cassette for switches with multiple possible sources.
        result_types = ("leftmost_exterior", "left_boundaries", "interior", "right_boundaries",
                        "rightmost_exterior", "exterior", "interior_and_boundaries")
        bootstrapping_results = {result: tuple({"mean": [], "std": [], "nobs": []}
                                               for _ in range(max_switches + 1))
                                 for result in result_types}
        all_switches_bootstrapping_results = {result: {"mean": [], "std": [], "nobs": []} for result in result_types}

        #   EXAMPLE:
        #   bootstrapping_results["rightmost_unswitched"][num_switches]["mean"] = list of means
        #   bootstrapping_results["rightmost_unswitched"][num_switches]["std"] = list of  of standard deviations
        #   bootstrapping_results["rightmost_unswitched"][num_switches]["nobs"] = list of sample sizes

        for i in range(bootstrap):
            ut.tprint("Bootstrapping %d of %d" % (i + 1, bootstrap), ontop=True)
            with open(base_report_path + ".boundary_changes.bootstrap%04d.tsv" % i, 'w') as handle:
                event_writer = csv.writer(handle, delimiter="\t")
                event_writer.writerow(
                    ["Read", "Alignment", "Number of Alignments", "Number of switches on Read", "Switch Start",
                     "Switch End", "Boundary", "Sequence", "Post-mutation Sequence"])


                # count non-templated SNPs by number of switches and location relative to switch.
                results = {result: tuple(([], []) for _ in range(max_switches + 1)) for result in result_types}

                for read in read_subset:
                    if unambiguous_switches_only:
                        # check if all alignments give the same templated alignment
                        if not len({al.templated(aln.transform, reference) for aln in read.alns}) == 1:
                            continue
                    for aln_number, aln in enumerate(read.alns, 1):
                        templated_tf = al.templated(aln.transform, reference)
                        nontemplated_tf = al.nontemplated(aln.transform, reference)
                        # filter nontemplated_tf for optypes as specified in parameters
                        nontemplated_tf = tuple(op for op in nontemplated_tf if op[1] in changetypes)
                        switch_sets = switches[(templated_tf, reference.name)]
                        if isinstance(switch_sets, int):
                            continue
                        if unambiguous_switches_only and len(switch_sets) > 1:
                            continue
                        switch_set = random.choice(switch_sets)
                        num_switches = len(switch_set)
                        weight = 1 / len(read.alns)

                        # filter out reads with overlapping switches
                        if nonoverlapping_switches_only:
                            continue_flag = False
                            for x, (start_idx, stop_idx, sources) in enumerate(switch_set[:-1]):
                                stop_op = templated_tf[stop_idx - 1]
                                max_right_pos = max([burs.right[source][stop_op] for source in sources])
                                next_start_idx, next_stop_idx, next_sources = switch_set[x+1]
                                next_start_op = templated_tf[next_start_idx]
                                max_left_pos = min([burs.left[source][next_start_op] for source in next_sources])
                                if not max_right_pos < max_left_pos:
                                    continue_flag = True
                            if continue_flag:
                                continue

                        sampled_switches = []
                        for start_idx, stop_idx, sources in switch_set:
                            ev = {}
                            source = random.sample(sources, 1)[0]
                            start_op = templated_tf[start_idx]
                            stop_op = templated_tf[stop_idx - 1]
                            ev["start"] = start_op[0]
                            ev["stop"] = stop_op[0]
                            ev["left"] = burs.left[source][start_op]
                            ev["right"] = burs.right[source][stop_op]
                            sampled_switches.append(ev)
                        if not all(ev["stop"] - ev["start"] >= minimum_switch_length for ev in sampled_switches):
                            continue
                        # reads with no switches
                        if num_switches == 0:
                            nontemplated_snps = len(nontemplated_tf)
                            length = len(reference.seq)
                            frequency = nontemplated_snps / length
                            results["exterior"][0][0].append(frequency)
                            results["exterior"][0][1].append(weight)
                            event_writer.writerow([read.name, aln_number, len(read.alns), num_switches, "", "", "",
                                                   reference.seq, read.seq])
                        # reads with switches
                        for x, ev in enumerate(sampled_switches):
                            # Interior
                            nontemplated_snps = len([op for op in nontemplated_tf
                                                     if ev["start"] <= op[0] <= ev["stop"]])
                            length = ev["stop"] - ev["start"] + 1
                            if length > 0:
                                frequency = nontemplated_snps / length
                                results["interior"][num_switches][0].append(frequency)
                                results["interior"][num_switches][1].append(weight)
                                event_writer.writerow([read.name, aln_number, len(read.alns), num_switches, ev["start"],
                                                       ev["stop"], "Interior", reference.seq[ev["start"]:ev["stop"]+1],
                                                       al.transform(reference=reference.seq, mapping=nontemplated_tf,
                                                                    start=ev["start"], stop=ev["stop"]+1)])
                            # Left Boundary Uncertainty Region
                            nontemplated_snps = len([op for op in nontemplated_tf if ev["left"] <= op[0] < ev["start"]])
                            length = ev["start"] - ev["left"]
                            if length > 0:
                                frequency = nontemplated_snps/length
                                results["left_boundaries"][num_switches][0].append(frequency)
                                results["left_boundaries"][num_switches][1].append(weight)
                                event_writer.writerow([read.name, aln_number, len(read.alns), num_switches, ev["start"],
                                                       ev["stop"], "Left BUR", reference.seq[ev["start"]:ev["stop"] + 1],
                                                       al.transform(reference=reference.seq, mapping=nontemplated_tf,
                                                                    start=ev["start"], stop=ev["stop"] + 1)])
                            # Right Boundary Uncertainty Region
                            nontemplated_snps = len([op for op in nontemplated_tf if ev["stop"] < op[0] <= ev["right"]])
                            length = ev["right"] - ev["stop"]
                            if length > 0:
                                frequency = nontemplated_snps / length
                                results["right_boundaries"][num_switches][0].append(frequency)
                                results["right_boundaries"][num_switches][1].append(weight)
                                event_writer.writerow([read.name, aln_number, len(read.alns), num_switches, ev["start"],
                                                       ev["stop"], "Right BUR", reference.seq[ev["start"]:ev["stop"] + 1],
                                                       al.transform(reference=reference.seq, mapping=nontemplated_tf,
                                                                    start=ev["start"], stop=ev["stop"] + 1)])
                            # Interior and boundaries
                            nontemplated_snps = len([op for op in nontemplated_tf if ev["left"] <= op[0] <= ev["right"]])
                            length = ev["right"] - ev["left"] + 1
                            if length > 0:
                                frequency = nontemplated_snps / length
                                results["interior_and_boundaries"][num_switches][0].append(frequency)
                                results["interior_and_boundaries"][num_switches][1].append(weight)

                            # Exterior (between switches only)
                            if x > 0:
                                last_ev = sampled_switches[x-1]
                                if last_ev["right"] < ev["left"]:
                                    nontemplated_snps = len([op for op in nontemplated_tf
                                                             if last_ev["left"] < op[0] < ev["right"]])
                                    length = ev["right"] - last_ev["left"] - 1
                                    if length > 0:
                                        frequency = nontemplated_snps / length
                                        results["exterior"][num_switches][0].append(frequency)
                                        results["exterior"][num_switches][1].append(weight)
                                        event_writer.writerow(
                                            [read.name, aln_number, len(read.alns), num_switches, ev["start"],
                                             ev["stop"], "Exterior", reference.seq[ev["start"]:ev["stop"] + 1],
                                             al.transform(reference=reference.seq, mapping=nontemplated_tf,
                                                          start=ev["start"], stop=ev["stop"] + 1)])
                            # First switch: do leftmost exterior region
                            if x == 0:
                                nontemplated_snps = len([op for op in nontemplated_tf if 0 <= op[0] < ev["left"]])
                                length = ev["left"]
                                if length > 0:
                                    frequency = nontemplated_snps / length
                                    results["leftmost_exterior"][num_switches][0].append(frequency)
                                    results["leftmost_exterior"][num_switches][1].append(weight)
                                    results["exterior"][num_switches][0].append(frequency)
                                    results["exterior"][num_switches][1].append(weight)
                                    event_writer.writerow([read.name, aln_number, len(read.alns), num_switches, ev["start"],
                                                           ev["stop"], "Exterior",
                                                           reference.seq[ev["start"]:ev["stop"] + 1],
                                                           al.transform(reference=reference.seq, mapping=nontemplated_tf,
                                                                        start=ev["start"], stop=ev["stop"] + 1)])
                            # Last switch: do rightmost_exterior region
                            if x == len(sampled_switches) - 1:
                                nontemplated_snps = len([op for op in nontemplated_tf
                                                         if ev["right"] < op[0] <= len(reference.seq)])
                                length = len(reference.seq) - ev["right"]
                                if length > 0:
                                    frequency = nontemplated_snps / length
                                    results["rightmost_exterior"][num_switches][0].append(frequency)
                                    results["rightmost_exterior"][num_switches][1].append(weight)
                                    results["exterior"][num_switches][0].append(frequency)
                                    results["exterior"][num_switches][1].append(weight)
                                    event_writer.writerow([read.name, aln_number, len(read.alns), num_switches, ev["start"],
                                                           ev["stop"], "Exterior",
                                                           reference.seq[ev["start"]:ev["stop"] + 1],
                                                           al.transform(reference=reference.seq, mapping=nontemplated_tf,
                                                                        start=ev["start"], stop=ev["stop"] + 1)])

            # calculate stats for bootstrapping results
            for result_type, by_num_switches in results.items():
                all_freqs = []
                all_weights = []
                for num_switches, (freqs, weights) in enumerate(by_num_switches):
                    # combine data for "Any" category
                    all_freqs.extend(freqs)
                    all_weights.extend(weights)
                    # add to bootstrap results
                    if freqs:
                        stats = DescrStatsW(data=freqs, weights=weights)
                        bootstrapping_results[result_type][num_switches]["mean"].append(stats.mean)
                        bootstrapping_results[result_type][num_switches]["std"].append(stats.std)
                        bootstrapping_results[result_type][num_switches]["nobs"].append(stats.nobs)
                if all_freqs:
                    all_stats = DescrStatsW(data=all_freqs, weights=all_weights)
                    all_switches_bootstrapping_results[result_type]["mean"].append(all_stats.mean)
                    all_switches_bootstrapping_results[result_type]["std"].append(all_stats.std)
                    all_switches_bootstrapping_results[result_type]["nobs"].append(all_stats.nobs)

        with open(base_report_path + ".boundaries.tsv", 'w') as handle:
            tsv_writer = csv.writer(handle, delimiter="\t")
            tsv_writer.writerow([None, "Leftmost exterior", None, None, "Left BURs", None, None,
                                 "Interior", None, None, "Right BURs", None, None, "Rightmost exterior", None, None,
                                 "All unswitched", None, None, "Interior + Boundaries", None, None])
            tsv_writer.writerow(["Number of switches"] + ["Mean", "SD", "N"]*6)
            for num_switches in range(max_switches + 1):
                row = [num_switches]
                for result_type in result_types:
                    if bootstrapping_results[result_type][num_switches]["mean"]:
                        mean = np.mean(bootstrapping_results[result_type][num_switches]["mean"])
                        std = np.mean(bootstrapping_results[result_type][num_switches]["std"])
                        nobs = np.mean(bootstrapping_results[result_type][num_switches]["nobs"])
                        row.extend([mean, std, nobs])
                    else:
                        row.extend([None, None, None])
                tsv_writer.writerow(row)

            # "Any" category
            row = ["Any"]
            for result_type in result_types:
                if all_switches_bootstrapping_results[result_type]["mean"]:
                    mean = np.mean(all_switches_bootstrapping_results[result_type]["mean"])
                    std = np.mean(all_switches_bootstrapping_results[result_type]["std"])
                    nobs = np.mean(all_switches_bootstrapping_results[result_type]["nobs"])
                    row.extend([mean, std, nobs])
                else:
                    row.extend([None, None, None])
            tsv_writer.writerow(row)


def switching_and_slippage(data, reportdir, database, bootstrap=100, minimum_switch_length=0,
                           unambiguous_switches_only=False, nonoverlapping_switches_only=False):
    """

    :param data:
    :param reportdir:
    :param database:
    :param unambiguous_switches_only: If True, only uses switches that are unambiguously located.
    :return:
    """
    switches = database.get("switches")
    slips = database.get("slips")
    references = database.get("references")
    max_switches = max(len(s) for ss in switches.values() for s in ss)
    if not switches:
        raise ValueError("switches.p is empty. Run \"vls label_switches\" to calculate switches "
                         "before exporting report.")
    if not slips:
        raise ValueError("slips.p not found. Run \"vls label_slippage\" to calculate polymerase slippage "
                         "before exporting report.")

    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]
    for refid, tags, read_subset in data:
        reference = references[refid]
        base_report_path = os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags))
        # count slips for reads with 0, 1, 2... switch events
        slips_per_read = [[] for _ in range(max_switches + 1)]
        paired_switch_slips = []
        for read in read_subset:
            best_slips, best_aln = al.get_slips(read, slips, reference)
            templated_tf = al.templated(best_aln.transform, reference)
            num_switches = len(switches[(templated_tf, reference.name)][0])
            slips_per_read[num_switches].append(len(best_slips))
            paired_switch_slips.append((read.name, num_switches, len(best_slips)))


        # output report
        with open(base_report_path + ".tsv", 'w') as handle:
            tsv_writer = csv.writer(handle, delimiter="\t")
            tsv_writer.writerow(["Number of switches", "Number of reads",
                                 "Slips per read (mean)", "Slips per read (SD)"])
            for num_switches in range(max_switches + 1):
                data = slips_per_read[num_switches]
                if data:
                    row = [num_switches, len(data), np.mean(data), np.std(data)]
                else:
                    row = [num_switches, len(data)]
                tsv_writer.writerow(row)

        # Paired data report
        with open(base_report_path + ".paired.tsv", "w") as handle:
            tsv_writer = csv.writer(handle, delimiter="\t")
            tsv_writer.writerow(["Read name", "Switch Count", "Slip Count"])
            for rname, swcount, slcount in paired_switch_slips:
                tsv_writer.writerow([rname, swcount, slcount])

        # assemble left and right boundary uncertainty regions (BURs) around each SNP in the cassettes
        burs = al.get_burs(reference.seq, reference.cassettes_aln)

        # rather than computing all possible switch combinations and source cassettes (complex), this randomly chooses
        # a switch set as well as a source cassette for switches with multiple possible sources.
        result_types = ("leftmost_exterior", "left_boundaries", "interior", "right_boundaries",
                        "rightmost_exterior", "exterior", "interior_and_boundaries")
        bootstrapping_results = {result: tuple({"mean": [], "std": [], "nobs": []}
                                               for _ in range(max_switches + 1))
                                 for result in result_types}
        all_switches_bootstrapping_results = {result: {"mean": [], "std": [], "nobs": []} for result in result_types}

        #   EXAMPLE:
        #   bootstrapping_results["rightmost_unswitched"][num_switches]["mean"] = list of means
        #   bootstrapping_results["rightmost_unswitched"][num_switches]["std"] = list of  of standard deviations
        #   bootstrapping_results["rightmost_unswitched"][num_switches]["nobs"] = list of sample sizes

        for i in range(bootstrap):
            ut.tprint("Bootstrapping %d of %d" % (i + 1, bootstrap), ontop=True)
            # count non-templated SNPs by number of switches and location relative to switch.
            results = {result: tuple([] for _ in range(max_switches + 1)) for result in result_types}

            for read in read_subset:
                if unambiguous_switches_only:
                    # check if all alignments give the same templated alignment
                    if not len({al.templated(aln.transform, reference) for aln in read.alns}) == 1:
                        continue

                best_slips, best_aln = al.get_slips(read, slips, reference)
                templated_tf = al.templated(best_aln.transform, reference)
                nontemplated_tf = al.nontemplated(best_aln.transform, reference)
                switch_sets = switches[(templated_tf, reference.name)]
                if isinstance(switch_sets, int):
                    continue
                if unambiguous_switches_only and len(switch_sets) > 1:
                    continue
                switch_set = random.choice(switch_sets)
                num_switches = len(switch_set)
                if nonoverlapping_switches_only:
                    continue_flag = False
                    for x, (start_idx, stop_idx, sources) in enumerate(switch_set[:-1]):
                        stop_op = templated_tf[stop_idx - 1]
                        max_right_pos = max([burs.right[source][stop_op] for source in sources])
                        next_start_idx, next_stop_idx, next_sources = switch_set[x+1]
                        next_start_op = templated_tf[next_start_idx]
                        max_left_pos = min([burs.left[source][next_start_op] for source in next_sources])
                        if not max_right_pos < max_left_pos:
                            continue_flag = True
                    if continue_flag:
                        continue

                sampled_switches = []
                for start_idx, stop_idx, sources in switch_set:
                    ev = {}
                    source = random.sample(sources, 1)[0]
                    start_op = templated_tf[start_idx]
                    stop_op = templated_tf[stop_idx - 1]
                    ev["start"] = start_op[0]
                    ev["stop"] = stop_op[0]
                    ev["left"] = burs.left[source][start_op]
                    ev["right"] = burs.right[source][stop_op]
                    sampled_switches.append(ev)
                if not all(ev["stop"] - ev["start"] >= minimum_switch_length for ev in sampled_switches):
                    continue
                # reads with no switches
                if num_switches == 0:
                    frequency = len(best_slips) / len(reference.seq)
                    results["exterior"][0].append(frequency)
                # reads with switches
                for x, ev in enumerate(sampled_switches):
                    # Interior
                    rslips = len([slip for slip in best_slips if ev["start"] <= slip[2] + 0.5 * slip[3] <= ev["stop"]])
                    length = ev["stop"] - ev["start"] + 1
                    if length > 0:
                        frequency = rslips / length
                        results["interior"][num_switches].append(frequency)
                    # Left Boundary Uncertainty Region
                    rslips = len([slip for slip in best_slips if ev["left"] <= slip[2] + 0.5 * slip[3] < ev["start"]])
                    length = ev["start"] - ev["left"]
                    frequency = rslips/length
                    results["left_boundaries"][num_switches].append(frequency)
                    # Right Boundary Uncertainty Region
                    rslips = len([slip for slip in best_slips if ev["stop"] < slip[2] + 0.5 * slip[3] <= ev["right"]])
                    length = ev["right"] - ev["stop"]
                    frequency = rslips / length
                    results["right_boundaries"][num_switches].append(frequency)
                    # Interior and boundaries
                    rslips = len([slip for slip in best_slips if ev["left"] <= slip[2] + 0.5 * slip[3]  <= ev["right"]])
                    length = ev["right"] - ev["left"] + 1
                    frequency = rslips / length
                    results["interior_and_boundaries"][num_switches].append(frequency)

                    # Exterior (between switches only)
                    if x > 0:
                        last_ev = sampled_switches[x-1]
                        if last_ev["right"] < ev["left"]:
                            rslips = len([slip for slip in best_slips
                                          if last_ev["left"] < slip[2] + 0.5 * slip[3] < ev["right"]])
                            length = ev["right"] - last_ev["left"] - 1
                            frequency = rslips / length
                            results["exterior"][num_switches].append(frequency)
                    # First switch: do leftmost exterior region
                    if x == 0:
                        rslips = len([slip for slip in best_slips if 0 <= slip[2] + 0.5 * slip[3] < ev["left"]])
                        length = ev["left"]
                        frequency = rslips / length
                        results["leftmost_exterior"][num_switches].append(frequency)
                        results["exterior"][num_switches].append(frequency)
                    # Last switch: do rightmost_exterior region
                    if x == len(sampled_switches) - 1:
                        rslips = len([slip for slip in best_slips
                                                 if ev["right"] < slip[2] + 0.5 * slip[3] <= len(reference.seq)])
                        length = len(reference.seq) - ev["right"]
                        frequency = rslips / length
                        results["rightmost_exterior"][num_switches].append(frequency)
                        results["exterior"][num_switches].append(frequency)

            # calculate stats for bootstrapping results
            for result_type, by_num_switches in results.items():
                all_freqs = []
                for num_switches, freqs in enumerate(by_num_switches):
                    # combine data for "Any" category
                    all_freqs.extend(freqs)
                    # add to bootstrap results
                    if freqs:
                        bootstrapping_results[result_type][num_switches]["mean"].append(np.mean(freqs))
                        bootstrapping_results[result_type][num_switches]["std"].append(np.std(freqs))
                        bootstrapping_results[result_type][num_switches]["nobs"].append(len(freqs))
                if all_freqs:
                    all_switches_bootstrapping_results[result_type]["mean"].append(np.mean(all_freqs))
                    all_switches_bootstrapping_results[result_type]["std"].append(np.mean(all_freqs))
                    all_switches_bootstrapping_results[result_type]["nobs"].append(len(all_freqs))

        with open(base_report_path + ".boundaries.tsv", 'w') as handle:
            tsv_writer = csv.writer(handle, delimiter="\t")
            tsv_writer.writerow([None, "Leftmost exterior", None, None, "Left BURs", None, None,
                                 "Interior", None, None, "Right BURs", None, None, "Rightmost exterior", None, None,
                                 "All unswitched", None, None, "Interior + Boundaries", None, None])
            tsv_writer.writerow(["Number of switches"] + ["Mean", "SD", "N"]*6)
            for num_switches in range(max_switches + 1):
                row = [num_switches]
                for result_type in result_types:
                    if bootstrapping_results[result_type][num_switches]["mean"]:
                        mean = np.mean(bootstrapping_results[result_type][num_switches]["mean"])
                        std = np.mean(bootstrapping_results[result_type][num_switches]["std"])
                        nobs = np.mean(bootstrapping_results[result_type][num_switches]["nobs"])
                        row.extend([mean, std, nobs])
                    else:
                        row.extend([None, None, None])
                tsv_writer.writerow(row)

            # "Any" category
            row = ["Any"]
            for result_type in result_types:
                if all_switches_bootstrapping_results[result_type]["mean"]:
                    mean = np.mean(all_switches_bootstrapping_results[result_type]["mean"])
                    std = np.mean(all_switches_bootstrapping_results[result_type]["std"])
                    nobs = np.mean(all_switches_bootstrapping_results[result_type]["nobs"])
                    row.extend([mean, std, nobs])
                else:
                    row.extend([None, None, None])
            tsv_writer.writerow(row)


def mutation_categories(data, reportdir, database):
    """

        :param data:
        :param reportdir:
        :param database:
        :return:
        """

    references = database.get("references")
    switches = database.get("switches")

    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]

    for refid, tags, read_subset in data:
        reference = references[refid]
        if reference.cassettes_aln is not None:  # only do analysis for those bins where cassettes are aligned.
            insertions = {}
            deletions = {}
            substitutions = {}
            for read in read_subset:
                for aln in read.alns:
                    nontemplated_tf = al.nontemplated(aln.transform, reference)
                    for op in nontemplated_tf:
                        if op[1] == "I":
                            if op[2] not in insertions:
                                insertions[op[2]] = 0
                            insertions[op[2]] += 1/len(read.alns)
                        elif op[1] == "D":
                            delseq = reference.seq[op[0]:op[0]+op[2]]
                            if delseq not in deletions:
                                deletions[delseq] = 0
                            deletions[delseq] += 1/len(read.alns)
                        elif op[1] == "S":
                            replacing = reference.seq[op[0]]
                            key = (replacing, op[2])
                            if key not in substitutions:
                                substitutions[key] = 0
                            substitutions[key] += 1/len(read.alns)

            # write report on the frequency of each insertion and deletion
            base_report_path = os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags))

            with open(base_report_path + ".insertions.tsv", "w") as handle:
                tsv_writer = csv.writer(handle, delimiter="\t")
                tsv_writer.writerow(["Number of Reads:", str(len(read_subset))])
                tsv_writer.writerow([])
                tsv_writer.writerow(["Sequence", "Length", "Frequency"])
                for sequence, frequency in insertions.items():
                    tsv_writer.writerow([sequence, str(len(sequence)), frequency])

            with open(base_report_path + ".deletions.tsv", "w") as handle:
                tsv_writer = csv.writer(handle, delimiter="\t")
                tsv_writer.writerow(["Number of Reads:", str(len(read_subset))])
                tsv_writer.writerow([])
                tsv_writer.writerow(["Sequence", "Length", "Frequency"])
                for sequence, frequency in deletions.items():
                    tsv_writer.writerow([sequence, str(len(sequence)), frequency])

            with open(base_report_path + ".substitutions.tsv", "w") as handle:
                tsv_writer = csv.writer(handle, delimiter="\t")
                tsv_writer.writerow(["Number of Reads:", str(len(read_subset))])
                tsv_writer.writerow([])
                tsv_writer.writerow(["Original Base", "Substitution", "Frequency"])
                for (frombase, tobase), frequency in substitutions.items():
                    tsv_writer.writerow([frombase, tobase, frequency])

reporters = {
    "snp_frequency": partial(simple_functions, metric="snp_frequency"),
    "distinct_variants": partial(simple_functions, metric="distinct_variants"),
    "parentals": partial(simple_functions, metric="parentals"),
    "nontemp_indel_sizes": partial(simple_functions, metric="nontemp_indel_sizes"),
    "snp_frequency_vr_cr": partial(simple_functions, metric="snp_frequency_vr_cr"),
    "nontemplated_snp_frequency_vr_cr": partial(simple_functions, metric="nontemplated_snp_frequency_vr_cr"),
    "snp_frequency_annotated_vr": partial(simple_functions, metric="annotated_vr_snp_frequency"),
    "switch_length": partial(simple_functions, metric="switch_length"),
    "switches_per_read": partial(simple_functions, metric="switches_per_read"),
    "unique_variants": partial(simple_functions, metric="unique_variants"),
    "slips_per_read": partial(simple_functions, metric="slips_per_read"),
    "slipped_snps_per_nontemplated_snp": partial(simple_functions, metric="slipped_snps_per_nontemplated_snp"),
    "frameshifts": partial(simple_functions, metric="frameshifts"),
    "dn_ds": partial(simple_functions, metric="dn_ds"),
    "snp_positions": snp_positions,
    "snp_positions_cassettes": snp_positions_cassettes,
    "ids_colocation": ids_colocation,
    "nontemplated_reads_bam": nontemplated_reads_bam,
    "two_subset_comparison": two_subset_comparison,
    "slippage": slippage,
    "detailed_switch_length": detailed_switch_length,
    "switch_length_simulation": switch_length_simulation,
    "cassette_usage": cassette_usage,
    "detailed_switches_per_read": detailed_switches_per_read,
    "variant_frequency": variant_frequency,
    "distance_between_switches": distance_between_switches,
    "max_variants": max_variants,
    "cassette_similarity": cassette_similarity,
    "aa_positions": aa_positions,
    "aa_op_frequency": aa_op_frequency,
    "dna_op_frequency": dna_op_frequency,
    "switch_boundaries": switch_boundaries,
    "switch_boundary_g_run_coincidence": switch_boundary_g_run_coincidence,
    "cassette_sequence_identity": cassette_sequence_identity,
    "switching_and_nontemplated": switching_and_nontemplated,
    "switching_and_slippage": switching_and_slippage,
    "list_of_slips": list_of_slips,
    "mutation_categories": mutation_categories,
    "long_switches": long_switches
}
