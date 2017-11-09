#!/usr/bin/env python3

# builtin
import os
import csv
import math
import string
from itertools import product, combinations, chain
from functools import reduce, partial
from operator import mul

# dependencies
import numpy as np
from matplotlib import rcParams
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import get_cmap, colors
try:  # pysam is optional
    import pysam
except ImportError:
    pysam = False

# included
import vlstools.alignments as al
import vlstools.utils as ut

# Decrease font size to prevent clashing in larger plots.
rcParams['font.sans-serif'] = "Arial"
rcParams['pdf.fonttype'] = 42


def simple_functions(metric, data, reportdir, database, count_indels_once=False):
    references = database.get("references")
    switches = database.get("switches")
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
                   "snp_frequency_vr_ir": ["IR Size", "VR Size", "IR SNPs (Mean)", "IR SNPs (SD)", "IR SNPs (N)",
                                           "VR SNPs (Mean)", "VR SNPs (SD)", "VR SNPs (N)"],
                   "annotated_vr_snp_frequency": ["IR Size", "VR Size", "IR SNPs (Mean)", "IR SNPs (N)",
                                                  "VR SNPs (Mean)", "VR SNPs (N)",
                                                  "Templated IR SNPs (Mean)", "Templated IR SNPs (N)",
                                                  "Templated VR SNPs (Mean)", "Templated VR SNPs (N)"],
                   "unique_variants": ["Variants in subset", "Variants in other subsets", "Unique Variants",
                                       "Shared Variants"]
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

            elif metric == "snp_frequency_vr_ir":
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
            else:
                raise NotImplementedError

            tagdict = dict(tags)
            tsv_writer.writerow([refid, len(read_subset)] + [str(tagdict[l]) for l in tag_labels] + outputs)


def snp_positions(data, reportdir, database):
    references = database.get("references")

    # filter for non-empty bins, and require references with aligned cassettes
    data = [(refid, tags, read_subset) for refid, tags, read_subset in data
            if references[refid].cassettes_aln is not None and read_subset]

    for refid, tags, read_subset in data:
        reference = references[refid]
        # writes TSV report
        with open(os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags) + ".tsv"), "w") as handle:
            tsv_writer = csv.writer(handle, delimiter="\t")
            coords = list(range(reference.offset + 1, reference.offset + 1 + len(reference.seq)))
            tsv_writer.writerow([""] + coords)
            all_snps = [0] * len(reference.seq)
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

            # count the SNP frequency for actual templated SNPs, and the number of SNPs in the cassettes
            if reference.cassettes_aln is not None:
                # make a list of all the SNPs that are templated
                templated_ops = [op for aln in reference.cassettes_aln.values() for op in
                                 al.trim_transform(aln.transform, len(reference.seq))]

                # bin the data by position
                templated_hist = [0] * len(reference.seq)
                for read in read_subset:
                    for aln in read.alns:
                        for op in aln.transform:
                            if op in templated_ops:
                                if op[1] in "S":
                                    templated_hist[op[0]] += 1 / len(read_subset) / len(read.alns)
                                elif op[1] == "D":
                                    for x in range(op[2]):
                                        templated_hist[op[0] + x] += 1 / len(read_subset) / len(read.alns)
                                elif op[1] == "I":
                                    templated_hist[op[0]] += len(op[2]) / len(read_subset) / len(read.alns)
                tsv_writer.writerow(["Templated SNP frequency"] + templated_hist)

                # number of SNPs at each position in the cassettes
                cassette_snps = [0] * len(reference.seq)
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


        # Plot the data and send to PDF files.
        with PdfPages(os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags) + ".pdf")) as pdf:
            # Page 1: All SNPs vs position.
            fig, (ax_all_snps, ax_cassettes) = plt.subplots(2, figsize=(8, 8), sharex=True)
            ax_all_snps.bar(height=all_snps, left=coords, width=1, linewidth=0, color="black")
            ax_all_snps.set_title("Observed Variants")
            ax_all_snps.set_ylabel("SNP Frequency")
            ax_all_snps.set_xlabel("vlsE position (bp)")
            if reference.cassettes_aln is not None:
                ax_cassettes.bar(height=cassette_snps, left=coords, width=1, linewidth=0, color="black")
                ax_cassettes.set_title("Reference Cassettes")
                ax_cassettes.set_ylabel("Frequency of SNPs in silent cassettes")
                ax_cassettes.set_xlabel("vlsE position (bp)")
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            if reference.cassettes_aln is not None:
                # Page 2: Mirror plot comparing the distribution of Templated to Cassettes
                fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 8), sharex=True)
                plt.subplots_adjust(hspace=0)
                ax1.bar(height=templated_hist, left=coords, width=1, linewidth=0, color="green")
                ax1.set_ylabel("Frequency of Templated SNPs", color="green")
                ax1.set_xlabel("vlsE position (bp)")
                ax1.set_xlim(min(coords), max(coords))
                ax1.spines['bottom'].set_visible(False)
                ax1.xaxis.set_ticks_position('none')
                ax1.yaxis.tick_left()
                for tl in ax1.get_yticklabels():
                    tl.set_color("green")

                ax2.bar(height=cassette_snps, left=coords, width=1, linewidth=0, color="red")
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
            plt.clf()


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
    coords = list(range(reference.offset + 1, reference.offset + 1 + len(reference.seq)))

    with PdfPages(pdfpath) as pdf:
        # data1 vs data2 Templated SNP Frequencies
        fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 8), sharex=True)
        plt.subplots_adjust(hspace=0)
        ax1.bar(height=hist1, left=coords, width=1, linewidth=0, color="green")
        ax1.set_ylabel("Templated SNP density (%s)" % name1, color="green")
        ax1.set_xlabel("vlsE position (bp)")
        ax1.set_xlim(min(coords), max(coords))
        ax1.spines['bottom'].set_visible(False)
        ax1.xaxis.set_ticks_position('none')
        ax1.yaxis.tick_left()
        for tl in ax1.get_yticklabels():
            tl.set_color("green")

        ax2.bar(height=hist2, left=coords, width=1, linewidth=0, color="red")
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
                tsv_writer.writerow([tagname] + [dict(tags)[tagname] for tags in tagsets])
            tsv_writer.writerow([])
            tsv_writer.writerow(["Number of Reads"] + readcounts)
            tsv_writer.writerow([])
            tsv_writer.writerow(["Variant"])
            for variant in outputs:
                tsv_writer.writerow([repr(variant)]
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
            ax1.bar(height=hist, left=coords, width=1, linewidth=0, color="darkblue")
            ax1.set_ylabel("Frequency of Templated SNPs", color="darkblue")
            ax1.set_xlabel("vlsE position (bp)")
            ax1.set_xlim(min(coords), max(coords))
            ax1.spines['bottom'].set_visible(False)
            ax1.xaxis.set_ticks_position('none')
            ax1.yaxis.tick_left()
            for tl in ax1.get_yticklabels():
                tl.set_color("darkblue")

            ax2.bar(height=cassette_snps, left=coords, width=1, linewidth=0, color="firebrick")
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
        reference = references[refid]
        read_subset = [r for r in read_subset if r.protein_aln]
        methods = set(method for r in read_subset for method in r.protein_aln.keys())
        refmethods = set(method for method in references_aa_positions[refid].keys())

        # writes TSV report
        if read_subset:
            coords = list(range(math.ceil(reference.offset / 3) + 1,  # start at first full amino acid
                                (reference.offset + len(reference.seq)) // 3 + 1))  # end at last full amino acid
            with open(os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags) + ".tsv"), "w") as handle:
                tsv_writer = csv.writer(handle, delimiter="\t")

                # Part 1: actual frequencies
                tsv_writer.writerow([])
                tsv_writer.writerow(["Actual frequencies"])
                tsv_writer.writerow([])
                tsv_writer.writerow(["Position:"] + coords)
                tsv_writer.writerow(["Amino Acid:"] + list(reference.protein))
                for method in methods:
                    aa_freq = [0] * len(coords)
                    for read in read_subset:
                        for aln in read.protein_aln[method]:
                            for op in aln.transform:
                                if op[1] == "S":
                                    aa_freq[op[0]] += 1 / len(read_subset) / len(read.protein_aln[method])
                                if op[1] == "D":
                                    for x in range(op[2]):
                                        aa_freq[op[0] + x] += 1 / len(read_subset) / len(read.protein_aln[method])
                                elif op[1] == "I":
                                    aa_freq[op[0]] += len(op[2]) / len(read_subset) / len(read.protein_aln[method])

                    tsv_writer.writerow(["Frequency of variant amino acid (Method: %s)" % method] + aa_freq)
                    # write a file that can be used by data2bfactor.py to colour 3D structures
                    with open(os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags)
                              + "_data2b_%s.txt" % method), "w") as d2bhandle:
                        d2b_writer = csv.writer(d2bhandle, delimiter=" ")
                        for coord in range(coords[0]):  # prefix with zeros to overwrite b-factors in pdb
                            d2b_writer.writerow([coord, 0])
                        for coord, one_letter, freq in zip(coords, reference.protein, aa_freq):
                            d2b_writer.writerow([coord, freq])
                        for coord in range(coords[-1]+1, 10000):  # suffix with zeros to overwrite b-factors in pdb
                            d2b_writer.writerow([coord, 0])

                # Part 2: theoretical frequencies
                tsv_writer.writerow([])
                tsv_writer.writerow(["Theoretical frequencies"])
                tsv_writer.writerow([])
                tsv_writer.writerow(["Position:"] + coords)
                tsv_writer.writerow(["Amino Acid:"] + list(reference.protein))
                for method in refmethods:
                    aa_freq = [0] * len(coords)
                    for cassette in references_aa_positions[refid][method]:
                        for alignment in cassette:
                            for op in alignment.transform:
                                if op[1] == "S":
                                    aa_freq[op[0]] += 1 / len(references_aa_positions[refid][method]) / len(cassette)
                                if op[1] == "D":
                                    for x in range(op[2]):
                                        aa_freq[op[0] + x] += (1 / len(references_aa_positions[refid][method])
                                                               / len(cassette))
                                elif op[1] == "I":
                                    aa_freq[op[0]] += (len(op[2]) / len(references_aa_positions[refid][method])
                                                       / len(cassette))

                    tsv_writer.writerow(["Frequency of variant amino acid (Method: %s)" % method] + aa_freq)
                    # write a file that can be used by data2bfactor.py to colour 3D structures
                    with open(os.path.join(reportdir, ut.get_tagstring(refid=refid, tags=tags)
                              + "_data2b_theoretical_%s.txt" % method), "w") as d2bhandle:
                        d2b_writer = csv.writer(d2bhandle, delimiter=" ")
                        for coord in range(coords[0]):  # prefix with zeros to overwrite b-factors in pdb
                            d2b_writer.writerow([coord, 0])
                        for coord, one_letter, freq in zip(coords, reference.protein, aa_freq):
                            d2b_writer.writerow([coord, freq])
                        for coord in range(coords[-1]+1, 10000):  # suffix with zeros to overwrite b-factors in pdb
                            d2b_writer.writerow([coord, 0])

                # Part 3: read counts
                tsv_writer.writerow([])
                tsv_writer.writerow(["Read Counts"])
                tsv_writer.writerow([])
                tsv_writer.writerow(["Number of Reads:", len(read_subset)])
                tsv_writer.writerow(["Position:"] + coords)
                tsv_writer.writerow(["Amino Acid:"] + list(reference.protein))
                for method in methods:
                    aa_freq = [0] * len(coords)
                    for read in read_subset:
                        for aln in read.protein_aln[method]:
                            for op in aln.transform:
                                if op[1] == "S":
                                    aa_freq[op[0]] += 1 / len(read.protein_aln[method])
                                if op[1] == "D":
                                    for x in range(op[2]):
                                        aa_freq[op[0] + x] += 1 / len(read.protein_aln[method])
                                elif op[1] == "I":
                                    aa_freq[op[0]] += len(op[2]) / len(read.protein_aln[method])
                    tsv_writer.writerow(["Counts of variant amino acid (Method: %s)" % method] + aa_freq)


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

reporters = {
    "snp_frequency": partial(simple_functions, metric="snp_frequency"),
    "distinct_variants": partial(simple_functions, metric="distinct_variants"),
    "parentals": partial(simple_functions, metric="parentals"),
    "snp_frequency_vr_ir": partial(simple_functions, metric="snp_frequency_vr_ir"),
    "snp_frequency_annotated_vr": partial(simple_functions, metric="annotated_vr_snp_frequency"),
    "unique_variants": partial(simple_functions, metric="unique_variants"),
    "snp_positions": snp_positions,
    "two_subset_comparison": two_subset_comparison,
    "variant_frequency": variant_frequency,
    "max_variants": max_variants,
    "aa_positions": aa_positions,
    "aa_op_frequency": aa_op_frequency
}
