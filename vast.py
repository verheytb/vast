#!/usr/bin/env python3

"""
Variable Antigen Switching Tracer is a set of tools for analyzing PacBio sequencing data of vls variants. vast allows
FASTA files from multiple experiments and with different experimental conditions to be entered into a central database,
and comes with a variety of tools to extract biologically meaningful measurements from that data.
"""
# builtins
import argparse
import csv
import os
import pickle
import shutil
import sys
import multiprocessing
from types import SimpleNamespace
from time import sleep
from datetime import datetime
# dependencies
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
# included
import vlstools.utils as utl
import vlstools.reports as rp
import vlstools.alignments as al


__author__ = "Theodore (Ted) B. Verhey"
__version__ = "4.0"
__email__ = "verheytb@gmail.com"
__status__ = "Production"


class Database(object):
    def __init__(self, user_working=os.path.expanduser("~/.vastdb")):
        # check that a database is loaded
        try:
            with open(user_working, "r") as handle:
                self.db_location = os.path.abspath(handle.__next__())
        except OSError:
            utl.tprint('Error: No database loaded. Use "vast load" to load an existing database or "vast new" to '
                       'create a new database.')
            sys.exit()

        # check for valid database
        if not os.path.isdir(self.db_location):
            utl.tprint("Error: %s is not a directory" % self.db_location)
            sys.exit()

        # create subdirectories if not there
        self.reportdir = os.path.join(self.db_location, "Reports")
        self.cassette_aln_dir = os.path.join(self.db_location, "Cassette Alignments")
        self.exported_references = os.path.join(self.db_location, "References")
        if not (os.path.exists(self.reportdir) and os.path.isdir(self.reportdir)):
            os.mkdir(self.reportdir)
        if not (os.path.exists(self.cassette_aln_dir) and os.path.isdir(self.cassette_aln_dir)):
            os.mkdir(self.cassette_aln_dir)

        # report success!
        utl.tprint("Database loaded: %s" % self.db_location)

    def get(self, table_name):
        table_path = os.path.join(self.db_location, table_name + ".p")
        with open(table_path, "rb") as handle:
            table = pickle.load(handle)
        return table

    def save(self, list, table_name):
        table_path = os.path.join(self.db_location, table_name + ".p")
        with open(table_path, "wb") as handle:
            pickle.dump(list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_and_check_reads(self):
        try:
            reads = self.get("reads")
        except FileNotFoundError:
            utl.tprint('No reads in database. Use "vast add_reads" to add reads.')
            sys.exit()
        return reads

    def get_and_check_references(self):
        try:
            references = self.get("references")
        except FileNotFoundError:
            utl.tprint('No references in database. Use "vast add_reference" to add.')
            sys.exit()
        return references


if __name__ == "__main__":
    # parses the arguments and displays usage information
    parser = argparse.ArgumentParser(
        description="Variable Antigen Switching Tracer is a set of tools for analyzing PacBio sequencing data of vls "
                    "variants. VAST allows FASTA files from multiple experiments and with different experimental "
                    "conditions to be entered into a central database, and comes with a variety of tools to extract "
                    "biologically meaningful measurements from that data.")
    subparsers = parser.add_subparsers(title="Commands", help="Command Help", dest="subcommand")
    # loads a previously created database
    load_subparser = subparsers.add_parser("load", help="Load a database")
    load_subparser.add_argument("dbdir")
    # creates a new database
    new_subparser = subparsers.add_parser("new", help="Create a new database")
    new_subparser.add_argument("-f", "--force", action="store_true",
                               help="Force overwriting an existing database with a new one.")
    new_subparser.add_argument("dbdir", help="A database filename to store all generated and imported data.")

    # add reads to database from a list of tabular list of fasta files
    add_reads_subparser = subparsers.add_parser("add_reads", help="Add a set of sample reads and the corresponding "
                                                                  "reference and cassette sequences for analysis.")
    add_reads_subparser.add_argument("csv_file", type=str,
                                     help="A csv file with a header row and the following columns: "
                                          "1. Sample FASTA/Q filename 2. Reference name 3+. Sample Tags (eg. Strain, "
                                          "Time Point, Conditions). Note that the reference name and the cassettes "
                                          "name must refer to sequences already entered in the database.")
    add_reads_subparser.add_argument("-r", "--replace", action="store_true",
                                     help="For reads already in the database, tags will be entirely replaced to "
                                          "reflect those being imported. Overrides --force and --merge.")
    add_reads_subparser.add_argument("-f", "--force", action="store_true",
                                     help="Overwrites tags for reads already in the database. Combine with --merge to "
                                          "retain tags not present in the reads being imported.")
    add_reads_subparser.add_argument("-m", "--merge", action="store_true",
                                      help="If the reads being imported already exist in the database, any new tags will"
                                           " be added, but not overwritten unless --force is also specified.")

    # add a vlsE reference and associated vls cassettes to the database from a fasta file
    add_reference_subparser = subparsers.add_parser("add_reference", help="Add a reference file to the database.")
    add_reference_subparser.add_argument("name", type=str, help="A name for the reference file (avoid all whitespace "
                                                                "characters).")
    add_reference_subparser.add_argument("reference_fasta", type=str,
                                         help="A FASTA file with a single reference sequence.")
    add_reference_subparser.add_argument("offset", type=int, help="The 0-based starting position of the reference "
                                                                  "sequence relative to the start of the gene.")
    add_reference_subparser.add_argument("-c", "--cassettes", metavar="cassettes_fasta", type=str,
                                         default=None, help="A FASTA file with the cassette sequences.")

    # add the positions of the traditionally defined variable regions to a reference.
    add_vr_subparser = subparsers.add_parser("annotate_vr",
                                             help="Annotate a reference with the positions of variable regions.")
    add_vr_subparser.add_argument("refid", type=str, help="The name of a reference already in the database.")
    add_vr_subparser.add_argument("-r", "--relative", action="store_true",
                                  help="When specified, coordinates are relative to reference, not vlsE gene.")
    add_vr_subparser.add_argument("-f", "--force", action="store_true",
                                  help="Overwrite variable region annotations already in the database.")
    add_vr_subparser.add_argument("positions", metavar="x", nargs="+", type=int,
                                  help="a list of start and stop coordinates (eg., for two regions: 17 29 35 63). Each "
                                       "pair is [start, stop) in 0-based coordinates and is relative to the start of "
                                       "the gene (not the reference sequence).")

    # align cassettes from scratch or input pre-aligned cassettes
    align_cassettes_subparser = subparsers.add_parser("align_cassettes",
                                                      help="Make the multiple alignment of the cassettes for each set "
                                                           "of cassettes provided with \"vast add_reference\".")
    align_cassettes_subparser.add_argument("-p", "--prealigned", metavar="pyc_file", type=str,
                                           help="A .pyc file representing previously computed multiple alignment.")
    align_cassettes_subparser.add_argument("-f", "--force", action="store_true",
                                           help="Force recomputing alignments already in the database.")

    # display stats about the db
    dbstats_subparser = subparsers.add_parser("dbstats",
                                              help="Display information about the contents of the database.")
    dbstats_subparser.add_argument("--references", "-r", action="store_true",
                                   help="Display information about the reference and cassette sequences in the "
                                        "database.")
    dbstats_subparser.add_argument("--ontology", "-o", action="store_true",
                                   help="Display the ontology for reads in the database.")

    # export data in the database to human-readable and other formats
    export_subparser = subparsers.add_parser("export_references",
                                             help="Export references in the database as fasta files (for the sequence) "
                                                  "and bam files (for the aligned cassette sequences).")

    # map variants
    map_variants_subparser = subparsers.add_parser("map",
                                                   help="Map the variant sequences to their references, retaining "
                                                        "any alignments that are both objectively closest to the "
                                                        "reference and are also most similar to the aligned cassettes.")
    map_variants_subparser.add_argument("-j", "--justify", action="store_true",
                                        help="Do not use all equivalent alignments; instead, use a single alignment "
                                             "that is the left-justified one. This eliminates read mapping ambiguity "
                                             "and decreases memory usage, but introduces a bias into alignments.")
    map_variants_subparser.add_argument("-f", "--force", action="store_true", help="Remap already mapped reads.")
    map_variants_subparser.add_argument("-c", "--cpus", default=multiprocessing.cpu_count(), type=int,
                                        help="Specify the number of CPUs to use for processing.")
    # map proteins
    map_proteins_subparser = subparsers.add_parser("map_proteins",
                                                   help="Translate the variants into protein sequences and map to the "
                                                        "reference.")

    # generates report of measurements based on customized groupings of the data
    report_subparser = subparsers.add_parser("report", help="Make measurements on groupings of data and export a CSV "
                                                            "report.")
    report_subparser.add_argument("metric", type=str, choices=rp.reporters,
                                  help="Specify the quantity to be measured.")
    report_subparser.add_argument("--groupby", metavar="tag", type=str, nargs="+",
                                  help="Group samples by the supplied tag(s). Specify \"-g +\" to group by all "
                                       "possible tags (ie. to separate all samples from each other).")
    report_subparser.add_argument("--where", metavar="tag==val1,val2 | tag!=val1,val2", type=str, nargs="+",
                                  help="Select a subset of the data by including or samples with certain tags.")
    report_subparser.add_argument("--custom", metavar="arg=value", type=str, nargs="+",
                                  help="Specify advanced options for specific reports, where supported. See "
                                       "documentation for details.")
    args = parser.parse_args()

    if args.subcommand is None:
        db = Database()
        parser.print_help()

    elif args.subcommand == "new":
        if os.path.exists(args.dbdir) and not args.force:
            utl.tprint('Error: %s already exists. If you want to overwrite the old one, use "vast new -f <dbdir>" '
                       'to overwrite.' % args.dbdir)
        else:
            if os.path.exists(args.dbdir):
                utl.tprint("Overwriting database: %s" % args.dbdir)
                shutil.rmtree(args.dbdir)
            os.mkdir(args.dbdir)
            with open(".vlsdb", "w") as vlsdbname:
                vlsdbname.write(args.dbdir)
            db = Database()

    elif args.subcommand == "load":
        user_working = os.path.expanduser("~/.vastdb")
        with open(user_working, "w") as vlsdbname:
            vlsdbname.write(args.dbdir)
        db = Database()

    # adds a reference sequence and cassette sequences to the database
    elif args.subcommand == "add_reference":
        db = Database()
        try:
            references = db.get("references")
        except FileNotFoundError:
            references = {}
        # gets reference and cassettes
        with open(args.reference_fasta, "r") as handle:
            newseq = str(SeqIO.read(handle, "fasta").seq)
        if args.cassettes:
            with open(args.cassettes, "r") as handle:
                newcassettes = {c.id: str(c.seq) for c in SeqIO.parse(handle, "fasta")}
        else:
            newcassettes = None

        protein = al.translate(newseq, args.offset)

        # makes new entry, tests whether it exists already
        new_reference = SimpleNamespace(name=args.name, seq=newseq, offset=args.offset, protein=protein,
                                        cassettes=newcassettes, cassettes_aln=None, variable_regions=None)
        if args.name in references:
            utl.tprint("Error: A reference with the same name already exists in the database.")
            sys.exit()
        for r in references.values():
            if newseq == r.seq and newcassettes == r.cassettes:
                utl.tprint("Error: An identical reference entry with the same reference and cassette sequences already "
                           "exists in the database:")
                sys.exit()
        # writes to db
        references[new_reference.name] = new_reference
        db.save(references, "references")

    elif args.subcommand == "annotate_vr":
        db = Database()
        references = db.get_and_check_references()
        ref = references[args.refid]
        if ref.variable_regions and not args.force:
            vrstring = ", ".join(str(pair[0] + ref.offset) + "–" + str(pair[1] + ref.offset)
                                 for pair in ref.variable_regions)
            utl.tprint("Error: Variable Regions have already been annotated for the %s reference:\n" % args.refid +
                       vrstring + "\nPlease re-run with the -f option to force overwriting them.")
            sys.exit()
        if args.relative:
            offset = 0
        else:
            offset = ref.offset
        ref.variable_regions = [(args.positions[x]-offset, args.positions[x+1]-offset)
                                for x in range(0, len(args.positions), 2)]
        utl.tprint("%s Variable regions: %s" % (ref.name, repr(ref.variable_regions)))
        db.save(references, "references")

    # prepares a multiple alignment of the cassettes that is optimally close to the reference and the other cassettes.
    elif args.subcommand == "align_cassettes":
        db = Database()
        references = db.get_and_check_references()
        if args.preset:
            with open(args.preset, "rb") as handle:
                msa = pickle.load(handle)
            cassette_set = {x[0]: x[1] for x in msa}
            if cassette_set not in (r.cassettes for r in references.values()):
                utl.tprint("No matching cassettes found, %s was not imported." % args.preset)
                sys.exit()
            for r in references.values():
                if cassette_set == r.cassettes:
                    r.cassettes_aln = {x[0]: x[2] for x in msa}
        else:  # fallback to calculating optimal cassette multiple alignments by hand.
            for r in references.values():
                if r.cassettes is not None and r.cassettes_aln is None:
                    reference_dir_name = os.path.join(db.cassette_aln_dir, r.name)
                    if not os.path.isdir(reference_dir_name):
                        os.mkdir(reference_dir_name)
                    r.cassettes_aln = al.align_cassettes(reference=r, outputdir=reference_dir_name)
        db.save(references, "references")

    elif args.subcommand == "add_reads":
        db = Database()
        # loads database tables
        references = db.get_and_check_references()
        try:
            reads = {read.name: read for read in db.get("reads")}  # dict is handy for getting reads by name
        except FileNotFoundError:
            reads = {}

        # assembles list of reads to be added from input files
        with open(args.csv_file, "r") as csvhandle:
            csvreader = csv.reader(csvhandle)
            new_tag_set = csvreader.__next__()[2:]
            newreads = []
            for row in csvreader:
                # check file format
                if any(row[0].lower().endswith(x) for x in (".fna", ".fa", ".fasta")):
                    fastaq_format = "fasta"
                elif any(row[0].lower().endswith(x) for x in (".fnq", ".fq", ".fastq")):
                    fastaq_format = "fastq"
                else:
                    utl.tprint("File \"%s\" does not have a recognizable fasta/fastq extension." % row[0])
                    sys.exit()
                # check that references are added
                if row[1] not in references:
                    utl.tprint("The \"%s\" reference does not exist in the database but is required for importing "
                               "associated reads. References in the database: %s" % (row[1], ", ".join(references)))
                    sys.exit()
                # add reads
                with open(row[0], "rU") as fastaqfile:
                    for entry in SeqIO.parse(fastaqfile, format=fastaq_format):
                        if fastaq_format == "fastq":
                            phred_quality = entry.letter_annotations[list(entry.letter_annotations)[0]]  # untested
                        else:
                            phred_quality = None
                        tags = {new_tag_set[x]: tag_value for x, tag_value in enumerate(row[2:])}
                        newreads.append(SimpleNamespace(name=entry.id, seq=str(entry.seq), qual=phred_quality,
                                                        refid=row[1], tags=tags, alns=None))

        # go through and add reads and/or tags if appropriate.
        for newread in newreads:
            if newread.name in reads:
                oldread = reads[newread.name]
                # Check that ID matches the same sequence, quality, and reference in the database
                if not (newread.name == oldread.name and newread.seq == oldread.seq and
                        newread.qual == oldread.qual and newread.refid == oldread.refid):
                    utl.tprint("Reads with the same IDs as imported reads but different sequences or quality "
                               "scores were found. Read IDs must match sequences.")
                    sys.exit()
                if args.replace:
                    # replace old tags completely with new ones
                    reads[newread.name].tags = newread.tags
                else:
                    # add or replace tags if force/merge is employed
                    for tag, tagvalue in newread.tags.items():
                        if (tag in oldread.tags and args.force) or args.merge:
                            reads[newread.name].tags[tag] = tagvalue

            else:
                reads[newread.name] = newread

        # goes through all reads and makes sure all of them have the same tag categories; if a read doesn't have a
        # particular tag, this creates it, and fills it with a 'None' value.
        all_tags = [tag for read in reads.values() for tag in read.tags]
        for read in reads.values():
            for tag in all_tags:
                if tag not in read.tags:
                    read.tags[tag] = None

        # converts reads dictionary back to list and saves.
        db.save(list(reads.values()), "reads")

    elif args.subcommand == "dbstats":
        db = Database()
        if args.references:
            references = db.get_and_check_references()
            utl.tprint("Reference Information:")
            for r in references.values():
                print("*"*shutil.get_terminal_size()[0])
                print("Reference name:", r.name)
                print("Reference sequence:", r.seq)
                print("Reference offset:", int(r.offset))
                if r.variable_regions:
                    vrstring = ", ".join(str(pair[0]+r.offset) + "–" + str(pair[1]+r.offset)
                                         for pair in r.variable_regions)
                else:
                    vrstring = "None"
                print("VR Annotations:", vrstring)
                if r.cassettes is None:
                    print("No cassettes associated with this reference.")
                else:
                    print("Cassettes:")
                    for name, casseq in r.cassettes.items():
                        print("\t%s | %sbp | %s" % (name, len(casseq),
                                                    ("aligned" if r.cassettes_aln else "not aligned")))
                        print(casseq)

        if args.ontology:
            reads = db.get_and_check_reads()
            tagsets = {}
            for read in reads:
                tagset = tuple(tag for tag in read.tags.keys() if tag is not None)
                if tagset not in tagsets:
                    tagsets[tagset] = 0
                tagsets[tagset] += 1
            for x, (tagset, count) in enumerate(tagsets.items(), start=1):
                print("Ontology %d (%d reads): %s" % (x, count, str(tagset)))

    elif args.subcommand == "export_references":
        db = Database()
        export_dir = db.exported_references
        if not (os.path.exists(export_dir) and os.path.isdir(export_dir)):
            os.mkdir(export_dir)
        references = db.get("references")
        for ref in references.values():
            basename = os.path.join(export_dir, ref.name)
            # export FASTA file of reference
            with open(basename + ".fasta", "w") as handle:
                SeqIO.write(SeqRecord(seq=Seq(ref.seq), id=ref.name, description=""), handle, "fasta")
            if ref.cassettes_aln is not None:
                # exports a bam file for the aligned cassettes of each reference.
                al.write_bam(filename=basename + ".bam", refid=ref.name, refseq=ref.seq,
                             reads_dict=ref.cassettes_aln)


    elif args.subcommand == "map":
        db = Database()
        reads = db.get_and_check_reads()
        if args.force:
            for read in reads:
                read.alns = None
        references = db.get_and_check_references()
        # assembles a list of unique sequence-reference pairs that have not already been aligned (except with -f).
        # this is a time saving step that is optional
        unique_mappables = set((r.seq, r.refid) for r in reads if r.alns is None)
        arg_generator = ((seq, references[refid]) for seq, refid in unique_mappables)
        with multiprocessing.Pool(args.cpus) as P:
            for count, (seq, refid, alns) in enumerate(P.imap(al.read_align_worker, arg_generator), start=1):
                for r in reads:
                    if r.seq == seq and r.refid == refid:
                        if args.justify:
                            ref_length = len(references[refid].seq)
                            alns.sort(key=lambda aln: al.score_left_justification(aln.transform, ref_length))
                            scores = [al.score_left_justification(aln.transform, ref_length) for aln in alns]
                            r.alns = [alns[0]]
                        else:
                            r.alns = alns
                utl.tprint("%d of %d distinct reads mapped (%d saved)" %
                           (count, len(unique_mappables), count // 100 * 100), ontop=True)
                if count % 100 == 0 or count == len(unique_mappables):
                    db.save(reads, "reads")

    elif args.subcommand == "map_proteins":
        db = Database()
        reads = db.get_and_check_reads()
        if args.force:
            for read in reads:
                read.protein_aln = None
        references = db.get_and_check_references()
        cached_alignments = {}  # stores already computed protein alignments to save time
        for x, read in enumerate(reads):
            reference = references[read.refid]
            read.protein_aln = {"inframe_native_all": [],
                                "inframe_native_templated": [],
                                "inframe_native_nontemplated": [],
                                "inframe_corrected_all": [],
                                "inframe_corrected_templated": [],
                                "inframe_corrected_nontemplated": []}
            if reference.cassettes_aln:
                for aln in read.alns:
                    mapping = aln.transform
                    # 1. Proteins from all in-frame alns, no error-correction.
                    protein_alns = al.translate_mapping(mapping=mapping, reference=reference, templ=True, nontempl=True,
                                                        correctframe=False, filterframe=True, filternonsense=True)
                    read.protein_aln["inframe_native_all"].extend(protein_alns)

                    # 2. Proteins from templated changes only, no error-correction.
                    protein_alns = al.translate_mapping(mapping=mapping, reference=reference, templ=True,
                                                        nontempl=False, correctframe=False, filterframe=True,
                                                        filternonsense=True)
                    read.protein_aln["inframe_native_templated"].extend(protein_alns)

                    # 3. Proteins from nontemplated changes only, no error-correction.
                    protein_alns = al.translate_mapping(mapping=mapping, reference=reference, templ=False,
                                                        nontempl=True, correctframe=False, filterframe=True,
                                                        filternonsense=True)
                    read.protein_aln["inframe_native_nontemplated"].extend(protein_alns)

                    # 4. Proteins from all in-frame alns, with error-correction.
                    protein_alns = al.translate_mapping(mapping=mapping, reference=reference, templ=True, nontempl=True,
                                                        correctframe=True, filterframe=True, filternonsense=True)
                    read.protein_aln["inframe_corrected_all"].extend(protein_alns)

                    # 5. Proteins from templated changes only, with error-correction.
                    protein_alns = al.translate_mapping(mapping=mapping, reference=reference, templ=True,
                                                        nontempl=False, correctframe=True, filterframe=True,
                                                        filternonsense=True)
                    read.protein_aln["inframe_corrected_templated"].extend(protein_alns)

                    # 6. Proteins from nontemplated changes only, with error-correction.
                    protein_alns = al.translate_mapping(mapping=mapping, reference=reference, templ=False,
                                                        nontempl=True, correctframe=True, filterframe=True,
                                                        filternonsense=True)
                    read.protein_aln["inframe_corrected_nontemplated"].extend(protein_alns)
            utl.tprint("Finished %d of %d reads" % (x, len(reads)), ontop=True)
        db.save(reads, "reads")

    elif args.subcommand == "report":
        db = Database()
        reads = db.get_and_check_reads()
        references = db.get_and_check_references()

        # separate data into bins. Bins are always separated by sequences with different references; the --groupby
        # parameter also allows the data to be further split by any or all of the sample tags.
        valid_tags = {tag for read in reads for tag in read.tags.keys()}
        if args.groupby:
            if args.groupby == ["+"]:
                bins = frozenset((r.refid, frozenset(t for t in r.tags.items())) for r in reads)
            else:
                bins = frozenset((r.refid, frozenset(t for t in r.tags.items() if t[0] in args.groupby)) for r in reads)
                for g in args.groupby:
                    if g not in valid_tags:
                        raise ValueError(
                            "%s is not a valid sample tag. Valid tags are: %s" % (g, ", ".join(valid_tags)))
        else:
            # no specified tags to group data by; groups only by reference.
            bins = frozenset((r.refid, frozenset()) for r in reads)

        # parse the --where parameter and set up filters
        include_filter = {}
        exclude_filter = {}
        if args.where:
            for where_statement in args.where:
                # split tag from values by operator
                if "==" in where_statement:
                    tag, values = where_statement.split('==')
                elif "!=" in where_statement:
                    tag, values = where_statement.split('!=')
                else:
                    raise ValueError("One or more of the where statements did not contain exactly 1 == or != operator. "
                                     "Proper formatting of the --where option looks like "
                                     "\"--where week==3,4,5 mstrain!=WT\"")
                # check for valid tag
                if tag not in valid_tags:
                    raise ValueError("%s is not a valid sample tag. Valid tags are: %s" % (tag, ", ".join(valid_tags)))

                # split values by comma and test for valid values
                values = values.split(",")
                valid_values = {read.tags[tag] for read in reads}
                if not all(v in valid_values for v in values):
                    utl.tprint("Warning: The \"%s\" tag does not have the following values in imported reads: %s"
                               % (tag, ",".join(v for v in values if v not in valid_values)))

                # create filters
                if "==" in where_statement:
                    include_filter[tag] = set(values)
                elif "!=" in where_statement:
                    exclude_filter[tag] = set(values)

        # add reads to each bin, filtering with filters supplied by the --where parameter
        binned_data = [(refid, tags, []) for refid, tags in bins]
        for refid, tags, read_subset in binned_data:
            for r in reads:
                # check the bin
                if r.refid == refid and tags <= frozenset(r.tags.items()):
                    # check the include filter
                    if all(r.tags[ifilt] in include_filter[ifilt] for ifilt in include_filter):
                        # check the exclude filter
                        if all(r.tags[efilt] not in exclude_filter[efilt] for efilt in exclude_filter):
                            read_subset.append(r)

        # sort by order of groupby
        for word in (reversed(args.groupby) if args.groupby else []):
            binned_data.sort(key=lambda x: dict(x[1])[word])
        # sort by reference
        binned_data.sort(key=lambda x: x[0])

        # create report directory. As it currently stands, parameter details are all included in the folder name.
        # this may change if this becomes problematic.

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        reportdir = os.path.join(db.reportdir, timestamp + "_" + args.metric)
        if args.groupby:
            reportdir += "_groupby_" + ".".join(args.groupby)
        for ifilt in include_filter:
            reportdir += "_include_" + ifilt + "_" + ".".join(include_filter[ifilt])
        for efilt in exclude_filter:
            reportdir += "_exclude_" + efilt + "_" + ".".join(exclude_filter[efilt])
        if not (os.path.exists(reportdir) and os.path.isdir(reportdir)):
            os.mkdir(reportdir)

        # parse any custom args
        if args.custom:
            custom_args = {c.split("=")[0]: c.split("=")[1] for c in args.custom}
            for arg, value in custom_args.items():
                if value == "True":
                    custom_args[arg] = True
                elif value == "False":
                    custom_args[arg] = False
                elif value == "None":
                    custom_args[arg] = None
                else:
                    try:  # converts values to integers if possible
                        custom_args[arg] = int(value)
                    except ValueError:
                        pass

        else:
            custom_args = {}

        # log the report arguments. This is a backup in case the filename identification is not clear.
        with open(os.path.join(reportdir, "info"), "w") as handle:
            handle.write("VAST Run Report\n\n")
            handle.write(timestamp + "\n")
            if args.groupby:
                handle.write("Groupby: " + " ".join(args.groupby) + "\n")
            else:
                handle.write("Groupby: None\n")
            if args.where:
                handle.write("Where: " + " ".join(args.where) + "\n\n")
            else:
                handle.write("Where: None\n\n")
            handle.write("Custom Arguments:\n")
            for arg, value in custom_args.items():
                handle.write("    %s: %s\n" % (arg, value))

        # call the appropriate function
        rp.reporters[args.metric](data=binned_data, reportdir=reportdir, database=db, **custom_args)