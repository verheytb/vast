[![DOI](https://zenodo.org/badge/DOI/10.1016/j.celrep.2018.04.117.svg)](https://doi.org/10.1016/j.celrep.2018.04.117)

# VAST: Variable Antigen Switching Tracer
**VAST** is for analyzing sequenced variants of the *vlsE* gene found in *Borrelia* species, as well as similar antigenic variation systems in other pathogens.

## Case Study: the VlsE antigen from *B. burgdorferi*
The vls (**V**MP-**l**ike **s**equence) antigenic variation system is composed of a *vlsE* expression locus, and a set of approximately 15 unexpressed, partial-length vls cassettes.

![Diagram of vlsE](https://github.com/verheytb/vast/blob/master/images/vls%20part%201.png)

Unidirectional, segmental recombination of the unexpressed cassettes into the expression locus is responsible for the massive repertoire of vlsE
proteins that protect *Borrelia* from clearance by adaptive immunity.

<p align="center">
    <img src="https://github.com/verheytb/vast/blob/master/images/vls%20part%202.png" alt="Diagram of switching" width="600"/>
</p>

VAST was developed to quantify the behaviour of switching from NGS or Sanger sequencing of full-length or partial vlsE sequences. With large datasets in mind (<100 000 full-length variants), it is written entirely in Python with support for high-performance computing (HPC) environments.

## Workflow

1. Import the reference (unswitched) *vlsE* sequence, and the sequence of the reference silent cassettes. Multiple
reference vls systems can be imported to the database, so that switching in different *Borrelia* strains can be
compared.
1. Align the cassette sequences to the reference and to each other.
1. Import the reads, categorizing bins of reads with sample labels (Eg. to distinguish replicates, time points, strains, etc.).
1. Align reads to the reference and cassettes. Aligning the cassettes first means that variations in
the read sequences will map to the reference in the same ways that those same variants in the silent
cassettes map to the reference.
1. Choose a type of analysis to do and how to group the data for the analysis.

## Dependencies
VAST requires [Python 3.4+](https://www.python.org) and the following packages:
+ [NumPy](https://github.com/numpy/numpy)
+ [SciPy](https://github.com/scipy/scipy)
+ [pandas](http://pandas.pydata.org/)
+ [matplotlib](http://matplotlib.org/index.html)
+ [Biopython](https://github.com/biopython/biopython)
+ [pysam](https://github.com/pysam-developers/pysam) is not strictly required, but if absent, no BAM-format outputs will
be produced.

## Installation

1. Clone the repository.

    ```shell
    git clone https://github.com/verheytb/vast
    ```
1. Create a link to vast.py in a directory in your [PATH](http://www.linfo.org/path_env_var.html) so that it can be
accessible from anywhere. (Linux only)

    ```shell
    ln -s vast/vast.py /usr/local/bin/vast
    ```

1. Run VAST as follows:

    ```shell
    vast
    ```

## Documentation
Documentation is found on the [wiki](https://github.com/verheytb/vast/wiki).
