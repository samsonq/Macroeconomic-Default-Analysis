"""
Perform ETL on used datasets. Methods to pull, clean, and reformat data. Mainly useful to extract
large datasets from LendingClub source and load them into a format workable with Pandas. Preprocessing
methods are contained in the preprocessing.py file.
"""
import pandas as pd
import re
import json


def remove_header(path, filename):
    """
    Removes annoying link header on data sets.

    :param path: path to data
    :param filename: file name to save as
    """
    with open(path, "r") as r:
        next_line = next(r)
        with open(".././data/LendingClub/processed/" + filename, "w") as w:
            for line in r:
                w.write(line)
