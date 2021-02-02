import os
import pickle
import re
from glob import glob
from pathlib import Path


def save_pickled(obj, file):
    file = _file_ending(file, "p")
    create_directories(file)
    with open(f"{file}", 'wb') as f:
        pickle.dump(obj, f)


def load_pickled(file):
    file = _file_ending(file, "p")
    with open(file, 'rb') as f:
        return pickle.load(f)


def _file_ending(file, ending):
    return f"{file}.{ending}" if f".{ending}" not in file else file


def create_directories(path):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)


def list_file_numbering(directory, prefix, suffix=None):
    """
    Finds all files in the specified directory that match the given {prefix}{number}{suffix} pattern.
    All found {number}s are returned as a list in ascending order.

    Parameters
    ----------
        directory: where to search for file numberings
        prefix: prefix of files to be considered
        suffix: (optional) suffix of files to be considered

    Returns
    -------
        a list of numbers (without leading zeros) that appear in the matched file names in between `prefix` and `suffix`.
    """
    if suffix is None or suffix.count('.') == 1 and suffix[0] == '.':
        suffix = ""
    regex = re.compile(f"{prefix}(-?\d+){Path(suffix).stem}")
    file_names = glob(f"{directory}/{prefix}*{suffix}")
    file_names = [Path(file_name).stem for file_name in file_names]
    numbering = sorted([int(regex.search(file_name).group(1)) for file_name in file_names if regex.match(file_name)])
    return numbering


def generate_run_name(directory, run_prefix):
    """
    Assumes that runs will be stored in folder run_dir and have format "{run_prefix}-{run_id}".
    Generates a new run name by searching for existing runs and adding 1 to the one with the highest ID
    """
    run_ids = list_file_numbering(directory, f"{run_prefix}-")
    # regex = re.compile(f"{run_prefix}-(\d+)$")
    # run_names = glob(f"{run_dir}/{run_prefix}-*")
    # run_names = [Path(run_name).stem for run_name in run_names]
    # run_ids = [int(regex.search(run_name).group(1)) for run_name in run_names if regex.match(run_name)]

    run_id = max(run_ids) + 1 if len(run_ids) > 0 else 1
    return f"{run_prefix}-{run_id}"
