# -*- coding: utf-8 -*-
from pathlib import Path
import re
from zipfile import ZipFile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Relevant data paths.
DIR_DATA = Path("data")
DIR_ORIG = DIR_DATA / "orig"
DIR_PROC = DIR_DATA / "proc"

# Check sumission columns.
submission = pd.read_csv(DIR_ORIG / "submission_format.csv")
print(submission.columns)

# Check training labels.
train_labels = pd.read_csv(DIR_ORIG / "phase1/train_labels.csv")
print(train_labels.columns)

# Confirm label ordering in submission and training sets.
order_match = all(train_labels.columns == submission.columns)
print(f"Submission/labels are ordered in the same way: {order_match}")

# Load training data contents. List files with archive.namelist().
archive = ZipFile(DIR_ORIG / "phase1/train_features.zip")

# Identify how many compounds (get *pure* substance picture)
train_labels["n_comp"] = train_labels.iloc[:, 1:].sum(axis=1)
train_pures = train_labels.loc[train_labels.n_comp == 1]

# Get subsets per compound.
compounds_ids = [train_pures.loc[train_pures[col] != 0].sample_id
                 for col in train_pures.columns[1:]]


def get_row_data(cid):
    """ Read data for a single database row. """
    with archive.open(f"train_features/{cid}.csv") as fp:
        df = pd.read_csv(fp)
    return df


def get_all_data_pure_compound(cid_list):
    """ Read data for all labels in given list. """
    return [get_row_data(cid) for cid in cid_list]


# Get datasets of pure compounds.
compounds_dfs = [get_all_data_pure_compound(cid_list)
                 for cid_list in compounds_ids]

# Number of digits in mass spectra.
digits = 2


def reduce_mass_spectra(df, digits=digits):
    """ Reduce number of masses assuming detector tolerance. """
    df["mass"] = df.mass.round(digits)
    return df


# Let's get simpler spectra.
reduced_mass_dfs = [[reduce_mass_spectra(df) for df in group]
                    for group in compounds_dfs]


# Standardized mass index.
masses = np.arange(5.0, 600.0+1.0e-09, pow(10, -digits))


def get_mass_signature(df_orig):
    """ Get a time-independent mass spectra signature. """
    df = df_orig.copy()
    df.sort_values("mass", inplace=True)

    df = df.groupby("mass").sum()
    df.drop(columns=["time"], inplace=True)

    index = df.index.union(masses)
    df = df.reindex(index, fill_value=0.0)

    df.intensity /= df.intensity.max()
    df.intensity = np.clip(df.intensity, 0.0, 1.0)

    return df


# Get mass spectra signature of data.
mass_sign_dfs = [[get_mass_signature(df) for df in group]
                 for group in reduced_mass_dfs]


# Maximum number of spectra to plot (avoid overflow).
max_lines = 10

# Check signature of pure compounds.
for cid_group, cid_list in enumerate(mass_sign_dfs):
    print(f"Processing cid_group = {cid_group}")
    if not len(cid_list):
        print(f"cid_group {cid_group} is empty")
        continue

    plt.close("all")
    plt.style.use("seaborn-white")

    fig, ax = plt.subplots()

    for k, df in enumerate(cid_list):
        if k > max_lines:
            break

        x = df.index.to_numpy()
        y = df.intensity.to_numpy()
        ax.plot(x, y + 0.5 * k)

    ax.grid(linestyle=":")
    fig.tight_layout()

    plt.savefig(DIR_PROC / f"phase1/pic{cid_group}", dpi=200)


# from IPython import embed; embed(colors="Linux")
