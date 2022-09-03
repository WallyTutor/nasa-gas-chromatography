# -*- coding: utf-8 -*-
from pathlib import Path
from zipfile import ZipFile
from IPython import embed
import matplotlib.pyplot as plt
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


# # Check signature of pure compounds.
# for cid_group, cid_list in enumerate(compounds_ids):
#     print(f"Processing cid_group = {cid_group}")
#     cid_data = get_all_data_pure_compound(cid_list)

#     plt.close("all")
#     plt.style.use("seaborn-white")

#     fig, ax = plt.subplots()

#     for k, df in enumerate(cid_data):
#         # Here we don't consider different exit times for same mass
#         # so that data is ordered by mass, that is checked elsewhere.
#         df = df.sort_values("mass")

#         x = df["mass"].to_numpy()
#         y = df["intensity"].to_numpy()
#         ax.plot(x, y)

#     ax.grid(linestyle=":")
#     fig.tight_layout()
#     plt.savefig(DIR_PROC / f"phase1/pic{cid_group}", dpi=200)


embed(colors="Linux")
