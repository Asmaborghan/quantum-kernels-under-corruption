"""Corruption of training data: missing values, measurement noise and label flips.

Rules for every function:
  1. It works on a copy; the input is never changed.
  2. A fixed seed gives the same result every time.
  3. Only training rows are corrupted; the experiment code never passes validation or test rows.
  4. Severities are nested: with the same seed, everything damaged at a low level is also damaged at every
     higher level (noise is one random draw, scaled up). Levels differ only by the extra damage.

Measurement noise is added only to measured quantities (amounts, sizes, durations, ages, counts).
Category codes, yes/no flags, ordered levels and calendar positions get no noise; they are listed below.
"""
import zlib
import numpy as np
import pandas as pd
from paths import PROCESSED

MCAR_RATES = (0.10, 0.20, 0.30)     # share of cells hidden
NOISE_LEVELS = (0.05, 0.15, 0.30)   # noise SD = level x feature SD
FLIP_RATES = (0.05, 0.10, 0.15)     # share of labels flipped
LEVELS = {"missing": MCAR_RATES, "noise": NOISE_LEVELS, "labels": FLIP_RATES}

# Columns that are measured quantities in German Credit; its other 21 columns are integer codes
# for categories and ordered levels (dataset documentation).
GERMAN_MEASURED = ["f1", "f3", "f9"]          # duration (months), credit amount, age (years)

# Code columns of the other datasets (no noise). Datasets not listed are measured in every column.
CODE_COLUMNS = {
    "concrete": ["Age (day)"],                                       # fixed test schedule, 12 set values
    "steel_faults": ["TypeOfSteel_A400", "Outside_Global_Index"],     # yes/no flag, 3-level index
    "bank_marketing": ["job", "marital", "education", "default", "housing", "loan", "contact",
                       "poutcome", "day", "month"],
    "electricity": ["day", "period"],                                 # weekday, half-hour of the day
}


def feature_names(dataset):
    """Feature names of a prepared dataset, in column order."""
    return [c for c in pd.read_csv(PROCESSED / f"{dataset}_train_ready.csv", nrows=0).columns if c != "target"]


def noise_columns(dataset):
    """True for every feature that receives measurement noise."""
    names = feature_names(dataset)
    if dataset == "german_credit":
        return np.array([n in GERMAN_MEASURED for n in names])
    return np.array([n not in CODE_COLUMNS.get(dataset, []) for n in names])


def corruption_seed(dataset, kind, base_seed, fold):
    """One fixed seed per (dataset, corruption type, CV seed, fold). The level is not part of it,
    so the levels stay nested."""
    return zlib.crc32(f"{dataset}|{kind}|{base_seed}|{fold}".encode())


# ---------------- 1. Missing values (MCAR) + training-median fill ----------------
def add_mcar(X, rate, seed):
    """Hide exactly round(rate x cells) values, chosen completely at random. Returns (X_with_nan, mask)."""
    X = np.array(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix (rows x features).")
    if not 0 <= rate < 1:
        raise ValueError("rate must be in [0, 1).")
    if np.isnan(X).any():
        raise ValueError("X already contains missing values.")
    order = np.random.default_rng(seed).permutation(X.size)
    mask = np.zeros(X.size, dtype=bool)
    mask[order[:int(round(rate * X.size))]] = True
    mask = mask.reshape(X.shape)
    X[mask] = np.nan
    return X, mask


def fit_median_imputer(X_train):
    """One median per feature from the (corrupted) training rows, missing cells ignored."""
    med = np.nanmedian(np.asarray(X_train, dtype=float), axis=0)
    if np.isnan(med).any():
        raise ValueError("A feature is missing in every training row; cannot fill it.")
    return med


def apply_median_imputer(X, medians):
    """Fill missing cells with the given medians (returns a copy)."""
    X = np.array(X, dtype=float)
    rows, cols = np.where(np.isnan(X))
    X[rows, cols] = medians[cols]
    return X


def mcar_then_impute(X_train, rate, seed):
    """Hide cells, then fill them with medians fitted on the same training rows."""
    X_nan, mask = add_mcar(X_train, rate, seed)
    medians = fit_median_imputer(X_nan)
    return apply_median_imputer(X_nan, medians), mask, medians


# ---------------- 2. Measurement noise (Gaussian) ----------------
def add_gaussian_noise(X, level, seed, columns=None):
    """Add N(0, (level x SD_j)^2) to every cell of each measured feature j (SD from the rows given).
    `columns` is a boolean mask of the features that receive noise (default: all).
    The random draw covers the whole matrix, so the noise in a measured column does not depend on
    which other columns are measured. No clipping: noisy values may leave [0, pi], like real errors.
    Returns (X_noisy, noise, sd)."""
    X = np.array(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix (rows x features).")
    if level < 0:
        raise ValueError("level must be >= 0.")
    if np.isnan(X).any():
        raise ValueError("X contains missing values.")
    columns = np.ones(X.shape[1], dtype=bool) if columns is None else np.asarray(columns, dtype=bool)
    if columns.shape != (X.shape[1],):
        raise ValueError("columns must have one entry per feature.")
    sd = X.std(axis=0, ddof=1)
    z = np.random.default_rng(seed).standard_normal(X.shape)     # same draw for every level (nested)
    noise = z * (level * sd) * columns
    return X + noise, noise, sd


# ---------------- 3. Label noise (random flips) ----------------
def flip_labels(y, rate, seed):
    """Flip exactly round(rate x n) binary labels, chosen at random (symmetric). Returns (y_flipped, idx)."""
    y = np.array(y).astype(int)
    if y.ndim != 1:
        raise ValueError("y must be a 1D label vector.")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("labels must be 0/1.")
    if not 0 <= rate < 0.5:
        raise ValueError("rate must be in [0, 0.5).")
    idx = np.sort(np.random.default_rng(seed).permutation(y.size)[:int(round(rate * y.size))])
    y[idx] = 1 - y[idx]
    return y, idx


# ---------------- one entry point for the experiments ----------------
def corrupt_training(X_train, y_train, kind, level, seed, noise_cols=None):
    """Corrupt one training fold. Returns (X_corrupted, y_corrupted).
      missing : hide cells completely at random, fill them with this fold's medians
      noise   : Gaussian noise on the measured columns (`noise_cols`), SD from this fold
      labels  : random label flips, features unchanged
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train).astype(int)
    if kind == "missing":
        return mcar_then_impute(X_train, level, seed)[0], y_train.copy()
    if kind == "noise":
        return add_gaussian_noise(X_train, level, seed, noise_cols)[0], y_train.copy()
    if kind == "labels":
        return X_train.copy(), flip_labels(y_train, level, seed)[0]
    raise ValueError(f"Unknown corruption kind: {kind!r} (use {list(LEVELS)})")
