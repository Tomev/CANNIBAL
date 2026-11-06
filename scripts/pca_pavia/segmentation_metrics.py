"""Segmentation metrics of random forests on the selected bands of PCA-reduced Pavia University.

For each dataset folder (as written by ``make_pca_dataset.py``) the script reads
``<results-root>/<dataset folder name>/bands/manifest.csv`` (as written by
``select_bands.py``), trains a random forest on the selected bands of the
training samples and scores it on the test pixels with a known class. The
protocol is the one of ``scripts/runner.py``: ``RandomForestClassifier(
n_estimators=4, random_state=seed)``, default settings otherwise, the same
scikit-learn metric calls, raw (unscaled) PCA values, background (class 0)
removed from the test set.

Pairing of band sets and forest seeds:

* ``random`` has one band set per seed (0-19) and count; band set ``i`` is
  evaluated with forest seed ``i`` only (20 runs per count).
* every other method has one band set per configuration, evaluated with forest
  seeds 0-19 (20 runs per configuration). The manifest ``seed`` of ``bombs``
  rows is the retry seed of the selection; it is kept as ``band_seed`` for
  information only.

Results go to ``<results-root>/<dataset folder name>/metrics/``; see the README
("Segmentation metrics on the PCA datasets") for the layout. Rerunning
overwrites that folder.

The script does not depend on the working directory.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True  # tracked .pyc files exist in the repo; keep them untouched

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "experimental_results" / "pca_pavia"

TRAIN_NAME = "train_samples.dat"
TEST_SAMPLES_NAME = "test_samples.npy"
TEST_TARGETS_NAME = "test_targets.npy"
HEADER_LINES = 6
EXPECTED_TEST_PIXELS = 40641  # test pixels with a known class (background removed)
METHODS = ("bombs", "mi", "random", "cannibal", "affinity_prop", "gawll_sel")
N_ESTIMATORS = 4
FOREST_SEEDS = range(20)
METRICS = (
    "acc", "avg_acc",
    "f1_macro", "f1_micro", "f1_weighted",
    "precision_macro", "precision_micro", "precision_weighted",
    "recall_macro", "recall_micro", "recall_weighted",
    "time",
)
RUN_COLUMNS = (
    "dataset", "method", "requested_n", "returned_n", "band_seed", "rf_seed", "status", "band_file",
) + METRICS
SUMMARY_COLUMNS = (
    ("dataset", "method", "requested_n", "returned_n", "status", "n_runs")
    + tuple(f"{m}_{stat}" for m in METRICS for stat in ("mean", "std"))
)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``x_train, y_train, x_test, y_test``; test pixels of class 0 are dropped."""
    data = np.loadtxt(path / TRAIN_NAME, skiprows=HEADER_LINES, dtype=np.float64)
    x_train = data[:, :-1]
    labels = data[:, -1]
    if not np.array_equal(labels, np.round(labels)):
        raise ValueError(f"{path / TRAIN_NAME}: last column is not made of integer labels")
    y_train = labels.astype(np.int64)

    x_test = np.load(path / TEST_SAMPLES_NAME)
    y_test = np.load(path / TEST_TARGETS_NAME).astype(np.int64)
    if x_test.shape[0] != y_test.shape[0] or x_test.shape[1] != x_train.shape[1]:
        raise ValueError(f"{path}: train and test shapes do not match")
    known = y_test != 0
    x_test, y_test = x_test[known], y_test[known]
    if y_test.shape[0] != EXPECTED_TEST_PIXELS:
        raise SystemExit(
            f"{path}: {y_test.shape[0]} test pixels with a known class, "
            f"expected {EXPECTED_TEST_PIXELS}"
        )
    return x_train, y_train, x_test, y_test


def load_bands(results_root: Path, band_file: str, row: dict[str, str], k: int) -> np.ndarray:
    bands = np.atleast_1d(np.loadtxt(results_root / band_file, dtype=np.int64))
    if bands.size != int(row["returned_n"]):
        raise ValueError(f"{band_file}: {bands.size} bands, manifest says {row['returned_n']}")
    if np.unique(bands).size != bands.size or bands.min() < 0 or bands.max() >= k:
        raise ValueError(f"{band_file}: band indices must be distinct and within 0-{k - 1}")
    return bands


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
    bands: np.ndarray, seed: int,
) -> tuple[dict[str, float], int]:
    """Fit a forest on ``bands``, score it on the test set.

    Returns the metrics (``time`` covers fit, predict and metrics) and the number of
    ``UndefinedMetricWarning`` raised by the metric calls. Other warnings are passed on.
    """
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=seed)
        model.fit(x_train[:, bands], y_train)
        y_pred = model.predict(x_test[:, bands])
        metrics = {
            "acc": accuracy_score(y_true=y_test, y_pred=y_pred),
            "avg_acc": balanced_accuracy_score(y_true=y_test, y_pred=y_pred),
        }
        for name, score in (("f1", f1_score), ("precision", precision_score), ("recall", recall_score)):
            for average in ("macro", "micro", "weighted"):
                metrics[f"{name}_{average}"] = score(y_true=y_test, y_pred=y_pred, average=average)
        metrics["time"] = time.perf_counter() - started
    for item in caught:  # show the warnings the way Python would have
        warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
    undefined = sum(issubclass(item.category, UndefinedMetricWarning) for item in caught)
    return {name: float(metrics[name]) for name in METRICS}, undefined


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

def _n_key(row: dict[str, str]) -> int:
    return -1 if row["requested_n"] == "auto" else int(row["requested_n"])


def _seed_key(row: dict[str, str]) -> int:
    return -1 if row["seed"] == "" else int(row["seed"])


def read_manifest(results_root: Path, name: str) -> list[dict[str, str]]:
    """Usable manifest rows, sorted by method, count and seed."""
    path = results_root / name / "bands" / "manifest.csv"
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    usable = []
    for row in rows:
        if row["dataset"] != name:
            raise ValueError(f"{path}: row of dataset {row['dataset']!r} in the manifest of {name}")
        if row["status"] == "failed" or not row["band_file"]:
            print(f"  skipping {row['method']} requested_n={row['requested_n']}: "
                  f"status {row['status']!r}, no band file")
            continue
        usable.append(row)
    usable.sort(key=lambda r: (METHODS.index(r["method"]), _n_key(r), _seed_key(r)))
    random_rows = [r for r in usable if r["method"] == "random"]
    for n in {r["requested_n"] for r in random_rows}:
        seeds = sorted(int(r["seed"]) for r in random_rows if r["requested_n"] == n)
        if seeds != list(FOREST_SEEDS):
            raise ValueError(f"{path}: random n={n} has seeds {seeds}, expected 0-19")
    return usable


def forest_seeds_for(row: dict[str, str]) -> list[int]:
    """The manifest seed of a random band set is its pairing key; elsewhere all 20 seeds run."""
    return [int(row["seed"])] if row["method"] == "random" else list(FOREST_SEEDS)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def summarize(runs: list[dict]) -> list[dict]:
    """One row per method and requested count."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        groups.setdefault((run["method"], run["requested_n"]), []).append(run)
    summary = []
    for (method, requested), group in groups.items():
        for column in ("returned_n", "status"):
            if len({g[column] for g in group}) != 1:
                raise ValueError(f"{method} n={requested}: mixed {column} within one group")
        row = {
            "dataset": group[0]["dataset"], "method": method, "requested_n": requested,
            "returned_n": group[0]["returned_n"], "status": group[0]["status"],
            "n_runs": len(group),
        }
        for metric in METRICS:
            values = np.array([g[metric] for g in group])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        summary.append(row)
    return summary  # runs are already sorted, and dicts keep the insertion order


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter=";", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run_dataset(dataset_dir: Path, results_root: Path) -> None:
    name = dataset_dir.name
    print(f"== {name}")
    manifest = read_manifest(results_root, name)
    x_train, y_train, x_test, y_test = load_dataset(dataset_dir)
    k = x_train.shape[1]
    print(f"  train {x_train.shape}, test {x_test.shape} (background removed), "
          f"{len(manifest)} band sets")

    runs: list[dict] = []
    undefined_by_method: Counter = Counter()
    runs_by_method: Counter = Counter()
    for row in manifest:
        bands = load_bands(results_root, row["band_file"], row, k)
        for seed in forest_seeds_for(row):
            metrics, undefined = evaluate(x_train, y_train, x_test, y_test, bands, seed)
            runs.append({
                "dataset": name, "method": row["method"], "requested_n": row["requested_n"],
                "returned_n": row["returned_n"], "band_seed": row["seed"], "rf_seed": seed,
                "status": row["status"], "band_file": row["band_file"], **metrics,
            })
            undefined_by_method[row["method"]] += undefined
            runs_by_method[row["method"]] += 1
    runs.sort(key=lambda r: (
        METHODS.index(r["method"]),
        -1 if r["requested_n"] == "auto" else int(r["requested_n"]),
        -1 if r["band_seed"] == "" else int(r["band_seed"]),
        r["rf_seed"],
    ))

    out_dir = results_root / name / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "all_runs.csv", RUN_COLUMNS, runs)
    write_csv(out_dir / "summary.csv", SUMMARY_COLUMNS, summarize(runs))

    for method in METHODS:
        if runs_by_method[method]:
            print(f"  {method}: {runs_by_method[method]} runs, "
                  f"{undefined_by_method[method]} UndefinedMetricWarning")
    print(f"  total: {len(runs)} runs, {sum(undefined_by_method.values())} UndefinedMetricWarning")
    print(f"  metrics: {out_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dataset-dir", type=Path, nargs="+", required=True,
        help="PCA dataset folder(s) holding train_samples.dat, test_samples.npy and test_targets.npy.",
    )
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
        help=f"Folder with the band selection results (default: {DEFAULT_RESULTS_ROOT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results_root = args.results_root.resolve()
    dataset_dirs = [path.resolve() for path in args.dataset_dir]
    if len({d.name for d in dataset_dirs}) != len(dataset_dirs):
        raise SystemExit("dataset folders must have distinct names")
    for dataset_dir in dataset_dirs:
        run_dataset(dataset_dir, results_root)


if __name__ == "__main__":
    main()
