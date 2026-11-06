"""Run every band selection method on PCA-reduced Pavia University datasets.

For each dataset folder (as written by ``make_pca_dataset.py``) the script
selects bands with six methods, using the TRAINING data only (the test files
are never opened):

* ``bombs``         existing code in ``experimental_results/bombs`` (subprocess)
* ``mi``            existing code in ``experimental_results/mi``
* ``random``        uniform random, 20 seeds (0-19) per requested count
* ``cannibal``      spectral clustering of the GAwLL variable interaction graph
* ``affinity_prop`` affinity propagation on the same graph (chooses its count)
* ``gawll_sel``     bands set in the first GAwLL run (chooses its count)

Parametric methods run for every band count from 5 to the number of features
``k``. At ``n == k`` all bands are recorded without running the algorithm.

Results go to ``<results-root>/<dataset folder name>/bands/`` together with a
``manifest.csv``; see the README ("PCA datasets") for the layout. GAwLL outputs
are kept in ``<results-root>/<dataset folder name>/gawll/`` and are reused on
reruns unless ``--force`` is given.

The script does not depend on the working directory.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True  # tracked .pyc files exist in the repo; keep them untouched
os.environ.setdefault("MPLBACKEND", "Agg")  # non-GUI backend for the mi code

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from random_bands import select_random_bands  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTAL_DIR = REPO_ROOT / "experimental_results"
GAWLL_SRC_DIR = REPO_ROOT / "gawll_with_hu_and_seg"
DEFAULT_RESULTS_ROOT = EXPERIMENTAL_DIR / "pca_pavia"

TRAIN_NAME = "train_samples.dat"
HEADER_LINES = 6
METHODS = ("bombs", "mi", "random", "cannibal", "affinity_prop", "gawll_sel")
GAWLL_METHODS = ("cannibal", "affinity_prop", "gawll_sel")
MIN_BANDS = 5
RANDOM_SEEDS = range(20)
BOMBS_MAX_ATTEMPTS = 20
GAWLL_OUTPUT_PATTERNS = ("eVIG_*_r*.csv", "bind_*", "bfi_*", "gen_*", "nedges_*", "time_*")
SCALER_NAME = "scaler.json"
MANIFEST_COLUMNS = (
    "dataset", "method", "requested_n", "returned_n", "seed", "status", "band_file", "note",
)

# Runs inside the bombs subprocess. Its load_data ignores its arguments and reads
# ./others/dataset_pavia.dat, so the working directory is a temp folder.
BOMBS_WORKER = """
import random, sys
import numpy as np
seed, n, dest = int(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.argv = ["bombs", "--bands_per_antibody", n, "--dest_path", dest]
from bombs.runner import main
from bombs.utils import arguments
random.seed(seed)
np.random.seed(seed)
main(arguments())
"""


class Dataset:
    """Training data of one PCA dataset folder."""

    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        train_path = path / TRAIN_NAME
        with open(train_path) as handle:
            self.header = [handle.readline() for _ in range(HEADER_LINES)]
        data = np.loadtxt(train_path, skiprows=HEADER_LINES, dtype=np.float64)
        self.x = data[:, :-1]
        labels = data[:, -1]
        if not np.array_equal(labels, np.round(labels)):
            raise ValueError(f"{train_path}: last column is not made of integer labels")
        self.y = labels.astype(np.int64)
        self.n_samples, self.k = self.x.shape
        attributes = _header_value(self.header, "N_ATTRIBUTES")
        examples = _header_value(self.header, "N_EXAMPLES")
        if attributes != self.k or examples != self.n_samples:
            raise ValueError(f"{train_path}: header does not match the data shape")
        self.col_min = self.x.min(axis=0)
        self.col_max = self.x.max(axis=0)
        if np.any(self.col_max <= self.col_min):
            raise ValueError(f"{train_path}: a component is constant, cannot min-max scale it")
        self.x_scaled = (self.x - self.col_min) / (self.col_max - self.col_min)
        self.gawll_name = f"dataset_pca{self.k}_pavia"

    def write_scaled_dat(self, path: Path) -> None:
        """Write the min-max scaled training data in the GAwLL .dat format."""
        with open(path, "w", newline="\n") as handle:
            handle.writelines(self.header)
            for row, label in zip(self.x_scaled, self.y):
                handle.write(" ".join(f"{value:1.6f}" for value in row) + f" {label}\n")


def _header_value(header: list[str], key: str) -> int:
    for line in header:
        if line.startswith(key + ":"):
            return int(line.split(":", 1)[1])
    raise ValueError(f"header has no {key}")


# --------------------------------------------------------------------------- #
# Selection functions
# --------------------------------------------------------------------------- #

def cannibal_bands(matrix: np.ndarray, n: int) -> list[int]:
    """CANNIBAL selection: spectral clustering on the GAwLL graph, one band per cluster.

    Same procedure as ``get_band_indices_based_on_VIGs`` in ``scripts/runner.py``.
    """
    from sklearn.cluster import SpectralClustering

    labels = SpectralClustering(
        n_clusters=n, affinity="precomputed", random_state=0
    ).fit_predict(matrix)
    bands = []
    for cluster in range(n):
        members = np.where(labels == cluster)[0]
        if members.size == 0:
            raise RuntimeError(f"spectral clustering returned an empty cluster ({cluster} of {n})")
        bands.append(int(members[np.argmax(matrix[members].sum(axis=1))]))
    return sorted(bands)


def affinity_prop_bands(matrix: np.ndarray) -> list[int]:
    """Exemplars of affinity propagation, as in ``get_affinity_prop_clusters``."""
    from sklearn.cluster import affinity_propagation

    return sorted(int(i) for i in affinity_propagation(matrix, random_state=0)[0])


def gawll_sel_bands(bind_path: Path) -> list[int]:
    """Indices of the 1s in the first row (run 0) of a GAwLL ``bind_*`` file."""
    rows = np.loadtxt(bind_path, ndmin=2)
    return [int(i) for i in np.flatnonzero(rows[0] == 1)]


def _load_mi_module():
    path = EXPERIMENTAL_DIR / "mi" / "runner.py"
    spec = importlib.util.spec_from_file_location("cannibal_mi_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MiSelector:
    """Mutual-information selection, with a ranking fallback where the original asserts."""

    def __init__(self, dataset: Dataset):
        self.module = _load_mi_module()
        self.dataset = dataset
        self.ranking = self._rank()

    def _fresh(self, n: int):
        selector = self.module.MutualInformation(designed_band_size=n, bandwidth=0, eta=0)
        # prep_bands drops label 0 after the shift, so the first class is excluded (known quirk).
        selector.prep_bands(data=self.dataset.x.copy(), ref_map=self.dataset.y - 1)
        selector.calculate_mi(dest_path=None)
        return selector

    def _rank(self) -> list[int]:
        scores = [float(s) for s in self._fresh(n=1).return_mi_scores()]
        return sorted(range(len(scores)), key=lambda i: (-scores[i], i))

    def select(self, n: int) -> tuple[list[int], str]:
        top = sorted(self.ranking[:n])
        try:
            selector = self._fresh(n)
            selector.perform_search()
        except AssertionError:
            return top, "mi_ranking_fill"
        original = sorted(int(i) for i in selector.set_of_selected_bands)
        if original != top:
            raise RuntimeError(
                f"mi: original result {original} differs from the top-{n} ranking {top}"
            )
        return original, "selected"


def bombs_bands(
    workdir: Path, n: int, bombs_env: dict[str, str]
) -> tuple[list[int] | None, int | None, str]:
    """Run bombs for ``n`` bands; retry with the next seed on failure.

    Returns (bands, seed used, note). bands is None if all attempts fail.
    A band file left by a failed attempt is never used.
    """
    failed_seeds = []
    for seed in range(BOMBS_MAX_ATTEMPTS):
        dest = workdir / f"out_n{n}_seed{seed}"
        proc = subprocess.run(
            [sys.executable, "-c", BOMBS_WORKER, str(seed), str(n), str(dest)],
            cwd=workdir, env=bombs_env, capture_output=True, text=True,
        )
        band_file = dest / f"best_individual_bands_{n}"
        if proc.returncode == 0 and band_file.exists():
            bands = [int(i) for i in np.atleast_1d(np.loadtxt(band_file, dtype=np.int64))]
            note = f"retried after failed seeds {failed_seeds}" if failed_seeds else ""
            return bands, seed, note
        failed_seeds.append(seed)
        last_error = proc.stderr.strip().splitlines()[-1:] or ["no stderr"]
        print(f"    bombs n={n} seed={seed} failed (exit {proc.returncode}): {last_error[0]}")
    return None, None, f"all {BOMBS_MAX_ATTEMPTS} seeds failed"


# --------------------------------------------------------------------------- #
# GAwLL
# --------------------------------------------------------------------------- #

def gawll_dir_for(results_root: Path, dataset: Dataset) -> Path:
    return results_root / dataset.name / "gawll"


def _gawll_complete(directory: Path, dataset: Dataset) -> bool:
    stem = f"{dataset.gawll_name}_c1_a1"
    needed = [
        f"eVIG_{stem}_r0.csv", f"eVIG_{stem}_r1.csv", f"bind_{stem}.dat", SCALER_NAME,
    ]
    return all((directory / name).is_file() for name in needed)


def _scaler_matches(directory: Path, dataset: Dataset) -> bool:
    saved = json.loads((directory / SCALER_NAME).read_text())
    return np.allclose(saved["min"], dataset.col_min, rtol=0, atol=0) and np.allclose(
        saved["max"], dataset.col_max, rtol=0, atol=0
    )


def build_gawll(build_dir: Path) -> Path:
    """Compile GAwLL in a temp copy of the sources; nothing is built inside the repo."""
    for pattern in ("*.cpp", "*.h", "makefile"):
        for source in GAWLL_SRC_DIR.glob(pattern):
            shutil.copy2(source, build_dir)
    subprocess.run(["make"], cwd=build_dir, check=True, capture_output=True, text=True)
    return build_dir / "gawll_fs"


def ensure_gawll(
    datasets: list[Dataset], results_root: Path, force: bool, tmp_root: Path
) -> None:
    """Make sure GAwLL outputs exist for every dataset, running missing ones concurrently."""
    to_run = []
    for dataset in datasets:
        directory = gawll_dir_for(results_root, dataset)
        if not force and _gawll_complete(directory, dataset):
            if not _scaler_matches(directory, dataset):
                raise RuntimeError(
                    f"{directory}: saved scaler differs from the training data; use --force"
                )
            print(f"GAwLL: reusing outputs in {directory}")
        else:
            to_run.append(dataset)
    if not to_run:
        return

    binary = build_gawll(_new_dir(tmp_root, "gawll_build"))
    jobs = []
    for dataset in to_run:
        run_dir = _new_dir(tmp_root, f"gawll_run_{dataset.name}")
        shutil.copy2(binary, run_dir)
        dataset.write_scaled_dat(run_dir / f"{dataset.gawll_name}.dat")
        log = open(run_dir / "gawll.log", "w")
        proc = subprocess.Popen(
            ["./gawll_fs", dataset.gawll_name, "1", "1"],
            cwd=run_dir, stdout=log, stderr=subprocess.STDOUT,
        )
        print(f"GAwLL: started {dataset.name} (pid {proc.pid}) in {run_dir}")
        jobs.append((dataset, run_dir, proc, log, time.time()))

    for dataset, run_dir, proc, log, started in jobs:  # wait for all before handling any
        proc.wait()
        log.close()
        print(f"GAwLL: {dataset.name} exited {proc.returncode} after {time.time() - started:.0f} s wall")

    for dataset, run_dir, proc, log, started in jobs:
        if proc.returncode != 0:
            raise RuntimeError(f"GAwLL failed for {dataset.name}; see {run_dir / 'gawll.log'}")
        directory = gawll_dir_for(results_root, dataset)
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        for pattern in GAWLL_OUTPUT_PATTERNS:
            for produced in sorted(run_dir.glob(pattern)):
                shutil.copy2(produced, directory)
        (directory / SCALER_NAME).write_text(
            json.dumps(
                {"min": dataset.col_min.tolist(), "max": dataset.col_max.tolist()}, indent=2
            ) + "\n"
        )
        if not _gawll_complete(directory, dataset):
            raise RuntimeError(f"GAwLL outputs incomplete for {dataset.name} in {run_dir}")
        for time_file in sorted(directory.glob("time_*")):
            print(f"GAwLL: {dataset.name} {time_file.name}: {time_file.read_text().strip()} (CPU s per run)")


def _new_dir(root: Path, prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix + "_", dir=root))


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

class Writer:
    """Writes band files and collects manifest rows for one dataset."""

    def __init__(self, results_root: Path, dataset: Dataset):
        self.results_root = results_root
        self.dataset = dataset
        self.bands_dir = results_root / dataset.name / "bands"
        self.rows: list[dict[str, str]] = []

    def add(
        self, method: str, requested, bands, seed, status: str, relative: str, note: str = ""
    ) -> None:
        path = self.bands_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if bands is not None:
            bands = np.array(sorted(int(b) for b in bands), dtype=np.int64)
            np.savetxt(path, bands, fmt="%d")
        self.rows.append({
            "dataset": self.dataset.name,
            "method": method,
            "requested_n": str(requested),
            "returned_n": "" if bands is None else str(len(bands)),
            "seed": "" if seed is None else str(seed),
            "status": status,
            "band_file": "" if bands is None else (path.relative_to(self.results_root)).as_posix(),
            "note": note,
        })

    def write_manifest(self, methods_run: set[str]) -> None:
        manifest = self.bands_dir / "manifest.csv"
        kept = []
        if manifest.exists():
            with open(manifest, newline="") as handle:
                kept = [r for r in csv.DictReader(handle) if r["method"] not in methods_run]
        rows = kept + self.rows
        rows.sort(key=_manifest_key)
        self.bands_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def _manifest_key(row: dict[str, str]):
    requested = row["requested_n"]
    return (
        METHODS.index(row["method"]),
        -1 if requested == "auto" else int(requested),
        -1 if row["seed"] == "" else int(row["seed"]),
    )


def _fmt(n: int) -> str:
    return f"{n:02d}"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run_dataset(
    dataset: Dataset, methods: list[str], results_root: Path, tmp_root: Path
) -> None:
    print(f"== {dataset.name}: {dataset.n_samples} training samples, {dataset.k} features")
    writer = Writer(results_root, dataset)
    counts = range(MIN_BANDS, dataset.k + 1)
    all_bands = list(range(dataset.k))

    if "bombs" in methods:
        workdir = _new_dir(tmp_root, f"bombs_{dataset.name}")
        (workdir / "others").mkdir()
        dataset.write_scaled_dat(workdir / "others" / "dataset_pavia.dat")
        bombs_env = dict(os.environ, PYTHONPATH=str(EXPERIMENTAL_DIR), PYTHONDONTWRITEBYTECODE="1")
        for n in counts:
            started = time.time()
            rel = f"bombs/n_{_fmt(n)}.txt"
            if n == dataset.k:
                writer.add("bombs", n, all_bands, None, "trivial_all_bands", rel)
                continue
            bands, seed, note = bombs_bands(workdir, n, bombs_env)
            if bands is None:
                writer.add("bombs", n, None, None, "failed", "", note)
            else:
                _check_count("bombs", n, bands)
                writer.add("bombs", n, bands, seed, "selected", rel, note)
            print(f"  bombs n={n}: {time.time() - started:.1f} s")

    if "mi" in methods:
        selector = MiSelector(dataset)
        for n in counts:
            started = time.time()
            rel = f"mi/n_{_fmt(n)}.txt"
            if n == dataset.k:
                writer.add("mi", n, all_bands, None, "trivial_all_bands", rel)
                continue
            bands, status = selector.select(n)
            _check_count("mi", n, bands)
            note = "original search asserts; top-n of the mutual information ranking" \
                if status == "mi_ranking_fill" else ""
            writer.add("mi", n, bands, None, status, rel, note)
            print(f"  mi n={n}: {status} ({time.time() - started:.1f} s)")

    if "random" in methods:
        for n in counts:
            trivial = n == dataset.k
            for seed in RANDOM_SEEDS:
                bands = select_random_bands(dataset.k, n, seed)
                writer.add(
                    "random", n, bands, seed,
                    "trivial_all_bands" if trivial else "selected",
                    f"random/n_{_fmt(n)}/seed_{seed:02d}.txt",
                )
        print(f"  random: {len(list(counts)) * len(RANDOM_SEEDS)} band sets")

    if any(m in methods for m in GAWLL_METHODS):
        gawll_dir = gawll_dir_for(results_root, dataset)
        stem = f"{dataset.gawll_name}_c1_a1"
        if "cannibal" in methods:
            matrix = np.loadtxt(gawll_dir / f"eVIG_{stem}_r1.csv", delimiter=",")
            if matrix.shape != (dataset.k, dataset.k):
                raise RuntimeError(f"eVIG has shape {matrix.shape}, expected {dataset.k} square")
            for n in counts:
                started = time.time()
                rel = f"cannibal/n_{_fmt(n)}.txt"
                if n == dataset.k:
                    writer.add("cannibal", n, all_bands, None, "trivial_all_bands", rel)
                    continue
                bands = cannibal_bands(matrix, n)
                _check_count("cannibal", n, bands)
                writer.add("cannibal", n, bands, None, "selected", rel)
                print(f"  cannibal n={n}: {time.time() - started:.1f} s")
        if "affinity_prop" in methods:
            matrix = np.loadtxt(gawll_dir / f"eVIG_{stem}_r1.csv", delimiter=",")
            bands = affinity_prop_bands(matrix)
            note = "count chosen by the method" + (
                f"; fewer than {MIN_BANDS} bands" if len(bands) < MIN_BANDS else "")
            writer.add("affinity_prop", "auto", bands, None, "selected", "affinity_prop/auto.txt", note)
            print(f"  affinity_prop: {len(bands)} bands")
        if "gawll_sel" in methods:
            bands = gawll_sel_bands(gawll_dir / f"bind_{stem}.dat")
            note = "count chosen by the method (run 0)" + (
                f"; fewer than {MIN_BANDS} bands" if len(bands) < MIN_BANDS else "")
            writer.add("gawll_sel", "auto", bands, None, "selected", "gawll_sel/auto.txt", note)
            print(f"  gawll_sel: {len(bands)} bands")

    writer.write_manifest(set(methods))
    print(f"  manifest: {writer.bands_dir / 'manifest.csv'}")


def _check_count(method: str, n: int, bands: list[int]) -> None:
    if len(set(bands)) != n or len(bands) != n:
        raise RuntimeError(f"{method}: requested {n} bands, got {sorted(bands)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dataset-dir", type=Path, nargs="+", required=True,
        help="PCA dataset folder(s) holding train_samples.dat (test files are never read).",
    )
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS),
        help="Methods to run (default: all six).",
    )
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
        help=f"Output root (default: {DEFAULT_RESULTS_ROOT}).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rerun GAwLL even if its outputs already exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    methods = [m for m in METHODS if m in set(args.methods)]
    results_root = args.results_root.resolve()
    datasets = [Dataset(path.resolve()) for path in args.dataset_dir]
    if len({d.name for d in datasets}) != len(datasets):
        raise SystemExit("dataset folders must have distinct names")

    with tempfile.TemporaryDirectory(prefix="select_bands_") as tmp:
        tmp_root = Path(tmp)
        if any(m in methods for m in GAWLL_METHODS):
            ensure_gawll(datasets, results_root, args.force, tmp_root)
        for dataset in datasets:
            run_dataset(dataset, methods, results_root, tmp_root)


if __name__ == "__main__":
    main()
