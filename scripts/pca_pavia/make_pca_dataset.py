"""Create PCA-reduced versions of the Pavia University train/test dataset.

PCA is fitted on the training features only (``PCA(svd_solver="full")``, no
whitening) and then applied to both the training and the test samples. The
saved values are raw PCA scores: they are NOT rescaled to 0-1 and may be
negative.

For every requested number of components ``n`` the script writes the folder
``<output-root>/PCA<n>-Pavia/`` with:

* ``train_samples.dat``  - GAwLL-style text file (6 header lines, LF endings)
* ``test_samples.npy``   - float64, row order preserved
* ``test_targets.npy``   - unchanged copy of the source labels
* ``pca_model.npz``      - components, mean, explained variance ratio
* ``metadata.json``      - n components, explained variance, source hashes,
                           library versions

The script has no default paths and does not depend on the working directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.decomposition import PCA

TRAIN_NAME = "train_samples.dat"
TEST_SAMPLES_NAME = "test_samples.npy"
TEST_TARGETS_NAME = "test_targets.npy"
MODEL_NAME = "pca_model.npz"
METADATA_NAME = "metadata.json"
OUTPUT_FILES = (TRAIN_NAME, TEST_SAMPLES_NAME, TEST_TARGETS_NAME, MODEL_NAME, METADATA_NAME)
HEADER_LINES = 6


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_train(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a GAwLL-style .dat file (header lines, then features + label)."""
    data = np.loadtxt(path, skiprows=HEADER_LINES, dtype=np.float64)
    features, labels = data[:, :-1], data[:, -1]
    if not np.array_equal(labels, np.round(labels)):
        raise ValueError(f"{path}: last column is not made of integer labels")
    return features, labels.astype(np.int64)


def write_train_dat(path: Path, n: int, features: np.ndarray, labels: np.ndarray) -> None:
    """Write the training set in the source format, with LF line endings."""
    header = (
        f"MODEL: PCA{n}PAVIA \n"
        "TYPE: 1 \n"
        f"N_ATTRIBUTES: {n} \n"
        f"N_EXAMPLES: {features.shape[0]} \n"
        f"N_CLASSES: {len(np.unique(labels))} \n"
        "DATASET:\n"
    )
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        for row, label in zip(features, labels):
            line = " ".join(f"{value:.10f}" for value in row) + f" {int(label)}\n"
            handle.write(line.encode("ascii"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit PCA on training data only and write PCA-reduced train/test datasets."
    )
    parser.add_argument("--source-dir", type=Path, required=True,
                        help=f"folder with {TRAIN_NAME}, {TEST_SAMPLES_NAME} and {TEST_TARGETS_NAME}")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="folder in which PCA<n>-Pavia/ directories are created")
    parser.add_argument("--n-components", type=int, nargs="+", required=True, metavar="N",
                        help="one or more numbers of principal components")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow writing into an existing output folder (default: error)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    source_dir: Path = args.source_dir
    output_root: Path = args.output_root
    sources = {name: source_dir / name for name in (TRAIN_NAME, TEST_SAMPLES_NAME, TEST_TARGETS_NAME)}
    for name, path in sources.items():
        if not path.is_file():
            parser.error(f"missing source file: {path}")

    # Preserve the given order, drop duplicates.
    sizes = list(dict.fromkeys(args.n_components))

    train_features, train_labels = load_train(sources[TRAIN_NAME])
    n_features = train_features.shape[1]
    for n in sizes:
        if not 1 <= n <= n_features:
            parser.error(f"--n-components value {n} is outside 1..{n_features} (the number of features)")

    # Fail before doing any work if an output folder is already there.
    out_dirs = {n: output_root / f"PCA{n}-Pavia" for n in sizes}
    if not args.overwrite:
        existing = [str(d) for d in out_dirs.values() if d.exists()]
        if existing:
            print("error: output folder already exists (use --overwrite to replace its files): "
                  + ", ".join(existing), file=sys.stderr)
            return 1

    test_samples = np.load(sources[TEST_SAMPLES_NAME])
    test_targets = np.load(sources[TEST_TARGETS_NAME])
    if test_samples.ndim != 2 or test_samples.shape[1] != n_features:
        parser.error(f"test samples have shape {test_samples.shape}, expected (*, {n_features})")
    if test_targets.shape != (test_samples.shape[0],):
        parser.error(f"test targets have shape {test_targets.shape}, expected ({test_samples.shape[0]},)")

    source_hashes = {name: sha256_of(path) for name, path in sources.items()}
    print(f"train {train_features.shape}, test {test_samples.shape}, targets {test_targets.shape}")

    for n in sizes:
        out_dir = out_dirs[n]
        out_dir.mkdir(parents=True, exist_ok=args.overwrite)

        pca = PCA(n_components=n, svd_solver="full", whiten=False)
        train_pca = pca.fit(train_features).transform(train_features)  # fitted on train only
        test_pca = np.ascontiguousarray(pca.transform(test_samples), dtype=np.float64)
        total_variance = float(np.sum(pca.explained_variance_ratio_))

        write_train_dat(out_dir / TRAIN_NAME, n, train_pca, train_labels)
        np.save(out_dir / TEST_SAMPLES_NAME, test_pca)
        np.save(out_dir / TEST_TARGETS_NAME, test_targets)
        np.savez(out_dir / MODEL_NAME, components=pca.components_, mean=pca.mean_,
                 explained_variance_ratio=pca.explained_variance_ratio_)
        metadata = {
            "n_components": n,
            "total_explained_variance": total_variance,
            "source_sha256": source_hashes,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
        }
        (out_dir / METADATA_NAME).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"PCA{n}: total explained variance {total_variance:.6f} -> {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
