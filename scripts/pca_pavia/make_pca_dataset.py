"""Create PCA-reduced versions of the Pavia University train/test dataset.

PCA is fitted on the training features only and applied to both the training and
the test samples. For every --n-components value N the script writes
<output-root>/PCA<N>-Pavia/ with train_samples.dat, test_samples.npy,
test_targets.npy, pca_model.npz and metadata.json. Folders are created if
missing and existing files are overwritten.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.decomposition import PCA


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_train(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a GAwLL-style .dat file (6 header lines, then features + label)."""
    data = np.loadtxt(path, skiprows=6, dtype=np.float64)
    return data[:, :-1], data[:, -1].astype(np.int64)


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
    with open(path, "w", newline="\n") as f:
        f.write(header)
        for row, label in zip(features, labels):
            f.write(" ".join(f"{value:.10f}" for value in row) + f" {label}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="folder with train_samples.dat, test_samples.npy and test_targets.npy")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--n-components", type=int, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_path = args.source_dir / "train_samples.dat"
    test_samples_path = args.source_dir / "test_samples.npy"
    test_targets_path = args.source_dir / "test_targets.npy"

    train_features, train_labels = load_train(train_path)
    test_samples = np.load(test_samples_path)
    test_targets = np.load(test_targets_path)
    source_sha256 = {path.name: sha256_of(path)
                     for path in (train_path, test_samples_path, test_targets_path)}
    print(f"train {train_features.shape}, test {test_samples.shape}, targets {test_targets.shape}")

    for n in args.n_components:
        out_dir = args.output_root / f"PCA{n}-Pavia"
        out_dir.mkdir(parents=True, exist_ok=True)

        pca = PCA(n_components=n, svd_solver="full", whiten=False)
        train_pca = pca.fit(train_features).transform(train_features)  # fitted on train only
        test_pca = pca.transform(test_samples)
        total_variance = float(np.sum(pca.explained_variance_ratio_))

        write_train_dat(out_dir / "train_samples.dat", n, train_pca, train_labels)
        np.save(out_dir / "test_samples.npy", test_pca)
        np.save(out_dir / "test_targets.npy", test_targets)
        np.savez(out_dir / "pca_model.npz", components=pca.components_, mean=pca.mean_,
                 explained_variance_ratio=pca.explained_variance_ratio_)
        metadata = {
            "n_components": n,
            "total_explained_variance": total_variance,
            "source_sha256": source_sha256,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"PCA{n}: total explained variance {total_variance:.6f} -> {out_dir}")


if __name__ == "__main__":
    main()
