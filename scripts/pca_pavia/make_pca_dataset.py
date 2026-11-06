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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-dir", type=Path, required=True,
                    help="folder with train_samples.dat, test_samples.npy and test_targets.npy")
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--n-components", type=int, nargs="+", required=True)
args = parser.parse_args()

train_path = args.source_dir / "train_samples.dat"
test_samples_path = args.source_dir / "test_samples.npy"
test_targets_path = args.source_dir / "test_targets.npy"

data = np.loadtxt(train_path, skiprows=6, dtype=np.float64)  # 6 header lines
train_features, train_labels = data[:, :-1], data[:, -1].astype(np.int64)
test_samples = np.load(test_samples_path)
test_targets = np.load(test_targets_path)
source_sha256 = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (train_path, test_samples_path, test_targets_path)}
print(f"train {train_features.shape}, test {test_samples.shape}, targets {test_targets.shape}")

for n in args.n_components:
    out_dir = args.output_root / f"PCA{n}-Pavia"
    out_dir.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=n, svd_solver="full", whiten=False)
    train_pca = pca.fit(train_features).transform(train_features)  # fitted on train only
    test_pca = np.ascontiguousarray(pca.transform(test_samples), dtype=np.float64)
    total_variance = float(np.sum(pca.explained_variance_ratio_))

    header = (
        f"MODEL: PCA{n}PAVIA \n"
        "TYPE: 1 \n"
        f"N_ATTRIBUTES: {n} \n"
        f"N_EXAMPLES: {train_pca.shape[0]} \n"
        f"N_CLASSES: {len(np.unique(train_labels))} \n"
        "DATASET:\n"
    )
    with open(out_dir / "train_samples.dat", "wb") as f:
        f.write(header.encode("ascii"))
        for row, label in zip(train_pca, train_labels):
            f.write((" ".join(f"{value:.10f}" for value in row) + f" {label}\n").encode("ascii"))

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
