# GAwLL

*** Genetic Algorithm with Linkage Learning (GAwLL) for the Wrapper Feature Selection Problem  ***

Description: This is the source code for running GAwLL in the feature selection problems described in the paper:

Reference:  Tinos, R.; Przewozniczek, M.; Whitley, D. & Chicano, F. (2023). "Genetic Algorithm with Linkage Learning", Submitted to GECCO'2023.    

It also includes the code for running the standard Genetic Algorithm described in the paper.

Contact: Renato Tinos `<rtinos@ffclrp.usp.br>`

Running the code: `./gawll_fs` `<problem name>` `<classifier>` `<GA_type>`

`<problem name>`: name of the instance (dataset), without extension. An example of the dataset format is given in file ionosphere.dat.
 In the file with the dataset, the inputs are normalized between 0 and 1. The outputs for classification are labels of the classes 
 indicated by integers (starting at 1). The outputs for regression are normalized between 0 and 1. It is also recommended to shuffle the
  examples. The first lines of the file indicate the name of the dataset, the type of the machine learning problem (1-classification; 2-regression), 
  the number of attributes, the number of examples, and the number of classes (only for classification; in regression, only one output is considered). 
  Example for the ionosphere dataset:
`MODEL: IONOSPHERE`
`TYPE: 1`
`N_ATTRIBUTES: 34` 
`N_EXAMPLES: 351` 
`N_CLASSES: 2` 

`<classifier>`: machine learning model (here, only KNN is used). 1: KNN with K=3; 2: KNN with K=5.

`<GA_type>`: genetic algorithm (GA) model. 0: standard GA; 1: GA with linkage learning (GAwLL)

Example for running the code for: ionosphere (dataset ionosphere) 1 (KNN with K=3) 1 (gawll)

`make`

`./gawll_fs ionosphere 1 1`

	
Observation 1: file global.cpp contains the parameters of the GA (examples: number of runs, population size, and crossover rate).

Observation 2: gawll generates 4 main files
 
- `"bfi_%s_c%d_a%d.dat"`,prob_name,classifier_type,GA_type: best fitness found in each run
	
- `"bind_%s_c%d_a%d.dat"`,prob_name,classifier_type,GA_type: best individuals found in each run

- `"time_%s_c%d_a%d.dat"`,prob_name,classifier_type,GA_type: time for each runx

- `"gen_%s_c%d_a%d.dat"`,prob_name,classifier_type,GA_type: number of generations for each run

- `"nedges_%s_c%d_a%d.dat"`,prob_name,classifier_type,GA_type: mean number of edges of the empirical VIG for each run (only for GAwLL)

- `"eVIG_%s_c%d_a%d_r%d.csv"`,prob_name,classifier_type,GA_type,n_run: save empirical VIG (only for GAwLL) for run n_run

# CANNIBAL

## Setup

The Python environment is managed with [uv](https://docs.astral.sh/uv/) and uses Python 3.11 (the original experiments ran on 3.11.5; see `.python-version`). Direct dependencies are pinned to the original versions, because library versions affect the results.

```bash
uv sync --locked
```

`jupyterlab` and `ipykernel` are in the `dev` dependency group and are installed by default by `uv sync`. Run scripts with `uv run python ...`.

To build GAwLL (requires a C++ compiler such as `g++`/clang), run `make` inside `gawll_with_hu_and_seg/`. This produces the `gawll_fs` executable; see the GAwLL section above for how to run it.

`requirements.txt` is the original Windows Anaconda export (win-64) and is kept for reference only; it is not used to create the environment.

Before running experiments be sure to install all required packages. Also, please note, that GAwLL requires C++ compiler installed.

The code and script for each investigated band selection algorithm is present in the `"experimental_results/<METHOD-NAME>"` folder, e.g., `"experimental_results/bombs"` or `"experimental_results/mi"`.
The file for running the experiment of band selection ends with `"_bat.py"` suffix.
Note that the paths are hardcoded can thus could have been changed in the process of experiments.
In each of those folders the selected bands are present under directories names `"out_<BAND-NUMBER>"`.

The scripts to run all of the experiments are present in "scripts" directory.
There are modules to run experiments (`runner.py`), process results (`metrics.py`), calculate metrics (`metrics.py`) and draw different visualizations in the paper (`draw_mse.ipynb`). The specified names are only examples, analogous modules follow the same naming template to avoid confusion.

The main scripts for running the hyperspectral unmixing and segmentation experiments are located in `scripts` folder and are named `runner_reg.py` and `runner.py`, respectively. Both incorporate `run` function. Those modules additionally contain load data methods and CANNIBAL-specific `get_band_indices_based_on_VIGs` methods which executed the band selection process.

## PCA datasets

`scripts/pca_pavia/make_pca_dataset.py` fits PCA on the training samples only and writes a PCA-reduced train and test dataset for each requested size to `<output-root>/PCA<n>-Pavia/`. The saved values are raw PCA scores (not rescaled to 0-1, may be negative). There are no default paths, and the script works from any directory.

```bash
uv run python scripts/pca_pavia/make_pca_dataset.py \
    --source-dir "<folder with train_samples.dat, test_samples.npy, test_targets.npy>" \
    --output-root "<output folder>" \
    --n-components 10 12
```

Each folder holds `train_samples.dat` (GAwLL format), `test_samples.npy`, `test_targets.npy`, `pca_model.npz` and `metadata.json`. An existing output folder is an error unless `--overwrite` is given.

### Band selection on the PCA datasets

`scripts/pca_pavia/select_bands.py` runs six band selection methods (`bombs`, `mi`, `random`, `cannibal`, `affinity_prop`, `gawll_sel`) on one or more of those datasets. It uses `train_samples.dat` only and never opens the test files. There are no default dataset paths, and the script works from any directory.

```bash
caffeinate -i uv run python scripts/pca_pavia/select_bands.py \
    --dataset-dir "<PCA10-Pavia folder>" "<PCA12-Pavia folder>"
```

Options: `--methods` (default: all six), `--results-root` (default `experimental_results/pca_pavia/`) and `--force`. `cannibal`, `affinity_prop` and `gawll_sel` need GAwLL, which the script builds in a temporary folder and runs on both datasets at the same time. Its stop rule is CPU time, so expect about 25-30 minutes per dataset. GAwLL outputs are saved in `<results-root>/PCA<k>-Pavia/gawll/` and reused on later runs; `--force` reruns GAwLL.

Parametric methods (`bombs`, `mi`, `random`, `cannibal`) run for every band count from 5 to the number of features `k`. At `n = k` all bands are recorded without running the algorithm (status `trivial_all_bands`). `random` draws 20 band sets per count (seeds 0-19). `affinity_prop` and `gawll_sel` choose their own count and run once. Scaling (per-component min-max fitted on the training data) is applied only to the GAwLL and `bombs` inputs; the saved datasets are never changed.

Results are written to `<results-root>/PCA<k>-Pavia/bands/`; band files hold 0-based indices, sorted, one per line:

* `<method>/n_<NN>.txt` for `bombs`, `mi` and `cannibal`
* `random/n_<NN>/seed_<SS>.txt`
* `<method>/auto.txt` for `affinity_prop` and `gawll_sel`
* `manifest.csv` with columns `dataset,method,requested_n,returned_n,seed,status,band_file,note`. `band_file` is relative to the results root, `requested_n` is `auto` for self-counting methods, and `status` is `selected`, `trivial_all_bands`, `mi_ranking_fill` (the original `mi` search asserts for `n >= k - 2`, so the top `n` bands of its mutual-information ranking are used) or `failed`.

`bombs` runs in a subprocess and is retried with the next seed (starting from 0, at most 20 attempts) if it exits with an error; `seed` records the seed used.
