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
