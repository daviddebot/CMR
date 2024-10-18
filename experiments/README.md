# Reproducing experiments
The code in this repository supports running experiments on the different datasets to evaluate the proposed Concept-based Memory Reasoner (CMR) model and compare it with competitors. Instructions for running the experiments and aggregating results are provided below.

## Getting Started

### Prerequisites
A conda virtual environment can be set up by doing
```
conda env create -f environment.yml
```
**Note:** This might install a different version of Pytorch than was used for the paper. The version used for the paper was `pytorch=2.1.1`.

### Dataset
The MNIST dataset will be automatically downloaded when you run the experiments, so no additional setup is required for experiments requiring this dataset (MNIST+, MNIST+*, ColorMNIST). The same applies for CEBAB. For CUB, you first need to manually download the dataset from [here](https://github.com/mateoespinosa/cem/tree/main/cem/data/CUB200/class_attr_data_10). Put the `.pkl` files inside `./experiments/cub/embeddings/cub`. For CelebA, see the instructions below.


## Reproducing Experiments

### Running Experiments

1. **Modify your Python path environment variable:** Replace  `dir` such that the path points to `./CMR`:
   ```bash
   export PYTHONPATH=$PYTHONPATH:dir/CMR
   ```

2. **Navigate to the corresponding experiment directory:** Replace `dir` with `mnist` for MNIST+, `colormnist` for ColorMNIST, `cub` for CUB, `celeba` for CelebA and `cebab` for Cebab.
   ```bash
   cd ./experiments/dir
   ```
3. **(CelebA / CEBAB only) Get embeddings:** Use the command below to get pretrained embeddings of the images (CelebA) or text (CEBAB). CMR and the competitors will use these embeddings as input instead of the image/text itself.
   For CelebA, edit `dataset.py` and put `download=True`. Then, do
   ```bash
   python dataset.py
   ```
   This needs to be done twice. First, edit the file and set `split='test'`. Then, repeat but set `split='train'`. If this does not work, you instead need to manually download the dataset. This can be done using the links [torchvision](https://github.com/pytorch/vision/blob/main/torchvision/datasets/celeba.py) provides, i.e. the following links: ([1](https://drive.usercontent.google.com/download?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM&authuser=0), [2](https://drive.usercontent.google.com/download?id=0B7EVK8r0v71pblRyaVFSWGxPY0U&authuser=0), [3](https://drive.usercontent.google.com/download?id=1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS&authuser=0), [4](https://drive.usercontent.google.com/download?id=0B7EVK8r0v71pbThiMVRxWXZ4dU0&authuser=0), [5](https://drive.usercontent.google.com/download?id=0B7EVK8r0v71pd0FJY3Blby1HUTQ&authuser=0), [6](https://drive.usercontent.google.com/download?id=0B7EVK8r0v71pY0NSMzRuSXJEVkk&authuser=0)). Put these files in `./experiments/data/celeba`. Additionally, extract the zip file in the same location. Afterwards, run the file with `download=False` twice like described above.
   
   For CEBAB, do
   ```bash
   python pretrain.py
   ```
4. **Run experiments:** Use the command below, replacing `id` with a number between <1> and <6> depending on the experiment you wish to run:
   ```bash
   python run_experiments.py id
   ```
* General ID Descriptions:
  + 1: Run CMR. Saves results in a CSV file.
  + 2: Run competitors. Saves results in a CSV file.
* MNIST+ ID Descriptions:
  + 3: Run the CMR rule intervention experiment. Prints results to console.
  + 4: Run CMR on the MNIST+* dataset (with missing concepts for digits 0 and 1). Saves results in a CSV file.
  + 5: Run competitors on the MNIST+* dataset. Saves results in a CSV file.
  + 6: Run concept intervention experiments for both CMR and competitors. Prints results to console. **Note:** You need to update the paths in the code to point to the correct CSV files (of runs of ID 1 and 2).
* CEBAB ID Descriptions:
  + 3: Run the ablation study where decision tree rules are injected into CMR. Saves results in a CSV file.
  + 4: Run the experiment investigating CMR's robustness to the number of rules hyperparameter. Saves results in a CSV file.

Each experiment will save models and results to disk as specified.


### Aggregating Results
For obtaining aggregated results (task and concept accuracy) across all runs (i.e. compute mean and standard-deviation of the metrics), do the following:
1. Navigate to the results directory of the corresponding dataset. For example, for MNIST+:
    ```bash
   cd ./experiments/mnist/results
2. Edit paths in `calc_stats.py`, updating them to point to the appropriate CSV files containing the results to aggregate.
3. Run the aggregation script:
    ```bash
   python calc_stats.py
Directories per experiment:
  * MNIST+ IDs 1-3: `./experiments/mnist/results`
  * MNIST+ ID 4: `./experiments/mnist/results`
  * MNIST+ IDs 5-6: `./experiments/mnis_missing_digit_concepts`
  * ColorMNIST: `./experiments/colormnist/results`
  * CUB: `./experiments/cub/results`
  * CelebA: see below
  * Cebab IDs 1-2: `./experiments/cebab/cebab_base/results`
  * Cebab ID 3: run `calc_stats1.py` in `./experiments/cebab/cebab_base/results` (creates and saves the figure)
  * Cebab ID 4: run `calc_stats2.py` in `./experiments/cebab/cebab_base/results` (creates and saves the figure)

The results and figure for CelebA can be obtained by doing (after putting correct references to CSV files):
```bash
   cd ./experiments/celeba/results
   python plotter.py
```
