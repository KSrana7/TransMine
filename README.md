# TransMine
An Attention-based Transformer Architecture to Decode Chemical Transformations from Process Spectroscopic Data

## Requirements

- Python 3.9
- matplotlib == 3.3.4
- numpy == 1.20.3
- pandas == 1.1.5
- scikit_learn == 0.24.2
- seaborn==0.11.1
- h5py==3.1.0
- torch == 2.1.0


## Data

The dataset used in the work are also available in the repo (https://doi.org/10.5281/zenodo.19099471).


## Run the script
After installation of the packages, main_transformer.py file can be run in python. This file, trains the attention based model, test to get the attention scores across the layers and heads, and performs kinetics to obtain the causal relations. To obain results for differnet systems used in the work, change the data location at appropriate places. The file expects the data at absorbance mode for causal estimation.



## Citation
If you find this repository useful in your research, please consider citing the following papers: UNDER REVIEW

