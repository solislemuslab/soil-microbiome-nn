# Julia codes
This .md file gives an overview of the Julia codes used in this study and instruction for setting up the Julia environment to reproduce the results of the study. The Julia codes are mainly used for pre and post process of BNN models as the actual models are ran by shell scripts in Linux terminal. The Julia code also did data-augmentation, which is used in both BNN models and python scripts.
## File description
All the following files should finish running in less than 5 minutes.
|File name|Description|
|---|---|
|`data-augmentation.ipynb`|The code for data augmentation of raw count OTUs from the original data.|
|`file-process.ipynb`|The code for pre-processing of BNN models. It combine normalized/selected data from `NetComi` and `python-code` with binarized labels. The train-test-split is done in R as no realiable spliting package was found in Julia when the code is written.
|`result-computation.ipynb`| The code for post-processing of BNN models. It reads in the .txt files produced by the shell scripts and produce a table of measurements identical to the python scripts for Random-Forest models.|

## Enivornment setup
The codes are compiled in Julia 1.8.2, this version or later versions are available on [the Julia webpage](https://julialang.org/downloads/). Note that versions prior to 1.8.2 will not work for those scripts.
Once installed the appropriate Julia version, the `IJulia` package is needed to open the `Jupyter Notebook`. The instruction for installing and using the package is available on [the `IJulia` page](https://julialang.github.io/IJulia.jl/stable/manual/installation/). 
Some additional packages are needed for running all the code: `CSV`, `DataFrames`, `XLSX`,`Statistics`, `Distributions`, `Random`, `Tables`, `Glob`, `MLJ`, `MLJBase`, `DelimitedFiles`, and `CategoricalArrays`. Those packages could be easily added in the Julia terminal.
The folders that are needed for those code to run are described in the README file of the`shell-script` folder. Once the folders are set up, the code could be run in the `IJulia` interface easily.

## Additional note
The workflow of the Julia codes are as follows:
Run `data-augmentation.ipynb` first before all other scripts in this repository as it be solely based on the original raw data.
Run `file-process.ipynb` after the `NetComi` packages and the python scripts, and before all shell scripts and `train-test-split.R`. This code is based on the normalized/selected data from other scripts, and produce the data to train and test for BNN models.
Run `result-computation.ipynb` after all shell scripts as this code reads in the .txt results of the BNN models, and provides a easy-to-read table for BNN results.
