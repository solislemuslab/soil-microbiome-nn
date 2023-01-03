# Shell scripts
This .md files contains an overview of shell scripts that runs the BNN models, a short instruction to run those scripts, and descriptions of scripts in all the folders.

## Environment setup
The code in all the folders here are ran in the [CHTC System](https://chtc.cs.wisc.edu/) adminstered by UW-Madison. Each folder here correspond to scripts that runs the models based on predictors indicated by the folder name. The following table gives full predictor names to avoid confusion:

|Folder name|Predictor description|
|---|---|
|`all_otu_non_augmented`|all OTUs after normalization and zero treatments, with four levels (Phylum, Class, Order, Family)|
|`all_otu_augmented` |all OTUs after data augmentation, then normalization and zero treatments, with two levels (Phylum, Class)|
|`alpha`| Alpha diversity indices with normalization methods|
|`alpha_soil`| Alpha diversity indices and soil chemistry with normalization methods|
|`alpha_soil_disease`|Alpha diversity indices ,soil chemistry, and disease suppression ability with normalization methods|
|`chemistry`| Soil chemistry with normalizationn methods|
|`OTU_0`| OTUs with score 0 after feature selection with normalization by row sum|
|`OTU_1`| OTUs with score 1 after feature selection with normalization by row sum|
|`OTU_2`| OTUs with score 2 after feature selection with normalization by row sum|
|`OTU_3`| OTUs with score 3 after feature selection with normalization by row sum|
|`OTU_disease`|OTUs with score 3 and normalization by row sum, and disease suppression abilities with normalization methods|
|`OTU_soil`|OTUs with score 3 and normalization by row sum, and soil chemistry with normalization methods|
|`OTU_soil_disease`|OTUs with score 3 and normalization by row sum, soil chemistry, and disease suppression abilities with normalization methods|
|`soil_disease`| soil chemistry and disease suppression abilities, with normalization methods|
|`suppression`| disease suppression abilities with normalization methods|

The `.sub` files in each folder is the submission file for the CHTC system, detailed instructions for using the CHTC system could be found on the [CHTC website](https://chtc.cs.wisc.edu/uw-research-computing/guides.html). 

The `.sh` files are scripts that runs the actual BNN models. [Software for Flexible Bayesian Modeling and Markov Chain Sampling](https://glizen.com/radfordneal/fbm.software.html) is needed for all those scripts to run. Note that in the `.sub` submission files for the CHTC system, a Docker environment is used, so the user do not have to manually install the software themselves as it is pre-installed in the Docker container. 

However, if the user wish to run those scripts in a Bash command environment, the installation is needed and the instruction of the installation could be found in [this page](https://glizen.com/radfordneal/fbm.2022-04-21.doc/Install.html). When running on the Bash command, the argument `$1` in the `.sh` files need to be replaced by the actual file names of the trainning and testing data. A detailed explanation of every command in the script could be found on [this page](https://glizen.com/radfordneal/fbm.2022-04-21.doc/index.html).

## Getting the results
The result of each script(model) would be in the form of a `.txt` file. Those `.txt` files need to be placed in the corresponding folder described in the `julia-code` README file. After that, run `result-computation.ipynb` and the different measures of results will be generated in spreadsheets.
