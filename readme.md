# Neural network model on soil microbiome data
- Rosa Aghdam
- Xudong Tang
- Claudia Solís-Lemus
- Shan Shan
- Richard Lankau

This repository provides companion source code to reprooduce models results and visualization plots as presented in *Unlocking the predictive power of soil microbiome data on disease and yield outcomes using deep learning models.*

## Folder Structure
The detailed instruction of setting up and running the codes are provided in the README files of each of the following folder.
| Folder | Description|
|---|---|
|`julia-code`|The Julia code for data-augmentation, as well as pre and post process of BNN models.|
|`python-code`| The Python code for feature selections, comparison to random, Random Forest models, and result visualization.|
|`r-code`| The R code for normalization of OTUs and feature selection based on `NerComi`, and a train-test-split for BNN.|
|`shell-script`|The script for running all BNN models in `Software for Flexible Bayesian Modeling and Markov Chain Sampling`, and submission files for `CHTC` system of UW-Madison.|
|`unused-scripts`| Julia code for models and methods that did not make into the final paper.|

## Issues and questions
Issues and questions are encouraged through the [GitHub issue tracker](https://github.com/solislemuslab/soil-microbiome-nn/issues). For bugs or errors, please make sure to provide enough details for us to reproduce the error on our end.
