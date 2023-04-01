
# Human limits in Machine Learning: Prediction of plant phenotypes using soil microbiome data
- Rosa Aghdam*
- Xudong Tang*
- Shan Shan
- Richard Lankau
- Claudia Solís-Lemus


This repository provides companion source code to reproduce models results and visualization plots as presented in Aghdam et al, 2023.

Citation:
```
(upcoming)
```

## Folder Structure
The detailed instruction of setting up and running the codes are provided in the README files of each of the following folder.
| Folder | Description|
|---|---|
|`julia-code`|The Julia code for data-augmentation, as well as pre and post process of BNN models.|
|`python-code`| The Python code for feature selections, comparison to random, Random Forest models, and result visualization.|
|`r-code`| The R code for normalization of OTUs and feature selection based on `NerComi`, and a train-test-split for BNN.|
|`shell-script`|The script for running all BNN models in `Software for Flexible Bayesian Modeling and Markov Chain Sampling`, and submission files for `CHTC` system of UW-Madison.|
|`unused-scripts`| Julia code for models and methods that did not make into the final paper.|

Please read the description in `julia-code` first as an overview of workflow, and then proceed to `r-code` and `python-code`. The`shell-script` could be went over last, as all the works in other folders need to be done first.

## Issues and questions
Issues and questions are encouraged through the [GitHub issue tracker](https://github.com/solislemuslab/soil-microbiome-nn/issues). For bugs or errors, please make sure to provide enough details for us to reproduce the error on our end.
