# R codes
This .md file provides a brief explanation on the R scripts used in this repository and instruction for reproducing the experiment result. The R code in this study are used mainly on normalization of OTUs and feature selection. A train test split script is written here for producing the exact same split as python scripts as a realiable train-test-split package is not available in Julia at the time when this study is conducted.

## Run ``NetComi`` code
For Normalization and feature selection, we used ``NetComi`` R package. The ``netcomi_code.R`` code contains 3 parts, the details are summarized in the following table. To run R code, first install ``NetComi``, ``SpiecEasi``, and ``SPRING`` packages. The running times are shows in hours:minutes:seconds orders. For example, 01:17:39 shows 1 hours, 17 minutes and 39 seconds.
| Description | Time Duration |
| ------|------|
| Part1: Generate 20 normalized OTUs from count OTUs matrices| 00:8:14|
| Part2: Generate 20 normalized OTUs from count augmented OTUs matrices|01:17:39|
| Part3: Feature selection|34:46:03|

## Run `train-test-split.R`
This script would be used in-between Julia code. For more detailed information, please check the *Workflow* section in the readme file of `julia-code`.
