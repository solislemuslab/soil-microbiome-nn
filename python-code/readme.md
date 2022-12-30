#  Python codes
This .md file provides brief explanations to the Python code in this repository, as well as a short tutorial on reproducing the experiement results.
The python codes mainly focus on feature selections, comparison to random, Random Forest models, and result visualization.
## Setup
### Import the python environment
import the conda environment named soil_env2.yaml, which used by the project.
### Go the project’s folder
In terminal Active the environment (soil_env2.yaml) by
```{sh}
source activate soil_env2
```
### Run the 6 python code for this project
In terminal Run the py files by
```{sh}
python 1-RF.py
python 2-feature-selectionML.py
python 3-RF-selectedOTU+Env.py
python 4-RF-Env.py
python 5-RF-Aug.py
python 6-compareTorandom.py
python 7-visualization.py
```
## Running time
The running time for each algorithm is listed in the following table. The running times are shows in hours:minutes:seconds orders. For example, 01:28:53 shows 1 hours, 28 minutes and 53 seconds. Benchmarking was done in Mac M1 on a 10 core, with 500GB available RAM.
| Description | File name | Time Duration|
| ------------- | ------------- | ------------|
|RF based on OTUs predictors for 20 normalized data and 5 levels| 1-RF.py | 04:53:02|
|Feature selection for OTU predictors|2-feature-selectionML.py| 01:28:53|
|RF based on selected OTU and Env features|3-RF-selectedOTU+Env.py|03:07:46|
|RF based on Environmental features|4-RF-Env.py|05:56:50|
|RF based on augmented OTUs for 20 normalized data and 5 levels| 5-RF-Aug.py|12:05:09.61|
|Compare to random based on four strategies| 6-RF-compareTorandom.py|32:16:41.60|

## Run the codes
For running these codes, create the following folder as input of algorithms. The name of folders with their description are summarized in the following table.
|Folder name|Sub-folder name|Description|
|--------------|--------------|--------------|
|**OTU**|alpha_diversity|alpha_diversity information|
| | augumented_otu_count| Generated augumented_otu information|
| | Count_data | Original OTU information|
| | normalized_aug | Normalized OTU datasets by 20 methods|
| | normalized_otu | Normalized augmented OTU datasets by 20 methods |
| | OTUData-1-1 | Alpha diversity information |
|**Env**| disease_suppression | disease_suppression information which normalized by 6 methods|
| | field_information | field_information |
| | soil_chemistry | soil_chemistry information which normalized by 6 methods|
|**response**| response_aug|Contains the response values and resulted weighted F1-score for augmented data|
| | response_netcomi| Selected features by SPRING algorithm in NetComi package|
| | response_original | Contains the response values and resulted weighted F1-score|

For running 7-visualization.py code for results of RF and BNN methods for 6 responses (Black_Scurf, Scab, Scabpit, Scabsuper, Yield_Meter, Yield_Plant), the following folder are required:
|Folder name|Sub-folder name|Description|
|--------------|--------------|--------------|
|**response**|response_aug|Result of augmentation method for 6 responses|
| |response_netcomi|Result of running netcomi algorithm|
| |response_original| Result of running all models for 6 responses|
|**Figure**|RF |This folder is used to create figures for RF|
| | BNN| This folder is used to create figures for BNN|
| |figurefile|Result of running 7-visualization.py code are saved in this folder|
