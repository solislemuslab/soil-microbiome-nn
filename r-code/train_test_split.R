library("creditmodel")
# set the working directory to where the current script is located
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


################################################################################
################### Run this section before data-augmentation ##################
################################################################################
load_aug <- function(response){
  location = paste("../processed-data/data-augmentation/raw-otu/", response,
                  sep = "")
  path = paste(location, "/full-data", sep = "")
  file_name = dir(path)
  path_name = dir(path, full.names = TRUE)
  all_files = lapply(path_name, read.csv, header = FALSE)
  num = length(file_name)
  # for each file, split and save
  for (i in 1:num) {
    header <- all_files[[i]][1,]
    split_aug(all_files[[i]][-1,], file_name[i], location, header)
  }
}

split_aug <- function(full_data, filename, location, header) {
  name = substr(filename, 1, nchar(filename) - 4)
  split_data <- train_test_split(
    full_data, 
    prop = 0.8,
    split_type = "Random",
    occur_time = NULL,
    cut_date = NULL,
    start_date = NULL,
    save_data = FALSE,
    dir_path = tempdir(),
    file_name = NULL,
    note = FALSE,
    seed = 42
  )
  fullpath_train = paste(location, "/train-test-split/"
                         , name, "_train.csv", sep = "")
  fullpath_test = paste(location, "/train-test-split/", name
                        , "_test.csv", sep = "")

  write.table(rbind(header, split_data[2]$train), file = fullpath_train, 
              row.names = FALSE, col.names = FALSE, sep = ",")
  write.table(rbind(header,split_data[1]$test), file = fullpath_test, 
              row.names = FALSE, col.names = FALSE, sep = ",")
}

res <- cbind("no_tuber_scab", "no_tuber_scabpit", "no_tuber_scabsuper", 
                "pctg_black_scurf", "yield_per_meter",  "yield_per_plant")

for (i in 1:length(res)){
  load_aug(res[i])
}


################################################################################
######## Run the rest of these code after running file-process.ipynb ###########
################################################################################

load <- function(data_name, score, level) {
  if (score != -1) {
    location = paste("../processed-data/", data_name, "/", score,
                 "/", level, sep = "")
  } else if (level != "null") {
    location = paste("../processed-data/", data_name, "/", level, sep = "")
  } else {
    location = paste("../processed-data/", data_name, sep = "")
  }
  # set the path to the directory where the data is located
  path = paste(location, "/full-data", sep = "")
  # get all the file names in the directory
  file_name = dir(path)
  # read all files and save into a data frame
  path_name = dir(path, full.names = TRUE)
  all_files = lapply(path_name, read.csv, header = FALSE)
  # number of datasets for this level
  num = length(file_name)
  # for each file, split and save
  for (i in 1:num) {
    split(all_files[[i]], file_name[i], location)
  }
}

split <- function(full_data, filename, location) {
  # trim the .csv from the original file name for naming's sake
  name = substr(filename, 1, nchar(filename) - 4)
  # Split the data using 80% as training and the rest as testing
  # the seed is 42 to match Rosa's split
  split_data <- train_test_split(
    full_data, 
    prop = 0.8,
    split_type = "Random",
    occur_time = NULL,
    cut_date = NULL,
    start_date = NULL,
    save_data = FALSE,
    dir_path = tempdir(),
    file_name = NULL,
    note = FALSE,
    seed = 42
  )
  # name the training and testing CSVs
  fullpath_train = paste(location, "/train-test-split/"
                         , name, "_train.csv", sep = "")
  fullpath_test = paste(location, "/train-test-split/", name
                        , "_test.csv", sep = "")
  # Write the 2 sets to 2 files
  write.table(split_data[2]$train, file = fullpath_train, 
              row.names = FALSE, col.names = FALSE, sep = ",")
  write.table(split_data[1]$test, file = fullpath_test, 
              row.names = FALSE, col.names = FALSE, sep = ",")
}

levels <- cbind("Phylum", "Class", "Order", "Family", "Genus")


for (i in 1:length(levels)){
  load("alpha_index_data", -1, levels[i])
  load("alpha_soil", -1, levels[i])
  load("alpha_soil_disease", -1, levels[i])
  load("otu_soil_disease", -1, levels[i])
  load("otu_soil", -1, levels[i])
  load("otu_disease", -1, levels[i])
}

for (i in 1:4) {
  load("all_otu_non_augmented", -1, levels[i])
}

for (i in 1:length(levels)){
  for (j in 0:3) {
    load("otu_selection", j, levels[i])
  }
}

load("disease_suppression_data", -1, "null")
load("soil_chemistry_data", -1, "null")
load("soil_disease", -1, "null")
