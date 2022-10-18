library("creditmodel")
# set the working directory to where the current script is located
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# This function reads all data (normalization+responses) within a level
# then split the data into a training set and a testing set.
# It then saves the 2 sets in 2 seperate CSVs
otu_split <- function(level, score) {
  # set the path to the directory where the data is located
  path = paste("../processed-data/otu_data/non_augumented/",score, "/", level, "/full-data", sep = "")
  
  # get all the file names in the directory
  file_name = dir(path)
  # read all files and save into a data frame
  path_name = dir(path, full.names = TRUE)
  all_files = lapply(path_name, read.csv, header = FALSE)
  # number of datasets for this level
  num = length(file_name)
  
  # for each file, split and save
  for (i in 1:num) {
    split_otu(all_files[[i]], level, file_name[i], score)
  }
}

# This function split the dataset into training and testing sets
# and then save them in 2 CSVs
split_otu <- function(full_data, level, filename, score){
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
  fullpath_train = paste("../processed-data/otu_data/non_augumented/"
                         , score, "/", level, "/train-test-split/"
                         , name, "_train.csv", sep = "")
  fullpath_test = paste("../processed-data/otu_data/non_augumented/"
                        , score, "/", level, "/train-test-split/"
                        , name, "_test.csv", sep = "")
  # Write the 2 sets to 2 files
  write.table(split_data[2]$train, file = fullpath_train, 
              row.names = FALSE, col.names = FALSE, sep = ",")
  write.table(split_data[1]$test, file = fullpath_test, 
              row.names = FALSE, col.names = FALSE, sep = ",")
}

alpha_split <- function(level) {
  # set the path to the directory where the data is located
  path = paste("../processed-data/alpha_index_data/non_augumented/", level, "/full-data", sep = "")
  
  # get all the file names in the directory
  file_name = dir(path)
  # read all files and save into a data frame
  path_name = dir(path, full.names = TRUE)
  all_files = lapply(path_name, read.csv, header = FALSE)
  # number of datasets for this level
  num = length(file_name)
  
  # for each file, split and save
  for (i in 1:num) {
    split_alpha(all_files[[i]], level, file_name[i])
  }
}

# This function split the dataset into training and testing sets
# and then save them in 2 CSVs
split_alpha <- function(full_data, level, filename){
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
  fullpath_train = paste("../processed-data/alpha_index_data/non_augumented/"
                        , "/", level, "/train-test-split/"
                         , name, "_train.csv", sep = "")
  fullpath_test = paste("../processed-data/alpha_index_data/non_augumented/"
                        , "/", level, "/train-test-split/"
                        , name, "_test.csv", sep = "")
  # Write the 2 sets to 2 files
  write.table(split_data[2]$train, file = fullpath_train, 
              row.names = FALSE, col.names = FALSE, sep = ",")
  write.table(split_data[1]$test, file = fullpath_test, 
              row.names = FALSE, col.names = FALSE, sep = ",")
}

other_split <- function(type) {
  # set the path to the directory where the data is located
  path = paste("../processed-data/", type, "/non_augumented/full-data", sep = "")
  # get all the file names in the directory
  file_name = dir(path)
  # read all files and save into a data frame
  path_name = dir(path, full.names = TRUE)
  all_files = lapply(path_name, read.csv, header = FALSE)
  # number of datasets for this level
  num = length(file_name)
  
  # for each file, split and save
  for (i in 1:num) {
    split_other(all_files[[i]], type, file_name[i])
  }
}

# This function split the dataset into training and testing sets
# and then save them in 2 CSVs
split_other <- function(full_data, type, filename){
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
  fullpath_train = paste("../processed-data/", type, "/non_augumented/train-test-split/"
                         , name, "_train.csv", sep = "")
  fullpath_test = paste("../processed-data/", type, "/non_augumented/train-test-split/"
                        , name, "_test.csv", sep = "")
  # Write the 2 sets to 2 files
  write.table(split_data[2]$train, file = fullpath_train, 
              row.names = FALSE, col.names = FALSE, sep = ",")
  write.table(split_data[1]$test, file = fullpath_test, 
              row.names = FALSE, col.names = FALSE, sep = ",")
}

levels <- cbind("Phylum", "Class", "Order", "Family", "Genus")

# One only need to specify the level of the dataset to make everything works
for (i in 1:length(levels)){
  for (j in 1:3) {
    otu_split(levels[i], j)
  }
}

for (i in 1:length(levels)){
    alpha_split(levels[i])
}

other_split("disease_suppression_data")
other_split("soil_chemistry_data")
