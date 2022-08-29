library("creditmodel")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

level = "Phylum"

the_one_that_do_the_work <- function(level) {
  path = paste("../processed-data/",level, "/full-data", sep = "")
  file_name = dir(path)
  path_name = dir(path, full.names = TRUE)
  all_files = lapply(path_name, read.csv, header = FALSE)
  num = length(file_name)
  
  for (i in 1:num) {
    split(all_files[[i]], level, file_name[i])
  }
}

split <- function(full_data, level, filename){
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
  
  fullpath_train = paste("../processed-data/", level, 
                   "/train-test-split/", name, "_train.csv",
                   sep = "")
  fullpath_test = paste("../processed-data/", level, 
                         "/train-test-split/", name, "_test.csv",
                         sep = "")
  write.table(split_data[2]$train, file = fullpath_train, 
              row.names = FALSE, col.names = FALSE, sep = ",")
  write.table(split_data[1]$test, file = fullpath_test, 
              row.names = FALSE, col.names = FALSE, sep = ",")
}

the_one_that_do_the_work("Phylum")
