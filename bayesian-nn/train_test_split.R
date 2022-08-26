library("creditmodel")

P1_1 <- read.csv(file = './P1_1SP.csv', header = FALSE)
P1_2 <- read.csv(file = './P4_4SP.csv', header = FALSE)
#P1_3 <- read.csv(file = './P1_3.csv', header = FALSE)
#P1_4 <- read.csv(file = './P1_4.csv', header = FALSE)


a <- train_test_split(
  P1_1, 
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

b <- train_test_split(
  P1_2, 
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

c <- train_test_split(
  P1_3, 
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

d <- train_test_split(
  P1_4, 
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


P1_1 <- rbind(a[2]$train,a[1]$test)
P1_2 <- rbind(b[2]$train,b[1]$test)
P1_3 <- rbind(c[2]$train,c[1]$test)
P1_4 <- rbind(d[2]$train,d[1]$test)

write.csv(P1_1, file = './P1_1SP.csv', row.names = FALSE)
write.csv(P1_2, file = './P4_4SP.csv', row.names = FALSE)
write.csv(P1_3, file = './P1_3.csv', row.names = FALSE)
write.csv(P1_4, file = './P1_4.csv', row.names = FALSE)
