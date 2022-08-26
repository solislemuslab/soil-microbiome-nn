library("caret")

# calculate F score
result <- read.table("./result/result_p11_yield_per_meter.txt", header = TRUE, sep = "", dec = ".")
actual <- as.factor(result$Targets)
predict <- as.factor(result$Guesses)

statistics <- function(y, x, full) {
  zero <- full[full$Targets == 0, ]
  one <- full[full$Targets == 1, ]
  n_zero <- nrow(zero)
  n_one <- nrow(one)
  acc_zero <- 1 - sum(zero$Wrong.)/n_zero
  acc_one <- 1 - sum(one$Wrong.)/n_one
  acc_total <- 1 - sum(full$Wrong.)/(n_zero + n_one)
  
  print(acc_zero)
  print(acc_one)
  print(acc_total)
  
  precision <- posPredValue(x, y, positive = "1")
  recall <- sensitivity(x, y, positive = "1")
  F1 <- (2 * precision * recall) / (precision + recall)
  
  precision <- posPredValue(x, y, positive = "0")
  recall <- sensitivity(x, y, positive = "0")
  F0 <- (2 * precision * recall) / (precision + recall)
  
  f <- ((n_one*F1+n_zero*F0)/(n_zero + n_one))
  print(f)
}


statistics(actual,predict, result)
