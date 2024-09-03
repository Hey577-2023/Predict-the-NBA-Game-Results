# Install packages
install.packages("dplyr")
install.packages("caret")
install.packages("pROC")
install.packages("lubridate")
install.packages("randomForest")
install.packages("e1071")
install.packages("nnet")
install.packages("caretEnsemble")

# Load the necessary packages
library(dplyr)
library(caret)
library(pROC)
library(lubridate)
library(randomForest)
library(e1071)
library(nnet)
library(caretEnsemble)

# Read data
data <- read.csv("NBA Dataset.csv")

# Check data structure
str(data)

# Check the 'WINorLOSS' column in the raw data for NA values
cat("Original data NA count in WINorLOSS:", sum(is.na(data$WINorLOSS)), "\n")

# Extract the Year from the Date column and create the Year column
# Make sure the Date column is in date format
data$Date <- as.Date(data$Date, format="%Y-%m-%d")
# Extraction year
data$Year <- year(data$Date)

# Sort the data by year
data <- data %>% arrange(Year)

# Convert the "Home" property
data$Home <- ifelse(data$Home == "Home", 1, 0)
# Convert the "WINorLOSS" property
data$WINorLOSS <- ifelse(data$WINorLOSS == "W", 1, 0)
# Turn "WINorLOSS" into a factor
data$WINorLOSS <- factor(data$WINorLOSS, levels = c(0, 1), labels = c("Loss", "Win"))

# Check the converted 'WINorLOSS' column for NA values
cat("After conversion, NA count in WINorLOSS:", sum(is.na(data$WINorLOSS)), "\n")

# Deletes rows that contain missing values
data <- na.omit(data)

# Check that the converted 'WINorLOSS' column has NA values again
cat("After na.omit, NA count in WINorLOSS:", sum(is.na(data$WINorLOSS)), "\n")

# Define the selected variable
selected_vars <- c("Home", "WINorLOSS", "Assists", "TotalRebounds", 
                   "Blocks", "Steals", "Opp.Assists", 
                   "Opp.TotalRebounds", "Opp.Blocks", "Opp.Steals")

# Initialization variable
years <- unique(data$Year)
results <- data.frame(Year = integer(), Algorithm = character(), Accuracy = numeric(), Precision = numeric(), Recall = numeric(), F1 = numeric(), stringsAsFactors = FALSE)

# Define a function to calculate F1 scores
calc_f1 <- function(precision, recall) {
  return(2 * (precision * recall) / (precision + recall))
}

# Initializes the ROC list
roc_list <- list()

# ------------Logistic Regression
results_lr <- data.frame(Year = integer(), Accuracy = numeric(), Precision = numeric(), Recall = numeric(), F1 = numeric(), stringsAsFactors = FALSE)
roc_list_lr <- list()

for (i in 1:(length(years) - 3)) {
  train_years <- years[i:(i + 2)]
  test_year <- years[i + 3]
  
  # Create training and test sets
  train_set <- data %>% filter(Year %in% train_years)
  test_set <- data %>% filter(Year == test_year)
  
  cat("Logistic Regression: Processing year:", test_year, "\n")
  cat("Train set size:", nrow(train_set), "\n")
  cat("Test set size:", nrow(test_set), "\n")
  
  # Ensure that the training set and the test set have the same 'WINorLOSS' column factor levels
  train_set$WINorLOSS <- factor(train_set$WINorLOSS, levels = c("Win", "Loss"))
  test_set$WINorLOSS <- factor(test_set$WINorLOSS, levels = c("Win", "Loss"))
  
  # Select the specified variable
  train_set <- train_set %>% select(all_of(selected_vars))
  test_set <- test_set %>% select(all_of(selected_vars))
  
  # The model is trained and cross-verified
  train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)
  model_lr <- train(WINorLOSS ~ ., data = train_set, method = "glm", family = binomial, trControl = train_control, metric = "ROC")
  
  pred_lr <- predict(model_lr, test_set, type = "prob")[, "Win"]
  pred_lr_class <- ifelse(pred_lr > 0.5, "Win", "Loss")
  pred_lr_class <- factor(pred_lr_class, levels = c("Win", "Loss"))
  
  # Calculate confusion matrices and metrics
  conf_matrix_lr <- confusionMatrix(pred_lr_class, test_set$WINorLOSS, positive = "Win")
  accuracy_lr <- conf_matrix_lr$overall['Accuracy']
  recall_lr <- conf_matrix_lr$byClass['Recall']
  precision_lr <- conf_matrix_lr$byClass['Precision']
  f1_lr <- calc_f1(precision_lr, recall_lr)
  roc_lr <- roc(test_set$WINorLOSS, pred_lr, levels = rev(levels(test_set$WINorLOSS)))
  auc_lr <- roc_lr$auc
  
  results_lr <- rbind(results_lr, data.frame(Year = test_year, Accuracy = accuracy_lr, Precision = precision_lr, Recall = recall_lr, F1 = f1_lr, AUC = auc_lr))
  
  print(paste("Confusion Matrix for Logistic Regression (Year:", test_year, ")"))
  print(conf_matrix_lr)
  
  # Calculated ROC curve
  roc_list_lr[[paste("Logistic Regression Year", test_year)]] <- roc_lr
}

# Plot the ROC curve for Logistic Regression
colors <- rainbow(length(roc_list_lr))
if (length(roc_list_lr) > 0) {
  plot(roc_list_lr[[1]], col = colors[1], main = "ROC Curve for Logistic Regression")
  for (i in 2:length(roc_list_lr)) {
    plot(roc_list_lr[[i]], col = colors[i], add = TRUE)
  }
}
legend("bottomright", legend = names(roc_list_lr), col = colors, lwd = 2)

# Print the Logistic Regression results
results_str_lr <- capture.output(print(results_lr))
cat("Logistic Regression Results:", "\n", paste(results_str_lr, collapse = "\n"), "\n")

# ------------------Random Forest
results_rf <- data.frame(Year = integer(), Accuracy = numeric(), Precision = numeric(), Recall = numeric(), F1 = numeric(), stringsAsFactors = FALSE)
roc_list_rf <- list()

for (i in 1:(length(years) - 3)) {
  train_years <- years[i:(i + 2)]
  test_year <- years[i + 3]
  
  # Create training and test sets
  train_set <- data %>% filter(Year %in% train_years)
  test_set <- data %>% filter(Year == test_year)
  
  cat("Random Forest: Processing year:", test_year, "\n")
  cat("Train set size:", nrow(train_set), "\n")
  cat("Test set size:", nrow(test_set), "\n")
  
  # Ensure that the training set and the test set have the same 'WINorLOSS' column factor levels
  train_set$WINorLOSS <- factor(train_set$WINorLOSS, levels = c("Win", "Loss"))
  test_set$WINorLOSS <- factor(test_set$WINorLOSS, levels = c("Win", "Loss"))
  
  # Select the specified variable
  train_set <- train_set %>% select(all_of(selected_vars))
  test_set <- test_set %>% select(all_of(selected_vars))
  
  # The model is trained and cross-verified
  model_rf <- train(WINorLOSS ~ ., data = train_set, method = "rf", trControl = train_control, metric = "ROC")
  
  pred_rf <- predict(model_rf, test_set, type = "prob")[, "Win"]
  pred_rf_class <- ifelse(pred_rf > 0.5, "Win", "Loss")
  pred_rf_class <- factor(pred_rf_class, levels = c("Win", "Loss"))
  
  # Calculate confusion matrices and metrics
  conf_matrix_rf <- confusionMatrix(pred_rf_class, test_set$WINorLOSS, positive = "Win")
  accuracy_rf <- conf_matrix_rf$overall['Accuracy']
  recall_rf <- conf_matrix_rf$byClass['Recall']
  precision_rf <- conf_matrix_rf$byClass['Precision']
  f1_rf <- calc_f1(precision_rf, recall_rf)
  roc_rf <- roc(test_set$WINorLOSS, pred_rf, levels = rev(levels(test_set$WINorLOSS)))
  auc_rf <- roc_rf$auc
  
  results_rf <- rbind(results_rf, data.frame(Year = test_year, Accuracy = accuracy_rf, Precision = precision_rf, Recall = recall_rf, F1 = f1_rf, AUC = auc_rf))
  
  print(paste("Confusion Matrix for Random Forest (Year:", test_year, ")"))
  print(conf_matrix_rf)
  
  # Calculated ROC curve
  roc_list_rf[[paste("Random Forest Year", test_year)]] <- roc_rf
}

# Plot the ROC curve of Random Forest
colors <- rainbow(length(roc_list_rf))
if (length(roc_list_rf) > 0) {
  plot(roc_list_rf[[1]], col = colors[1], main = "ROC Curve for Random Forest")
  for (i in 2:length(roc_list_rf)) {
    plot(roc_list_rf[[i]], col = colors[i], add = TRUE)
  }
}
legend("bottomright", legend = names(roc_list_rf), col = colors, lwd = 2)

# Print the Random Forest results
results_str_rf <- capture.output(print(results_rf))
cat("Random Forest Results:", "\n", paste(results_str_rf, collapse = "\n"), "\n")

# ----------------Support Vector Machine
results_svm <- data.frame(Year = integer(), Accuracy = numeric(), Precision = numeric(), Recall = numeric(), F1 = numeric(), AUC = numeric(), stringsAsFactors = FALSE)
roc_list_svm <- list()

for (i in 1:(length(years) - 3)) {
  train_years <- years[i:(i + 2)]
  test_year <- years[i + 3]
  
  # Create training and test sets
  train_set <- data %>% filter(Year %in% train_years)
  test_set <- data %>% filter(Year == test_year)
  
  cat("Support Vector Machine: Processing year:", test_year, "\n")
  cat("Train set size:", nrow(train_set), "\n")
  cat("Test set size:", nrow(test_set), "\n")
  
  # Ensure that the training set and the test set have the same 'WINorLOSS' column factor levels
  train_set$WINorLOSS <- factor(train_set$WINorLOSS, levels = c("Win", "Loss"))
  test_set$WINorLOSS <- factor(test_set$WINorLOSS, levels = c("Win", "Loss"))
  
  # Select the specified variable
  train_set <- train_set %>% select(all_of(selected_vars))
  test_set <- test_set %>% select(all_of(selected_vars))
  
  # The model is trained and cross-verified
  model_svm <- train(WINorLOSS ~ ., data = train_set, method = "svmRadial", trControl = train_control, metric = "ROC")
  
  pred_svm <- predict(model_svm, test_set, type = "prob")[, "Win"]
  pred_svm_class <- ifelse(pred_svm > 0.5, "Win", "Loss")
  pred_svm_class <- factor(pred_svm_class, levels = c("Win", "Loss"))
  
  # Calculate confusion matrices and metrics
  conf_matrix_svm <- confusionMatrix(pred_svm_class, test_set$WINorLOSS, positive = "Win")
  accuracy_svm <- conf_matrix_svm$overall['Accuracy']
  recall_svm <- conf_matrix_svm$byClass['Recall']
  precision_svm <- conf_matrix_svm$byClass['Precision']
  f1_svm <- calc_f1(precision_svm, recall_svm)
  roc_svm <- roc(test_set$WINorLOSS, pred_svm, levels = rev(levels(test_set$WINorLOSS)))
  auc_svm <- roc_svm$auc
  
  results_svm <- rbind(results_svm, data.frame(Year = test_year, Accuracy = accuracy_svm, Precision = precision_svm, Recall = recall_svm, F1 = f1_svm, AUC = auc_svm))
  
  print(paste("Confusion Matrix for Support Vector Machine (Year:", test_year, ")"))
  print(conf_matrix_svm)
  
  # Calculated ROC curve
  roc_list_svm[[paste("Support Vector Machine Year", test_year)]] <- roc_svm
}

# Plot the ROC curve of the SVM
colors <- rainbow(length(roc_list_svm))
if (length(roc_list_svm) > 0) {
  plot(roc_list_svm[[1]], col = colors[1], main = "ROC Curve for Support Vector Machine")
  for (i in 2:length(roc_list_svm)) {
    plot(roc_list_svm[[i]], col = colors[i], add = TRUE)
  }
}
legend("bottomright", legend = names(roc_list_svm), col = colors, lwd = 2)

# Print the SVM results
results_str_svm <- capture.output(print(results_svm))
cat("Support Vector Machine Results:", "\n", paste(results_str_svm, collapse = "\n"), "\n")

# ----------------------Naive Bayes
results_nb <- data.frame(Year = integer(), Accuracy = numeric(), Precision = numeric(), Recall = numeric(), F1 = numeric(), stringsAsFactors = FALSE)
roc_list_nb <- list()

for (i in 1:(length(years) - 3)) {
  train_years <- years[i:(i + 2)]
  test_year <- years[i + 3]
  
  # Create training and test sets
  train_set <- data %>% filter(Year %in% train_years)
  test_set <- data %>% filter(Year == test_year)
  
  cat("Naive Bayes: Processing year:", test_year, "\n")
  cat("Train set size:", nrow(train_set), "\n")
  cat("Test set size:", nrow(test_set), "\n")
  
  # Ensure that the training set and the test set have the same 'WINorLOSS' column factor levels
  train_set$WINorLOSS <- factor(train_set$WINorLOSS, levels = c("Win", "Loss"))
  test_set$WINorLOSS <- factor(test_set$WINorLOSS, levels = c("Win", "Loss"))
  
  # Select the specified variable
  train_set <- train_set %>% select(all_of(selected_vars))
  test_set <- test_set %>% select(all_of(selected_vars))
  
  # The model is trained and cross-verified
  model_nb <- train(WINorLOSS ~ ., data = train_set, method = "nb", trControl = train_control, metric = "ROC")
  
  pred_nb <- predict(model_nb, test_set, type = "prob")[, "Win"]
  pred_nb_class <- ifelse(pred_nb > 0.5, "Win", "Loss")
  pred_nb_class <- factor(pred_nb_class, levels = c("Win", "Loss"))
  
  # Calculate confusion matrices and metrics
  conf_matrix_nb <- confusionMatrix(pred_nb_class, test_set$WINorLOSS, positive = "Win")
  accuracy_nb <- conf_matrix_nb$overall['Accuracy']
  recall_nb <- conf_matrix_nb$byClass['Recall']
  precision_nb <- conf_matrix_nb$byClass['Precision']
  f1_nb <- calc_f1(precision_nb, recall_nb)
  roc_nb <- roc(test_set$WINorLOSS, pred_nb, levels = rev(levels(test_set$WINorLOSS)))
  auc_nb <- roc_nb$auc
  
  results_nb <- rbind(results_nb, data.frame(Year = test_year, Accuracy = accuracy_nb, Precision = precision_nb, Recall = recall_nb, F1 = f1_nb, AUC = auc_nb))
  
  print(paste("Confusion Matrix for Naive Bayes (Year:", test_year, ")"))
  print(conf_matrix_nb)
  
  # Calculated ROC curve
  roc_list_nb[[paste("Naive Bayes Year", test_year)]] <- roc_nb
}

# Plot ROC curves for Naive Bayes
colors <- rainbow(length(roc_list_nb))
if (length(roc_list_nb) > 0) {
  plot(roc_list_nb[[1]], col = colors[1], main = "ROC Curve for Naive Bayes")
  for (i in 2:length(roc_list_nb)) {
    plot(roc_list_nb[[i]], col = colors[i], add = TRUE)
  }
}
legend("bottomright", legend = names(roc_list_nb), col = colors, lwd = 2)

# Print the Naive Bayes results
results_str_nb <- capture.output(print(results_nb))
cat("Naive Bayes Results:", "\n", paste(results_str_nb, collapse = "\n"), "\n")

# --------------------Neural Network
results_nn <- data.frame(Year = integer(), Accuracy = numeric(), Precision = numeric(), Recall = numeric(), F1 = numeric(), stringsAsFactors = FALSE)
roc_list_nn <- list()

for (i in 1:(length(years) - 3)) {
  train_years <- years[i:(i + 2)]
  test_year <- years[i + 3]
  
  # Create training and test sets
  train_set <- data %>% filter(Year %in% train_years)
  test_set <- data %>% filter(Year == test_year)
  
  cat("Neural Network: Processing year:", test_year, "\n")
  cat("Train set size:", nrow(train_set), "\n")
  cat("Test set size:", nrow(test_set), "\n")
  
  # Ensure that the training set and the test set have the same 'WINorLOSS' column factor levels
  train_set$WINorLOSS <- factor(train_set$WINorLOSS, levels = c("Win", "Loss"))
  test_set$WINorLOSS <- factor(test_set$WINorLOSS, levels = c("Win", "Loss"))
  
  # Select the specified variable
  train_set <- train_set %>% select(all_of(selected_vars))
  test_set <- test_set %>% select(all_of(selected_vars))
  
  # Standardized data
  preProcValues <- preProcess(train_set[, -2], method = c("center", "scale"))
  train_set[, -2] <- predict(preProcValues, train_set[, -2])
  test_set[, -2] <- predict(preProcValues, test_set[, -2])
  
  # Define the tuneGrid parameter
  tune_grid <- expand.grid(size = 5, decay = 0.1)
  
  # The model is trained and cross-verified
  model_nn <- tryCatch({
    train(WINorLOSS ~ ., data = train_set, method = "nnet", linout = FALSE, trace = FALSE, maxit = 200, tuneGrid = tune_grid, trControl = train_control, metric = "ROC")
  }, warning = function(w) {
    cat("Warning: ", w$message, "\n")
    NULL
  }, error = function(e) {
    cat("Error: ", e$message, "\n")
    NULL
  })
  
  if (!is.null(model_nn)) {
    pred_nn <- predict(model_nn, test_set, type = "prob")[, "Win"]
    pred_nn_class <- ifelse(pred_nn > 0.5, "Win", "Loss")
    pred_nn_class <- factor(pred_nn_class, levels = c("Win", "Loss"))
    
    # Calculate confusion matrices and metrics
    conf_matrix_nn <- confusionMatrix(pred_nn_class, test_set$WINorLOSS, positive = "Win")
    accuracy_nn <- conf_matrix_nn$overall['Accuracy']
    recall_nn <- conf_matrix_nn$byClass['Recall']
    precision_nn <- conf_matrix_nn$byClass['Precision']
    f1_nn <- calc_f1(precision_nn, recall_nn)
    roc_nn <- roc(test_set$WINorLOSS, pred_nn, levels = rev(levels(test_set$WINorLOSS)))
    auc_nn <- roc_nn$auc
    
    results_nn <- rbind(results_nn, data.frame(Year = test_year, Accuracy = accuracy_nn, Precision = precision_nn, Recall = recall_nn, F1 = f1_nn, AUC = auc_nn))
    
    print(paste("Confusion Matrix for Neural Network (Year:", test_year, ")"))
    print(conf_matrix_nn)
    
    # Calculated ROC curve
    roc_list_nn[[paste("Neural Network Year", test_year)]] <- roc_nn
  } else {
    cat("Model training failed for year", test_year, "\n")
  }
}

# Draw the ROC curve of the Neural Network
colors <- rainbow(length(roc_list_nn))
if (length(roc_list_nn) > 0) {
  plot(roc_list_nn[[1]], col = colors[1], main = "ROC Curve for Neural Network")
  for (i in 2:length(roc_list_nn)) {
    plot(roc_list_nn[[i]], col = colors[i], add = TRUE)
  }
}
legend("bottomright", legend = names(roc_list_nn), col = colors, lwd = 2)

# Print the Neural Network results
results_str_nn <- capture.output(print(results_nn))
cat("Neural Network Results:", "\n", paste(results_str_nn, collapse = "\n"), "\n")

# Set graphic layout
par(mfrow = c(2, 3))

# Plot the ROC curve for Logistic Regression
if (length(roc_list_lr) > 0) {
  plot(roc_list_lr[[1]], col = "green", main = "ROC Curve for Logistic Regression")
  for (i in 2:length(roc_list_lr)) {
    plot(roc_list_lr[[i]], col = "green", add = TRUE)
  }
}

# Plot the ROC curve for Random Forest
if (length(roc_list_rf) > 0) {
  plot(roc_list_rf[[1]], col = "blue", main = "ROC Curve for Random Forest")
  for (i in 2:length(roc_list_rf)) {
    plot(roc_list_rf[[i]], col = "blue", add = TRUE)
  }
}

# Plot the ROC curve for SVM
if (length(roc_list_svm) > 0) {
  plot(roc_list_svm[[1]], col = "red", main = "ROC Curve for Support Vector Machine")
  for (i in 2:length(roc_list_svm)) {
    plot(roc_list_svm[[i]], col = "red", add = TRUE)
  }
}

# Plot the ROC curve for Naive Bayes
if (length(roc_list_nb) > 0) {
  plot(roc_list_nb[[1]], col = "purple", main = "ROC Curve for Naive Bayes")
  for (i in 2:length(roc_list_nb)) {
    plot(roc_list_nb[[i]], col = "purple", add = TRUE)
  }
}

# Plot the ROC curve for Neural Network
if (length(roc_list_nn) > 0) {
  plot(roc_list_nn[[1]], col = "orange", main = "ROC Curve for Neural Network")
  for (i in 2:length(roc_list_nn)) {
    plot(roc_list_nn[[i]], col = "orange", add = TRUE)
  }
}

# Reset graphic layout
par(mfrow = c(1, 1))

# Plot ROC curves of all algorithms in the same graph
plot(NULL, xlim = c(1, 0), ylim = c(0, 1), xlab = "Specificity", ylab = "Sensitivity", main = "ROC Curves for All Algorithms")
colors <- c("green", "blue", "red", "purple", "orange")
algorithms <- c("Logistic Regression", "Random Forest", "SVM", "Naive Bayes", "Neural Network")
roc_lists <- list(roc_list_lr, roc_list_rf, roc_list_svm, roc_list_nb, roc_list_nn)

for (j in 1:length(roc_lists)) {
  if (length(roc_lists[[j]]) > 0) {
    for (i in 1:length(roc_lists[[j]])) {
      plot(roc_lists[[j]][[i]], col = colors[j], add = TRUE)
    }
  }
}

legend("bottomright", legend = algorithms, col = colors, lwd = 2)

# Print the results of all algorithms
results_str_lr <- capture.output(print(results_lr))
cat("Logistic Regression Results:", "\n", paste(results_str_lr, collapse = "\n"), "\n")

results_str_rf <- capture.output(print(results_rf))
cat("Random Forest Results:", "\n", paste(results_str_rf, collapse = "\n"), "\n")

results_str_svm <- capture.output(print(results_svm))
cat("Support Vector Machine Results:", "\n", paste(results_str_svm, collapse = "\n"), "\n")

results_str_nb <- capture.output(print(results_nb))
cat("Naive Bayes Results:", "\n", paste(results_str_nb, collapse = "\n"), "\n")

results_str_nn <- capture.output(print(results_nn))
cat("Neural Network Results:", "\n", paste(results_str_nn, collapse = "\n"), "\n")

# ----------------------Stacking
# Read data
data <- read.csv("NBA Dataset.csv")

# Data preprocessing
data$Date <- as.Date(data$Date, format="%Y-%m-%d")
data$Year <- year(data$Date)
data <- data %>% arrange(Year)
data$Home <- ifelse(data$Home == "Home", 1, 0)
data$WINorLOSS <- ifelse(data$WINorLOSS == "W", 1, 0)
data$WINorLOSS <- factor(data$WINorLOSS, levels = c(0, 1), labels = c("Loss", "Win"))
data <- na.omit(data)

selected_vars <- c("Home", "WINorLOSS", "Assists", "TotalRebounds", 
                   "Blocks", "Steals", "Opp.Assists", 
                   "Opp.TotalRebounds", "Opp.Blocks", "Opp.Steals")

years <- unique(data$Year)
results_stack <- data.frame(Year = integer(), Accuracy = numeric(), Precision = numeric(), Recall = numeric(), F1 = numeric(), AUC = numeric(), stringsAsFactors = FALSE)
roc_list_stack <- list()

# Define function to calculate F1 score
calc_f1 <- function(precision, recall) {
  return(2 * (precision * recall) / (precision + recall))
}

# Stacking method
train_control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = "final")

for (i in 1:(length(years) - 3)) {
  train_years <- years[i:(i + 2)]
  test_year <- years[i + 3]
  
  # Create training and test sets
  train_set <- data %>% filter(Year %in% train_years)
  test_set <- data %>% filter(Year == test_year)
  
  cat("Stacking: Processing year:", test_year, "\n")
  cat("Train set size:", nrow(train_set), "\n")
  cat("Test set size:", nrow(test_set), "\n")
  
  # Ensure that the training set and the test set have the same 'WINorLOSS' column factor levels
  train_set$WINorLOSS <- factor(train_set$WINorLOSS, levels = c("Win", "Loss"))
  test_set$WINorLOSS <- factor(test_set$WINorLOSS, levels = c("Win", "Loss"))
  
  # Select specified variables
  train_set <- train_set %>% select(all_of(selected_vars))
  test_set <- test_set %>% select(all_of(selected_vars))
  
  # Define base learners
  models <- caretList(
    WINorLOSS ~ ., data = train_set,
    trControl = train_control,
    methodList = c("rf", "nb")
  )
  
  # Define meta-learner
  stack_control <- trainControl(method = "cv", number = 5, savePredictions = "final", classProbs = TRUE)
  stack_model <- caretStack(models, method = "glm", trControl = stack_control)
  
  pred_stack <- predict(stack_model, newdata = test_set, type = "prob")
  
  # Ensure the prediction output is a data frame with a "Win" column
  if (!is.data.frame(pred_stack) || !"Win" %in% colnames(pred_stack)) {
    pred_stack <- data.frame(Win = pred_stack)
  }
  
  pred_stack <- pred_stack[, "Win"]
  pred_stack_class <- ifelse(pred_stack > 0.5, "Win", "Loss")
  pred_stack_class <- factor(pred_stack_class, levels = c("Win", "Loss"))
  
  # Calculate confusion matrix and metrics
  conf_matrix_stack <- confusionMatrix(pred_stack_class, test_set$WINorLOSS, positive = "Win")
  accuracy_stack <- conf_matrix_stack$overall['Accuracy']
  recall_stack <- conf_matrix_stack$byClass['Recall']
  precision_stack <- conf_matrix_stack$byClass['Precision']
  f1_stack <- calc_f1(precision_stack, recall_stack)
  roc_stack <- roc(test_set$WINorLOSS, pred_stack, levels = rev(levels(test_set$WINorLOSS)))
  auc_stack <- roc_stack$auc
  
  results_stack <- rbind(results_stack, data.frame(Year = test_year, Accuracy = accuracy_stack, Precision = precision_stack, Recall = recall_stack, F1 = f1_stack, AUC = auc_stack))
  
  print(paste("Confusion Matrix for Stacking (Year:", test_year, ")"))
  print(conf_matrix_stack)
  
  # Calculate ROC curve
  roc_list_stack[[paste("Stacking Year", test_year)]] <- roc_stack
}

# Plot ROC curve for Stacking
colors <- rainbow(length(roc_list_stack))
if (length(roc_list_stack) > 0) {
  plot(roc_list_stack[[1]], col = colors[1], main = "ROC Curve for Stacking")
  for (i in 2:length(roc_list_stack)) {
    plot(roc_list_stack[[i]], col = colors[i], add = TRUE)
  }
}
legend("bottomright", legend = names(roc_list_stack), col = colors, lwd = 2)

# Print the Stacking results
results_str_stack <- capture.output(print(results_stack))
cat("Stacking Results:", "\n", paste(results_str_stack, collapse = "\n"), "\n")

