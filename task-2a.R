# Load required libraries
library(readr)     # For reading data from csv file
library(dplyr)     # For data manipulation and transformation
library(neuralnet) # For building neural network models
library(scales)    # For scaling and normalization of data

# Read the data from csv file
df <- read_csv("uow_consumption.csv")

# Convert the 'date' column to date format
df <- df %>% mutate(date = as.Date(date, format = "%m/%d/%y"))

# Split data into training and testing sets
train_data <- df[1:380, ] # First 380 rows for training
test_data <- df[381:470, ] # Last 90 rows for testing

# Function to create input/output matrices for different time-delayed input vectors
create_io_matrix <- function(data, t_lags) {
  n <- nrow(data) # Number of rows in the data
  max_lag <- max(t_lags) # Maximum time lag value
  
  # Create an empty matrix with n - max_lag rows and length(t_lags) + 1 columns
  io_matrix <- data.frame(matrix(ncol = length(t_lags) + 1, nrow = n - max_lag))
  # Assign column names as 't_lag values' and 'output'
  colnames(io_matrix) <- c(paste0("t_", t_lags), "output")
  
  # Loop through rows from max_lag + 1 to n
  for (i in (max_lag + 1):n) { 
    # Assign input and output values to the matrix
    io_matrix[i - max_lag, ] <- c(data$`20`[(i - t_lags)], data$`20`[i]) 
  }
  
  return(io_matrix) # Return the input/output matrix
}

# Function to normalize data
normalize <- function(x) {
  # Scale the input values between 0 and 1
  return((x - min(x)) / (max(x) - min(x)))
}

# Function to denormalize data
denormalize <- function(x, original_data) {
  # Scale back the input values to the original scale
  return(x * (max(original_data) - min(original_data)) + min(original_data))
}

# Create and train MLP models for different input vectors and internal network structures
train_mlp <- function(train_data, test_data, t_lags, hidden_layers, linear_output, threshold = 0.01) {
  # Create input/output matrices for training and testing data with different time-lags
  io_train <- create_io_matrix(train_data, t_lags)
  io_test <- create_io_matrix(test_data, t_lags)
  
  # Normalize input/output matrices
  io_train_norm <- as.data.frame(lapply(io_train, normalize))
  io_test_norm <- as.data.frame(lapply(io_test, normalize))
  
  # Create formula for neural network using the column names of the input/output matrices
  nn_formula <- as.formula(paste("output ~", paste(colnames(io_train_norm)[-ncol(io_train_norm)], collapse = " + ")))
  
  # Train MLP model using neuralnet package
  mlp_model <- neuralnet(nn_formula, data = io_train_norm, hidden = hidden_layers, linear.output = linear_output, threshold = threshold)
  
  # Compute predicted values for the test set using the trained MLP model and denormalize them
  predictions_norm <- compute(mlp_model, io_test_norm[, -ncol(io_test_norm)])$net.result
  predictions <- denormalize(predictions_norm, io_test$output)
  
  # Return trained MLP model and predicted values
  return(list("model" = mlp_model, "predictions" = predictions))
}

# Calculate performance metrics
calc_metrics <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  mape <- mean(abs((actual - predicted) / actual)) * 100
  smape <- mean(2 * abs(actual - predicted) / (abs(actual) + abs(predicted))) * 100
  
  # Return calculated metrics as a list
  return(list("RMSE" = rmse, "MAE" = mae, "MAPE" = mape, "sMAPE" = smape))
}

train_n_evaluate = function(t_lags, hidden_layers, linear_output){
  mlp_result <- train_mlp(train_data, test_data, t_lags, hidden_layers, linear_output)
  metrics <- calc_metrics(test_data$`20`[-(1:max(t_lags))], mlp_result$predictions)
  print(paste("Results : RMSE=", metrics$RMSE, "MAE=", metrics$MAE, "MAPE=", metrics$MAPE, "sMAPE=", metrics$sMAPE))
}


# Example of calling train_n_evaluate with different parameter values
train_n_evaluate(t_lags=c(2,5,3), hidden_layers=c(3,5,2), linear_output=TRUE)
train_n_evaluate(t_lags=c(1,4,5,2), hidden_layers=c(4,6), linear_output=FALSE)
train_n_evaluate(t_lags=c(2,4,6), hidden_layers=c(3,4,5,2), linear_output=TRUE)
train_n_evaluate(t_lags=c(3,2), hidden_layers=c(7,3,4), linear_output=FALSE)
train_n_evaluate(t_lags=c(1,4,2), hidden_layers=c(4,6,2,5), linear_output=TRUE)
train_n_evaluate(t_lags=c(2,3,5,4), hidden_layers=c(5), linear_output=FALSE)
train_n_evaluate(t_lags=c(1,3,5), hidden_layers=c(3,6,2,4), linear_output=TRUE)
train_n_evaluate(t_lags=c(2,4), hidden_layers=c(4,3), linear_output=FALSE)
train_n_evaluate(t_lags=c(1,2,4), hidden_layers=c(6,4,2,5), linear_output=TRUE)
train_n_evaluate(t_lags=c(2,3,4), hidden_layers=c(3,5), linear_output=FALSE)
train_n_evaluate(t_lags=c(1,5), hidden_layers=c(5,2), linear_output=TRUE)
train_n_evaluate(t_lags=c(2,4,3), hidden_layers=c(4,6,2), linear_output=FALSE)
train_n_evaluate(t_lags=c(1,3), hidden_layers=c(3,4,2,6), linear_output=TRUE)
train_n_evaluate(t_lags=c(4,2), hidden_layers=c(7,3), linear_output=FALSE)
train_n_evaluate(t_lags=c(1,2,5), hidden_layers=c(4,6), linear_output=TRUE)


# Optimal time lags and hidden layers for the MLP model
t_lags <- c(1,5)
hidden_layers <- c(5, 2)

# Train MLP model and calculate performance metrics using the training and testing data
mlp_result <- train_mlp(train_data, test_data, t_lags, hidden_layers, linear_output=TRUE)
metrics <- calc_metrics(test_data$`20`[-(1:max(t_lags))], mlp_result$predictions)

# Print calculated performance metrics
print(metrics)


