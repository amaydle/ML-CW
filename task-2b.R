# Load required libraries
library(readr)     # For reading data from csv file
library(dplyr)     # For data manipulation and transformation
library(neuralnet) # For building neural network models
library(scales)    # For scaling and normalization of data
library(ggplot2)   # For data visualization


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
  io_matrix <- data.frame(matrix(ncol = length(t_lags) + 1, nrow = n - max_lag)) # Create an empty matrix with n - max_lag rows and length(t_lags) + 1 columns
  colnames(io_matrix) <- c(paste0("t_", t_lags), "output") # Assign column names as 't_lag values' and 'output'
  
  for (i in (max_lag + 1):n) { # Loop through rows from max_lag + 1 to n
    io_matrix[i - max_lag, ] <- c(data$`20`[(i - t_lags)], data$`20`[i]) # Assign input and output values to the matrix
  }
  
  return(io_matrix) # Return the input/output matrix
}

# Function to normalize data
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x))) # Scale the input values between 0 and 1
}

# Function to denormalize data
denormalize <- function(x, original_data) {
  return(x * (max(original_data) - min(original_data)) + min(original_data)) # Scale back the input values to the original scale
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
  # Calculate Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Symmetric Mean Absolute Percentage Error (sMAPE)
  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  mape <- mean(abs((actual - predicted) / actual)) * 100
  smape <- mean(2 * abs(actual - predicted) / (abs(actual) + abs(predicted))) * 100
  
  # Return calculated metrics as a list
  return(list("RMSE" = rmse, "MAE" = mae, "MAPE" = mape, "sMAPE" = smape))
}


# Define a function to plot predictions vs desired output
plot_predictions <- function(test_data, predictions, t_lags, method) {
  # Extract desired output from test data
  desired_output <- test_data$`20`[-(1:max(t_lags))]
  
  # Prepare data for plotting
  plot_data <- data.frame("Time" = 1:length(desired_output), "Desired" = desired_output, "Predicted" = predictions)
  
  # Create plot using ggplot2
  ggplot(plot_data, aes(x = Time)) +
    geom_line(aes(y = Desired, color = "Desired")) +
    geom_line(aes(y = Predicted, color = "Predicted")) +
    labs(title = paste("Predictions vs Desired Output (", method, " method )"), x = "Time", y = "Electricity Consumption (kWh)", color = "Legend") +
    theme_minimal()
}

# Optimal time lags and hidden layers for the MLP model
t_lags <- c(1,5)
hidden_layers <- c(5, 2)

# Train MLP model and obtain performance metrics
mlp_result <- train_mlp(train_data, test_data, t_lags, hidden_layers, TRUE)
metrics <- calc_metrics(test_data$`20`[-(1:max(t_lags))], mlp_result$predictions)

# Print metrics for the MLP model trained on the test set
print(metrics)

# Use the best MLP model's predictions and plot them against the desired output
best_predictions <- mlp_result$predictions
plot_predictions(test_data, best_predictions, t_lags, "AR")

# Function to create input/output matrices for NARX configuration
create_narx_io_matrix <- function(data, t_lags, narx_lags) {
  n <- nrow(data)
  max_lag <- max(t_lags)
  io_matrix <- data.frame(matrix(ncol = length(t_lags) + length(narx_lags) * 2 + 1, nrow = n - max_lag))
  colnames(io_matrix) <- c(paste0("t_", t_lags), paste0("t_19_", narx_lags), paste0("t_18_", narx_lags), "output")
  
  for (i in (max_lag + 1):n) {
    io_matrix[i - max_lag, ] <- c(data$`20`[(i - t_lags)], data$`19`[(i - narx_lags)], data$`18`[(i - narx_lags)], data$`20`[i])
  }
  
  return(io_matrix)
}

# Function to train MLP models for NARX configuration
train_narx_mlp <- function(train_data, test_data, t_lags, narx_lags, hidden_layers, linear_output, threshold = 0.01) {
  # Create input/output matrices for training and test sets
  io_train <- create_narx_io_matrix(train_data, t_lags, narx_lags)
  io_test <- create_narx_io_matrix(test_data, t_lags, narx_lags)
  
  # Normalize the input/output matrices
  io_train_norm <- as.data.frame(lapply(io_train, normalize))
  io_test_norm <- as.data.frame(lapply(io_test, normalize))
  
  # Define the formula for the neural network
  nn_formula <- as.formula(paste("output ~", paste(colnames(io_train_norm)[-ncol(io_train_norm)], collapse = " + ")))
  
  # Train the MLP model
  mlp_model <- neuralnet(nn_formula, data = io_train_norm, hidden = hidden_layers, linear.output = linear_output, threshold = threshold)
  
  # Get predictions on the test set using the trained MLP model
  predictions_norm <- compute(mlp_model, io_test_norm[, -ncol(io_test_norm)])$net.result
  predictions <- denormalize(predictions_norm, io_test$output)
  
  return(list("model" = mlp_model, "predictions" = predictions))
}

train_n_evaluate_narx_mlp = function(t_lags, narx_lags, hidden_layers, linear_output){
  mlp_result <- train_narx_mlp(train_data, test_data, t_lags, narx_lags, hidden_layers, linear_output)
  metrics <- calc_metrics(test_data$`20`[-(1:max(t_lags))], mlp_result$predictions)
  print(paste("Results : RMSE=", metrics$RMSE, "MAE=", metrics$MAE, "MAPE=", metrics$MAPE, "sMAPE=", metrics$sMAPE))
}

train_n_evaluate_narx_mlp(t_lags=c(1,2), narx_lags=c(1, 1), hidden_layers=c(3, 3), linear_output=TRUE)
train_n_evaluate_narx_mlp(t_lags=c(2,3), narx_lags=c(1, 2), hidden_layers=c(5, 4), linear_output=FALSE)
train_n_evaluate_narx_mlp(t_lags=c(1,2,3), narx_lags=c(3, 2, 1), hidden_layers=c(8, 6, 4), linear_output=TRUE)
train_n_evaluate_narx_mlp(t_lags=c(2,3,4), narx_lags=c(2, 1, 3), hidden_layers=c(10, 8, 6), linear_output=FALSE)
train_n_evaluate_narx_mlp(t_lags=c(1,2), narx_lags=c(1, 2), hidden_layers=c(4, 3), linear_output=TRUE)
train_n_evaluate_narx_mlp(t_lags=c(2,3,4), narx_lags=c(3, 2, 1), hidden_layers=c(7, 5, 3), linear_output=FALSE)
train_n_evaluate_narx_mlp(t_lags=c(1,2,3), narx_lags=c(2, 1, 3), hidden_layers=c(6, 4, 2), linear_output=TRUE)
train_n_evaluate_narx_mlp(t_lags=c(3,4), narx_lags=c(2, 3), hidden_layers=c(8, 6), linear_output=FALSE)
train_n_evaluate_narx_mlp(t_lags=c(2,3), narx_lags=c(1, 1), hidden_layers=c(4, 4), linear_output=TRUE)
train_n_evaluate_narx_mlp(t_lags=c(1,2,3), narx_lags=c(1, 2, 3), hidden_layers=c(5, 5, 5), linear_output=FALSE)




# Define the time lags for autoregression (AR) and non-autoregressive exogenous (NARX) configurations
t_lags <- c(1,2,3)
narx_lags <- c(1,2,3)

# Define the number of hidden layers for MLP models in NARX configuration
hidden_layers <- c(5, 5, 5)


# Train a neural network with NARX architecture using MLP
narx_mlp_result <- train_narx_mlp(train_data, test_data, t_lags, narx_lags, hidden_layers, FALSE)

# Calculate performance metrics for the NARX model on the test data set
narx_metrics <- calc_metrics(test_data$`20`[-(1:max(t_lags))], narx_mlp_result$predictions)

# Print the calculated metrics for the NARX model
print(narx_metrics)

# Plot the predictions made by the NARX model against the actual values in the test data set
plot_predictions(test_data, narx_mlp_result$predictions, t_lags, "NARX")
