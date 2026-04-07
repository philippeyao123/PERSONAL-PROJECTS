# Assessed Homework Solution - Risk Forecasting with Machine Learning
#Candidate Number: 37279
# Install and load necessary libraries
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(quantmod, e1071, ggplot2, caret, randomForest, xgboost, TTR, PerformanceAnalytics)

# Load dataset
data <- read.csv("/Users/philippeyao/Downloads/DATASETFINALE.csv")

# Convert date to proper format
data$date <- as.Date(as.character(data$date), format = "%Y%m%d")

# Filter data for selected tickers
tickers <- c("C", "F", "AAPL")
filtered_data <- subset(data, TICKER %in% tickers)

# Split data by ticker
data_split <- split(filtered_data, filtered_data$TICKER)

# Function for Feature Engineering, Model Training, and Risk Metrics
process_stock <- function(stock_data, stock_name) {
  # Calculate daily log returns
  stock_data$LogReturns <- c(NA, diff(log(stock_data$ASK)))
  
  # Remove NAs
  stock_data <- na.omit(stock_data)
  
  # Feature Engineering
  stock_data$Lag1 <- lag(stock_data$LogReturns, 1)
  stock_data$Lag2 <- lag(stock_data$LogReturns, 2)
  stock_data$Volatility <- runSD(stock_data$LogReturns, n = 20)
  macd <- MACD(stock_data$ASK, nFast = 12, nSlow = 26, nSig = 9, maType = "EMA")
  stock_data$MACD <- macd[, "macd"]
  stock_data$Signal <- macd[, "signal"]
  stock_data$RSI <- RSI(stock_data$ASK, n = 14)
  
  # Remove rows with NA values after feature engineering
  stock_data <- na.omit(stock_data)
  
  # Define predictors and target
  predictors <- stock_data[, c("Lag1", "Lag2", "MACD", "RSI", "ASK", "BID", "LogReturns", "Volatility")]
  target <- stock_data$Volatility
  
  # Ensure both predictors and target are complete and have matching rows
  valid_indices <- complete.cases(predictors, target)
  predictors <- predictors[valid_indices, ]
  target <- target[valid_indices]
  
  # Train-Test Split
  set.seed(123)
  train_index <- createDataPartition(target, p = 0.8, list = FALSE)
  train_data <- predictors[train_index, ]
  test_data <- predictors[-train_index, ]
  train_target <- target[train_index]
  test_target <- target[-train_index]
  
  ## RANDOM FOREST MODEL
  rf_model <- randomForest(
    x = train_data, 
    y = train_target, 
    ntree = 500, 
    mtry = floor(sqrt(ncol(train_data))), 
    importance = TRUE
  )
  rf_predictions <- predict(rf_model, test_data)
  
  # Evaluation Metrics for RF
  rf_mse <- mean((rf_predictions - test_target)^2)
  rf_r2 <- 1 - sum((rf_predictions - test_target)^2) / sum((test_target - mean(test_target))^2)
  
  ## XGBOOST MODEL
  train_matrix <- as.matrix(train_data)
  test_matrix <- as.matrix(test_data)
  
  xgb_model <- xgboost(
    data = train_matrix,
    label = train_target,
    max.depth = 6,
    eta = 0.1,  # Adjusted for lower learning rate
    nrounds = 100,
    objective = "reg:squarederror",
    verbose = 0
  )
  xgb_predictions <- predict(xgb_model, test_matrix)
  
  # Evaluation Metrics for XGBoost
  xgb_mse <- mean((xgb_predictions - test_target)^2)
  xgb_r2 <- 1 - sum((xgb_predictions - test_target)^2) / sum((test_target - mean(test_target))^2)
  
  # Value at Risk (VaR) and Expected Shortfall (ES)
  confidence_level <- 0.95
  var_rf <- quantile(rf_predictions, probs = 1 - confidence_level, na.rm = TRUE)
  var_xgb <- quantile(xgb_predictions, probs = 1 - confidence_level, na.rm = TRUE)
  
  es_rf <- mean(rf_predictions[rf_predictions <= var_rf], na.rm = TRUE)
  es_xgb <- mean(xgb_predictions[xgb_predictions <= var_xgb], na.rm = TRUE)
  
  # Violation Rates
  violation_rate_rf <- sum(test_target < var_rf) / length(test_target)
  violation_rate_xgb <- sum(test_target < var_xgb) / length(test_target)
  
  # Visualization: Side-by-side Plots for RF and XGB
  par(mfrow = c(2, 2))
  
  # RF Predicted vs Actual
  plot(test_target, rf_predictions, col = "blue", pch = 16,
       main = paste(stock_name, "RF Predicted vs Actual"),
       xlab = "Actual Volatility", ylab = "Predicted Volatility")
  abline(0, 1, col = "red")
  
  # RF Predictions Histogram
  hist(rf_predictions, breaks = 30, col = "lightblue", main = paste(stock_name, "RF Predictions"),
       xlab = "Predicted Volatility")
  abline(v = var_rf, col = "red", lwd = 2, lty = 2)
  legend("topright", legend = paste("VaR:", round(var_rf, 4), "\nES:", round(es_rf, 4), "\nVR:", round(violation_rate_rf, 4)), col = "red", lty = 2, lwd = 2)
  
  # XGB Predicted vs Actual
  plot(test_target, xgb_predictions, col = "green", pch = 16,
       main = paste(stock_name, "XGB Predicted vs Actual"),
       xlab = "Actual Volatility", ylab = "Predicted Volatility")
  abline(0, 1, col = "red")
  
  # XGB Predictions Histogram
  hist(xgb_predictions, breaks = 30, col = "lightgreen", main = paste(stock_name, "XGB Predictions"),
       xlab = "Predicted Volatility")
  abline(v = var_xgb, col = "blue", lwd = 2, lty = 2)
  legend("topright", legend = paste("VaR:", round(var_xgb, 4), "\nES:", round(es_xgb, 4), "\nVR:", round(violation_rate_xgb, 4)), col = "blue", lty = 2, lwd = 2)
  
  return(list(
    RandomForest = list(
      MSE = rf_mse,
      R2 = rf_r2,
      VaR = var_rf,
      ES = es_rf,
      ViolationRate = violation_rate_rf
    ),
    XGBoost = list(
      MSE = xgb_mse,
      R2 = xgb_r2,
      VaR = var_xgb,
      ES = es_xgb,
      ViolationRate = violation_rate_xgb
    )
  ))
}

# Process Each Stock
final_results <- setNames(lapply(names(data_split), function(ticker) {
  process_stock(data_split[[ticker]], ticker)
}), names(data_split))

# Summary Table for Comparison
summary_table <- do.call(rbind, lapply(names(final_results), function(ticker) {
  rf <- final_results[[ticker]]$RandomForest
  xgb <- final_results[[ticker]]$XGBoost
  
  data.frame(
    Ticker = ticker,
    RF_MSE = rf$MSE,
    RF_R2 = rf$R2,
    RF_VaR = rf$VaR,
    RF_ES = rf$ES,
    RF_ViolationRate = rf$ViolationRate,
    XGB_MSE = xgb$MSE,
    XGB_R2 = xgb$R2,
    XGB_VaR = xgb$VaR,
    XGB_ES = xgb$ES,
    XGB_ViolationRate = xgb$ViolationRate
  )
}))

print(summary_table)


# Save Summary
write.csv(summary_table, "model_comparison_summary_with_risk.csv")

