rm(list = ls())

# The following packages were loaded and used for this project:
# For data manipulation/processing
require(dplyr) 
require(caret)
require(MASS)
require(lubridate)

# For implementation
require(glmnet) # Logistic
# install.packages('kernlab')
require(kernlab) # KLR
require(e1071) # SVM

# For graphics
require(plotly)
require(ggplot2)

setwd("~/Brock University related/Master's/Year 1/Winter/STAT 5P87/Final Project")
mydata = read.csv('Scotia_2022_data.csv')
mydata$time = ymd_hm(mydata$time)
summary(mydata)

# ---- Data preprocessing and filtering ----
# Filter out negative values of oxygen and depth
mydata = mydata[mydata$oxygen >= 0 & mydata$depth >= 0, ]

# Create new variable - pressure, and calculate it using existing variables
mydata$pressure = mydata$depth * 9.80665 * mydata$density * 0.0001
# This is depth * acceleration due to gravity * density * 0.0001 (to convert from Pa to dBar)

mydata = mydata[,-c(4,7)] # Remove depth and density, which were used to calculate pressure in Excel

# Aggregate data by hours, and use hourly averages of observations of independent 
# variables in place of raw observations
mydata = mydata %>%
  mutate(time = floor_date(time, 'hour')) %>%
  group_by(time) %>%
  summarize(oxygen_avg = mean(oxygen, na.rm = TRUE),
            temp_avg = mean(temperature, na.rm = TRUE),
            salinity_avg = mean(salinity, na.rm = TRUE),
            pressure_avg = mean(pressure, na.rm = TRUE),
            .groups = "drop")

# Next, set threshold for classification
threshold = 156
mydata$anomaly = factor(ifelse(mydata$oxygen_avg > threshold, 'Normal','Hypoxic'))

length(which(mydata$anomaly == 'Normal')) # 659
length(which(mydata$anomaly == 'Hypoxic')) # 258
summary(mydata) # Check new summary statistics


# ---- Visualization of distributions of data by anomaly class per variable ----
ggplot(mydata, aes(x = temp_avg, fill = anomaly)) +
  geom_histogram(position = "dodge", bins = 20) +
  labs(title = "Temperature Distribution by Anomaly", x = "Temperature", y = "Count") +
  theme_minimal()

ggplot(mydata, aes(x = salinity_avg, fill = anomaly)) +
  geom_histogram(position = "dodge", bins = 20) +
  labs(title = "Salinity Distribution by Anomaly", x = "Salinity", y = "Count") +
  theme_minimal()

ggplot(mydata, aes(x = pressure_avg, fill = anomaly)) +
  geom_histogram(position = "dodge", bins = 20) +
  labs(title = "Pressure Distribution by Anomaly", x = "Pressure", y = "Count") +
  theme_minimal()

mydata$timestamp = as.Date(mydata$time)

# Calculate the daily anomaly rates
daily_anomaly = mydata %>%
  group_by(timestamp) %>%
  summarise(
    normal_count = sum(anomaly == "Normal"),
    hypoxic_count = sum(anomaly == "Hypoxic"),
    total_count = n(),
    normal_rate = normal_count / total_count,
    hypoxic_rate = hypoxic_count / total_count
  )

# Plot 1: Daily anomaly rate (Normal and Hypoxic over time)
barplot(
  height = rbind(daily_anomaly$normal_rate, daily_anomaly$hypoxic_rate),
  beside = TRUE,
  names.arg = as.character(daily_anomaly$timestamp),
  col = c("skyblue", "lightcoral"),
  main = "Daily Anomaly Rates Over Time",
  xlab = "Date",
  ylab = "Anomaly Rate",
  legend.text = c("Normal", "Hypoxic"),
  args.legend = list(x = "topright")
)

par(mfrow = c(1, 3))

# Plot 2: Temperature vs Anomaly Status (histogram)
hist(mydata$temp_avg[mydata$anomaly == 'Normal'], 
     main = "Temperature", 
     xlab = "Temperature (°C)", 
     col = "skyblue", 
     xlim = range(mydata$temp_avg), 
     breaks = 10)
hist(mydata$temp_avg[mydata$anomaly == 'Hypoxic'], 
     add = TRUE, 
     col = rgb(1, 0, 0, 0.5), 
     breaks = 10)

# Plot 3: Salinity vs Anomaly Status (histogram)
hist(mydata$salinity_avg[mydata$anomaly == 'Normal'], 
     main = "Salinity", 
     xlab = "Salinity (PSU)", 
     col = "seagreen4", 
     xlim = range(mydata$salinity_avg), 
     breaks = 10)
hist(mydata$salinity_avg[mydata$anomaly == 'Hypoxic'], 
     add = TRUE, 
     col = rgb(1, 0, 0, 0.5), 
     breaks = 10)

# Plot 4: Pressure vs Anomaly Status (histogram)
hist(mydata$pressure_avg[mydata$anomaly == 'Normal'], 
     main = "Pressure", 
     xlab = "Pressure (dbar)", 
     col = "lightpink", 
     xlim = range(mydata$pressure_avg), 
     breaks = 10)
hist(mydata$pressure_avg[mydata$anomaly == 'Hypoxic'], 
     add = TRUE, 
     col = rgb(1, 0, 0, 0.5), 
     breaks = 10)

# Reset to default layout
par(mfrow = c(1, 1))
# Aggregate daily average oxygen levels
mydata$date = as.Date(mydata$time)
mydata$hour = hour(mydata$time)
daily_data = mydata %>%
  group_by(date) %>%
  summarise(oxygen_avg = mean(oxygen_avg, na.rm = TRUE),
            anomaly_rate = mean(anomaly == "Hypoxic"))

# Time series plot (anomaly rate over time)
ggplot(daily_data, aes(x = date, y = anomaly_rate)) +
  geom_line(color = "violetred4") +
  geom_point(alpha = 0.5) +
  labs(title = "Daily Hypoxia Rate Over Time",
       x = "Date", y = "Proportion of Hypoxic Readings") +
  theme_minimal()

ggplot(mydata, aes(x = anomaly)) +
  geom_bar(fill = "cornflowerblue", color = "black") +
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5) +
  labs(title = "Class Distribution of Oxygen Levels",
       x = "Oxygen Class",
       y = "Count") +
  theme_minimal()

# ---- CV function ----
make_folds = function(Y, nFolds, stratified = FALSE, seed = 0){
  # K-Fold cross validation
  # Input:
  #   Y (either sample size, or vector of outputs)
  #   stratified (boolean): whether the folds should 
  #     be stratified. Requires Y to be a vector of outputs
  # Output: list of vectors of fold indices
  set.seed(seed)
  if(stratified & length(Y) == 1){
    stop('For stratified folds, Y must be a vector of outputs')
  }
  n = ifelse(length(Y) > 1, length(Y), Y)
  index = c(1:n)
  if(stratified){
    Y = factor(Y)
    classes = levels(Y)
    nClasses = length(classes)
    if(nClasses == 1){
      stop('stratified requires more than one class')
    }
    classfolds = list()
    for(class in 1:nClasses){
      classfolds[[class]] = list()
      classIndex = index[Y == classes[class]]
      n_class = sum(Y == classes[class])
      n_per_fold = floor(n_class / nFolds)
      shuffled_index = sample(classIndex)
      for(fold in c(1:(nFolds - 1))){
        classfolds[[class]][[fold]] = shuffled_index[c((1 + (fold - 1) * n_per_fold):(fold * n_per_fold))]
      }
      classfolds[[class]][[nFolds]] = shuffled_index[c(((nFolds - 1)*n_per_fold + 1):n_class)]
    }
    folds = list()
    for(fold in 1:nFolds){
      folds[[fold]] = classfolds[[1]][[fold]]
      for(class in 2:nClasses){
        folds[[fold]] = c(folds[[fold]], classfolds[[class]][[fold]])
      }
    }
  }else{
    folds = list()
    n_per_fold = floor(n / nFolds)
    shuffled_index = sample(index)
    for(fold in c(1:(nFolds - 1))){
      folds[[fold]] = shuffled_index[c((1 + (fold - 1) * n_per_fold):(fold * n_per_fold))]
    }  
    folds[[nFolds]] = shuffled_index[c(((nFolds - 1)*n_per_fold + 1):n)]
  }
  return(folds)
}

# ---- Model Implementations ----
set.seed(0)

X = model.matrix(anomaly ~ 0 + temp_avg + salinity_avg + pressure_avg, data=mydata)
Y = ifelse(mydata$anomaly == 'Normal', 1, 0)

n = dim(mydata)[1]

# ---- Logistic ----
lambda_values = seq(from = 0, to = 0.2, by = 0.002) 
# Checked from 0 to 1, but accuracy is lower than 65% consistently after lambda > 0.4
n_lambda_values = length(lambda_values)

nFolds = 5
folds = make_folds(Y, nFolds, FALSE, 0)

cv_accuracy = matrix(NA, nrow = n_lambda_values, ncol = nFolds)

for(i in 1:n_lambda_values){
  lambda = lambda_values[i]
  
  for (fold in 1:nFolds){
    train_fold = mydata[-folds[[fold]], ]
    test_fold = mydata[folds[[fold]], ]
    # alpha = 1 for LASSO penalty
    
    y_train = ifelse(train_fold$anomaly == "Normal", 1, 0)
    x_train = as.matrix(train_fold[, c("temp_avg", "salinity_avg", "pressure_avg")])
    
    log_model = glmnet(x_train, y_train, family = 'binomial', lambda = lambda, alpha = 1)
    
    # Compute testing accuracy
    x_test = as.matrix(test_fold[, c("temp_avg", "salinity_avg", "pressure_avg")])
    
    yHat_log = predict(log_model, x_test, type = 'class')
    
    y_test = ifelse(test_fold$anomaly == 'Normal', 1, 0)
    
    cv_accuracy[i, fold] = mean(yHat_log == y_test)
  }
}

mean_cv_accuracy = apply(cv_accuracy, 1, mean)

plot(lambda_values, mean_cv_accuracy, bty = 'n', col = 'slateblue4',pch = 19,
     xlab = 'Lambda Values', ylab = 'Accuracy', 
     main= 'Plot of Accuracy Against Lambda')

which.max(mean_cv_accuracy) # 1
best_lambda = lambda_values[1] # Best lambda = 0
best_accuracy = max(mean_cv_accuracy) # 92.69%

# Refit whole model using optimal lambda
final_log_model = glmnet(X, Y, family = 'binomial',
                         lambda = best_lambda, alpha = 1)

y_pred_log = predict(final_log_model, X, type = 'class')
mean(Y == y_pred_log) # Overall accuracy - 93.02%

# Confusion matrix to get some other important metrics
D = confusionMatrix(as.factor(Y), as.factor(y_pred_log)) 
print(D)
# Recall of 88.80% and specificity of 94.60%

set.seed(0)
temp_range = seq(from = min(mydata$temp_avg), to = max(mydata$temp_avg), length.out = 30)  # Adjust length.out for resolution
salinity_range = seq(from = min(mydata$salinity_avg), to = max(mydata$salinity_avg), length.out = 30)
press_range = seq(from = min(mydata$pressure_avg), to = max(mydata$pressure_avg), length.out = 30)

grid = expand.grid(temp_avg = temp_range, salinity_avg = salinity_range, pressure_avg = press_range)

y0Hat_log_grid = as.numeric(predict(final_log_model, as.matrix(grid), type = 'class'))

grid$pred = y0Hat_log_grid

# 3D Plot of Decision Boundary
fig = plot_ly(x = grid$temp_avg, y = grid$salinity_avg, z = grid$pressure_avg, 
              color = grid$pred, colors = c("violetred4", "gold"), type = "scatter3d", mode = "markers",
              marker = list(size = 2)) %>%
  add_trace(type = "mesh3d", opacity = 0.1, color = "seagreen", 
            x = grid$temp_avg, y = grid$salinity_avg, z = grid$pressure_avg, 
            intensity = grid$pred, showscale = TRUE)

fig = fig %>% layout(title = "3D Logistic Regression Decision Boundary",
                     scene = list(xaxis = list(title = 'Temperature'),
                                  yaxis = list(title = 'Salinity'),
                                  zaxis = list(title = 'Pressure')))
fig

# Based on plot, it is evident hypoxia occurs accross all temps and pressure conditions,
# but really only sets in for salinity > 32.5
# And to check:
high_sal_hypoxia = grid %>%
  filter(salinity_avg > 32.5, pred == 0)
summary(high_sal_hypoxia)
# This thus confirms it, leading to believe that salinity may be the most important variable,
# according to the logistic regression model.


# ---- Kernel Logistic Regression ----
lambda_values = seq(from = 0.001, to = 0.03, by = 0.002) 
n_lambda_values = length(lambda_values)

sigma_values = seq(from = 0.002, to = 0.1, by = 0.01) # Kernel width to iterate over
n_sigma_values = length(sigma_values) 

cv_accuracy = array(0, dim = c(n_lambda_values, n_sigma_values, nFolds))

for(i in 1:n_lambda_values){
  for (j in 1:n_sigma_values){
    lambda = lambda_values[i]
    sigma = sigma_values[j]
    
    for (fold in 1:nFolds){
      train_fold = mydata[-folds[[fold]], ]
      test_fold = mydata[folds[[fold]], ]
      # alpha = 1 for LASSO penalty
      
      y_train = ifelse(train_fold$anomaly == "Normal", 1, 0)
      x_train = as.matrix(train_fold[, c("temp_avg", "salinity_avg", "pressure_avg")])
      
      x_test = as.matrix(test_fold[, c("temp_avg", "salinity_avg", "pressure_avg")])
      y_test = ifelse(test_fold$anomaly == 'Normal', 1, 0)
      
      kernel = rbfdot(sigma = sigma)
      
      kernel_mat_train = kernelMatrix(kernel, x_train) 
      kernel_mat_test = kernelMatrix(kernel, x_test, x_train)
      
      
      klog = glmnet(kernel_mat_train, y_train, family = 'binomial', lambda = lambda, alpha = 0.5, maxit = 1e+06)
      
      probs = predict(klog, kernel_mat_test, type = 'response')
      # Compute testing accuracy
      preds = ifelse(probs > 0.5, 1, 0)
      
      yHat_klr = preds
      
      cv_accuracy[i, j, fold] = mean(yHat_klr == y_test)
    }
  }
}

mean_cv_accuracy = apply(cv_accuracy, c(1,2), mean)
max(mean_cv_accuracy) # 88.77%

which(mean_cv_accuracy == max(mean_cv_accuracy), arr.ind = T)

lambda = lambda_values[6] # 0.011
sigma = sigma_values[6] # 0.052

kernel = rbfdot(sigma = 0.052)
kernel_matrix = kernelMatrix(kernel, X) 

klr_model = glmnet(kernel_matrix, Y, family = 'binomial', lambda = lambda, alpha = 0.5)
pred_probs = predict(klr_model, kernel_matrix, type = 'response')
pred_probs
yHat_klr = ifelse(pred_probs > 0.5, 1, 0)
klr_accuracy = mean(as.factor(yHat_klr) == Y) # 89.31%
G = confusionMatrix(as.factor(Y), as.factor(yHat_klr))
G
# Recall score of 85.71%

set.seed(0)
temp_range = seq(from = min(mydata$temp_avg), to = max(mydata$temp_avg), length.out = 20) 
salinity_range = seq(from = min(mydata$salinity_avg), to = max(mydata$salinity_avg), length.out = 20)
press_range = seq(from = min(mydata$pressure_avg), to = max(mydata$pressure_avg), length.out = 20)

grid = expand.grid(temp_avg = temp_range, salinity_avg = salinity_range, pressure_avg = press_range)

kernel_grid = kernelMatrix(kernel, as.matrix(grid), X)
y0Hat_klr = predict(klr_model, kernel_grid, type = 'response')
grid$pred = ifelse(y0Hat_klr > 0.5, 1, 0)

x = grid$temp_avg
y = grid$salinity_avg
z = grid$pressure_avg
pred = grid$pred 

# Plot DB
plot_ly(data = grid,
        x = ~temp_avg,
        y = ~salinity_avg,
        z = ~pressure_avg,
        color = ~as.factor(pred),  # Color by predicted class
        colors = c("violetred", "slateblue"), # Class 0 = red, Class 1 = blue
        type = "scatter3d",
        mode = "markers",
        marker = list(size = 3, opacity = 0.7)) %>%
  layout(title = "3D Scatter of KLR Predicted Classes",
         scene = list(
           xaxis = list(title = "Temp"),
           yaxis = list(title = "Salinity"),
           zaxis = list(title = "Pressure")
         ),
         legend = list(title = list(text = "Predicted Class")))

hypoxia_conditions_klr = grid[grid$pred == 0, ]

print(hypoxia_conditions_klr)
summary(hypoxia_conditions_klr)

# ---- SVM ----
mydata$anomaly_svm = ifelse(mydata$anomaly == 'Normal', 1, -1)

set.seed(0)
folds = make_folds(mydata$anomaly_svm, nFolds, F, seed = 0)

C_values = seq(from = 40, to = 50, by = 0.5) # Tested multiple times and reduced range
n_C_values = length(C_values)

gamma_values = seq(from = 0.075, to = 0.175, by = 0.0075)
n_gamma_values = length(gamma_values)

accuracy = array(0, dim = c(n_C_values, n_gamma_values, nFolds))

for(fold in 1:nFolds){
  
  train_fold = mydata[-folds[[fold]],]
  test_fold = mydata[folds[[fold]],]
  
  for(i in 1:n_C_values){
    for(j in 1:n_gamma_values){
      fit = svm(anomaly_svm ~ temp_avg + salinity_avg + pressure_avg, data=train_fold, scale = TRUE, type = 'C',
                kernel = 'radial', cost = C_values[i], gamma = gamma_values[j])
      yHat_svm = predict(fit, newdata = test_fold)
      accuracy[i, j, fold] = mean(yHat_svm == test_fold$anomaly_svm)
    }
  }
}

accuracy = apply(accuracy, c(1,2), mean)

A = which(accuracy == max(accuracy), arr.ind = T)
choice = A[which.min(accuracy[A]),]

C = C_values[choice[1]] # 44
gamma = gamma_values[choice[2]] # 0.09

max(accuracy) # 93.34%

fit1 = svm(anomaly_svm ~ temp_avg + salinity_avg + pressure_avg, data = mydata, scale = TRUE, 
           type = 'C', kernel = 'radial', cost = C, gamma = gamma)
y_svm = predict(fit1, mydata)
accuracy_svm = mean(y_svm == mydata$anomaly_svm)
K = confusionMatrix(as.factor(y_svm), as.factor(mydata$anomaly_svm))
K
# Accuracy of 92.911%, recall score of 86.82%

x_range = seq(min(mydata$temp_avg), max(mydata$temp_avg), length.out = 20)
y_range = seq(min(mydata$salinity_avg), max(mydata$salinity_avg), length.out = 20)
z_range = seq(min(mydata$pressure_avg), max(mydata$pressure_avg), length.out = 20)

grid = expand.grid(temp_avg = x_range, salinity_avg = y_range, pressure_avg = z_range)
grid$anomaly = predict(fit1, newdata = grid)
y0Hat_grid = grid$anomaly

plot_ly() %>%
  add_markers(data = mydata, x = ~temp_avg, y = ~salinity_avg, z = ~pressure_avg, 
              color = ~factor(anomaly_svm), colors = c("violetred4", "gold", "seagreen"), 
              marker = list(size = 3)) %>%
  add_markers(data = grid, x = ~temp_avg, y = ~salinity_avg, z = ~pressure_avg, 
              color = ~factor(anomaly), opacity = 0.2, marker = list(size = 2)) %>%
  layout(scene = list(xaxis = list(title = "Temperature"), 
                      yaxis = list(title = "Salinity"), 
                      zaxis = list(title = "Pressure")),
         title = "SVM Decision Boundary in 3D")

hypoxia_conditions_svm = grid[grid$anomaly == -1, ]

print(hypoxia_conditions_svm)
summary(hypoxia_conditions_svm)
