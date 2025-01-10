# -------------------------
# PM2.5 Prediction Project
# -------------------------

# Install required libraries
required_libraries <- c(
  "ggplot2", "ISLR2", "stargazer", "caret", "leaps", 
  "dplyr", "forcats", "tidyr", "gridExtra", "viridisLite"
)

for (lib in required_libraries) {
  if (!require(lib, character.only = TRUE)) {
    install.packages(lib, dependencies = TRUE)
  }
}

# Load libraries
library(tidyverse)
library(ggplot2)
library(MASS)
library(dplyr)
library(stargazer)
library(caret)
library(leaps)
library(hrbrthemes)
library(viridis)
library(forcats)
library(tidyr)
library(gridExtra)
library(readxl)

# -------------------------
# Load and Explore Dataset
# -------------------------

# Load dataset
Mydata <- read_xlsx("PRSA_data.xlsx")

# Display structure and summary of the dataset
str(Mydata)
summary(Mydata)

# -------------------------
# Data Cleaning
# -------------------------

# Replace 'cv' in 'cbwd' column with 'SW'
Mydata$cbwd <- sub("cv", "SW", Mydata$cbwd)

# Save cleaned dataset to CSV
write.csv(Mydata, "PRSA_cleaned_data.csv", row.names = FALSE)

# -------------------------
# Descriptive Analytics
# -------------------------

# Compute correlation matrix for numeric variables
numeric_data <- Mydata[, sapply(Mydata, is.numeric)]
cor_matrix <- cor(numeric_data)
print(cor_matrix)

# Identify the variable with the strongest correlation with PM2.5
correlation <- cor(numeric_data$pm2.5, numeric_data[, -1])
best_correlated_index <- which.max(abs(correlation))
best_correlated_variable <- names(numeric_data)[best_correlated_index + 1]
cat("Variable with the strongest correlation:", best_correlated_variable, "\n")

# -------------------------
# Data Visualization
# -------------------------

# Reshape numeric data to long format for easier plotting
# Clean variable names by replacing dots with spaces
# Round numeric values to the nearest integer for cleaner plots
data <- numeric_data %>%
  gather(key = "variable", value = "value")

data$variable <- gsub("\\.", " ", data$variable)
data$value <- round(data$value, 0)

# Create histograms for all variables
# Use facets to separate plots by variable and apply a clean theme
data %>%
  ggplot(aes(x = value, fill = variable)) +
  geom_histogram(alpha = 0.6, binwidth = 5) +
  scale_fill_viridis(discrete = TRUE) +
  facet_wrap(~ variable, scales = "free") +
  theme_bw() +
  theme(strip.text = element_text(size = 8),
        legend.position = "none") +
  xlab("") +
  ylab("Frequency")

# Create a boxplot for the `pm2.5` variable
numeric_data %>% 
  ggplot(aes(x = "", y = pm2.5)) + 
  geom_point() + 
  geom_boxplot()

# Create a scatterplot matrix to analyze relationships between `pm2.5` and other variables
pairs(pm2.5 ~ year + month + day + hour + DEWP + TEMP + PRES + Iws + Is + Ir, 
      data = Mydata)

# -----------------------------------
# First Model: Linear Regression
# -----------------------------------

# Modeling `pm2.5` with individual independent variables
attach(Mydata)

# Linear model with `year`
fm1 <- lm(pm2.5 ~ year)
summary(fm1)

# Linear model with `month`
fm2 <- lm(pm2.5 ~ month)
summary(fm2)

# Linear model with `hour`
fm3 <- lm(pm2.5 ~ hour)
summary(fm3)

# Linear model with `day`
fm4 <- lm(pm2.5 ~ day)
summary(fm4)

# Linear model with `PRES`
fm5 <- lm(pm2.5 ~ PRES)
summary(fm5)

# Linear model with `TEMP`
fm6 <- lm(pm2.5 ~ TEMP)
summary(fm6)

# Linear model with `DEWP`
fm7 <- lm(pm2.5 ~ DEWP)
summary(fm7)

# Linear model with `Iws`
fm8 <- lm(pm2.5 ~ Iws)
summary(fm8)

# Linear model with `Is`
fm9 <- lm(pm2.5 ~ Is)
summary(fm9)

# Linear model with `Ir`
fm10 <- lm(pm2.5 ~ Ir)
summary(fm10)

# Linear model with `cbwd`
fm11 <- lm(pm2.5 ~ cbwd)
summary(fm11)

# -----------------------------------
# Cross-Validation
# -----------------------------------

# Set seed for reproducibility
set.seed(12)

# Train a linear model with `pm2.5` as response and `Iws` as predictor
# Perform 10-fold cross-validation
firstCVModel <- train(
  form = pm2.5 ~ Iws,
  data = Mydata,
  method = "lm",
  trControl = trainControl(method = "cv", number = 10)
)

firstCVModel


# -----------------------------------
# Second Model: Linear Regression
# -----------------------------------

# Model excluding the categorical variable `cbwd`
sm1 <- lm(pm2.5 ~ year + month + day + hour + DEWP + TEMP + PRES + Iws + Is + Ir)
summary(sm1)

# Model excluding `cbwd` and time-series variables (`day`, `hour`, `month`, `year`)
sm2 <- lm(pm2.5 ~ DEWP + TEMP + PRES + Iws + Is + Ir)
summary(sm2)

# Model excluding time-series variables (`day`, `hour`, `month`, `year`) but including `cbwd`
sm3 <- lm(pm2.5 ~ DEWP + TEMP + PRES + Iws + Is + Ir + cbwd)
summary(sm3)

# Model including all predictors
sm4 <- lm(pm2.5 ~ year + month + day + hour + DEWP + TEMP + PRES + Iws + Is + Ir + cbwd)
summary(sm4)

# -----------------------------------
# Cross-Validation
# -----------------------------------

# Train a linear model with `pm2.5` as response and all predictors
# Perform 10-fold cross-validation
secondCVModel <- train(
  form = pm2.5 ~ year + month + day + hour + DEWP + TEMP + PRES + Iws + Is + Ir + cbwd,
  data = Mydata,
  method = "lm",
  trControl = trainControl(method = "cv", number = 10)
)

secondCVModel


# ------------------------------------------------
# Fitted vs Residuals and Residual Diagnostics
# ------------------------------------------------

# Fitted vs Residuals for the first model (highest adjusted R-squared)
fmResids <- fm8$residuals
fmlFitted <- fm8$fitted.values
plot(fmlFitted, fmResids, main = "Fitted vs Residuals (First Model)", 
     xlab = "Fitted Values", ylab = "Residuals")

# Fitted vs Residuals for the second model (highest adjusted R-squared)
smResids <- sm4$residuals
smlFitted <- sm4$fitted.values
plot(smlFitted, smResids, main = "Fitted vs Residuals (Second Model)", 
     xlab = "Fitted Values", ylab = "Residuals")

# Histogram of residuals for both models
hist(fmResids, main = "Residuals Histogram (First Model)", xlab = "Residuals", col = "lightblue")
hist(smResids, main = "Residuals Histogram (Second Model)", xlab = "Residuals", col = "lightblue")

# Q-Q Plot of residuals for both models
qqnorm(fmResids, main = "Q-Q Plot (First Model)")
qqline(fmResids, col = "red")
qqnorm(smResids, main = "Q-Q Plot (Second Model)")
qqline(smResids, col = "red")

# ------------------------------------------------
# Adding Polynomial Terms to Models
# ------------------------------------------------

# Adding polynomial term for `year`
smp1 <- lm(pm2.5 ~ year + I(year^2) + month + day + hour + DEWP + TEMP +
             PRES + Iws + Is + Ir + cbwd)
summary(smp1)

# Adding polynomial term for `month`
smp2 <- lm(pm2.5 ~ year + month + I(month^2) + day + hour + DEWP + TEMP +
             PRES + Iws + Is + Ir + cbwd)
summary(smp2)

# Adding polynomial term for `hour`
smp3 <- lm(pm2.5 ~ year + month + day + hour + I(hour^2) + DEWP + TEMP +
             PRES + Iws + Is + Ir + cbwd)
summary(smp3)

# Adding polynomial term for `day`
smp4 <- lm(pm2.5 ~ year + month + day + I(day^2) + hour + DEWP + TEMP +
             PRES + Iws + Is + Ir + cbwd)
summary(smp4)

# Adding polynomial term for `TEMP`
smp5 <- lm(pm2.5 ~ year + month + day + hour + DEWP + TEMP + I(TEMP^2) +
             PRES + Iws + Is + Ir + cbwd)
summary(smp5)

# Adding polynomial term for `PRES`
smp6 <- lm(pm2.5 ~ year + month + day + hour + DEWP + TEMP + PRES +
             I(PRES^2) + Iws + Is + Ir + cbwd)
summary(smp6)

# Adding polynomial term for `DEWP`
smp7 <- lm(pm2.5 ~ year + month + day + hour + DEWP + I(DEWP^2) + TEMP +
             PRES + Iws + Is + Ir + cbwd)
summary(smp7)

# Adding polynomial term for `Iws`
smp8 <- lm(pm2.5 ~ year + month + day + hour + DEWP + TEMP + PRES +
             Iws + I(Iws^2) + Is + Ir + cbwd)
summary(smp8)

# Adding polynomial term for `Is`
smp9 <- lm(pm2.5 ~ year + month + day + hour + DEWP + TEMP + PRES +
             Iws + Is + I(Is^2) + Ir + cbwd)
summary(smp9)

# Adding polynomial term for `Ir`
smp10 <- lm(pm2.5 ~ year + month + day + hour + DEWP + TEMP + PRES +
              Iws + Is + Ir + I(Ir^2) + cbwd)
summary(smp10)


# Train a linear regression model with pm2.5 as the response and month as a single predictor.
# Use 10-fold cross-validation to assess model performance.
thirdCVModel <- train(
  form = pm2.5 ~ month, 
  data = Mydata,        
  method = "lm",        
  trControl = trainControl(method = "cv", number = 10) 
)
# Display the summary of the cross-validated model
thirdCVModel

# Residual analysis for the third model (model with highest adjusted R-squared)
smpResids <- smp2$residuals        
smplFitted <- smp2$fitted.value    
# Plot fitted values vs residuals to check for patterns
plot(smplFitted, smpResids, main = "Residuals vs Fitted Values",
     xlab = "Fitted Values", ylab = "Residuals", pch = 20)

# Perform feature selection using stepwise AIC
# Forward stepwise selection: Adds variables iteratively
step <- stepAIC(smp2, direction = "forward", trace = FALSE)
step$anova 

# Backward stepwise selection: Removes variables iteratively
step1 <- stepAIC(smp2, direction = "backward", trace = FALSE)
step1$anova 

# After backward selection, the variable year is suggested to be removed.
# Refit the model without the year variable
smp2 <- lm(pm2.5 ~ month + I(month^2) + day + hour + DEWP + TEMP + PRES +
             Iws + Is + Ir + cbwd) 
# Display the summary of the updated model
summary(smp2)

# Add interaction terms to explore relationships between predictors
# Adding interaction between month and year
smpn1 <- lm(pm2.5 ~ year + month + I(month^2) + month:year + day + hour +
              DEWP + TEMP + PRES + Iws + Is + Ir + cbwd)
summary(smpn1)

# Adding interaction between month and day
smpn2 <- lm(pm2.5 ~ year + month + I(month^2) + month:day + day + hour +
              DEWP + TEMP + PRES + Iws + Is + Ir + cbwd)
summary(smpn2)

# Adding interaction between month and hour
smpn3 <- lm(pm2.5 ~ year + month + I(month^2) + month:hour + day + hour +
              DEWP + TEMP + PRES + Iws + Is + Ir + cbwd)
summary(smpn3)

# Adding interaction between month and TEMP
smpn4 <- lm(pm2.5 ~ year + month + I(month^2) + month:TEMP + day + hour +
              DEWP + TEMP + PRES + Iws + Is + Ir + cbwd)
summary(smpn4)

# Adding interaction between month and PRES
smpn5 <- lm(pm2.5 ~ year + month + I(month^2) + month:PRES + day + hour +
              DEWP + TEMP + PRES + Iws + Is + Ir + cbwd)
summary(smpn5)

# Adding interaction between month and DEWP
smpn6 <- lm(pm2.5 ~ year + month + I(month^2) + month:DEWP + day + hour +
              DEWP + TEMP + PRES + Iws + Is + Ir + cbwd)
summary(smpn6)

# Adding interaction between month and Iws
smpn7 <- lm(pm2.5 ~ year + month + I(month^2) + month:Iws + day + hour +
              DEWP + TEMP + PRES + Iws + Is + Ir + cbwd)
summary(smpn7)

# Adding interaction between month and Is
smpn8 <- lm(pm2.5 ~ year + month + I(month^2) + month:Is + day + hour +
              DEWP + TEMP + PRES + Iws + Is + Ir + cbwd)
summary(smpn8)

# Adding interaction between month and Ir
smpn9 <- lm(pm2.5 ~ year + month + I(month^2) + month:Ir + day + hour +
              DEWP + TEMP + PRES + Iws + Is + Ir + cbwd)
summary(smpn9)

# Adding interaction between month and cbwd
smpn10 <- lm(pm2.5 ~ year + month + I(month^2) + month:cbwd + day + hour +
               DEWP + TEMP + PRES + Iws + Is + Ir + cbwd)
summary(smpn10)

# Adding interaction between month and itself (quadratic interaction)
smpn11 <- lm(pm2.5 ~ year + month + I(month^2) + month:month + day + hour +
               DEWP + TEMP + PRES + Iws + Is + Ir + cbwd)
summary(smpn11)


# Train a linear model with pm2.5 as the response and TEMP as the predictor.
# Perform 10-fold cross-validation to evaluate the model's performance.
fourthCVModel <- train(
  form = pm2.5 ~ TEMP,  
  data = Mydata,        
  method = "lm",        
  trControl = trainControl(method = "cv", number = 10)  
)
fourthCVModel

# Perform residual analysis for the selected model with the highest adjusted R-squared.
# The residuals and fitted values are extracted and plotted to check for any patterns.
smpnResids <- smpn4$residuals
smpnlFitted <- smpn4$fitted.value
plot(smpnlFitted, smpnResids, main = "Residuals vs Fitted Values",
     xlab = "Fitted Values", ylab = "Residuals", pch = 20)

# Perform feature selection using stepwise AIC to select significant predictors.

# Forward stepwise selection
step <- stepAIC(smpn4, direction = "forward", trace = FALSE)
step$anova

# Backward stepwise selection
step1 <- stepAIC(smpn4, direction = "backward", trace = FALSE)
step1$anova

# Perform best subset selection to identify the "best" model for pm2.5.
# Use the regsubsets function to evaluate different subsets of predictors.
subsetModel <- regsubsets(pm2.5 ~ year + month + day + hour + DEWP + TEMP +
                            PRES + Iws + Is + Ir + cbwd, data = Mydata)
# Plot the best subsets based on adjusted R-squared.
plot(subsetModel, scale = "adjr2")

# The top model based on adjusted R-squared includes DEWP, TEMP, and cbwd.
bestCVModel1 <- train(
  form = pm2.5 ~ DEWP + TEMP + cbwd,  
  data = Mydata,  
  method = "lm",  
  trControl = trainControl(method = "cv", number = 10)  
)
bestCVModel1

# Prediction model: Create a dataset with different Iws values (10, 20, 30)
# Predict the pm2.5 values for these Iws values using the trained model (fm8).
Iws_predictions <- data.frame(Iws = c(10, 20, 30))
predict(fm8, Iws_predictions)  

# Calculate the prediction interval to give a range of predicted values with 95% confidence.
# This ensures the predicted value will fall within this range with 95% certainty.
prediction_interval <- predict(fm8, newdata = Iws_predictions, interval = "confidence", level = 0.95)
prediction_interval
