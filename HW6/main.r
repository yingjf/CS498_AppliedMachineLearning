library(glmnet)
library(MASS)
library(ISLR)

# Problem 1
summary(Boston)
fit1=lm(medv~.,Boston)
summary(fit1)
# par(mfrow=c(2,2))
plot(fit1, id.n = 10) # which gives 365, 369, 372, 373

# Calculate hat values
hv = hatvalues(fit1)
hv_max_index = which(hv==max(hv)) # which gives 381

# Calcualte standardized residuals
sr = rstandard(fit1)
sr_max_index = which(sr==max(sr)) # which gives 369

# Calculate cooks distance
ck = cooks.distance(fit1)
ck_max_index = which(ck==max(ck)) # which gives 369

# Problem 2
# Remove the 3 outliers 365, 370, 373, 369 observed from the plot
Boston1 <- Boston[-c(365, 369, 370, 371, 372,373, 366, 413), ] 
fit2=lm(medv~.,Boston1)
summary(fit2)
par(mfrow=c(2,2))
plot(fit2)
plot(fitted(fit2), residuals(fit2)); 
title("Residual vs Fit. value with outliers removed");

# Problem 3
# boxcox
bc <- boxcox(fit2)
lambda <- bc$x
lik <- bc$y
combined <- cbind(lambda, lik)
combined[order(-lik), ]
#lambda is  0.30303030

# regression with transformation
fit3=lm((medv^(1/3) - 1)/0.3~.,Boston1)
summary(fit3)
plot(fit3)
plot(fitted(fit3), residuals(fit3)); 
title("Residual vs Fit. value with boxcox transformation")

# PLot predicted against true values
plot(Boston1$medv, (predict(fit3) * 0.3 + 1 )^ 3, xlab="actual",ylab="predicted")
abline(a=0,b=1)
title("Predicted vs True values")
