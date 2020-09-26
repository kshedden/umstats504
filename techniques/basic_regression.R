library(dplyr)
library(splines)
library(purrr)
library(ggplot2)
library(stringr)

# Generate different types of covariates.
n = 2000
d = seq(0, 9.9, length.out=n)
month = floor((d * 12) %% 12)
year = floor(d)
female = as.integer(runif(n, 0, 1) < 1/2)
age = runif(n, 0, 80)

# Generate an outcome.  The mean structure is log-linear, with a month x female
# interaction.
monthx = sin(2*pi*d)
yearx = d / 3
logey = -1 + monthx + 0.5*yearx - monthx*(2*female-1) + ((age - 40) / 40)**2
ey = exp(logey)
y = rpois(lambda=ey, n=n)

df = data.frame(month=month, year=year, female=female, age=age, y=y)
df$month = as.factor(df$month)
df$year = as.factor(df$year)

# Fit three Poisson models and compare using AIC.
m1 = glm(y ~ month + year + female + bs(age, 5), family=poisson(), data=df)
m2 = glm(y ~ month*female + year + bs(age, 5), family=poisson(), data=df)
m3 = glm(y ~ (month + year)*female + bs(age, 5), family=poisson(), data=df)
mm = list(m1, m2, m3)
cat("AIC of three Poisson models:\n")
cat(map_dbl(mm, AIC))
cat("\n\n")

# Fit three quasi-Poisson models and consider their scale (dispersion) parameters.
# Note that the correct model (or models that contain it) have scale parameters close to 1.
qm1 = glm(y ~ month + year + female + bs(age, 5), family=quasipoisson(), data=df)
qm2 = glm(y ~ month*female + year + bs(age, 5), family=quasipoisson(), data=df)
qm3 = glm(y ~ (month + year)*female + bs(age, 5), family=quasipoisson(), data=df)
qm = list(qm1, qm2, qm3)
scales = map_dbl(qm, function(x)summary(x)$dispersion)
cat("Scale parameters:\n")
cat(scales)
cat("\n\n")

# Compare the three quasi-GLM models using the quasi-AIC (QAIC).
s = scales[3] # Scale of the parent model
llf = map_dbl(mm, function(m)as.numeric(logLik(m)))
degf = map_dbl(qm, function(m)length(coef(m)))
cat("QAIC for three quasi-Poisson models:\n")
qaic = -2*llf/s + 2*degf
cat(qaic)
cat("\n\n")

pdf("basic_regression.pdf")

# Plot the year effects, note that these are relative to the reference
# year.
p2 = coef(m2)
ii = str_detect(names(p2), "year")
dp = data.frame(year_effect=p2[ii], year=seq(sum(ii)))
p = ggplot(dp, aes(x=year, y=year_effect)) + geom_line()
print(p)

# Plot the month effects for females and males, note that these are relative
# to the reference month.
i1 = str_detect(names(p2), "month[0-9]*$")
i2 = str_detect(names(p2), "month[0-9]*:")
dp1 = data.frame(female=0, month_effect=p2[i1], month=seq(sum(i1)))
dp2 = data.frame(female=1, month_effect=p2[i1]+p2[i2], month=seq(sum(i2)))
dp = rbind(dp1, dp2)
dp$female = as.factor(dp$female)
p = ggplot(dp, aes(x=month, y=month_effect, color=female)) + geom_line()
print(p)

# Plot the age effects
dfx = df[1:100, ]
dfx$age = levels(df$year)[1]
dfx$month = levels(df$month)[1]
dfx$female = 0
dfx$age = seq(0, 80, length.out=100)
df$w = 1
dfx$w = 0
dfz = rbind(df, dfx)
qm2x = glm(y ~ month*female + year + bs(age, 5), family=quasipoisson(), weight=w, data=dfz)
yhat = predict(qm2x)
yhata = yhat[(length(yhat)-99):length(yhat)]
dp = data.frame(yhat=yhata, age=dfx$age)
p = ggplot(dp, aes(x=age, y=yhata)) + geom_line()
print(p)

dev.off()
