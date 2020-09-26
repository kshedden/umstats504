library(geepack)
library(splines)
library(purrr)
library(stringr)
library(ggplot2)

set.seed(4432532)

pdf("basic_gee.pdf")

# Generate different types of covariates.
n = 2000 # Overall sample size
m = 25 # Number of observations per group
d = seq(0, 9.9, length.out=n)
month = floor((d * 12) %% 12)
year = floor(d)
female = as.integer(runif(n, 0, 1) < 1/2)
age = runif(n, 0, 80)
group = kronecker(seq(n/m), array(1, m))

# Permute group so that it is independent of time.
group = sample(group, length(group), replace=FALSE)

# Generate the marginal mean structure with a log-linear form,
# including main effects and a month x female interaction.
monthx = sin(2*pi*d)
yearx = d / 3
logey = -1 + monthx + 0.5*yearx - monthx*(2*female-1) + ((age - 40) / 40)**2
ey = exp(logey)

# Use copula to generate Poisson-distributed values with marginal mean
# equal to ey, that are correlated within levels of group but
# independent between levels of group.
ar = 0.9
e = rnorm(n)
e = array(e, c(10, n/10))
for (k in 2:10) {
    e[k,] = ar*e[k-1,] + sqrt(1-ar^2)*e[k,]
}
e = array(e, n)
e = pnorm(e)
y = qpois(e, lambda=ey)

df = data.frame(month=month, year=year, female=female, age=age, y=y, group=group)
df$month = as.factor(df$month)
df$year = as.factor(df$year)
df$group = as.factor(df$group)
df$y = as.numeric(df$y) # Not clear why this is needed

# Fit a glm, ignoring the groups
m0 = glm(y ~ month*female + year + bs(age, 5), family=poisson(), data=df)

# Fit a glm, using fixed effects to account for the groups
m1 = glm(y ~ month*female + year + bs(age, 5) + group, family=poisson(), data=df)

# Fit three Poisson GEE models and inspect their scale parameters
m2 = geeglm(y ~ month + year + female + bs(age, 5), id=df$group, family=poisson(), corstr="ar1", data=df)
m3 = geeglm(y ~ month*female + year + bs(age, 5), id=df$group, family=poisson(), corstr="ar1", data=df)
m4 = geeglm(y ~ (month + year)*female + bs(age, 5), id=df$group, family=poisson(), corstr="ar1", data=df)
mm = list(m2, m3, m4)
cat("Estimated scale parameters for three Poisson models:\n")
scales = map_dbl(mm, function(x)summary(x)$dispersion$Estimate)
cat(scales)
cat("\n\n")

cat("QIC for three Poisson models:\n")
print(map(mm, QIC))

get_df = function(m, i) {
    p = coef(m)
    se = sqrt(diag(vcov(m)))
    a = sprintf("year%d", y)
    j0 = which(a %in% names(p))[[1]]
    est = NULL
    se = NULL

    # Construct contrasts between every year and year 0.
    for (y in 1:9) {
        a = sprintf("year%d", y)
        j = which(names(p) %in% a)
        c = array(0, length(p))
        c[j0] = -1
        c[j] = 1
        est[y] = sum(c * p)
        se[y] = sqrt(c %*% vcov(m) %*% c)
    }

    dp = data.frame(year_effect=est, year_se=se, year=seq(9)+i/10)
    dp$model = i
    return(dp)
}

# Create a dataframe with estimates and standard errors of the
# contrasts between each year and year 0, for three approaches: 0: glm
# with no consideration of groups, 1: glm with group fixed effects, 3:
# GEE with an autoregressive working correlation structure.
#
# This is not a simulation study, it only works with one simulated
# dataset.  Therefore the results may vary when changing the random
# seed (above).  But in general, the parameter estimates will be
# similar for the three methods.  The standard errors are generally
# largest for fixed effects (due to the "Neyman Scott" phenomenon),
# smallest for GLM (which are not valid standard errors), and
# intermediate for GEE, which gives standard errors that are robust to
# variance and (certain forms of) covariance estimation.
dp0 = get_df(m0, 0)
dp1 = get_df(m1, 1)
dp3 = get_df(m3, 3)

# Prepare a dataframe for plotting
dp = rbind(dp0, dp1, dp3)
dp$ymin = dp$year_effect - dp$year_se
dp$ymax = dp$year_effect + dp$year_se
dp$model = as.factor(dp$model)

# Plot the estimates with the standard errors.
p = ggplot(dp, aes(x=year, y=year_effect, color=model)) + geom_line()
p = p + geom_errorbar(aes(ymin=ymin, ymax=ymax))
print(p)

# Plot the standard errors to make it easier to see the differences.
p = ggplot(dp, aes(x=year, y=year_se, color=model)) + geom_line()
print(p)

dev.off()
