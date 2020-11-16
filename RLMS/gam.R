library(mgcv)

#source("data_prep.R")

dx$log2J10 = log2(dx$J10)
dx$log2J8 = log2(dx$J8)
dx$female = as.factor(dx$female)

m = gam(log2J10 ~ status + s(age, by=female) + female + s(year) + s(educ) + s(log2J8),
        data=dx)

pdf("rlms_gam.pdf")
plot(m)
dev.off()
