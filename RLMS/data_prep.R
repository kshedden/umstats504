library(dplyr)
library(readr)

# Variables to keep
#va = cols_only(idind, "J1", "age", "year", "J69_9C", "educ",
#      "status", "J8", "J10", "H5", "psu", "OCCUP08")

c = cols_only(idind=col_double(), J1=col_double(), age=col_double(), year=col_double(),
              J69_9C=col_double(), educ=col_double(), status=col_factor(), J8=col_double(),
              J10=col_double(), H5=col_double(), psu=col_factor(), OCCUP08=col_factor())

df = read_tsv("RLMS-HSE_IND_1994_2018_STATA.tab.gz", col_types=c)

# Drop people who are not working
df = df[df$J1==1,]

# Recode gender
df$female = as.integer(df$H5 == 2)

# Drop the original versions of variables that we no longer need.
df = subset(df, select=-c(J1, H5))

# Drop rows with missing values
dx = df[complete.cases(df),]

# Center year at 2000, making it more centered
dx$year = dx$year - 2000

# Remove special codes
for (v in c("J10", "J8", "educ", "age")) {
    ii = dx[[v]] < 99999997
    dx = dx[ii,]
}