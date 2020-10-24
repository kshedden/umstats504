library(Matrix)
library(SuperGauss)

# Covariance matrix for fractional Brownian motion
# of length n at Hurst parameter h.
getR = function(h, n) {
    k = seq(0, n-1)
    v = (k + 1)^(2*h) - 2*k^(2*h) + abs(k - 1)^(2*h)
    v = v / 2
    rt = Toeplitz$new(acf=v)
    return(list(matrix=rt, vector=v))
}

# Log-likelihood for data z at Hurst parameter h.
getL = function(h, degf, z) {

    # https://www.hindawi.com/journals/mpe/2014/490568/
    rv = getR(h, length(z))
    r = rv$matrix
    v = rv$vector
    ld = r$log_det()
    qf = z %*% solve(r, z)
    p = length(z)
    ll = -(degf+p) * log(1 + qf/degf) / 2
    ll = ll - ld/2
    return(ll[1,1])
}

estimateHurst = function(z, degf) {

    ll0 = optimize(function(h)-getL(h, 20, z), lower=0.1, upper=0.9)
    return(ll0)
}

test_hurst = function() {

    z = rnorm(10000)
    h = estimateHurst(z, 20)
    stopifnot(abs(h$minimum - 0.5) < 0.05)

    z = seq(0, 100*2*pi)
    z = sin(z)
    h = estimateHurst(z, 20)
    stopifnot(h$minimum > 0.7)
}

