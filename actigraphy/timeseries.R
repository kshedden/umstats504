library(Matrix)

# See packages Rssa and swdft for singular spectrum analysis
# and time/frequency representations.  Below is a basic SSA
# implementation if you don't want to use the package.

# Return the singular spectrum and components for a time
# series, using window width p.
ssa = function(ts, p) {

    # Work with the centered input series
    ts = ts - mean(ts)

    # The autocorrelation matrix of the series
    a = acf(ts, p-1)$acf
    am = toeplitz(as.vector(a))

    # The SSA spectrum
    ei = eigen(am)
    s = sqrt(ei$values)

    # Construct the shift matrix of the centered time series.
    m = floor(length(ts) - p + 1)
    xm = array(0, c(m, p))
    for (i in 1:m) {
        xm[i, ] = ts[i:(i+p-1)]
    }

    # The SSA components
    c = (xm %*% ei$vectors) / s

    return(list(spectrum=s, components=c))
}

test_ssa = function() {

    m = 100 # Number of cycles
    q = 10  # Number of points per cycle
    n = m * q
    x = seq(0, 2*pi*m, length.out=n)
    y = sin(x) + cos(x)

    ss = ssa(y, q)

    # A sinusoidal function should have a 2-dimensional
    # spectrum, modulo edge effects.
    s = ss$spectrum
    stopifnot(sum(s[1:2]) > 0.9*sum(s))

    # The dominant two components should be periodic like
    # the input series
    c = ss$components[,1]
    p = length(c)
    c1 = c[1:(p-q)]
    c2 = c[(q+1):p]
    stopifnot(var(c1) > 1000*var(c1-c2))
    stopifnot(var(c2) > 1000*var(c1-c2))
}
