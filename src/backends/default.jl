"""
    EADefault(n = 16)

The default backend: a `BayesBase.ExpectationFirstOf` that sends finite discrete distributions to
the exact sum, Gaussians (and log-normal and logit-normal distributions) to Gauss–Hermite, Gamma
distributions to Gauss–Laguerre and Beta distributions to Gauss–Jacobi, each with `n` points per
dimension, and any other continuous univariate distribution to Gauss–Legendre in probability
space. The choice is made by inference. Products of distributions are routed component by
component.
"""
function EADefault(n::Integer = 16)
    return ExpectationFirstOf(ExactSum(), GaussHermite(n), GaussLaguerre(n), GaussJacobi(n), QuantileLegendre(n))
end
