export Laplace, laplace

"""
    Laplace
This structure stores all information needed to perform expectation by Laplaces method.
it approximates integrals of the following form ∫f(x)exp(Mq(x))dx where q(x) is a scalar valued multivariate function twice differentiable. 
∫f(x)exp(Mq(x))dx ≈ (2pi/M)^(d/2) \frac{f(x_0)exp(Mq(x_0))}{det(-H(f)(x_0))^{1/2}}
# Fields
- `dimension`: dimension of integration 
- `M`        : constant in the exponent term multiplying the scalar valued function
- `point`    : point of expansion, x_0. Often chosen as the point where gradient is zero
"""

struct Laplace{T, R} <: AbstractApproximationMethod
    dimension :: Int64
    M         :: T
    point     :: R
end

laplace(d, M, point) = Laplace(d, M, point)

approximation_name(::Laplace)       = "Laplace"
approximation_short_name(::Laplace) = "LP"

using ForwardDiff
using Optim
import LinearAlgebra: det

"""
    approximate_expectation(method::Laplace, q, f::F) where {F}

This function approximates the expectation of f(x) with respect to q(x) by Laplace's method.

This methods is not robust if your function is not twice differentiable.
"""

function approximate_expectation(method::Laplace, q, f::F) where {F}
    M     = method.M
    point = method.point
    d     = method.dimension
    if point === nothing
        if d == 1
            dq = (x) -> ForwardDiff.derivative(q, x)
            neghessian = -ForwardDiff.derivative(dq, point)
        else
            neghessian = -ForwardDiff.hessian(q, point)
        end
    else
        if length(point) == 1
            dq = (x) -> ForwardDiff.derivative(q, x)
            neghessian = -ForwardDiff.derivative(dq, point)
        else
            neghessian = -ForwardDiff.hessian(q, point)
        end
    end

    if isnan(neghessian) || isless(neghessian, 0)
        error("Laplace's integration returned NaN or hessian is not negative definite")
    else
        return (2pi / M)^(d / 2) * f(point) * exp(M * q(point)) / sqrt(det(neghessian))
    end
end

function approximate_expectation(method::Laplace, q::Distribution, f::F) where {F}
    ## this function accepts q as a distribution and passes the logpdf to simulate
    ## expectation with respect to q
    dist = (x) -> logpdf(q, x)
    return approximate_expectation(method, dist, f)
end
