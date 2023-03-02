export Laplace, laplace

"""
    Laplace
This structure stores all information needed to perform expectation by Laplaces method.
it approximates integrals of the following form ∫f(x)exp(Mq(x))dx where q(x) is a scalar valued multivariate function twice differentiable. 
∫f(x)exp(Mq(x))dx ≈ (2pi/M)^(d/2) \frac{f(x_0)exp(Mq(x_0))}{det(-H(f)(x_0))^{1/2}}
# Fields
- `dimension`: dimension of integration 
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

function approximate_expectation(method::Laplace, q, f::F) where {F}
    M     = method.M
    point = method.point
    d     = method.dimension
    if point === nothing
        if d == 1
            point = 0.0
            dq = (x) -> ForwardDiff.derivative(q, x)
            neghessian = -ForwardDiff.derivative(dq, point)
        else
            point = zeros(d)
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
