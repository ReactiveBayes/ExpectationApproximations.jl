```@meta
CurrentModule = ExpectationApproximations
```

# Quadrature rules

The quadrature backends are built on four classical Gaussian rules. Each approximates an integral
against a weight function by a weighted sum over ``n`` nodes, and is exact for polynomials up to
degree ``2n - 1``:

| Rule | Integral | Used by |
|:-----|:---------|:--------|
| [`gausshermite`](@ref) | ``\int f(x)\, e^{-x^2} dx`` | [`GaussHermite`](@ref), for Gaussians |
| [`gausslaguerre`](@ref) | ``\int_0^\infty f(x)\, x^{\alpha} e^{-x} dx`` | [`GaussLaguerre`](@ref), for Gammas of any shape |
| [`gaussjacobi`](@ref) | ``\int_{-1}^1 f(x)\, (1 - x)^\alpha (1 + x)^\beta dx`` | [`GaussJacobi`](@ref), for Betas |
| [`gausslegendre`](@ref) | ``\int_{-1}^1 f(x)\, dx`` | [`QuantileLegendre`](@ref), for any continuous distribution |

```@example rules
using ExpectationApproximations
const EA = ExpectationApproximations
nodes, weights = EA.gausshermite(5)
sum(weights .* nodes .^ 4), 3 * sqrt(π) / 4   # ∫ x⁴ e^{-x²} dx
```

## How they are computed

The rules are computed with the Golub–Welsch algorithm. The nodes are the eigenvalues of the
Jacobi matrix, the symmetric tridiagonal matrix of the three-term recurrence of the orthogonal
polynomials, refined by Newton's method on that recurrence. The weights are the Christoffel
numbers ``1 / \sum_k p_k(x_i)^2`` of the orthonormal polynomials.

The usual alternative computes the weights from the first components of the eigenvectors, which
lose relative precision in the tails. With the Christoffel numbers, the smallest weights are
accurate to about ``10^{-13}`` of their size, measured against 256-bit arithmetic. The rules are
computed in `Float64` when a backend is constructed. For Gauss–Laguerre and Gauss–Jacobi, whose
weight function depends on the shape, they are computed when an expectation is prepared.

```@docs
gausshermite
gausslaguerre
gaussjacobi
gausslegendre
```
