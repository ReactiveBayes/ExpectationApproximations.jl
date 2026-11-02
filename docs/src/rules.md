# Quadrature rules

Gaussian quadrature rules, computed with the Golub–Welsch algorithm: the nodes are the
eigenvalues of the Jacobi matrix of the orthogonal polynomials' three-term recurrence, polished
by Newton's method, and the weights are the Christoffel numbers of the orthonormal polynomials.
The weights are accurate to about `1e-13` relative to their size, the smallest ones included.

Each function returns the nodes and weights in `Float64` for the weight function of its family.
The backends map them onto distributions.

```@docs
ExpectationApproximations.gausshermite
ExpectationApproximations.gausslaguerre
ExpectationApproximations.gaussjacobi
ExpectationApproximations.gausslegendre
```
