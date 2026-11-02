"""
    ExactSum()

The exact expectation over a finite support: a sum over every value, weighted by its
probability. For `Categorical` and other `DiscreteNonParametric` distributions, `Bernoulli`,
`Binomial`, `DiscreteUniform`, `Dirac` and `PointMass`. Labelled `Exact`.
"""
struct ExactSum end

const FiniteDiscrete = Union{DiscreteNonParametric, Bernoulli, Binomial, DiscreteUniform, Dirac, PointMass}

BayesBase.supports(::ExactSum, ::Type, ::Type{<:FiniteDiscrete}) = true
BayesBase.haslabel(::ExactSum, ::Exact) = true

BayesBase.prepare_expectation(::ExactSum, q::DiscreteNonParametric) = PointSet(support(q), probs(q))
BayesBase.prepare_expectation(::ExactSum, q::Bernoulli) = PointSet((false, true), (failprob(q), succprob(q)))
BayesBase.prepare_expectation(::ExactSum, q::Dirac) = PointSet((q.value,), (1.0,))
BayesBase.prepare_expectation(::ExactSum, q::PointMass) = PointSet((mean(q),), (1.0,))

function BayesBase.prepare_expectation(::ExactSum, q::Union{Binomial, DiscreteUniform})
    values = support(q)
    return PointSet(values, pdf.(q, values))
end
