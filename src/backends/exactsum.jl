"""
    ExactSum()

The exact expectation over a finite support: a sum over every value, weighted by its
probability. For `Categorical` and other `DiscreteNonParametric` distributions, `Bernoulli`,
`Binomial`, `DiscreteUniform`, `Dirac` and `PointMass`. Labelled `ExpectationExact`.
"""
struct ExactSum end

const FiniteDiscrete = Union{DiscreteNonParametric, Bernoulli, Binomial, DiscreteUniform, Dirac, PointMass}

BayesBase.supports_expectation(::ExactSum, ::Type, ::Type{<:FiniteDiscrete}) = true
BayesBase.has_expectation_label(::ExactSum, ::ExpectationExact) = true

BayesBase.prepare_expectation(::ExactSum, q::DiscreteNonParametric) = ExpectationPointSet(support(q), probs(q))
BayesBase.prepare_expectation(::ExactSum, q::Bernoulli) = ExpectationPointSet((false, true), (failprob(q), succprob(q)))
BayesBase.prepare_expectation(::ExactSum, q::Dirac) = ExpectationPointSet((q.value,), (1.0,))
BayesBase.prepare_expectation(::ExactSum, q::PointMass) = ExpectationPointSet((mean(q),), (1.0,))

function BayesBase.prepare_expectation(::ExactSum, q::Union{Binomial, DiscreteUniform})
    values = support(q)
    return ExpectationPointSet(values, pdf.(q, values))
end
