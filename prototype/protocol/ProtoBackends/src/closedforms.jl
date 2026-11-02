"""
    ClosedForms()

Exact expectations for a handful of pairs, a stand-in for ClosedFormExpectations: the first two
moments of a Gaussian, `E[log x]` under a Gamma and a log-normal, and `E_q[log p(x)]` of a
Gaussian `p` under a Gaussian `q`.
"""
struct ClosedForms end

BayesBase.haslabel(::ClosedForms, ::Exact) = true

BayesBase.supports(::ClosedForms, ::Type{typeof(identity)}, ::Type{<:Gaussian}) = true
BayesBase.supports(::ClosedForms, ::Type{typeof(abs2)}, ::Type{<:Gaussian}) = true
BayesBase.supports(::ClosedForms, ::Type{typeof(log)}, ::Type{<:Gamma}) = true
BayesBase.supports(::ClosedForms, ::Type{typeof(log)}, ::Type{<:LogNormal}) = true
BayesBase.supports(::ClosedForms, ::Type{<:Logpdf{<:Gaussian}}, ::Type{<:Gaussian}) = true

BayesBase.expectation(::ClosedForms, ::typeof(identity), q::Gaussian) = mean(q)
BayesBase.expectation(::ClosedForms, ::typeof(abs2), q::Gaussian) = mean(q)^2 + var(q)
BayesBase.expectation(::ClosedForms, ::typeof(log), q::Gamma) = digamma(shape(q)) + log(scale(q))
BayesBase.expectation(::ClosedForms, ::typeof(log), q::LogNormal) = first(params(q))

function BayesBase.expectation(::ClosedForms, f::Logpdf{<:Gaussian}, q::Gaussian)
    m, v = mean(f.dist), var(f.dist)
    return -(log2π + log(v)) / 2 - ((mean(q) - m)^2 + var(q)) / (2v)
end
