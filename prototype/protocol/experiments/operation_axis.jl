# The operation axis (U0.2, item 7 of the plan; an open question in PLAN.md): should `supports`
# and `haslabel` take the operation (`expectation`, `mean_cov`, `expectation_gradient`) as an
# argument now, because adding it later would be breaking?
#
# The experiment adds it *after the fact*, in a separate module, on top of the released
# three-argument protocol, and checks what a later addition would break. Run in the
# universal-dev environment:
#
#     julia --project=~/Projects/Julia/universal-dev operation_axis.jl

using BayesBase, ProtoBackends, Distributions, ExponentialFamily, Test

const P = BayesBase
const B = ProtoBackends

"The operation axis as a later, additive extension of the protocol."
module WithOperations

    using BayesBase: BayesBase, FirstOf
    const P = BayesBase

    "A second operation: mean and covariance of a pushforward, `(E[f], Cov[f])`."
    function mean_cov end

    # The four-argument form. For `expectation` it forwards to the three-argument form, so every
    # backend written against the released protocol keeps working, unchanged; every other operation
    # is opt-in
    supports(backend, ::typeof(P.expectation), ::Type{F}, ::Type{Q}) where {F, Q} = P.supports(backend, F, Q)
    supports(backend, op, ::Type, ::Type) = false

    # Routing per operation, decided by inference as in `FirstOf`
    choice(router::FirstOf, op, ::Type{F}, ::Type{Q}) where {F, Q} = find(router.backends, op, F, Q)
    find(::Tuple{}, op, ::Type, ::Type) = 0
    function find(backends::Tuple, op, ::Type{F}, ::Type{Q}) where {F, Q}
        supports(first(backends), op, F, Q) && return 1
        rest = find(Base.tail(backends), op, F, Q)
        return rest == 0 ? 0 : rest + 1
    end

end

# A backend written after the extension, opting into the new operation for Gaussians
struct MeanCovOnly end
WithOperations.supports(::MeanCovOnly, ::typeof(WithOperations.mean_cov), ::Type, ::Type{<:NormalMeanVariance}) = true

function folds_to(f, types, value)
    code = only(code_typed(f, types; optimize = true)).first.code
    returns = filter(statement -> statement isa Core.ReturnNode, code)
    length(returns) == 1 || return false
    returned = only(returns).val
    return (returned isa QuoteNode ? returned.value : returned) === value
end

const W = WithOperations
const OPS = (typeof(P.expectation), Type{Any}, Type{NormalMeanVariance{Float64}})

@testset "the operation axis, added later" begin
    gh, cf = B.GaussHermite(8), B.ClosedForms()

    # existing backends answer the four-argument query without a single change
    @test W.supports(gh, P.expectation, Any, NormalMeanVariance{Float64}) === true
    @test W.supports(cf, P.expectation, typeof(log), Gamma{Float64}) === true
    @test W.supports(gh, W.mean_cov, Any, NormalMeanVariance{Float64}) === false

    # existing callers of the three-argument form are untouched
    @test P.supports(gh, Any, NormalMeanVariance{Float64}) === true

    # a new backend opts into the new operation only
    @test W.supports(MeanCovOnly(), W.mean_cov, Any, NormalMeanVariance{Float64}) === true
    @test W.supports(MeanCovOnly(), P.expectation, Any, NormalMeanVariance{Float64}) === false

    # routing per operation is still decided by inference
    router = P.FirstOf(MeanCovOnly(), gh)
    @test W.choice(router, P.expectation, Any, NormalMeanVariance{Float64}) == 2
    @test W.choice(router, W.mean_cov, Any, NormalMeanVariance{Float64}) == 1
    @test folds_to(W.supports, (typeof(gh), OPS...), true)
    @test folds_to(W.choice, (typeof(router), typeof(P.expectation), Type{Any}, Type{NormalMeanVariance{Float64}}), 2)
    @test folds_to(W.choice, (typeof(router), typeof(W.mean_cov), Type{Any}, Type{NormalMeanVariance{Float64}}), 1)
end
