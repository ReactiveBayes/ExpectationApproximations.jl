"""
    ExpectationApproximations.Testing

The conformance suite for backends of the expectation protocol: every backend of this package
passes it, and a backend in another package can run it from its own tests.

```julia
using ExpectationApproximations.Testing: Case, test_backend
test_backend(MyBackend(), [Case(Normal(0.0, 1.0), abs2, 1.0), Case(Gamma(2.0, 1.0), log, digamma(2.0))])
```
"""
module Testing

using BayesBase: BayesBase, FactorizedJoint, expectation, prepare_expectation, supports, isexact, isstochastic, withrng
using Compat: @compat
using ForwardDiff: ForwardDiff
using Random: AbstractRNG, Xoshiro
using Test: @test, @testset

@compat public Case, Check, conformance, test_backend

"""
    Case(q, f, expected; rtol = 1e-10)

An expectation with a known value: the distribution `q`, the integrand `f` and
`expected = E_q[f]`. A stochastic backend needs a looser `rtol`.
"""
struct Case{Q, F, R}
    q::Q
    f::F
    expected::R
    rtol::Float64
end

Case(q, f, expected; rtol::Real = 1.0e-10) = Case(q, f, expected, Float64(rtol))

"""
    Check

The outcome of one check of [`conformance`](@ref): its `name`, the index of the `case` it
concerns (zero for checks across cases), whether it `passed`, and a `detail` for when it did not.
"""
struct Check
    name::String
    case::Int
    passed::Bool
    detail::String
end

"""
    conformance(backend, cases; rng = Xoshiro(1234)) -> Vector{Check}

Checks `backend` on every [`Case`](@ref), without recording tests:

- `supports`: the backend supports the case;
- `correct`: the expectation equals the expected value, to the case's `rtol`;
- `exact label holds`: where the backend is labelled `Exact`, to rounding;
- `prepared equals direct`: a prepared expectation gives what a direct one gives, for a
  deterministic backend;
- `reproducible`: the same random number generator gives the same result, for a stochastic one;
- `inferred`: evaluating a prepared expectation is type-stable;
- `allocation-free`: and allocates nothing;
- `duals pass`: ForwardDiff differentiates through the integrand of a prepared expectation;
- `product`: the expectation of `f₁(x) f₂(y)` under the product of the first two cases' distributions
  is the product of their expectations, for a deterministic backend.

Stochastic backends get `rng` from above with `withrng`.
"""
function conformance(backend, cases; rng::AbstractRNG = Xoshiro(1234))
    checks = Check[]
    for (i, case) in enumerate(cases)
        append!(checks, check_case(backend, case, i, rng))
    end
    if length(cases) >= 2
        push!(checks, check_product(backend, cases[1], cases[2]))
    end
    return checks
end

"""
    test_backend(backend, cases; rng = Xoshiro(1234))

Runs [`conformance`](@ref) and records every check as a test, in a test set named after the
backend; a failed check also logs its detail.
"""
function test_backend(backend, cases; kwargs...)
    @testset "conformance of $(nameof(typeof(backend)))" begin
        for check in conformance(backend, cases; kwargs...)
            check.passed || @error "Conformance check failed" check.name check.case check.detail
            @test check.passed
        end
    end
    return nothing
end

function check_case(backend, case::Case, i, rng)
    (; q, f, expected, rtol) = case
    checks = Check[]
    record(name, passed, detail = "") = push!(checks, Check(name, i, passed, detail))

    if !supports(backend, f, q)
        record("supports", false, "the backend does not support $(typeof(q))")
        return checks
    end
    record("supports", true)

    stochastic = isstochastic(backend, f, q)
    configured = stochastic ? withrng(backend, copy(rng)) : backend
    prepared = prepare_expectation(configured, q)
    value = expectation(prepared, f)
    record("correct", agrees(value, expected, rtol), "got $value, expected $expected, rtol $rtol")

    if isexact(backend, f, q)
        record("exact label holds", agrees(value, expected, 1.0e-12), "got $value, expected $expected exactly")
    end

    if stochastic
        again = expectation(prepare_expectation(withrng(backend, copy(rng)), q), f)
        record("reproducible", again == value, "got $value, then $again")
    else
        direct = expectation(backend, f, q)
        record("prepared equals direct", direct == value, "prepared $value, direct $direct")
    end

    types = Base.return_types(expectation, (typeof(prepared), typeof(f)))
    record("inferred", length(types) == 1 && isconcretetype(only(types)), "inferred $types")

    allocated = allocations(prepared, f)
    record("allocation-free", allocated == 0, "$allocated bytes")

    derivative = ForwardDiff.derivative(t -> expectation(prepared, x -> t * f(x)), 1.0)
    record("duals pass", agrees(derivative, value, 1.0e-12), "d/dt E[t f] = $derivative, E[f] = $value")

    return checks
end

function check_product(backend, a::Case, b::Case)
    joint = FactorizedJoint((a.q, b.q))
    f = ((x, y),) -> a.f(x) * b.f(y)
    supports(backend, f, joint) || return Check("product", 0, false, "the backend does not support the product")
    # a product of sampled components would multiply their sample sizes
    isstochastic(backend, f, joint) && return Check("product", 0, true, "skipped for a stochastic backend")
    value = expectation(backend, f, joint)
    expected = a.expected * b.expected
    rtol = 2 * max(a.rtol, b.rtol)
    return Check("product", 0, agrees(value, expected, rtol), "got $value, expected $expected, rtol $rtol")
end

agrees(value, expected, rtol) = isapprox(value, expected; rtol = rtol, atol = rtol)

# A function compiled for the types of its arguments, so that only the evaluation is measured
function allocations(prepared::P, f::F) where {P, F}
    expectation(prepared, f)
    return @allocated expectation(prepared, f)
end

end
