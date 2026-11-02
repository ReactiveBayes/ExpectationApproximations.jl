# Acceptance test of the protocol prototype (U0.2, item 8 of the plan): the notebooks' fallback
# rewritten with `BayesBase.prepare_expectation` and `BayesBase.expectation` must give the same
# posteriors as the hand-written quadrature, to 1e-10. Run in the universal-dev environment:
#
#     julia --project=~/Projects/Julia/universal-dev compare.jl

using Distributions, BayesBase, Printf

function load(path)
    mod = Module()
    seconds = @elapsed redirect_stdout(devnull) do
        Base.include(mod, joinpath(@__DIR__, path))
    end
    return mod, seconds
end

flat(d) = reduce(vcat, map(p -> vec(collect(float.(p))), params(d)))
flat(d::Union{BayesBase.PointMass, Real}) = [float(mean(d))]
flat(ds::AbstractVector) = reduce(vcat, map(flat, ds))

function difference(a, b)
    x, y = flat(a), flat(b)
    length(x) == length(y) || return Inf
    return maximum(abs.(x .- y) ./ max.(abs.(x), 1.0))
end

# the main inference call of each notebook, rerun to time it once compiled
const RERUN = Dict(
    "mixture" => :(infer(
        model = laplace_mixture(K = K), data = (y = y,), constraints = laplace_constraints,
        initialization = laplace_initialization, iterations = n_iterations,
        returnvars = (m = KeepEach(), s = KeepEach(), π = KeepLast()),
        options = (rulefallback = ExpectedLogPdfFallback(),),
    )),
    "zip" => :(infer(
        model = monastery(prior_p = prior_p, prior_λ = prior_λ), data = (y = y,),
        constraints = transformed_constraints, initialization = initialization, iterations = n_iterations,
        returnvars = KeepEach(), options = (rulefallback = ExpectedLogPdfFallback(),),
    )),
)

# posteriors to compare, by notebook
const COMPARED = Dict(
    "mixture" => [
        :(result.posteriors[:m]), :(result.posteriors[:s]), :(result.posteriors[:π]),
    ],
    "zip" => [
        :(result.posteriors[:p]), :(result.posteriors[:λ]), :(small_p), :(conjugate_result.posteriors[:λ]),
        :(moments(q_p)[1]), :(moments(q_λ)[2]),
    ],
)

passed = true
for name in ("mixture", "zip")
    original, t_original = load("$(name)_original.jl")
    protocol, t_protocol = load("$(name)_protocol.jl")
    println(name)
    for expression in COMPARED[name]
        # the notebooks load packages after this file was compiled, hence `invokelatest`
        δ = Base.invokelatest(difference, Core.eval(original, expression), Core.eval(protocol, expression))
        ok = δ <= 1e-10
        global passed &= ok
        @printf("  %-40s max relative difference %.1e  %s\n", string(expression), δ, ok ? "ok" : "FAILED")
    end
    rerun(mod) = minimum(_ -> @elapsed(Core.eval(mod, RERUN[name])), 1:3)
    @printf("  whole script: original %.1f s, protocol %.1f s (with compilation)\n", t_original, t_protocol)
    @printf("  main inference, compiled: original %.3f s, protocol %.3f s\n", rerun(original), rerun(protocol))
end

println(passed ? "acceptance: PASSED" : "acceptance: FAILED")
passed || exit(1)
