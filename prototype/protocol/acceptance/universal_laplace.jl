# Acceptance test of U4 (PLAN.md): the two notebooks' models inferred with the package pieces,
# DeltaMessagePassingRules' `ExpectationRuleFallback` and `LaplaceForm` given once as RxInfer's
# options, in place of the notebooks' own fallback and form constraints. The posteriors must be
# the notebooks'. Run in the universal-dev environment:
#
#     julia --project=~/Projects/Julia/universal-dev universal_laplace.jl

using Distributions, BayesBase, Printf

function load(path)
    mod = Module()
    redirect_stdout(devnull) do
        Base.include(mod, joinpath(@__DIR__, path))
    end
    return mod
end

# A Gaussian's or a transformed Gaussian's location and spread, in its unconstrained coordinates:
# the notebooks give `LogNormal`s and `LogitNormal`s, the package `TransformedGaussian`s.
coordinates(d::Union{LogNormal, LogitNormal}) = collect(params(d))
coordinates(d::AbstractMvNormal) = vcat(mean(d), vec(cov(d)))
coordinates(d::Dirichlet) = collect(params(d)[1])
coordinates(d) = hasproperty(d, :gaussian) ? coordinates(d.gaussian) : vcat(mean(d), std(d))
coordinates(d::Normal) = [mean(d), std(d)]

difference(a, b) = maximum(abs.(coordinates(a) .- coordinates(b)) ./ max.(abs.(coordinates(a)), 1.0))

const UNIVERSAL = Dict(
    "mixture" => :(
        infer(
            model = laplace_mixture(K = K), data = (y = y,),
            constraints = @constraints(
                begin
                    q(π, m, s, z) = q(π)q(m)q(s)q(z)
                end
            ),
            initialization = laplace_initialization, iterations = n_iterations, returnvars = (m = KeepLast(), s = KeepLast(), π = KeepLast()),
            options = (
                rulefallback = DeltaMessagePassingRules.ExpectationRuleFallback(),
                default_form_constraint = DeltaMessagePassingRules.LaplaceForm(), context = (expectation = notebook_router(5),),
            ),
        )
    ),
    "zip" => :(
        infer(
            model = monastery(prior_p = prior_p, prior_λ = prior_λ), data = (y = y,),
            constraints = @constraints(
                begin
                    q(p, λ) = q(p)q(λ)
                end
            ),
            initialization = initialization, iterations = n_iterations, returnvars = KeepLast(),
            options = (
                rulefallback = DeltaMessagePassingRules.ExpectationRuleFallback(),
                default_form_constraint = DeltaMessagePassingRules.LaplaceForm(), context = (expectation = notebook_router(16),),
            ),
        )
    ),
)

const COMPARED = Dict(
    "mixture" => [(:m, :q_m), (:s, :q_s), (:π, :q_π)],
    "zip" => [(:p, :q_p), (:λ, :q_λ)],
)

passed = true
for name in ("mixture", "zip")
    notebook = load("$(name)_protocol.jl")
    Core.eval(notebook, :(using DeltaMessagePassingRules))
    universal = Base.invokelatest(Core.eval, notebook, UNIVERSAL[name])
    println(name)
    for (variable, reference) in COMPARED[name]
        δ = Base.invokelatest(difference, universal.posteriors[variable], Core.eval(notebook, reference))
        ok = δ <= 1.0e-6
        global passed &= ok
        @printf("  %-4s max relative difference to the notebook %.1e  %s\n", variable, δ, ok ? "ok" : "FAILED")
    end
end

println(passed ? "universal Laplace: PASSED" : "universal Laplace: FAILED")
passed || exit(1)
