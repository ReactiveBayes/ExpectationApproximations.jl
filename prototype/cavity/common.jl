# Shared by the prototype's scripts: the engine, and checks that print ok or FAILED.

include(joinpath(@__DIR__, "Cavity.jl"))
using .Cavity
using BayesBase, Distributions, LinearAlgebra, Printf, StableRNGs

passed = true
function check(name, ok, detail)
    global passed &= ok
    @printf("  %-66s %s  %s\n", name, ok ? "ok" : "FAILED", detail)
end
function finish(title)
    println(passed ? "$title: PASSED" : "$title: FAILED")
    passed || exit(1)
end

rms(a) = sqrt(sum(abs2, a) / length(a))
scalar(q) = (only(mean(q)), sqrt(only(cov(q))))

const POLICIES = (LaplacePropagation(), ExpectationPropagation(), CVIPropagation())
policyname(p) = string(nameof(typeof(p)))

# A scalar chain x_1 ~ N(m1, v1), x_t ~ N(φ x_{t-1}, σ²), with one factor per observation, made
# by `observation(t)`
function scalar_chain(T, m1, v1, φ, σ², observation)
    graph = Graph()
    for t in 1:T
        variable!(graph, t, 1)
    end
    factor!(graph, Prior(MvNormal([m1], fill(v1, 1, 1))), 1)
    for t in 2:T
        factor!(graph, LinearGaussian(fill(φ, 1, 1), [0.0], fill(σ², 1, 1)), t, t - 1)
    end
    for t in 1:T
        factor!(graph, observation(t), t)
    end
    return graph
end
