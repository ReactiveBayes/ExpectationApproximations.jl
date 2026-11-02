@testsnippet BrokenBackends begin
    using Distributions
    const EA = ExpectationApproximations
    const P = ExpectationApproximations.BayesBase

    # Wrong points: shifted by one
    struct Shifted end
    P.supports(::Shifted, ::Type, ::Type{<:Normal}) = true
    P.prepare_expectation(::Shifted, q::Normal) = P.PointSet(mean(q) .+ 1 .+ std(q) .* [-1.0, 1.0], [0.5, 0.5])

    # Claims to be exact, is not: two points are exact up to degree three only
    struct Overclaims end
    P.supports(::Overclaims, ::Type, ::Type{<:Normal}) = true
    P.haslabel(::Overclaims, ::P.Exact) = true
    P.prepare_expectation(::Overclaims, q::Normal) = P.PointSet(mean(q) .+ std(q) .* [-1.0, 1.0], [0.5, 0.5])

    # Allocates on every evaluation
    struct Allocating{T}
        set::T
    end
    P.supports(::Allocating, ::Type, ::Type{<:Normal}) = true
    P.prepare_expectation(::Allocating, q::Normal) = Allocating(P.PointSet(mean(q) .+ std(q) .* [-1.0, 1.0], [0.5, 0.5]))
    P.expectation(prepared::Allocating, f) = sum(collect(map(f, prepared.set.points)) .* prepared.set.weights)

    failed(checks) = Set(check.name for check in checks if !check.passed)
end

@testitem "Testing: a correct backend passes every check" tags = [:testing] setup = [BrokenBackends] begin
    cases = [EA.Testing.Case(Normal(0.5, 1.5), x -> x^2, 0.5^2 + 1.5^2), EA.Testing.Case(Normal(0.0, 1.0), cos, exp(-1 / 2); rtol = 1.0e-8)]
    checks = EA.Testing.conformance(EA.GaussHermite(16), cases)
    @test !isempty(checks)
    @test isempty(failed(checks))
    names = Set(check.name for check in checks)
    for name in ("supports", "correct", "prepared equals direct", "inferred", "allocation-free", "duals pass", "product")
        @test name in names
    end
end

@testitem "Testing: each kind of broken backend fails the right check" tags = [:testing] setup = [BrokenBackends] begin
    case = [EA.Testing.Case(Normal(0.0, 1.0), x -> x^4, 3.0)]
    @test "correct" in failed(EA.Testing.conformance(Shifted(), case))
    @test "exact label holds" in failed(EA.Testing.conformance(Overclaims(), case))
    @test "allocation-free" in failed(EA.Testing.conformance(Allocating(nothing), [EA.Testing.Case(Normal(0.0, 1.0), x -> x^2, 1.0)]))
    @test "supports" in failed(EA.Testing.conformance(EA.GaussHermite(4), [EA.Testing.Case(Gamma(2.0, 1.0), identity, 2.0)]))
end

@testitem "Testing: stochastic backends are reproducible from the same RNG" tags = [:testing] setup = [BrokenBackends] begin
    checks = EA.Testing.conformance(EA.MonteCarlo(20_000), [EA.Testing.Case(Gamma(2.0, 2.0), identity, 4.0; rtol = 0.05)])
    @test "reproducible" in Set(check.name for check in checks)
    @test isempty(failed(checks))
end

@testitem "Testing: test_backend records every check as a test" tags = [:testing] setup = [BrokenBackends] begin
    using Test
    results = Test.@testset "inner" begin
        EA.Testing.test_backend(EA.GaussHermite(8), [EA.Testing.Case(Normal(0.0, 1.0), abs2, 1.0)])
    end
    @test results.n_passed + length(results.results) > 0
end
